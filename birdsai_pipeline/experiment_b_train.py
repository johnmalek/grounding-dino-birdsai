from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForZeroShotObjectDetection, AutoProcessor


STOP_REQUESTED = False
CLASS_NAMES = ["animal", "person"]
PROMPT = "animal. person."


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"Received signal {signum}; saving at the next optimizer boundary.", flush=True)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_coco(path: Path) -> tuple[list[dict], dict[int, list[dict]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in data["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    return data["images"], annotations_by_image


def balanced_subset(
    images: list[dict],
    annotations_by_image: dict[int, list[dict]],
    count: int,
    seed: int,
) -> list[dict]:
    if count < 2:
        raise ValueError("A balanced subset requires at least two images")

    by_category: dict[int, list[dict]] = {1: [], 2: []}
    for image in images:
        annotations = annotations_by_image[int(image["id"])]
        categories = {int(item["category_id"]) for item in annotations}
        if categories == {1}:
            by_category[1].append(image)
        elif categories == {2}:
            by_category[2].append(image)

    generator = random.Random(seed)
    for values in by_category.values():
        generator.shuffle(values)

    animal_count = count // 2
    person_count = count - animal_count
    if len(by_category[1]) < animal_count or len(by_category[2]) < person_count:
        raise ValueError("Not enough single-class images for the balanced subset")

    selected = by_category[1][:animal_count] + by_category[2][:person_count]
    generator.shuffle(selected)
    return selected


class BirdsAICocoDataset(Dataset):
    def __init__(
        self,
        images: list[dict],
        annotations_by_image: dict[int, list[dict]],
        dataset_root: Path,
    ) -> None:
        self.images = images
        self.annotations_by_image = annotations_by_image
        self.dataset_root = dataset_root

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_record = self.images[index]
        image_id = int(image_record["id"])
        image_path = self.dataset_root / image_record["file_name"]
        image = Image.open(image_path).convert("RGB")

        annotations = []
        for annotation in self.annotations_by_image[image_id]:
            bbox = [float(value) for value in annotation["bbox"]]
            annotations.append(
                {
                    "bbox": bbox,
                    "category_id": int(annotation["category_id"]) - 1,
                    "area": float(annotation.get("area", bbox[2] * bbox[3])),
                    "iscrowd": int(annotation.get("iscrowd", 0)),
                }
            )

        return {
            "image": image,
            "target": {"image_id": image_id, "annotations": annotations},
            "file_name": image_record["file_name"],
        }


class GroundingDinoCollator:
    def __init__(self, processor: Any, shortest_edge: int, longest_edge: int) -> None:
        self.processor = processor
        self.shortest_edge = shortest_edge
        self.longest_edge = longest_edge

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [sample["image"] for sample in samples]
        targets = [sample["target"] for sample in samples]

        vision = self.processor.image_processor(
            images=images,
            annotations=targets,
            size={
                "shortest_edge": self.shortest_edge,
                "longest_edge": self.longest_edge,
            },
            return_tensors="pt",
        )
        language = self.processor.tokenizer(
            [PROMPT] * len(samples),
            padding=True,
            return_tensors="pt",
        )

        batch = dict(vision)
        batch.update(language)
        batch["file_names"] = [sample["file_name"] for sample in samples]
        return batch


def freeze(module: torch.nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def configure_trainable_parameters(model: torch.nn.Module, configuration: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True

    freeze(model.model.text_backbone)

    if configuration == "B1":
        freeze(model.model.backbone)
    elif configuration == "B2":
        swin_backbone = model.model.backbone.conv_encoder.model
        swin = swin_backbone.swin
        freeze(swin.embeddings)
        freeze(swin.encoder.layers[0])
        freeze(swin.encoder.layers[1])
        freeze(swin_backbone.hidden_states_norms["stage2"])
    elif configuration == "B3":
        pass
    else:
        raise ValueError(f"Unsupported configuration: {configuration}")


def keep_frozen_modules_in_eval(model: torch.nn.Module, configuration: str) -> None:
    model.model.text_backbone.eval()
    if configuration == "B1":
        model.model.backbone.eval()
    elif configuration == "B2":
        swin_backbone = model.model.backbone.conv_encoder.model
        swin_backbone.swin.embeddings.eval()
        swin_backbone.swin.encoder.layers[0].eval()
        swin_backbone.swin.encoder.layers[1].eval()
        swin_backbone.hidden_states_norms["stage2"].eval()


def parameter_report(model: torch.nn.Module) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percentage": 100.0 * trainable / total,
    }


def optimizer_for(
    model: torch.nn.Module,
    learning_rate: float,
    backbone_learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    backbone = []
    detector = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("model.backbone"):
            backbone.append(parameter)
        else:
            detector.append(parameter)

    groups = []
    if detector:
        groups.append({"params": detector, "lr": learning_rate})
    if backbone:
        groups.append({"params": backbone, "lr": backbone_learning_rate})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denominator))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def move_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict, list[dict]]:
    model_inputs = {
        key: batch[key].to(device)
        for key in (
            "pixel_values",
            "pixel_mask",
            "input_ids",
            "attention_mask",
            "token_type_ids",
        )
        if key in batch
    }
    labels = [
        {
            "class_labels": target["class_labels"].to(device),
            "boxes": target["boxes"].to(device),
        }
        for target in batch["labels"]
    ]
    return model_inputs, labels


def trainable_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model_state": trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "training_state": state,
        "configuration": arguments.configuration,
        "model_id": arguments.model_id,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    arguments: argparse.Namespace,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["configuration"] != arguments.configuration:
        raise ValueError("Checkpoint configuration does not match the requested configuration")
    if payload["model_id"] != arguments.model_id:
        raise ValueError("Checkpoint model ID does not match the requested model")
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    return payload["training_state"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Grounding DINO on BIRDSAI.")
    parser.add_argument("--configuration", choices=["B1", "B2", "B3"], required=True)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--gradient-clip", type=float, default=0.1)
    parser.add_argument("--shortest-edge", type=int, default=384)
    parser.add_argument("--longest-edge", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--balanced-subset", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--log-steps", type=int, default=1)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "shortest_edge": args.shortest_edge,
        "longest_edge": args.longest_edge,
        "log_steps": args.log_steps,
        "save_steps": args.save_steps,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("Experiment B training requires an allocated CUDA GPU")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "last_checkpoint.pt"
    log_path = args.output_dir / "training_log.jsonl"
    metadata_path = args.output_dir / "metadata.json"

    images, annotations_by_image = load_coco(args.train_json)
    if args.balanced_subset is not None:
        images = balanced_subset(
            images,
            annotations_by_image,
            args.balanced_subset,
            args.seed,
        )

    dataset = BirdsAICocoDataset(images, annotations_by_image, args.dataset_root)
    processor = AutoProcessor.from_pretrained(args.model_id)
    collator = GroundingDinoCollator(
        processor,
        shortest_edge=args.shortest_edge,
        longest_edge=args.longest_edge,
    )

    config = AutoConfig.from_pretrained(args.model_id)
    config.disable_custom_kernels = True
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_id,
        config=config,
    )
    configure_trainable_parameters(model, args.configuration)
    if args.gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            raise ValueError(
                f"{model.__class__.__name__} does not support gradient "
                "checkpointing in the installed Transformers version. "
                "Remove --gradient-checkpointing."
            )
        model.gradient_checkpointing_enable()

    report = parameter_report(model)
    print(json.dumps(report, indent=2), flush=True)

    device = torch.device("cuda")
    model.to(device)
    optimizer = optimizer_for(
        model,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
    )

    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    planned_steps = args.epochs * updates_per_epoch
    if args.max_steps is not None:
        planned_steps = min(planned_steps, args.max_steps)
    scheduler = cosine_scheduler(optimizer, planned_steps, args.warmup_steps)

    state: dict[str, Any] = {
        "epoch": 0,
        "batch_in_epoch": 0,
        "global_step": 0,
        "optimizer_step": 0,
        "created_at_utc": utc_now(),
    }
    if args.resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Resume requested but checkpoint is missing: {checkpoint_path}")
        state = load_checkpoint(checkpoint_path, model, optimizer, scheduler, args)
        print(f"Resumed from {checkpoint_path}: {state}", flush=True)

    selected_summary = {
        "image_count": len(dataset),
        "annotation_count": sum(
            len(annotations_by_image[int(image["id"])]) for image in images
        ),
        "empty_image_count": sum(
            not annotations_by_image[int(image["id"])] for image in images
        ),
        "image_ids": [int(image["id"]) for image in images]
        if args.balanced_subset is not None
        else None,
    }
    run_metadata = {
        "experiment": "Experiment B",
        "configuration": args.configuration,
        "model_id": args.model_id,
        "prompt": PROMPT,
        "class_names": CLASS_NAMES,
        "disable_custom_kernels": True,
        "arguments": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(args).items()
        },
        "parameters": report,
        "dataset": selected_summary,
        "gpu": torch.cuda.get_device_name(0),
        "status": "running",
        "started_at_utc": utc_now(),
    }
    atomic_json(metadata_path, run_metadata)

    interrupted = False
    reached_limit = state["optimizer_step"] >= planned_steps
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(state["epoch"]), args.epochs):
        if reached_limit:
            break
        generator = torch.Generator()
        generator.manual_seed(args.seed + epoch)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

        model.train()
        keep_frozen_modules_in_eval(model, args.configuration)
        resume_batch = int(state["batch_in_epoch"]) if epoch == int(state["epoch"]) else 0

        for batch_index, batch in enumerate(loader):
            if batch_index < resume_batch:
                continue

            model_inputs, labels = move_batch(batch, device)
            outputs = model(**model_inputs, labels=labels)
            if not torch.isfinite(outputs.loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")

            remaining = len(loader) - (batch_index // args.gradient_accumulation) * args.gradient_accumulation
            accumulation_size = min(args.gradient_accumulation, remaining)
            (outputs.loss / accumulation_size).backward()
            state["global_step"] += 1

            update_now = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == len(loader)
            )
            if not update_now:
                continue

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad and parameter.grad is not None
                ],
                max_norm=args.gradient_clip,
            )
            if not torch.isfinite(gradient_norm):
                raise RuntimeError(
                    f"Non-finite gradient norm at epoch {epoch}, batch {batch_index}"
                )

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state["optimizer_step"] += 1
            state["epoch"] = epoch
            state["batch_in_epoch"] = batch_index + 1

            record = {
                "time_utc": utc_now(),
                "epoch": epoch,
                "batch": batch_index,
                "global_step": state["global_step"],
                "optimizer_step": state["optimizer_step"],
                "loss": float(outputs.loss.detach()),
                "losses": {
                    name: float(value.detach())
                    for name, value in outputs.loss_dict.items()
                },
                "gradient_norm": float(gradient_norm.detach()),
                "learning_rates": [group["lr"] for group in optimizer.param_groups],
                "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
            }
            append_jsonl(log_path, record)
            if state["optimizer_step"] % args.log_steps == 0:
                print(json.dumps(record), flush=True)

            if state["optimizer_step"] % args.save_steps == 0:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    state,
                    args,
                )
                print(f"Saved {checkpoint_path}", flush=True)

            reached_limit = state["optimizer_step"] >= planned_steps
            if reached_limit or STOP_REQUESTED:
                interrupted = STOP_REQUESTED and not reached_limit
                break

        epoch_complete = int(state["batch_in_epoch"]) >= len(loader)

        if epoch_complete:
            state["epoch"] = epoch + 1
            state["batch_in_epoch"] = 0

            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                state,
                args,
            )

            epoch_checkpoint_path = (
                args.output_dir
                / f"checkpoint_epoch_{epoch + 1}.pt"
            )

            save_checkpoint(
                epoch_checkpoint_path,
                model,
                optimizer,
                scheduler,
                state,
                args,
            )

            print(
                f"Saved completed epoch checkpoint: "
                f"{epoch_checkpoint_path}",
                flush=True,
            )

        if reached_limit or interrupted:
            break

    save_checkpoint(checkpoint_path, model, optimizer, scheduler, state, args)
    run_metadata["status"] = "interrupted" if interrupted else "complete"
    run_metadata["finished_at_utc"] = utc_now()
    run_metadata["training_state"] = state
    run_metadata["checkpoint"] = str(checkpoint_path)
    atomic_json(metadata_path, run_metadata)
    print(json.dumps(run_metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()

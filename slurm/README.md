# Slurm jobs

All jobs source common.sh and use BIRDSAI environment variables rather than
user-specific paths. Create logs/ before submitting because Slurm opens stdout
and stderr before the job script runs.

Recommended order:

1. prepare_birdsai.slurm
2. experiment_a_validation.slurm
3. experiment_a_test.slurm
4. experiment_b_train.slurm
5. experiment_b_validation.slurm
6. experiment_b_continue_b2.slurm for the epoch ablation
7. experiment_b_test.slurm for the frozen final evaluation

Cluster partition, GPU GRES, memory, and time directives are site-specific.
Adjust only the SBATCH resource lines when moving to another cluster.

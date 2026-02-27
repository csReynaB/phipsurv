#!/bin/bash
#SBATCH --job-name=survival             # Job name
#SBATCH --output=log/survival_%A_%a.out      # Output log file
#SBATCH --error=log/survival_%A_%a.err       # Error log file
#SBATCH --time=03:00:00                # Maximum runtime (hh:mm:ss)
#SBATCH --mem=4G
#SBATCH --ntasks=1                    # Number of tasks (1 task for a single script)
#SBATCH --nodes 1
#SBATCH --cpus-per-task=5             # Number of CPU cores per task


# run it like: sbatch --array=1-100 run_survival.sh PATH/seeds_file.txt PATH/config_file.yaml  $PATH/results/ project_results
set -euo pipefail

# Load the conda module if necessary
module load Conda
conda activate /lisc/data/scratch/ccr/conda_envs/ML_env

# -------------------------
# Inputs
# -------------------------
SEEDS_FILE=$1
CONFIG_FILE=$2
RESULTS_DIR=$3
RESULTS_NAME=$4

# -------------------------
# Compute seed from array ID
# -------------------------
line_number=${SLURM_ARRAY_TASK_ID}
seed=$(sed -n "${line_number}p" "${SEEDS_FILE}")

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Using seed: ${seed}"

# -------------------------
# Run Python module
# -------------------------
python -m phipsurv.cli.train_test \
                                  --seed $seed \
                                  --config $CONFIG_FILE \
                                  --run_nested_cv True  \
                                  --subgroup all \
                                  --with_oligos True \
                                  --with_additional_features True  \
                                  --prevalence_threshold_min 0 \
                                  --prevalence_threshold_max  100 \
                                  --train '{"treatment":"ICI"}' \
                                  --validate '{"treatment":"TKI"}' HCC-ICI_HCC-TKI \
                                  --validate '{"treatment":"e_ICI"}'  HCC-ICI_HCC-eICI \
                                  --out_name $RESULTS_NAME  \
                                  --out_dir  $RESULTS_DIR  \
                                  --outer_cv_split 5 \
                                  --inner_cv_split 3 \
                                  --max_timepoint 25

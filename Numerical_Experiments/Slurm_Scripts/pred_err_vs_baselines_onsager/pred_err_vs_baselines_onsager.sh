#!/bin/bash
#SBATCH -A stat-users
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH -t 10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --array=0-149                    # 3 n_values x 50 seeds = 150 tasks
#SBATCH -o /home/nandy.15/Research/Experiments_DAIF/GMM/Slurm_Scripts/pred_err_vs_baselines_onsager/Output_Messages/slurm_out_%A_%a_%x_%j_%t.out
#SBATCH -e /home/nandy.15/Research/Experiments_DAIF/GMM/Slurm_Scripts/pred_err_vs_baselines_onsager/Error_Messages/slurm_err_%A_%a_%x_%j_%t.err

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate multiview-regression || { echo "conda activate failed"; exit 1; }

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
export PYTHONNOUSERSITE=1
export R_HOME="$CONDA_PREFIX/lib/R"
export R_LIBS_USER="$CONDA_PREFIX/lib/R/library"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "Python  : $(which python)"
echo "Conda   : $CONDA_PREFIX"
python -V

cd /home/nandy.15/Research/Experiments_DAIF/GMM

# --- array math: map task ID to (n, seed) ---
n_values=(3000 3500 4000)
num_trials=50

idx=$SLURM_ARRAY_TASK_ID
n_idx=$(( idx / num_trials ))
seed=$(( idx % num_trials ))
n=${n_values[$n_idx]}

echo "Running n=$n; seed=$seed"

"$CONDA_PREFIX/bin/python" \
  "/home/nandy.15/Research/Experiments_DAIF/GMM/Python_Scripts/pred_err_vs_baselines_onsager.py" \
  "$n" "$seed"

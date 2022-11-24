#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=clipFT
#SBATCH --output=../logs/clipFT_train_%j.out

singularity \
    exec \
    --nv --overlay /scratch/sk8974/envs/cv/cv.ext3:ro \
    /scratch/sk8974/envs/cv/cv.sif \
    /bin/bash -c "
source /ext3/env.sh
python train_finetune_check.py --folder /vast/sk8974/experiments/cv_proj/data/fashion_FT/train/ --folder_val /vast/sk8974/experiments/cv_proj/data/fashion_FT/val/ --num_workers 10 --batch_size 50 --gpus 0,"
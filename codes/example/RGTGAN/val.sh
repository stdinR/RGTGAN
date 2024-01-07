ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH

partition=Test
your_model=
dataset=val_1st

srun -p ${partition} --gres=gpu:1 python -u val.py \
  --model_RGTNet_path=${your_model} \
  --exp_name=RGTGAN \
  --dataset=${dataset}$ \
  --save_path=./results



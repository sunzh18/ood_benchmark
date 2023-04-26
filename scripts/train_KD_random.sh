
in_dataset=$1
model=$2
wandb=$3
GPU=1
# out_dataset="SUN"
# in_dataset=CIFAR-100

model_path="checkpoints/network/baseline/CIFAR-10/resnet18_parameter.pth"
# model_path="checkpoints/network/baseline/CIFAR-10/wrn_parameter.pth"

CUDA_VISIBLE_DEVICES=${GPU} python train_distillation.py \
--model ${model} \
--name KD_random_init \
--batch 256 \
--in_dataset ${in_dataset} \
--lr 0.2 \
--epochs 200 \
--model_path ${model_path} \
--wandb ${wandb} 

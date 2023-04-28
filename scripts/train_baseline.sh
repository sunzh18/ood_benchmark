
in_dataset=$1
model=$2
wandb=$3

arch='norm_relu_x10'
GPU=0

# out_dataset="SUN"
# in_dataset=CIFAR-100

# model_path="checkpoints/network/resnet18_cifar10.pth"
model_path="checkpoints/network"

CUDA_VISIBLE_DEVICES=${GPU} python train_baseline.py \
--model ${model} \
--name baseline \
--batch 64 \
--in_dataset ${in_dataset} \
--lr 0.1 \
--epochs 100 \
# --wandb ${wandb} 
# --arch ${arch}

# --model_path ${model_path}


in_dataset=$1
model=$2
GPU=0
# out_dataset="SUN"
# in_dataset=CIFAR-100
name=$3
model_path="checkpoints/network/resnet18_cifar10.pth"

CUDA_VISIBLE_DEVICES=${GPU} python train_baseline.py \
--model ${model} \
--name ${name} \
--batch 128 \
--in_dataset ${in_dataset} \
--test True \
--model_path "checkpoints/network"

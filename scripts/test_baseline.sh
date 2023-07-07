
in_dataset=$1
model=$2
SCORE=$3
GPU=$4
ood_data="/data15/data15_5/Public/Datasets"
out_dataset="SUN"
# model_path="checkpoints/network/resnet18_cifar10.pth"
model_path="checkpoints/network"

# arch='norm_relu_x10'

CUDA_VISIBLE_DEVICES=${GPU} python test_baselines.py \
--model ${model} \
--name baseline \
--batch 64 \
--in_dataset ${in_dataset} \
--logdir result \
--score ${SCORE} \
--model_path ${model_path} \
--p 70
# --arch ${arch}
# --out_dataset ${out_dataset} \

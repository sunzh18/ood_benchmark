
in_dataset=$1
SCORE=$2
OUT_DATA=$3
ood_data="/data15/data15_5/Public/Datasets"
GPU=0
out_dataset="SUN"
# model_path="checkpoints/network/resnet18_cifar10.pth"
model_path="checkpoints/network"

# arch='norm_relu_x10'

CUDA_VISIBLE_DEVICES=${GPU} python test_dice.py \
--model densenet \
--name baseline \
--batch 128 \
--in_dataset ${in_dataset} \
--logdir result \
--score ${SCORE} \
--model_path ${model_path} \
 --p 90 \
# --arch ${arch}
# --out_dataset ${out_dataset} \


# in_dataset=$1
in_dataset=$1
model=$2
# SCORE=$2
# BATS=$3

GPU=$3
out_dataset="SUN"
# model_path="checkpoints/network/resnet18_cifar10.pth"
model_path="checkpoints/network"

# name='KD_teacher_init'
# name='KD_random_init'
name='baseline'

CUDA_VISIBLE_DEVICES=${GPU} python precompute.py \
--model ${model} \
--name ${name} \
--batch 32 \
--in_dataset ${in_dataset} \
--logdir result \
--model_path ${model_path} \
# --score ${SCORE} \
# --bats ${BATS} 

# --out_dataset ${OUT_DATA} \

# in_dataset=$1
in_dataset="CIFAR-10"
model=$1
BATS=$2

GPU=0
out_dataset="SUN"
# model_path="checkpoints/network/resnet18_cifar10.pth"
model_path="checkpoints/network"

# name='KD_teacher_init'
# name='KD_random_init'
name='baseline'

CUDA_VISIBLE_DEVICES=${GPU} python feature_analysis.py \
--model ${model} \
--name ${name} \
--batch 64 \
--in_dataset ${in_dataset} \
--logdir result \
--model_path ${model_path} \
--bats ${BATS} 
# --score ${SCORE} \
# --out_dataset ${OUT_DATA} \
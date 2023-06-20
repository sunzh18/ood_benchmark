
# in_dataset=$1
in_dataset=$1
model=$2
SCORE=$3
p=$4
# BATS=$4

GPU=$5
out_dataset="SUN"
# model_path="checkpoints/network/resnet18_cifar10.pth"
model_path="checkpoints/network"
# logdir='analysis_feature/prun_num'
logdir='result'
name='baseline'

CUDA_VISIBLE_DEVICES=${GPU} python my_gradcam_score.py \
--model ${model} \
--name ${name} \
--batch 1 \
--in_dataset ${in_dataset} \
--logdir ${logdir} \
--model_path ${model_path} \
--score ${SCORE} \
--p ${p}
# --bats ${BATS} 

# --out_dataset ${OUT_DATA} \
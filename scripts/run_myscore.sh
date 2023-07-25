
# in_dataset=$1
in_dataset=$1
model=$2
SCORE=$3
p=$4
cos=$5

GPU=$6
out_dataset="SUN"
# model_path="checkpoints/network/resnet18_cifar10.pth"
model_path="checkpoints/network"
# logdir='analysis_feature/prun_num'
logdir='result'
name='baseline'

CUDA_VISIBLE_DEVICES=${GPU} python my_classmean_score.py \
--model ${model} \
--name ${name} \
--batch 64 \
--in_dataset ${in_dataset} \
--logdir ${logdir} \
--model_path ${model_path} \
--score ${SCORE} \
--p ${p}    \
--cos ${cos} 

# --out_dataset ${OUT_DATA} \
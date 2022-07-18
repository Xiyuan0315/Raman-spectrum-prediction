#!bin/bash

# CNN
# CUDA_VISIBLE_DEVICES=1 python main_task2.py --isTrain --net CNN --dataset CancerDetection_CNN --isRandom \
#     --exp_name Task2_CNN_v1 --weight_decay 0 --lr 0.01

#-**********************************************************************************-#
# Taks 1
#-**********************************************************************************-#
#-********************************************
# MLP
# CE loss
CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --isTrain --net MLP_CE \
    --task Task_3 \
    --filename data_4.mat \
    --exp_name Task3_MLP_CE \
    --weight_decay 0 --lr 0.001 --epoch 250 --milestones 50_100 --reweight 1

# model - Task1_MLP_CE_v1
# test uniq and non-uniq
# CUDA_VISIBLE_DEVICES=1 python main_task1_CE.py --net MLP_CE \
#     --model_dir ./experiments/Task1_MLP_CE_v1/model_best.pth --isUniq

# CUDA_VISIBLE_DEVICES=1 python main_task1_CE.py --net MLP_CE \
#     --model_dir ./experiments/Task1_MLP_CE_v1/model_best.pth


#!bin/bash

#-********************************************
# MLP
# CE loss, 89.0 / 83.3
# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --net MLP_CE --task Task_1 --filename data_1.mat \
#     --exp_name Task1_MLP_CE --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1

# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --net MLP_CE_attention --task Task_1 --filename data_1.mat \
#     --exp_name Task1_MLP_CE_attention --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1


# CNN
# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --isCNN --net CNN_CE --task Task_1 --filename data_1.mat \
#     --exp_name Task1_CNN --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1 --milestones 50_100

# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --isCNN --net CNN_CE_attention --task Task_1 --filename data_1.mat \
#     --exp_name Task1_CNN_attention --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1 --milestones 50_100

# Testing
# CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --net MLP_CE --task Task_1 --filename data_1.mat --model_dir ./experiments/Task1_MLP_CE/model_best.pth
# CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --net MLP_CE_attention --task Task_1 --filename data_1.mat --model_dir ./experiments/Task1_MLP_CE_attention/model_best.pth
# CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --isCNN --net CNN_CE --task Task_1 --filename data_1.mat --model_dir ./experiments/Task1_CNN_2/model_best.pth
CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --isCNN --net CNN_CE_attention --task Task_1 --filename data_1.mat --model_dir ./experiments/Task1_CNN_attention/model_best.pthf
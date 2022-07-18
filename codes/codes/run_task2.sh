#!bin/bash

# CNN
# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --isCNN --net CNN_CE --task Task_2 --filename data.mat \
#     --exp_name Task2_CNN --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1 --milestones 50_100
# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --isCNN --net CNN_CE_2 --task Task_2 --filename data.mat \
#     --exp_name Task2_CNN_CE --weight_decay 5e-4 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1 --milestones 50_100
# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --isCNN --net CNN_CE_2 --task Task_2 --filename data.mat \
#     --exp_name Task2_CNN_CE_4 --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1 --milestones 50_100

#-********************************************
# MLP
# CE loss, 89.0 / 83.3
# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --net MLP_CE --task Task_2 --filename data.mat \
#     --exp_name Task2_MLP_CE --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1

# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --net MLP_CE_2 --task Task_2 --filename data.mat \
#     --exp_name Task2_MLP_CE --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1

# CUDA_VISIBLE_DEVICES=0 python main_task_CE.py --isTrain --isRandom --net MLP_CE_2 --task Task_2 --filename data2.mat \
#     --exp_name Task2_MLP_CE_2 --weight_decay 0 --lr 0.001 --epoch 150 --milestones 50_100 --reweight 1


# Testing
# CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --net MLP_CE_2 --task Task_2 --filename data.mat --model_dir ./experiments/Task2_MLP_CE/model_best.pth
# CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --isCNN --net CNN_CE --task Task_2 --filename data.mat --model_dir ./experiments/Task2_CNN_CE/model_best.pth

CUDA_VISIBLE_DEVICES=1 python main_task_CE.py --isCNN --net CNN_CE_2 --task Task_2 --filename data.mat --model_dir ./experiments/Task2_CNN_CE_4/model_best.pth
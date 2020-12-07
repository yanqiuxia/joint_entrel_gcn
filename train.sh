#!/bin/bash
PYTHON_HOME="/home/eleanor/anaconda3/envs/pytorch-gpu/bin"
APP_HOME="/home/eleanor/yan.qiuxia/PycharmProjects/joint_entrel_gcn"
CUDA_VISIBLE_DEVICES=0 nohup $PYTHON_HOME/python -u $APP_HOME/my_train.py > $APP_HOME/logs/baidu_1009.txt 2>&1 &
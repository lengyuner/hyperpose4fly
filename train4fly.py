





import os
import cv2
import sys
import time
import math
import json
import glob
import argparse
import matplotlib
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config,Model,Dataset




os.environ['CUDA_VISIBLE_DEVICES']='0,1'
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[1], True)



model_type = "Openpose"
# model_backbone = "Default"
# model_backbone = "Mobilenetv2"
model_backbone = "Resnet18"
# MobilenetV1.MobilenetV2.Vggtiny.Vgg16.Vgg19.Resnet18.Resnet50

dataset_type = "USERDEF"
model_name = dataset_type+'_'+model_type+'_' + model_backbone + '_'+time.strftime("%Y%m%d_%H%M")
dataset_version ="2020"
dataset_path = "./data"
train_type = "Single_train"
kf_optimizer ='Sync_avg'
userdef_dataset = 1
useradd_data_path = "./data"

domainadapt_data_path = None

optim_type ="Adam"
learning_rate =1e-4
log_interval =1e2
save_interval =5e3

Config.set_model_name(model_name)
Config.set_model_type(Config.MODEL[model_type])
Config.set_model_backbone(Config.BACKBONE[model_backbone])
Config.set_log_interval(log_interval)
Config.set_save_interval(save_interval)
# config train
Config.set_train_type(Config.TRAIN[train_type])
Config.set_learning_rate(learning_rate)
Config.set_optim_type(Config.OPTIM[optim_type])
Config.set_kungfu_option(Config.KUNGFU[kf_optimizer])

# config dataset
print(f"test enabling dataset:{userdef_dataset}")
Config.set_userdef_dataset(userdef_dataset)
Config.set_dataset_type(Config.DATA[dataset_type])
Config.set_dataset_path(dataset_path)
Config.set_dataset_version(dataset_version)

print('\n' * 4)



# train
config = Config.get_config()

config.pretrain.batch_size=4
config.eval.batch_size=1
config.train.batch_size=4
config.model.hin = 48*2
config.model.win = 64*2
config.model.hout = 6*2 #48/8
config.model.wout = 8*2 #64/8
config.train.save_interval = 5000
config.train.n_step = 60000
# config.train.lr_decay_every_step = 2000

print(config.model)
print('\n' * 2)
print(config.train)
print('\n' * 2)
print(config.data)
print('\n' * 4)




model = Model.get_model(config)

train = Model.get_train(config)

dataset = Dataset.get_dataset(config)


train(model, dataset)








import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from hyperpose import Config,Model,Dataset
from hyperpose.Dataset import imread_rgb_float,imwrite_rgb_float
matplotlib.get_backend()
import os
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
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[1], True)



model_type = "Openpose"
# model_backbone = "Default"
model_backbone = "Mobilenetv2"
# MobilenetV1.MobilenetV2.Vggtiny.Vgg16.Vgg19.Resnet18.Resnet50
dataset_type = "USERDEF"
# model_name = dataset_type+'_'+model_type+'_' + model_backbone + '_'+time.strftime("%Y%m%d_%H%M")
model_name = dataset_type+'_'+time.strftime("%Y%m%d_%H%M")
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

config.train.save_interval = 5000
config.train.n_step = 100000
print(config.model)
print('\n' * 2)
print(config.train)
print('\n' * 2)
print(config.data)
print('\n' * 4)




model = Model.get_model(config)

# Config.set_model_name("default_name")
# # Config.set_model_name("openpose_test_20201028")
# # Config.set_model_type(Config.MODEL.Openpose)
# Config.set_model_type(Config.MODEL.LightweightOpenpose)
# Config.set_model_backbone(Config.BACKBONE.Resnet50)
# config=Config.get_config()
# print(config.model)
# # config.model
#
#
# #get and load model
# model=Model.get_model(config)

weight_path = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_20201121_1621/model_dir/newest_model.npz'
weight_path = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_Openpose_Mobilenetv2_20201124_2030/model_dir/newest_model.npz'

'''
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
'''
weight_path = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_Openpose_Resnet18_20201201_1927/model_dir/newest_model.npz' # 这个结果很不错
# 只有四個點，最後一個點不准


'''






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

# train = Model.get_train(config)
# 
# dataset = Dataset.get_dataset(config)
# 
# 
# train(model, dataset)

'''

weight_path = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_Openpose_Resnet18_20201202_2325/model_dir/newest_model.npz' # 这个结果很不错


model.load_weights(weight_path)



import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
# C:\Users\ps\Desktop\djz\hyperpose\save_dir\USERDEF_Openpose_Resnet18_20201130_2259\train_vis_dir
# UserdefColor = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0]]
UserdefColor = [[1, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0, 1],[0, 0, 1], [0, 1, 0]]

save_all_pic = 1
if save_all_pic == 1:
    path_from = './data/egg/fly/train2020/'
    # C:\Users\ps\Desktop\djz\hyperpose\data\egg\fly\train2020
    # path_from = './data/fly/train2020/'
    # path_from = 'H:/data/picture_fly/2/'
    # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_20201121_1621/val/'
    # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_20201121_1621/val_head_output/'
    # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_Openpose_Mobilenetv2_20201124_2030/val_output/'
    # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/save_dir/USERDEF_Openpose_Mobilenetv2_20201124_2030/val_head_output/'
    # C:\Users\ps\Desktop\djz\hyperpose\save_dir\USERDEF_Openpose_Resnet18_20201130_1642\model_dir
    # path_to = 'C:/Users/ps/Desktop/djz/hyperpose/rubbish/val_head_output/'
    # path_ta = 'C:\Users\ps\Desktop\djz\hyperpose\save_dir\USERDEF_Openpose_Resnet18_20201130_2259\val_head_output/'
    # path_to = './save_dir/USERDEF_Openpose_Resnet18_20201202_2325/1/'
    path_to = './save_dir/USERDEF_Openpose_Resnet18_20201223_2319/train_all/'
    path_to = './save_dir/USERDEF_Openpose_Resnet18_20201223_2319/train_all_3/'

    read_name = os.listdir(path_from)
    number_of_pic = len(read_name)

    start_time = time.time()
    for K_0 in range(number_of_pic):#(10):#(number_of_pic):# K_0=0
        img_name = read_name[K_0]
        img0 = cv2.imread(path_from+img_name)
        # img1 = cv2.resize(img0, (432, 368))
        # img1 = cv2.resize(img0, (384, 288))
        img1 = cv2.resize(img0, (128, 96))
        # img1 = np.copy(img0)
        # img1 = cv2.resize(img0, (96, 96))#TODO(JZ)
        # plt.imshow(img1)
        ori_image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        input_image = ori_image.astype(np.float32) / 255.0
        if (model.data_format == "channels_first"):
            input_image = np.transpose(input_image, [2, 0, 1])
        img_c, img_h, img_w = input_image.shape

        conf_map, paf_map = model.infer(input_image[np.newaxis, :, :, :])
        find_all_part = 0
        if find_all_part == 1:
            visualize = Model.get_visualize(Config.MODEL.Openpose)
            # def visualize(img,conf_map,paf_map,save_name="maps",save_dir="./save_dir/vis_dir",data_format="channels_first",save_tofile=True):
            vis_parts_heatmap, vis_limbs_heatmap = visualize(input_image, conf_map[0], paf_map[0],
                                                         save_name=img_name[:-4],
                                                         save_dir=path_to,
                                                         data_format=model.data_format )

        find_all = 1
        if find_all == 1:
            conf_map_head = np.transpose(conf_map[0], [2, 0, 1])
            for K_1 in range(5):
                conf_head = np.abs(conf_map_head[K_1, :, :])
                heatmap_avg = cv2.resize(conf_head,
                                         (input_image.shape[1], input_image.shape[0]),
                                         interpolation=cv2.INTER_CUBIC)
                x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
                center = y, x  # x, y      #int(x*8),int(y*8)    #y, x
                radius = 3  # int(radius)
                # cv2.circle(input_image, center, radius, (1, 0, 0), -1)
                cv2.circle(input_image, center, radius, UserdefColor[K_1], -1)
            plt.imsave(path_to + img_name, input_image)
        find_head = 0
        if find_head == 1:
            # np.abs(conf_map[0, :, :])

            conf_map_head = np.transpose(conf_map[0], [2, 0, 1])
            conf_head=np.abs(conf_map_head[0, :, :])
            # show_conf_map = np.abs(conf_map[0, :, :])
            # plt.imshow(conf_head)
            heatmap_avg = cv2.resize(conf_head,
                                     (input_image.shape[1], input_image.shape[0]),
                                     interpolation=cv2.INTER_CUBIC)
            x, y = np.unravel_index(np.argmax(heatmap_avg), heatmap_avg.shape)
            # x, y = np.unravel_index(np.argmax(conf_head), conf_head.shape)
            center = y,x#x, y      #int(x*8),int(y*8)    #y, x
            radius = 3  # int(radius)
            cv2.circle(input_image, center, radius, (1, 0, 0), -1)
            plt.imsave(path_to + img_name, input_image)
            # plt.imsave(img_name, input_image)

        vis_and_head = 0
        if vis_and_head==1:

            img, conf_map, paf_map = input_image, conf_map[0], paf_map[0]
            save_name = img_name[:-4]
            save_dir = path_to
            data_format = model.data_format
            save_tofile = True
            if (type(img) != np.ndarray):
                img = img.numpy()
            if (type(conf_map) != np.ndarray):
                conf_map = conf_map.numpy()
            if (type(paf_map) != np.ndarray):
                paf_map = paf_map.numpy()

            if (data_format == "channels_last"):
                conf_map = np.transpose(conf_map, [2, 0, 1])
                paf_map = np.transpose(paf_map, [2, 0, 1])
            elif (data_format == "channels_first"):
                img = np.transpose(img, [1, 2, 0])
            os.makedirs(save_dir, exist_ok=True)
            ori_img = np.clip(img * 255.0, 0.0, 255.0).astype(np.uint8)
            vis_img = ori_img.copy()
            fig = plt.figure(figsize=(8, 8))
            # show input image
            a = fig.add_subplot(2, 2, 1)
            a.set_title("input image")
            plt.imshow(vis_img)
            # show conf_map
            show_conf_map = np.abs(conf_map[0, :, :])
            #np.amax(np.abs(conf_map[:-1, :, :]), axis=0)
            a = fig.add_subplot(2, 2, 3)
            a.set_title("conf_map")
            plt.imshow(show_conf_map)
            # show paf_map
            show_paf_map = np.amax(np.abs(paf_map[0:2, :, :]), axis=0)
            a = fig.add_subplot(2, 2, 4)
            a.set_title("paf_map")
            plt.imshow(show_paf_map)
            # save
            if (save_tofile):
                plt.savefig(f"{save_dir}/{save_name}_visualize.png")
                plt.close('all')

        if K_0%10 == 0:
            print(K_0)
    end_time = time.time()
    print("Time used: ", end_time - start_time, 's')
else:
    matplotlib.use('Agg')
    # infer on single image
    # ori_image=cv2.cvtColor(cv2.imread("./sample.jpg"),cv2.COLOR_BGR2RGB)
    img_name = "./data/fly/train2020/000000000182.jpg"

    ori_image = cv2.cvtColor(cv2.imread(img_name), cv2.COLOR_BGR2RGB)
    input_image = ori_image.astype(np.float32) / 255.0
    if (model.data_format == "channels_first"):
        input_image = np.transpose(input_image, [2, 0, 1])

    img_c, img_h, img_w = input_image.shape

    conf_map, paf_map = model.infer(input_image[np.newaxis, :, :, :])

    visualize = Model.get_visualize(Config.MODEL.Openpose)

    # def visualize(img,conf_map,paf_map,save_name="maps",save_dir="./save_dir/vis_dir",data_format="channels_first",save_tofile=True):
    vis_parts_heatmap, vis_limbs_heatmap = visualize(input_image, conf_map[0], paf_map[0],
                                                     data_format=model.data_format, save_tofile=False, )

    postprocess = Model.get_postprocess(Config.MODEL.Openpose)
    humans = postprocess(conf_map[0], paf_map[0], img_h, img_w, model.parts, model.limbs, model.data_format,
                         model.colors)
    # humans=postprocess(conf_map,paf_map,img_h,img_w,model.parts,model.limbs,model.data_format,model.colors)
    #

    # img = input_image
    conf_map = conf_map[0]
    paf_map = paf_map[0]
    if (type(conf_map) != np.ndarray):
        conf_map = conf_map.numpy()
    if (type(paf_map) != np.ndarray):
        paf_map = paf_map.numpy()
    parts = model.parts
    limbs = model.limbs
    colors = model.colors
    if (colors == None):
        colors = [[255, 0, 0]] * len(parts)
    from hyperpose.Model.openpose.infer import Post_Processor

    # from .infer import Post_Processor
    post_processor = Post_Processor(parts, limbs, colors)
    data_format = model.data_format
    humans = post_processor.process(conf_map, paf_map, img_h, img_w, data_format=data_format)

    # #draw all detected skeletons
    output_img = ori_image.copy()
    for human in humans:
        output_img = human.draw_human(output_img)

    # if you want to visualize all the images in one plot:
    # show image,part heatmap,limb heatmap and detected image
    # here we use 'transpose' because our data_format is 'channels_first'
    fig = plt.figure(figsize=(8, 8))
    # origin image
    origin_fig = fig.add_subplot(2, 2, 1)
    origin_fig.set_title("origin image")
    origin_fig.imshow(ori_image)
    # parts heatmap
    parts_fig = fig.add_subplot(2, 2, 2)
    parts_fig.set_title("parts heatmap")
    parts_fig.imshow(vis_parts_heatmap)
    # limbs heatmap
    limbs_fig = fig.add_subplot(2, 2, 3)
    limbs_fig.set_title("limbs heatmap")
    limbs_fig.imshow(vis_limbs_heatmap)
    # detected results
    result_fig = fig.add_subplot(2, 2, 4)
    result_fig.set_title("detect result")
    result_fig.imshow(output_img)
    # save fig
    plt.savefig("./sample_custome_infer.png")
    plt.close()



# import time
#
# start_time = time.time()
# for K in range(10):
#     conf_map, paf_map = model.infer(input_image[np.newaxis, :, :, :])
# end_time = time.time()
# print("Time used: ", end_time-start_time, 's')
#
#
# import matplotlib.pyplot as plt
# conf_temp=conf_map[0,:,:,0]
# plt.imshow(conf_temp,"gray")
# plt.imsave('a1234343124.jpg',conf_temp)
#get visualize function, which is able to get visualized part and limb heatmap image from inferred heatmaps
#
#
#
# img = input_image
# conf_map = conf_map[0]
# paf_map = paf_map[0]
#
# if(type(img)!=np.ndarray):
#     img=img.numpy()
# if(type(conf_map)!=np.ndarray):
#     conf_map=conf_map.numpy()
# if(type(paf_map)!=np.ndarray):
#     paf_map=paf_map.numpy()
#
# # import numpy as np
# data_format = model.data_format
# if(data_format=="channels_last"):
#     conf_map=np.transpose(conf_map,[2,0,1])
#     paf_map=np.transpose(paf_map,[2,0,1])
# elif(data_format=="channels_first"):
#     img=np.transpose(img,[1,2,0])
#
# # import os
# # os.makedirs(save_dir,exist_ok=True)
# #
# # import matplotlib
# # matplotlib.use('Qt5Agg')
#
# ori_img=np.clip(img*255.0,0.0,255.0).astype(np.uint8)
# vis_img=ori_img.copy()
# fig=plt.figure(figsize=(8,8))
# #show input image
# a=fig.add_subplot(2,2,1)
# a.set_title("input image")
# plt.imshow(vis_img)
# #show conf_map
# show_conf_map=np.amax(np.abs(conf_map[:-1,:,:]),axis=0)
# # show_conf_map=np.amax(conf_map[:-1,:,:],axis=0)
# # show_conf_map = np.abs(conf_map[0,:,:])
# a=fig.add_subplot(2,2,3)
# a.set_title("conf_map")
# plt.imshow(show_conf_map, alpha=0.8)
# #show paf_map
# show_paf_map=np.amax(np.abs(paf_map[:,:,:]),axis=0)
# # show_paf_map = np.abs(paf_map[0,:,:])
# a=fig.add_subplot(2,2,4)
# a.set_title("paf_map")
# plt.imshow(show_paf_map)



# plt.savefig("save_name_visualize.png")

#save
# if(save_tofile):
#     plt.savefig(f"{save_dir}/{save_name}_visualize.png")
#     plt.close('all')
#
#
#
#
#
#
#
# config.model
# #get postprocess function, which is able to get humans that contains assembled detected parts from inferred heatmaps
# postprocess=Model.get_postprocess(Config.MODEL.Openpose)
# humans=postprocess(conf_map[0],paf_map[0],img_h,img_w,model.parts,model.limbs,model.data_format,model.colors)
# # humans=postprocess(conf_map,paf_map,img_h,img_w,model.parts,model.limbs,model.data_format,model.colors)
# #
#
# # img = input_image
# conf_map = conf_map[0]
# paf_map = paf_map[0]
# if(type(conf_map)!=np.ndarray):
#     conf_map=conf_map.numpy()
# if(type(paf_map)!=np.ndarray):
#     paf_map=paf_map.numpy()
# parts= model.parts
# limbs = model.limbs
# colors = model.colors
# if(colors==None):
#     colors=[[255,0,0]]*len(parts)
# from hyperpose.Model.openpose.infer import Post_Processor
# # from .infer import Post_Processor
# post_processor=Post_Processor(parts,limbs,colors)
# data_format = model.data_format
# humans=post_processor.process(conf_map,paf_map,img_h,img_w,data_format=data_format)
#
# # #draw all detected skeletons
# output_img=ori_image.copy()
# for human in humans:
#     output_img=human.draw_human(output_img)
#
# #if you want to visualize all the images in one plot:
# #show image,part heatmap,limb heatmap and detected image
# #here we use 'transpose' because our data_format is 'channels_first'
# fig=plt.figure(figsize=(8,8))
# #origin image
# origin_fig=fig.add_subplot(2,2,1)
# origin_fig.set_title("origin image")
# origin_fig.imshow(ori_image)
# #parts heatmap
# parts_fig=fig.add_subplot(2,2,2)
# parts_fig.set_title("parts heatmap")
# parts_fig.imshow(vis_parts_heatmap)
# #limbs heatmap
# limbs_fig=fig.add_subplot(2,2,3)
# limbs_fig.set_title("limbs heatmap")
# limbs_fig.imshow(vis_limbs_heatmap)
# #detected results
# result_fig=fig.add_subplot(2,2,4)
# result_fig.set_title("detect result")
# result_fig.imshow(output_img)
# #save fig
# plt.savefig("./sample_custome_infer.png")
# plt.close()




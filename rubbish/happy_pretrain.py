



import os
import cv2
import sys
import math
import json
import time
import argparse
import matplotlib
import multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config,Model,Dataset


# os.environ['CUDA_VISIBLE_DEVICES']='-1'
#
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
#tf 2.0
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[1], True)

# python pretrain.py --
model_type="LightweightOpenpose"
# --
model_backbone="Vgg19"
# model_backbone="Resnet50"
# --lightweight_openpose_resnet50.npz
model_name="temp_pretrain_202011042002"
# --
dataset_path="./data"

# Config.set_batch_size(4)
Config.set_model_name(model_name)
Config.set_model_type(Config.MODEL[model_type])
Config.set_model_backbone(Config.BACKBONE[model_backbone])
Config.set_pretrain(True)
# config dataset
Config.set_pretrain_dataset_path(dataset_path)

# Config.set_batch_size(4)

config = Config.get_config()
# train

config.pretrain.batch_size=4
print(config.pretrain)


print(config.data)

print(config)
'''
rint(config)
{'model': {'n_pos': 19, 'num_channels': 128, 'hin': 368, 'win': 432, 'hout': 46, 'wout': 54, 'model_type': <MODEL.LightweightOpenpose: 1>, 'model_name': 'default_name', 'model_backbone': <BACKBONE.Resnet50: 4>, 'data_format': 'channels_first', 'model_dir': './save_dir/default_name/model_dir', 'userdef_parts': None, 'userdef_limbs': None, 'parts': <enum 'CocoPart'>, 'limbs': [(1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 2), (2, 3), (3, 4), (2, 16), (1, 5), (5, 6), (6, 7), (5, 17), (1, 0), (0, 14), (0, 15), (14, 16), (15, 17)]}, 'train': {'batch_size': 8, 'save_interval': 5000, 'n_step': 1000000, 'lr_init': 0.0001, 'lr_decay_every_step': 136120, 'lr_decay_factor': 0.666, 'weight_decay_factor': 0.0002, 'train_type': <TRAIN.Single_train: 0>, 'vis_dir': './save_dir/default_name/train_vis_dir', 'optim_type': <OPTIM.Adam: 0>}, 'eval': {'batch_size': 22, 'vis_dir': './save_dir/default_name/eval_vis_dir'}, 'data': {'dataset_type': <DATA.MSCOCO: 0>, 'dataset_version': '2017', 'dataset_path': './data', 'dataset_filter': None, 'vis_dir': './save_dir/default_name/data_vis_dir', 'official_flag': True, 'userdef_dataset': None, 'useradd_flag': False, 'useradd_scale_rate': 1, 'useradd_train_img_paths': None, 'useradd_train_targets': None, 'domainadapt_flag': False, 'domainadapt_scale_rate': 1, 'domainadapt_train_img_paths': None}, 'log': {'log_interval': 100, 'log_path': './save_dir/default_name/log.txt'}, 'pretrain': {'enable': True, 'lr_init': 0.0005, 'batch_size': 4, 'total_step': 370000000, 'log_interval': 100, 'val_interval': 5000, 'save_interval': 5000, 'weight_decay_factor': 1e-05, 'pretrain_dataset_path': './data', 'pretrain_model_dir': './save_dir/pretrain_backbone', 'val_num': 20000, 'lr_decay_step': 170000}}
'''

model = Model.get_model(config)
pretrain = Model.get_pretrain(config)
dataset = Dataset.get_pretrain_dataset(config)

# config.train_dataset_path
# pretrain(model, dataset)


val_pretrain=1
if val_pretrain==1:
    pretrain(model,dataset)
    #
else:
    import os
    import cv2
    import numpy as np
    import multiprocessing
    import tensorflow as tf
    import tensorlayer as tl
    from functools import partial
    from hyperpose.Model.common import log, regulize_loss
    from hyperpose.Model.common import KUNGFU, MODEL, init_log
    # TODO(JZ)
    # def single_pretrain(model, dataset, config):
    init_log(config)
    ##TODO(JZ)
    lr_init = config.pretrain.lr_init
    lr_decay_step = config.pretrain.lr_decay_step
    batch_size = config.pretrain.batch_size
    total_step = config.pretrain.total_step
    log_interval = config.pretrain.log_interval
    val_interval = config.pretrain.val_interval
    save_interval = config.pretrain.save_interval
    pretrain_model_dir = config.pretrain.pretrain_model_dir
    weight_decay_factor = config.pretrain.weight_decay_factor

    print(f"starting to pretrain model backbone with learning rate:{lr_init} batch_size:{batch_size}")
    print(f"pretraining model_type:{config.model.model_type} model_backbone:{model.backbone.name}")


    # training dataset configure with shuffle,augmentation,and prefetch
    train_dataset = dataset.get_train_dataset()
    train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
    data_aug = partial(_data_aug, hin=224, win=224, data_format=model.data_format)
    train_dataset = train_dataset.map(partial(train_map_fn, data_aug=data_aug),
                                      num_parallel_calls=max(multiprocessing.cpu_count() // 2, 1))
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(8)

    # train configure
    step = tf.Variable(1, trainable=False)
    lr = tf.Variable(lr_init, trainable=False)
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    ckpt = tf.train.Checkpoint(step=step, optimizer=opt, lr=lr)
    ckpt_manager = tf.train.CheckpointManager(ckpt, pretrain_model_dir, max_to_keep=3)

    train_model = model.backbone
    train_model.train()
    # load from ckpt
    try:
        ckpt.restore(ckpt_manager.latest_checkpoint)
    except:
        log("ckpt_path doesn't exist, learning rate, step and optimizer are initialized")
    try:
        train_model.load_weights(os.path.join(pretrain_model_dir, f"newest_{train_model.name}.npz"),
                                 format="npz_dict")
    except:
        log("model_path doesn't exist, model parameters are initialized")

    total_pd_loss, total_re_loss, total_top1_acc_num, total_top5_acc_num, total_img_num = 0, 0, 0, 0, 0
    max_eval_acc = 0
    stuck_time = 0


    # optimize one step
    @tf.function
    def one_step(image, label, train_model):
        step.assign_add(1)
        with tf.GradientTape() as tape:
            predict = train_model.forward(image)
            pd_loss = train_model.cal_loss(label, predict)
            re_loss = regulize_loss(train_model, weight_decay_factor)
            total_loss = pd_loss + re_loss

        top1_acc_num = tf.reduce_sum(tf.where(tf.math.in_top_k(label, predict, 1), 1, 0))
        top5_acc_num = tf.reduce_sum(tf.where(tf.math.in_top_k(label, predict, 5), 1, 0))
        gradients = tape.gradient(total_loss, train_model.trainable_weights)
        opt.apply_gradients(zip(gradients, train_model.trainable_weights))
        return top1_acc_num, top5_acc_num, pd_loss, re_loss, predict


    for image, label in train_dataset:
        top1_acc_num, top5_acc_num, pd_loss, re_loss, predict = one_step(image.numpy(), label.numpy(), train_model)
        total_pd_loss += pd_loss / log_interval
        total_re_loss += re_loss / log_interval
        total_top1_acc_num += top1_acc_num
        total_top5_acc_num += top5_acc_num
        total_img_num += batch_size

        if (step % lr_decay_step == 0):
            lr = lr / 5

        # log info
        if (step != 0 and step % log_interval == 0):
            print(
                "Train iteration {} / {}: Learning rate:{} total_loss:{} pd_loss:{} re_loss:{} accuracy_top1:{} accuracy_top5:{}".format(
                    step.numpy(), total_step, lr.numpy(), total_pd_loss + total_re_loss, total_pd_loss,
                    total_re_loss, total_top1_acc_num / total_img_num, total_top5_acc_num / total_img_num))
            total_pd_loss, total_re_loss = 0.0, 0.0
            total_top1_acc_num, total_top5_acc_num, total_img_num = 0, 0, 0

        # save model
        if (step != 0 and step % save_interval == 0):
            ckpt_save_path = ckpt_manager.save()
            log(f"ckpt save_path:{ckpt_save_path} saved!\n")
            model_save_path = os.path.join(pretrain_model_dir, f"newest_{train_model.name}.npz")
            train_model.save_weights(model_save_path, format="npz_dict")
            log(f"model save_path:{model_save_path} saved!\n")

        # validate model
        if (step != 0 and step % val_interval == 0):
            train_model.eval()
            eval_acc = single_val(train_model, dataset, config)
            print(f"current validate: eval_acc:{eval_acc} max_eval_acc:{max_eval_acc} stuck_time:{stuck_time}")
            if (eval_acc < max_eval_acc):
                stuck_time += 1
            else:
                max_eval_acc = eval_acc
            if (stuck_time >= 3):
                lr = lr / 5
                stuck_time = 0
            train_model.train()

        if (step == total_step):
            break



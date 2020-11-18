





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
model_backbone = "Default"

# model_name = "default_name"
# model_name = "test_20201105_1444"
# dataset_type = "MSCOCO"
dataset_type = "USERDEF" #"fly"
model_name = dataset_type+'_'+time.strftime("%Y%m%d_%H%M")

dataset_version ="2020"
dataset_path = "./data"
train_type = "Single_train"
kf_optimizer ='Sync_avg'
# use_official_dataset = 1
userdef_dataset = 1
# useradd_data_path = None
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
# Config.set_official_dataset(use_official_dataset)
Config.set_userdef_dataset(userdef_dataset)
Config.set_dataset_type(Config.DATA[dataset_type])
Config.set_dataset_path(dataset_path)
Config.set_dataset_version(dataset_version)


# sample add user data to train
# if (useradd_data_path != None):
#     useradd_train_image_paths = []
#     useradd_train_targets = []
#     image_dir = os.path.join(useradd_data_path, "images")
#     anno_path = os.path.join(useradd_data_path, "anno.json")
#     # generate image paths and targets
#     anno_json = json.load(open(anno_path, mode="r"))
#     for image_path in anno_json["annotations"].keys():
#         anno = anno_json["annotations"][image_path]
#         useradd_train_image_paths.append(os.path.join(image_dir, image_path))
#         useradd_train_targets.append({
#             "kpt": anno["keypoints"],
#             "mask": None,
#             "bbx": anno["bbox"],
#             "labeled": 1
#         })
#     Config.set_useradd_data(useradd_train_image_paths, useradd_train_targets, useradd_scale_rate=1)


# # sample use domain adaptation to train:
# domainadapt_data_path = None
# if (domainadapt_data_path != None):
#     domainadapt_image_paths = glob.glob(os.path.join(domainadapt_data_path, "images", "*"))
#     Config.set_domainadapt_dataset(domainadapt_train_img_paths=domainadapt_image_paths, domainadapt_scale_rate=1)
# # print('domainadapt_image_paths = ', domainadapt_image_paths)
print('\n' * 4)



# train
config = Config.get_config()

config.pretrain.batch_size=4
config.eval.batch_size=1
config.train.batch_size=4

# config.model.n_pos = 5
# config.model.num_channels = 128
# config.model.hin = 224
# config.model.win = 224
# config.model.hout = 46
# config.model.wout = 54
# from enum import Enum
# #
# class CocoPart(Enum):
#      Head = 0
#      Center = 1
#      Tail = 2
#      LWing = 3
#      RWing  = 4
#
# config.model.parts=CocoPart
#
# config.model.limbs =[(0, 1), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4)]
#
# print(config)

# config.train.save_interval = 500
# config.train.n_step=1000

if 1>0:
     print(config.model)
     print(config.train)
     print(config.data)
print('\n' * 4)




model = Model.get_model(config)

train = Model.get_train(config)

dataset = Dataset.get_dataset(config)



val_train = 0
if val_train == 1:
     train(model, dataset)
else:

     import math
     import multiprocessing
     import os
     import cv2
     import time
     import sys
     import json
     import numpy as np
     import matplotlib

     matplotlib.use('Agg')
     import tensorflow as tf
     import tensorlayer as tl
     from pycocotools.coco import maskUtils
     import _pickle as cPickle
     from functools import partial

     from hyperpose.Model.openpose.utils import tf_repeat, get_heatmap, get_vectormap, draw_results
     from hyperpose.Model.openpose.utils import get_parts, get_limbs, get_flip_list
     from hyperpose.Model.common import log, KUNGFU, MODEL, get_optim, init_log
     from hyperpose.Model.domainadapt import get_discriminator


     def regulize_loss(target_model, weight_decay_factor):
          re_loss = 0
          regularizer = tf.keras.regularizers.l2(l=weight_decay_factor)
          for weight in target_model.trainable_weights:
               re_loss += regularizer(weight)
          return re_loss


     def _data_aug_fn(image, ground_truth, hin, hout, win, wout, parts, limbs, flip_list=None,
                      data_format="channels_first"):
          """Data augmentation function."""
          # restore data
          concat_dim = 0 if data_format == "channels_first" else -1
          ground_truth = cPickle.loads(ground_truth.numpy())
          image = image.numpy()
          annos = ground_truth["kpt"]
          labeled = ground_truth["labeled"]
          mask = ground_truth["mask"]

          # decode mask
          h_mask, w_mask, _ = np.shape(image)
          mask_miss = np.ones((h_mask, w_mask), dtype=np.uint8)
          if (mask != None):
               for seg in mask:
                    bin_mask = maskUtils.decode(seg)
                    bin_mask = np.logical_not(bin_mask)
                    if (bin_mask.shape != mask_miss.shape):
                         print(f"test error mask shape mask_miss:{mask_miss.shape} bin_mask:{bin_mask.shape}")
                    else:
                         mask_miss = np.bitwise_and(mask_miss, bin_mask)

          # get transform matrix
          M_rotate = tl.prepro.affine_rotation_matrix(angle=(-30, 30))  # original paper: -40~40
          M_zoom = tl.prepro.affine_zoom_matrix(zoom_range=(0.5, 0.8))  # original paper: 0.5~1.1
          M_combined = M_rotate.dot(M_zoom)
          h, w, _ = image.shape
          transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=w, y=h)

          # apply data augmentation
          image = tl.prepro.affine_transform_cv2(image, transform_matrix)
          mask_miss = tl.prepro.affine_transform_cv2(mask_miss, transform_matrix, border_mode='replicate')
          # print(annos)
          # print(transform_matrix)
          # [array([[-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [105., 95.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.],
          #         [-1000., -1000.]])]

          # [[7.98101109e-01  8.92176826e-03  4.17691499e+01]
          #  [-8.92176826e-03  7.98101109e-01  6.64784064e+01]
          # [0.00000000e+00
          # 0.00000000e+00
          # 1.00000000e+00]]

          annos = tl.prepro.affine_transform_keypoints(annos, transform_matrix)

          ##TODO
          # temply ignore flip augmentation
          '''
          if(flip_list!=None):
              image, annos, mask_miss = tl.prepro.keypoint_random_flip(image,annos, mask_miss, prob=0.5, flip_list=flip_list)
          '''
          image, annos, mask_miss = tl.prepro.keypoint_resize_random_crop(image, annos, mask_miss,
                                                                          size=(hin, win))  # hao add

          # generate result which include keypoints heatmap and vectormap
          height, width, _ = image.shape
          heatmap = get_heatmap(annos, height, width, hout, wout, parts, limbs, data_format=data_format)
          vectormap = get_vectormap(annos, height, width, hout, wout, parts, limbs, data_format=data_format)
          resultmap = np.concatenate((heatmap, vectormap), axis=concat_dim)

          image = cv2.resize(image, (win, hin))
          mask_miss = cv2.resize(mask_miss, (win, hin))
          img_mask = mask_miss

          # generate output masked image, result map and maskes
          img_mask = mask_miss.reshape(hin, win, 1)
          image = image * np.repeat(img_mask, 3, 2)
          resultmap = np.array(resultmap, dtype=np.float32)
          mask_miss = np.array(cv2.resize(mask_miss, (wout, hout), interpolation=cv2.INTER_AREA), dtype=np.float32)[:,
                      :, np.newaxis]
          if (data_format == "channels_first"):
               image = np.transpose(image, [2, 0, 1])
               mask_miss = np.transpose(mask_miss, [2, 0, 1])
          labeled = np.float32(labeled)
          return image, resultmap, mask_miss, labeled


     def _map_fn(img_list, annos, data_aug_fn, hin, win, hout, wout, parts, limbs):
          """TF Dataset pipeline."""
          # load data
          image = tf.io.read_file(img_list)
          image = tf.image.decode_jpeg(image, channels=3)  # get RGB with 0~1
          image = tf.image.convert_image_dtype(image, dtype=tf.float32)
          # data augmentation using affine transform and get paf maps
          image, resultmap, mask, labeled = tf.py_function(data_aug_fn, [image, annos],
                                                           [tf.float32, tf.float32, tf.float32, tf.float32])
          # data augmentaion using tf
          image = tf.image.random_brightness(image, max_delta=35. / 255.)  # 64./255. 32./255.)  caffe -30~50
          image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # lower=0.2, upper=1.8)  caffe 0.3~1.5
          image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
          return image, resultmap, mask, labeled


     def get_paramed_map_fn(hin, win, hout, wout, parts, limbs, flip_list=None, data_format="channels_first"):
          paramed_data_aug_fn = partial(_data_aug_fn, hin=hin, win=win, hout=hout, wout=wout, parts=parts, limbs=limbs, \
                                        flip_list=flip_list, data_format=data_format)
          paramed_map_fn = partial(_map_fn, data_aug_fn=paramed_data_aug_fn, hin=hin, win=win, hout=hout, wout=wout,
                                   parts=parts, limbs=limbs)
          return paramed_map_fn


     init_log(config)

     # train hyper params
     # dataset params
     n_step = config.train.n_step
     batch_size = config.train.batch_size
     # learning rate params
     lr_init = config.train.lr_init
     lr_decay_factor = config.train.lr_decay_factor
     lr_decay_steps = [200000, 300000, 360000, 420000, 480000, 540000, 600000, 700000, 800000, 900000]
     weight_decay_factor = config.train.weight_decay_factor
     # log and checkpoint params
     log_interval = config.log.log_interval
     save_interval = config.train.save_interval
     vis_dir = config.train.vis_dir
     train_model = model
     # model hyper params
     n_pos = train_model.n_pos
     hin = train_model.hin
     win = train_model.win
     hout = train_model.hout
     wout = train_model.wout
     model_dir = config.model.model_dir
     pretrain_model_dir = config.pretrain.pretrain_model_dir
     pretrain_model_path = f"{pretrain_model_dir}/newest_{train_model.backbone.name}.npz"

     print(f"single training using learning rate:{lr_init} batch_size:{batch_size}")
     # training dataset configure with shuffle,augmentation,and prefetch
     train_dataset = dataset.get_train_dataset()
     dataset_type = dataset.get_dataset_type()
     parts, limbs, data_format = train_model.parts, train_model.limbs, train_model.data_format
     flip_list = get_flip_list(dataset_type)
     paramed_map_fn = get_paramed_map_fn(hin, win, hout, wout, parts, limbs, flip_list=flip_list,
                                         data_format=data_format)
     train_dataset = train_dataset.shuffle(buffer_size=4096).repeat()
     train_dataset = train_dataset.map(paramed_map_fn, num_parallel_calls=max(multiprocessing.cpu_count() // 2, 1))
     train_dataset = train_dataset.batch(config.train.batch_size)
     train_dataset = train_dataset.prefetch(64)

     # train configure
     step = tf.Variable(1, trainable=False)
     lr = tf.Variable(lr_init, trainable=False)
     lr_init = tf.Variable(lr_init, trainable=False)
     opt = tf.keras.optimizers.Adam(learning_rate=lr)
     # domain adaptation params
     domainadapt_flag = config.data.domainadapt_flag
     if (domainadapt_flag):
          print("domain adaptaion enabled!")
          discriminator = get_discriminator(train_model)
          opt_d = tf.keras.optimizers.Adam(learning_rate=lr)
          lambda_d = tf.Variable(1, trainable=False)
          ckpt = tf.train.Checkpoint(step=step, optimizer=opt, lr=lr, optimizer_d=opt_d, lambda_d=lambda_d)
     else:
          ckpt = tf.train.Checkpoint(step=step, optimizer=opt, lr=lr)
     ckpt_manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)

     # load from ckpt
     try:
          log("loading ckpt...")
          ckpt.restore(ckpt_manager.latest_checkpoint)
     except:
          log("ckpt_path doesn't exist, step and optimizer are initialized")
     # load pretrained backbone
     try:
          log("loading pretrained backbone...")
          tl.files.load_and_assign_npz_dict(name=pretrain_model_path, network=train_model.backbone, skip=True)
     except:
          log("pretrained backbone doesn't exist, model backbone are initialized")
     # load model weights
     try:
          log("loading saved training model weights...")
          train_model.load_weights(os.path.join(model_dir, "newest_model.npz"))
     except:
          log("model_path doesn't exist, model parameters are initialized")
     if (domainadapt_flag):
          try:
               log("loading saved domain adaptation discriminator weight...")
               discriminator.load_weights(os.path.join(model_dir, "newest_discriminator.npz"))
          except:
               log("discriminator path doesn't exist, discriminator parameters are initialized")

     for lr_decay_step in lr_decay_steps:
          if (step > lr_decay_step):
               lr = lr * lr_decay_factor


     # optimize one step
     @tf.function
     def one_step(image, gt_conf, gt_paf, mask, train_model):
          step.assign_add(1)
          with tf.GradientTape() as tape:
               pd_conf, pd_paf, stage_confs, stage_pafs = train_model.forward(image, is_train=True)
               pd_loss, loss_confs, loss_pafs = train_model.cal_loss(gt_conf, gt_paf, mask, stage_confs, stage_pafs)
               re_loss = regulize_loss(train_model, weight_decay_factor)
               total_loss = pd_loss + re_loss

          gradients = tape.gradient(total_loss, train_model.trainable_weights)
          opt.apply_gradients(zip(gradients, train_model.trainable_weights))
          return pd_conf, pd_paf, total_loss, re_loss, loss_confs, loss_pafs


     @tf.function
     def one_step_domainadpat(image, gt_conf, gt_paf, mask, labeled, train_model, discriminator, lambda_d):
          step.assign_add(1)
          with tf.GradientTape(persistent=True) as tape:
               # optimize train model
               pd_conf, pd_paf, stage_confs, stage_pafs, backbone_fatures = train_model.forward(image, is_train=True,
                                                                                                domainadapt=True)
               d_predict = discriminator.forward(backbone_fatures)
               pd_loss, loss_confs, loss_pafs = train_model.cal_loss(gt_conf, gt_paf, mask, stage_confs, stage_pafs)
               re_loss = regulize_loss(train_model, weight_decay_factor)
               gan_loss = lambda_d * tf.nn.sigmoid_cross_entropy_with_logits(logits=d_predict, labels=1 - labeled)
               total_loss = pd_loss + re_loss + gan_loss
               d_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_predict, labels=labeled)
          # optimize G
          g_gradients = tape.gradient(total_loss, train_model.trainable_weights)
          opt.apply_gradients(zip(g_gradients, train_model.trainable_weights))
          # optimize D
          d_gradients = tape.gradient(d_loss, discriminator.trainable_weights)
          opt_d.apply_gradients(zip(d_gradients, discriminator.trainable_weights))
          return pd_conf, pd_paf, total_loss, re_loss, loss_confs, loss_pafs, gan_loss, d_loss


     # train each step
     tic = time.time()
     train_model.train()
     conf_losses, paf_losses = np.zeros(shape=(6)), np.zeros(shape=(6))
     avg_conf_loss, avg_paf_loss, avg_total_loss, avg_re_loss = 0, 0, 0, 0
     avg_gan_loss, avg_d_loss = 0, 0
     log(
          'Start - n_step: {} batch_size: {} lr_init: {} lr_decay_steps: {} lr_decay_factor: {} weight_decay_factor: {}'.format(
               n_step, batch_size, lr_init.numpy(), lr_decay_steps, lr_decay_factor, weight_decay_factor))

     K = 0
     for image, gt_label, mask, labeled in train_dataset:
          K+=1
          # extract gt_label
          if K == 1:
               if (train_model.data_format == "channels_first"):
                    gt_conf = gt_label[:, :n_pos, :, :]
                    gt_paf = gt_label[:, n_pos:, :, :]
               else:
                    gt_conf = gt_label[:, :, :, :n_pos]
                    gt_paf = gt_label[:, :, :, n_pos:]
               break

          print(gt_conf.shape)
          print(gt_paf.shape)

          # print(gt_paf.numpy().shape())
          # print(gt_conf.numpy().shape())
          # learning rate decay
          if (step in lr_decay_steps):
               new_lr_decay = lr_decay_factor ** (lr_decay_steps.index(step) + 1)
               lr = lr_init * new_lr_decay

          # optimize one step
          if (domainadapt_flag):
               lambda_d = 2 / (1 + tf.math.exp(-10 * (step / n_step))) - 1
               pd_conf, pd_paf, total_loss, re_loss, loss_confs, loss_pafs, gan_loss, d_loss = \
                    one_step_domainadpat(image.numpy(), gt_conf.numpy(), gt_paf.numpy(), mask.numpy(), labeled.numpy(),
                                         train_model, discriminator, lambda_d)
               avg_gan_loss += gan_loss / log_interval
               avg_d_loss += d_loss / log_interval
          else:
               # image.numpy().shape
               # (4, 368, 432, 3)
               # gt_conf.numpy().shape
               # (4, 46, 54, 5)
               # gt_paf.numpy().shape
               # (4, 46, 54, 12)
               # mask.numpy().shape
               # (4, 46, 54, 1)
               pd_conf, pd_paf, total_loss, re_loss, loss_confs, loss_pafs = \
                    one_step(image.numpy(), gt_conf.numpy(), gt_paf.numpy(), mask.numpy(), train_model)
               # image, gt_conf, gt_paf, mask, train_model = image.numpy(), gt_conf.numpy(), gt_paf.numpy(), mask.numpy(), train_model

          avg_conf_loss += tf.reduce_mean(loss_confs) / batch_size / log_interval
          avg_paf_loss += tf.reduce_mean(loss_pafs) / batch_size / log_interval
          avg_total_loss += total_loss / log_interval
          avg_re_loss += re_loss / log_interval

          # debug
          for stage_id, (loss_conf, loss_paf) in enumerate(zip(loss_confs, loss_pafs)):
               conf_losses[stage_id] += loss_conf / batch_size / log_interval
               paf_losses[stage_id] += loss_paf / batch_size / log_interval

          # save log info periodly
          if ((step.numpy() != 0) and (step.numpy() % log_interval) == 0):
               tic = time.time()
               log(
                    'Train iteration {} / {}: Learning rate {} total_loss:{}, conf_loss:{}, paf_loss:{}, l2_loss {} stage_num:{} time:{}'.format(
                         step.numpy(), n_step, lr.numpy(), avg_total_loss, avg_conf_loss, avg_paf_loss, avg_re_loss,
                         len(loss_confs), time.time() - tic))
               for stage_id in range(0, len(loss_confs)):
                    log(f"stage_{stage_id} conf_loss:{conf_losses[stage_id]} paf_loss:{paf_losses[stage_id]}")
               if (domainadapt_flag):
                    log(f"adaptation loss: g_loss:{avg_gan_loss} d_loss:{avg_d_loss}")

               avg_total_loss, avg_conf_loss, avg_paf_loss, avg_re_loss = 0, 0, 0, 0
               avg_gan_loss, avg_d_loss = 0, 0
               conf_losses, paf_losses = np.zeros(shape=(6)), np.zeros(shape=(6))

          # save result and ckpt periodly
          if ((step.numpy() != 0) and (step.numpy() % save_interval) == 0):
               # save ckpt
               log("saving model ckpt and result...")
               ckpt_save_path = ckpt_manager.save()
               log(f"ckpt save_path:{ckpt_save_path} saved!\n")
               # save train model
               model_save_path = os.path.join(model_dir, "newest_model.npz")
               train_model.save_weights(model_save_path)
               log(f"model save_path:{model_save_path} saved!\n")
               # save discriminator model
               if (domainadapt_flag):
                    dis_save_path = os.path.join(model_dir, "newest_discriminator.npz")
                    discriminator.save_weights(dis_save_path)
                    log(f"discriminator save_path:{dis_save_path} saved!\n")
               # draw result
               draw_results(image.numpy(), gt_conf.numpy(), pd_conf.numpy(), gt_paf.numpy(), pd_paf.numpy(),
                            mask.numpy(), \
                            vis_dir, 'train_%d_' % step, data_format=data_format)

          # training finished
          if (step == n_step):
               break



#
# '''
# #!/usr/bin/env python3
#
# import os
# import cv2
# import sys
# import math
# import json
# import glob
# import argparse
# import matplotlib
# import numpy as np
# import tensorflow as tf
# import tensorlayer as tl
# from hyperpose import Config,Model,Dataset
#
#
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)
# gpu = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpu[1], True)
#
# # model_type = 'Openpose'
# # model_backbone = 'Default'
# # model_name = 'default_name'
# # dataset_type = 'MSCOCO'
# # dataset_version = '2017'
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='FastPose.')
#     parser.add_argument("--model_type",
#                         type=str,
#                         default="Openpose",
#                         # default="LightweightOpenpose",
#                         help="human pose estimation model type, available options: "
#                              "Openpose, LightweightOpenpose ,MobilenetThinOpenpose, PoseProposal")
#     parser.add_argument("--model_backbone",
#                         type=str,
#                         default="Default",
#                         # default="Vggtiny",
#                         help="model backbone, available options: Mobilenet, Vggtiny, Vgg19, Resnet18, Resnet50")
#     parser.add_argument("--model_name",
#                         type=str,
#                         default="default_name",
#                         help="model name,to distinguish model and determine model dir")
#     parser.add_argument("--dataset_type",
#                         type=str,
#                         default="MSCOCO",
#                         help="dataset name,to determine which dataset to use, available options: MSCOCO, MPII ")
#     parser.add_argument("--dataset_version",
#                         type=str,
#                         default="2017",
#                         help="dataset version, only use for MSCOCO and available for version 2014 and 2017")
#     parser.add_argument("--dataset_path",
#                         type=str,
#                         default="data",
#                         help="dataset path,to determine the path to load the dataset")
#     parser.add_argument('--train_type',
#                         type=str,
#                         default="Single_train",
#                         help='train type, available options: Single_train, Parallel_train')
#     parser.add_argument('--kf_optimizer',
#                         type=str,
#                         default='Sync_avg',
#                         help='kung fu parallel optimizor,available options: Sync_sgd, Sync_avg, Pair_avg')
#     parser.add_argument('--use_official_dataset',
#                         type=int,
#                         default=1,
#                         help='whether to use official dataset, could be used when only user data is needed')
#     parser.add_argument('--useradd_data_path',
#                         type=str,
#                         default=None,
#                         help='path to user data directory where contains images folder and annotation json file')
#     parser.add_argument('--domainadapt_data_path',
#                         type=str,
#                         default=None,
#                         help='path to user data directory where contains images for domain adaptation')
#     parser.add_argument('--optim_type',
#                         type=str,
#                         default="Adam",
#                         help='optimizer type used for training')
#     parser.add_argument('--learning_rate',
#                         type=float,
#                         default=1e-4,
#                         help='learning rate')
#     parser.add_argument('--log_interval',
#                         type=int,
#                         default=1e2,
#                         help='log frequency')
#     parser.add_argument('--save_interval',
#                         type=int,
#                         default=5e3,
#                         help='log frequency')
#
#     args=parser.parse_args()
#     #config model
#     Config.set_model_name(args.model_name)
#     Config.set_model_type(Config.MODEL[args.model_type])
#     Config.set_model_backbone(Config.BACKBONE[args.model_backbone])
#     Config.set_log_interval(args.log_interval)
#     Config.set_save_interval(args.save_interval)
#     #config train
#     Config.set_train_type(Config.TRAIN[args.train_type])
#     Config.set_learning_rate(args.learning_rate)
#     Config.set_optim_type(Config.OPTIM[args.optim_type])
#     Config.set_kungfu_option(Config.KUNGFU[args.kf_optimizer])
#     #config dataset
#     print(f"test enabling official dataset:{args.use_official_dataset}")
#     Config.set_official_dataset(args.use_official_dataset)
#     Config.set_dataset_type(Config.DATA[args.dataset_type])
#     Config.set_dataset_path(args.dataset_path)
#     Config.set_dataset_version(args.dataset_version)
#     #sample add user data to train
#     if(args.useradd_data_path!=None):
#         useradd_train_image_paths=[]
#         useradd_train_targets=[]
#         image_dir=os.path.join(args.useradd_data_path,"images")
#         anno_path=os.path.join(args.useradd_data_path,"anno.json")
#         #generate image paths and targets
#         anno_json=json.load(open(anno_path,mode="r"))
#         for image_path in anno_json["annotations"].keys():
#             anno=anno_json["annotations"][image_path]
#             useradd_train_image_paths.append(os.path.join(image_dir,image_path))
#             useradd_train_targets.append({
#                 "kpt":anno["keypoints"],
#                 "mask":None,
#                 "bbx":anno["bbox"],
#                 "labeled":1
#             })
#         Config.set_useradd_data(useradd_train_image_paths,useradd_train_targets,useradd_scale_rate=1)
#     #sample use domain adaptation to train:
#     if(args.domainadapt_data_path!=None):
#         domainadapt_image_paths=glob.glob(os.path.join(args.domainadapt_data_path,"images","*"))
#         Config.set_domainadapt_dataset(domainadapt_train_img_paths=domainadapt_image_paths,domainadapt_scale_rate=1)
#     print('domainadapt_image_paths = ',domainadapt_image_paths)
#     print('\n'*4)
#
#     #train
#     config=Config.get_config()
#     print(config)
#     print('\n' * 4)
#
#     model=Model.get_model(config)
#     train=Model.get_train(config)
#     dataset=Dataset.get_dataset(config)
#     train(model,dataset)
#
# '''




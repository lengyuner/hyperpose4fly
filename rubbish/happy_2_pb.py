#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config, Model, Dataset
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


# model_type ="Openpose"
# model_backbone ="Default"
# model_name ="default_name"
# lightweight_openpose_resnet50.npz
model_type ="LightweightOpenpose"
model_backbone ="Resnet50"
model_name ="default_name_1102_2032"

dataset_type ="MSCOCO"
dataset_version ="2017"
dataset_path ="data"
train_type ="Single_train"
# kf_optimizer ='Sync_avg'
kf_optimizer ='Sma'
output_dir="save_dir"
use_official_dataset =1
useradd_data_path =None
domainadapt_data_path =None
# optim_type ="Adam"
# learning_rate =1e-4
# log_interval =1e2
# save_interval =5e3
# "model backbone, available options: Mobilenet, Vgg19, Resnet18, Resnet50")

Config.set_model_name(model_name)
Config.set_model_type(Config.MODEL[model_type])
Config.set_model_backbone(Config.BACKBONE[model_backbone])
config = Config.get_config()
export_model = Model.get_model(config)

# input_path = f"{config.model.model_dir}/newest_model.npz"
input_path=f"{config.model.model_dir}/lightweight_openpose_resnet50.npz"

output_dir = f"{output_dir}/{config.model.model_name}"

output_path = f"{output_dir}/frozen_{config.model.model_name}.pb"

print(f"exporting model {config.model.model_name} from {input_path}...")

# lightweightopenpose(
#   (resnet50_backbone): resnet50_backbone(
#     (block_1_1): block_1_1(
#       (layerlist_15): LayerList(
#         (0): Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 1), strides=(1, 1), padding=SAME, bias=False, No Activation, name='block_1_1_ds_conv1')
#         (1): BatchNorm2d(num_features=256, decay=0.9, epsilon=1e-05, No Activation, name="block_1_1_ds_bn1")
#       )
# raise ValueError("Shapes %s and %s are incompatible" % (self, other))
# ValueError: Shapes (1, 1, 64, 256) and (64,) are incompatible


if (not os.path.exists(output_dir)):
    print("creating output_dir...")
    os.mkdir(output_dir)
if (not os.path.exists(input_path)):
    print("input model file doesn't exist!")
    print("conversion aborted!")
else:
    export_model.load_weights(input_path)
    export_model.eval()
    if (export_model.data_format == "channels_last"):
        input_signature = tf.TensorSpec(shape=(None, None, None, 3))
    else:
        input_signature = tf.TensorSpec(shape=(None, 3, None, None))
    concrete_function = export_model.infer.get_concrete_function(x=input_signature)
    frozen_graph = convert_variables_to_constants_v2(concrete_function)
    frozen_graph_def = frozen_graph.graph.as_graph_def()
    tf.io.write_graph(graph_or_graph_def=frozen_graph_def, logdir=output_dir, name=f"frozen_{model_name}.pb", \
                      as_text=False)
    print(f"exporting pb file finished! output file: {output_path}")


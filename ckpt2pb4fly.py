
import os
import argparse
import tensorflow as tf
import tensorlayer as tl
from hyperpose import Config,Model,Dataset
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


from train4fly import model
export_model = model

# model_name="USERDEF_Openpose_Resnet18_20201130_1642"
model_name = "USERDEF_Openpose_Resnet18_20201202_2325"

model_dir = f"./save_dir/{model_name}/model_dir"
input_path = f"{model_dir}/newest_model.npz"

output_dir = f"./save_dir/{model_name}"
output_path=f"{output_dir}/frozen_{model_name}.pb"



print(f"exporting model {model_name} from {input_path}...")
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





#
# model_type="Openpose"
#
# model_backbone="Default"
# model_name="default_name"
# dataset_type="MSCOCO",
#                         help="dataset name,to determine which dataset to use, available options: coco ")
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
#                         default='Sma',
#                         help='kung fu parallel optimizor,available options: Sync_sgd, Async_sgd, Sma')
#     parser.add_argument("--output_dir",
#                         type=str,
#                         default="save_dir

# output_dir
#
# input_path
# data_format
# model_dir
# model_name


# C:\Users\ps\Desktop\djz\hyperpose\save_dir\USERDEF_Openpose_Resnet18_20201130_1642
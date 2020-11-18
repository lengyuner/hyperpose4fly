# {
#     'model':
#         {
#             'n_pos': 19,
#             'num_channels': 128,
#             'hin': 368, 'win': 432,
#             'hout': 46, 'wout': 54,
#             'model_type': <MODEL.LightweightOpenpose: 1>,
#             'model_name': 'default_name',
#             'model_backbone': <BACKBONE.Vggtiny: 5>,
#             'data_format': 'channels_first',
#             'model_dir': './save_dir/default_name/model_dir',
#             'userdef_parts': None, 'userdef_limbs': None
#         },
#     'train': {'batch_size': 4, 'save_interval': 5000, 'n_step': 1000000, 'lr_init': 0.0001, 'lr_decay_every_step': 136120, 'lr_decay_factor': 0.666, 'weight_decay_factor': 0.0002, 'train_type': <TRAIN.Single_train: 0>, 'vis_dir': './save_dir/default_name/train_vis_dir', 'optim_type': <OPTIM.Adam: 0>},
# 'eval': {'batch_size': 22, 'vis_dir': './save_dir/default_name/eval_vis_dir'},
# 'data': {'dataset_type': <DATA.MSCOCO: 0>, 'dataset_version': '2017', 'dataset_path': './data', 'dataset_filter': None, 'vis_dir': './save_dir/default_name/data_vis_dir', 'official_flag': True, 'userdef_dataset': None, 'useradd_flag': False, 'useradd_scale_rate': 1, 'useradd_train_img_paths': None, 'useradd_train_targets': None, 'domainadapt_flag': False, 'domainadapt_scale_rate': 1, 'domainadapt_train_img_paths': None},
# 'log': {'log_interval': 100, 'log_path': './save_dir/default_name/log.txt'},
# 'pretrain': {'enable': True, 'lr_init': 0.0005, 'batch_size': 32,
# 'total_step': 370000000, 'log_interval': 100, 'val_interval': 5000,
# 'save_interval': 5000, 'weight_decay_factor': 1e-05, 'pretrain_dataset_path': './data', 'pretrain_model_dir': './save_dir/pretrain_backbone', 'val_num': 20000, 'lr_decay_step': 170000}}



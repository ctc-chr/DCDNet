params = dict()

params['num_classes'] = 101

params['dataset'] = 'dataset_path'

params['epoch_num'] = 20
params['batch_size'] = 32
params['step'] = 10
params['num_workers'] = 4
params['learning_rate'] = 3e-3
params['momentum'] = 0.9
params['weight_decay'] = 1e-3
params['display'] = 10
params['pretrained'] = r'r3d50_KM_200ep.pth'
# params['pretrained'] = None
params['gpu'] = [0]
params['log'] = 'resnet_V2_UCF101(new)'
params['best_model_path'] = 'best_model'
params['save_path'] = 'HMDB51'
params['clip_len'] = 16
params['frame_sample_rate'] = 1

# # UCF101
# params['epoch_num'] = 100
# params['batch_size'] = 32
# params['step'] = 10
# params['num_workers'] = 4
# params['learning_rate'] = 0.05
# params['momentum'] = 0.9
# params['weight_decay'] = 1e-4
# params['display'] = 10
# params['pretrained'] = r'r3d50_K_200ep.pth'
# # params['pretrained'] = None
# params['gpu'] = [0]
# params['log'] = 'resnet_V2_UCF101(new)'
# params['save_path'] = 'HMDB51'
# params['clip_len'] = 16
# params['frame_sample_rate'] = 1

# # ssv1
# params['epoch_num'] = 120
# params['batch_size'] = 32
# params['step'] = 40
# params['num_workers'] = 4
# params['learning_rate'] = 0.01
# params['momentum'] = 0.9
# params['weight_decay'] = 1e-33
# params['display'] = 10
# # params['pretrained'] = r'r3d50_K_200ep.pth'
# params['pretrained'] = None
# params['gpu'] = [0]
# params['log'] = 'resnet_V2_SSV1'
# params['best_model_path'] = 'best_model'
# params['save_path'] = 'HMDB51'
# params['clip_len'] = 16
# params['frame_sample_rate'] = 1

# params['epoch_num'] = 50
# params['batch_size'] = 32
# params['step'] = 10
# params['num_workers'] = 4
# params['learning_rate'] = 3e-3
# params['momentum'] = 0.9
# params['weight_decay'] = 1e-3
# params['display'] = 10
# params['pretrained'] = r'r3d50_K_200ep.pth'
# # params['pretrained'] = None
# params['gpu'] = [0]
# params['log'] = 'resnet_V2_include'
# params['best_model_path'] = 'best_model'
# params['save_path'] = 'HMDB51'
# params['clip_len'] = 16
# params['frame_sample_rate'] = 1

# # slowfast
# params['epoch_num'] = 20
# params['batch_size'] = 32
# params['step'] = 10
# params['num_workers'] = 4
# params['learning_rate'] = 0.003
# params['momentum'] = 0.9
# params['weight_decay'] = 1e-3
# params['display'] = 10
# # params['pretrained'] = r'/home/tian903/ZF/project/pre_training_file/slowfast_pretrain/slowfast_r50_8xb8-8x8x1-256e_kinetics400-rgb_20220818-1cb6dfc8.pth'
# # params['pretrained'] = r'x3d_s.pyth'
# params['pretrained'] = r'x3d_s_13x6x1_facebook-kinetics400-rgb_20201027-623825a0.pth'
# # params['pretrained'] = None
# params['gpu'] = [0]
# # params['log'] = r'/home/tian903/CTC/SlowFastNetworks-new/resnet_compare'
# params['log'] = r'/home/tian903/CTC/SlowFastNetworks-new/X3D_UCF101'
# params['best_model_path'] = 'best_model'
# params['save_path'] = 'HMDB51'
# params['clip_len'] = 16
# params['frame_sample_rate'] = 1

# r3d默认参数 SGD cosin    Would Mega-scale Datasets Further Enhance Spatiotemporal 3D CNNs?
# params['epoch_num'] = 30
# params['batch_size'] = 32
# params['step'] = 10
# params['num_workers'] = 4
# params['learning_rate'] = 3e-3
# params['momentum'] = 0.9
# params['weight_decay'] = 1e-3
# params['display'] = 10
# params['pretrained'] = r'r3d50_K_200ep.pth'
# # params['pretrained'] = None
# params['gpu'] = [0]
# params['log'] = 'resnet_HMDB'
# params['save_path'] = 'HMDB51'
# params['clip_len'] = 16
# params['frame_sample_rate'] = 1

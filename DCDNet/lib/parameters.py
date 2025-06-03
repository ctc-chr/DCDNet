import torch
from thop import profile
# from slowfastnet_two import SlowFast
# from slowfastnet_three import SlowFast
# from slowfastnet_TtS import SlowFast
# from slowfastnet_original import resnet50
# from modules.r3d import generate_model
from modules.r3d_v2 import generate_model
from modules.efficientent3d import efficientnet3d_b0


model = efficientnet3d_b0(num_classes=101)

input_tensor = torch.ones(4, 3, 16, 112, 112)
flops, params = profile(model, inputs=(input_tensor,))
print('********FLOPs = ' + str(flops / 1000 ** 3) + 'G')
print('********Params = ' + str(params / 1000 ** 2) + 'M')

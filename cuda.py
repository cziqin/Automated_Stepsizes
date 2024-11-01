import torch
x = torch.rand(5,3)
print(x)
print(torch.__version__)
num_gpu=torch.cuda.device_count()
print('gpu', torch.cuda.is_available(),num_gpu)
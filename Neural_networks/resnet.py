'''ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Implementation:
    https://github.com/kuangliu/pytorch-cifar
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class resnet1(nn.Module):
    def __init__(self):
        super(resnet1, self).__init__()
        self.res = models.resnet18()
        in_num = self.res.fc.in_features
        self.res.fc = nn.Linear(in_num, 1000) # 1000 is the class_num of imagenet dataset
        for param in self.res.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.res(x)



if __name__ == "__main__":
    # res = resnet1(100)
    # t = torch.randn((10, 3, 244, 244))
    # t = res(t)
    # print(t.shape)

    res = resnet1(10)
    t = torch.randn((10, 3, 32, 32))
    t = res(t)
    print(t.shape)

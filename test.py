"""
训练器模块
"""
import os
import unet
import torch
import os
import cv2
import torch.nn as nn
import torchvision

from torch.utils.data import Dataset
from torchvision.utils import save_image

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Datasets(Dataset):

    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的图片和标签
        self.name1 = os.listdir(os.path.join(path, "images"))
        self.name1.sort()
        self.count = 0
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.name1)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return new_img

    def __getitem__(self,index):
        # 拿到的图片和标签
        count = self.count
        name1 = self.name1[count]
        # 图片和标签的路径
        img_path = [os.path.join(self.path, i) for i in ('images', "labels")]
        # 读取原始图片和标签，并转RGB
        img_o = cv2.imread(os.path.join(img_path[0], name1))
        img_o = cv2.cvtColor(img_o, cv2.COLOR_BGR2RGB)
        # 转成网络需要的正方形
        img_o = self.__trans__(img_o, 512)
        self.count = self.count + 1
        return self.trans(img_o)

# 训练
class Trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络
        self.net = unet.UNet().to(self.device)
        # 优化器，这里用的Adam，跑得快点
        self.opt = torch.optim.Adam(self.net.parameters())
        # 这里直接使用二分类交叉熵来训练，效果可能不那么好
        # 可以使用其他损失，比如DiceLoss、FocalLoss之类的
        self.loss_func = nn.BCELoss()
        # 设备好，batch_size和num_workers可以给大点
        self.loader = DataLoader(Datasets(path), batch_size=1, shuffle=True, num_workers=0)

        # 判断是否存在模型
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("No Param!")
        os.makedirs(img_save_path, exist_ok=True)
    def head_test(self):
        epoch = 1
        while True:
            for inputs in tqdm(self.loader,ascii=True, total=len(self.loader)):
                # 图片和分割标签
                inputs = inputs.to(self.device)
                # 输出生成的图像
                out = self.net(inputs)
                # 输入的图像，取第一张
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = out[0]
                # 三张图，从第0轴拼接起来，再保存
                img = torch.stack([x, x_], 0)
                save_image(img.cpu(), os.path.join(self.img_save_path, f"{epoch}.png"))
                epoch += 1



if __name__ == '__main__':
    # 路径改一下
    t = Trainer(r"output", r'./version-1/model_300_0.02231956645846367.plt', r'./model_{}_{}.plt', img_save_path=r'./test_img')
    t.head_test()
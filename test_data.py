import os
from PIL import Image
import torchvision.transforms as transforms

class test_dataset:# 定义一个用于测试数据集的类
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith(('.png','.jpg'))]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        img_name=self.img_list[self.index]
        image = self.binary_loader(os.path.join(self.image_root,img_name+ '.png'))
        gt_path = os.path.join(self.gt_root, img_name + '.png')
        if not os.path.exists(gt_path):
            gt_path = os.path.join(self.gt_root, img_name + '.jpg')
        gt = self.binary_loader(gt_path)
        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


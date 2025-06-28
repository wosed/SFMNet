import os
import os.path
import torch.utils.data as data
from PIL import Image
from config import train_data

def make_dataset(root):
    img_path = os.path.join(root, 'RGB')
    print(img_path)
    depth_path = os.path.join(root, 'depth')
    gt_path = os.path.join(root, 'GT')
    file_extensions = ['.jpg', '.png']
    # 收集图像文件的文件名（不含后缀），去重处理
    img_names = set()
    for file_ext in file_extensions:
        for filename in os.listdir(img_path):
            if filename.endswith(file_ext):
                img_names.add(os.path.splitext(filename)[0])
    dataset = []
    for img_name in img_names:
        img_file_path = os.path.join(img_path,  f"{img_name}.jpg")
        if not os.path.exists(img_file_path):
            continue
        for depth_ext in file_extensions:
            depth_file_path = os.path.join(depth_path,f"{img_name}{depth_ext}")
            gt_file_path = os.path.join(gt_path, f"{img_name}.png")
            if os.path.exists(depth_file_path) and os.path.exists(gt_file_path):
                dataset.append((img_file_path, depth_file_path, gt_file_path))
                break
    return dataset

class ImageFolder(data.Dataset):
    def __init__(self, root, joint_transform=None, transform=None, target_transform=None):#初始化函数，接受根目录路径和可选的联合变换、图像变换和目标变换
        self.root = root
        #print(root)
        self.imgs = make_dataset(root)
        self.joint_transform = joint_transform
        self.transform = transform
        self.target_transform = target_transform
    def __getitem__(self, index):
        img_path, depth_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        target = Image.open(gt_path).convert('L')
        depth = Image.open(depth_path).convert('L')
        if self.joint_transform is not None:
            img, depth, target = self.joint_transform(img,depth, target)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
            depth = self.target_transform(depth)

        return img, depth,target

    def __len__(self):
        return len(self.imgs)
# 创建 ImageFolder 类的实例
if __name__ == "__main__":
    root = train_data  # 假设 train_data 是正确的根目录路径
    print("Root directory:", root)
    try:
        dataset = ImageFolder(root)
        print("Dataset length:", len(dataset))
    except Exception as e:
        print(f"Error: {e}")

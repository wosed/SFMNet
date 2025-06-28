import numpy as np
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from config import test_data
from misc import check_mkdir, crf_refine
from SFMNet import RGBD_sal

torch.manual_seed(2018)
# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

ckpt_path = 'model1'
exp_name = 'model_vgg16_DANet'
snapshot = '20500'
state_dict = torch.load(os.path.join(ckpt_path, exp_name, snapshot + '.pth'), map_location=device)# 指定设备映射

args = {
    'snapshot': '20500',
    'crf_refine':False,
    'save_results': True
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

])

depth_transform = transforms.ToTensor()
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

to_test = {'test':test_data}

def main():
    net = RGBD_sal().to(device)  # 将模型移到指定设备
    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    pretrained_dict = torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location=device)
    model_dict = net.state_dict()
    filtered_pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(filtered_pretrained_dict)
    net.load_state_dict(model_dict)
    net.eval()
    with torch.no_grad():

        for name, root in to_test.items():
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
            root1 = os.path.join(root,'GT')
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root1) if f.endswith(('.jpg','.png'))]

            for idx, img_name in enumerate(img_list):

                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                img1 = Image.open(os.path.join(root,'RGB',img_name + '.jpg')).convert('RGB')
                depth_file_path = os.path.join(root, 'depth', img_name + '.jpg')
                if os.path.exists(depth_file_path):
                    depth = Image.open(depth_file_path).convert('L')
                else:
                    depth_file_path = os.path.join(root, 'depth', img_name + '.png')
                    depth = Image.open(depth_file_path).convert('L')
                img = img1
                w_,h_ = img1.size
                img1 = img1.resize([384,384])
                depth = depth.resize([384,384])
                img_var = Variable(img_transform(img1).unsqueeze(0)).to(device) # 改将输入数据移到指定设备
                depth = Variable(depth_transform(depth).unsqueeze(0)).to(device) # 改将深度数据移到指定设备
                prediction = net(img_var,depth)
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w_, h_), Image.BILINEAR)
                if args['crf_refine']:
                    prediction = crf_refine(np.array(img), np.array(prediction))
                prediction = np.array(prediction)
                if args['save_results']:
                    Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name ,'(%s) %s_%s' % (
                            exp_name, name, args['snapshot'],), img_name + '.png'))




if __name__ == '__main__':
    main()

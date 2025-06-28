import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from multi_scale_module import NMS

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.ratio = ratio
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        in_planes = x.size(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False).to(x.device)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False).to(x.device)
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        ca=self.sigmoid(out)
        return ca*x

# 空间注意力模块，卷积核大小设置为 3
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_l = torch.cat([avg_out, max_out], dim=1)
        s_a = self.sigmoid(self.conv1(x_l))
        return s_a*x

# SAIE 模块
class SAIE(nn.Module):
    def __init__(self, in_channels):
        super(SAIE, self).__init__()
        self.ca_combined = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        self.reduce_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, x, y):
        combined=x+y
        # 计算通道注意力
        ca_out = self.ca_combined(self.reduce_conv(torch.cat((x, combined), dim=1)))
        # 计算空间注意力
        sa_out = self.sa(ca_out)
        return sa_out
# FSIE 模块
class DWT(nn.Module):
    def __init__(self, in_channels_list):
        super(DWT, self).__init__()
        # 根据不同通道数设置不同的 J 值，高层对应小 J 值，低层对应大 J 值
        self.J_values = [1, 2, 3, 4, 5]
        self.DWT_list = nn.ModuleList([
            DTCWTForward(J=J, biort='near_sym_b', qshift='qshift_b') for J in self.J_values
        ])
        self.IWT_list = nn.ModuleList([
            DTCWTInverse(biort='near_sym_b', qshift='qshift_b') for _ in in_channels_list
        ])
        self.conv3_list = nn.ModuleList()
        for in_channels in in_channels_list:
            self.conv3_list.append(BasicConv2d(2*in_channels, in_channels))
    def forward(self, x, y,index):
        Xl, Xh = self.DWT_list[index](x)
        enhancement_factor = 1.5
        enhanced_Xh = []
        for scale in Xh:
            enhanced_scale = scale * enhancement_factor
            enhanced_Xh.append(enhanced_scale)
        Yl, Yh = self.DWT_list[index](y)
        enhancement_factor = 1.5
        enhanced_Yh = []
        for scale in Yh:
            enhanced_scale = scale * enhancement_factor
            enhanced_Yh.append(enhanced_scale)
        x_y = Xl+Yl
        x_m = self.IWT_list[index]((x_y, enhanced_Xh))
        y_m = self.IWT_list[index]((x_y,enhanced_Yh))
        out = self.conv3_list[index](torch.cat([x_m, y_m], dim=1))
        return out

class RGBD_sal(nn.Module):

    def __init__(self):
        super(RGBD_sal, self).__init__()
        ################################vgg16#######################################
        # 加载预训练的VGG16模型（带BN层）的特征提取部分
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv0 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(*feats[1:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])
        self.dem1 = NMS(512, 512)
        # 定义一系列卷积层用于特征映射和降维
        self.dem2f =nn.Conv2d(512, 256, kernel_size=3, padding=1)
        # 新增的 PAFEM 模块，在 dem2 操作后引入
        self.dem2 = NMS(256, 256)
        self.dem3f = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.dem4 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.dem5 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.PReLU())
        # 用于显著分支和背景分支的融合卷积层
        self.fuse_1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1), nn.BatchNorm2d(256), nn.PReLU(),nn.Conv2d(256, 1, kernel_size=3, padding=1))
        self.fuse_2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.PReLU(),nn.Conv2d(128, 1, kernel_size=3, padding=1))
        self.fuse_3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU(),nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.fuse_4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=1), nn.BatchNorm2d(32), nn.PReLU(),nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.fuse_5 = nn.Sequential(nn.Conv2d(32, 16, kernel_size=1), nn.BatchNorm2d(16), nn.PReLU(),nn.Conv2d(16, 1, kernel_size=3, padding=1))
        # 输出层相关卷积层
        self.output1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output1_rev = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.PReLU())
        self.output2 = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output2_rev = nn.Sequential(nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.PReLU())
        self.output3 = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output3_rev = nn.Sequential(nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU())
        self.output4 = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32), nn.PReLU())
        self.output4_rev = nn.Sequential(nn.Conv2d(64, 32, kernel_size=3, padding=1),nn.BatchNorm2d(32), nn.PReLU())
        self.output5 = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.output5_rev = nn.Sequential(nn.Conv2d(32, 1, kernel_size=3, padding=1))
        self.fuseout = nn.Sequential(nn.Conv2d(2, 1, kernel_size=3, padding=1), nn.PReLU())
        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True
        # 添加 PDWT 模块
        in_channels_list = [512,256, 128, 64, 32]
        self.dwt = DWT(in_channels_list)
        # 实例化SAIE模块
        self.SAIE = nn.ModuleList([SAIE(in_channels) for in_channels in in_channels_list])
    def forward(self, x,depth,return_attention=False):
        c0 = self.conv0(torch.cat((x,depth),1))
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        ################################PAFEM#######################################
        dem1 = self.dem1(c5)
        dem2 = self.dem2(self.dem2f(c4))
        dem3 = self.dem3f(c3)
        dem4 = self.dem4(c2)
        dem5 = self.dem5(c1)
        ################################DAM for Saliency branch&Background branch#######################################
        # 计算显著分支的注意力图，结合深度图和上采样后的特征
        dem1_attention = F.sigmoid(self.fuse_1(self.dwt(dem1, self.SAIE[0](dem1,F.upsample(depth, size=dem1.size()[2:], mode='bilinear')), 0)))
        output1 = self.output1(dem1 *(dem1_attention+ (dem1_attention *  dem1_attention)))
        output1_rev = self.output1_rev(dem1 *((1-dem1_attention)+ (1-dem1_attention) *  (1-dem1_attention)))
        del dem1
        upsampled_output1 = F.upsample(output1, size=dem2.size()[2:], mode='bilinear')
        dem2_attention = F.sigmoid(self.fuse_2(self.dwt(upsampled_output1,self.SAIE[1](dem2,F.upsample(depth, size=dem2.size()[2:], mode='bilinear')),1)))
        output2 = self.output2(upsampled_output1 * dem2_attention + dem2 * (dem2_attention + (dem2_attention * dem2_attention)))
        output2_rev = self.output2_rev(F.upsample(output1_rev, size=dem2.size()[2:], mode='bilinear') * (1 - dem2_attention) + dem2 * ((1 - dem2_attention) + (1 - dem2_attention) * (1 - dem2_attention)))
        del dem2, output1, output1_rev,upsampled_output1
        upsampled_output2 = F.upsample(output2, size=dem3.size()[2:], mode='bilinear')
        dem3_attention = F.sigmoid(self.fuse_3(self.dwt(upsampled_output2,self.SAIE[2](dem3,F.upsample(depth, size=dem3.size()[2:], mode='bilinear')), 2)))
        output3 = self.output3(upsampled_output2 * dem3_attention + dem3 * (dem3_attention + (dem3_attention * dem3_attention)))
        output3_rev = self.output3_rev(F.upsample(output2_rev, size=dem3.size()[2:], mode='bilinear') * (1 - dem3_attention) + dem3 * ((1 - dem3_attention) + (1 - dem3_attention) * (1 - dem3_attention)))
        del dem3, output2, output2_rev,upsampled_output2  # 删除不再使用的dem3, output2, output2_rev张量
        upsampled_output3 = F.upsample(output3, size=dem4.size()[2:], mode='bilinear')
        dem4_attention = F.sigmoid(self.fuse_4(self.dwt(upsampled_output3,self.SAIE[3](dem4,F.upsample(depth, size=dem4.size()[2:], mode='bilinear')), 3)))
        output4 = self.output4(upsampled_output3 * dem4_attention + dem4 * (dem4_attention + (dem4_attention * dem4_attention)))
        output4_rev = self.output4_rev(F.upsample(output3_rev, size=dem4.size()[2:], mode='bilinear') * (1 - dem4_attention) + dem4 * ((1 - dem4_attention) + (1 - dem4_attention) * (1 - dem4_attention)))
        del dem4, output3, output3_rev,upsampled_output3
        upsampled_output4=F.upsample(output4, size=dem5.size()[2:], mode='bilinear')
        dem5_attention = F.sigmoid(self.fuse_5(self.dwt(upsampled_output4,self.SAIE[4](dem5,F.upsample(depth, size=dem5.size()[2:], mode='bilinear')), 4)))
        output5 = self.output5(upsampled_output4 * dem5_attention + dem5 * (dem5_attention + (dem5_attention * dem5_attention)))
        output5_rev = self.output5_rev(F.upsample(output4_rev, size=dem5.size()[2:], mode='bilinear') * (1 - dem5_attention) + dem5 * ((1 - dem5_attention) + (1 - dem5_attention) * (1 - dem5_attention)))
        del dem5, upsampled_output4
        ################################Dual Branch Fuse#######################################
        output5 = F.upsample(output5, size=x.size()[2:], mode='bilinear')
        output5_rev = F.upsample(output5_rev, size=x.size()[2:], mode='bilinear')
        output = self.fuseout(torch.cat((output5, -output5_rev), 1))  # 双重前景增强
        output = -output5_rev + output

        if return_attention:
            return dem1_attention, dem2_attention, dem3_attention, dem4_attention, dem5_attention
        if self.training:
            return output, output5, output5_rev, dem1_attention, dem2_attention, dem3_attention, dem4_attention, dem5_attention
        return F.sigmoid(output)


if __name__ == "__main__":
    model = RGBD_sal()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    depth = torch.autograd.Variable(torch.randn(4, 1, 384, 384))
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input = input.to(device)
    depth = depth.to(device)
    output = model(input,depth)

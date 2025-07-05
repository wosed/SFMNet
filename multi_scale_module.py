import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Conv2d, Parameter, Softmax

class SpatialAttention(nn.Module):
        def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(cat)
        x = self.sigmoid(out)

        return x
###NMS
class NMS(nn.Module):
    def __init__(self, dim,in_dim):
        super(NMS, self).__init__()
        self.down_conv = nn.Sequential(nn.Conv2d(dim, in_dim, 3, padding=1), nn.BatchNorm2d(in_dim),nn.PReLU())
        down_dim = in_dim // 2
        # 添加空间注意力模块实例化
        self.spatial_attn = SpatialAttention()
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=2, padding=2), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv2 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma2 = Parameter(torch.zeros(1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=4, padding=4), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv3 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma3 = Parameter(torch.zeros(1))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_dim, down_dim, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(down_dim), nn.PReLU()
        )
        self.query_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.key_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim//8, kernel_size=1)
        self.value_conv4 = Conv2d(in_channels=down_dim, out_channels=down_dim, kernel_size=1)
        self.gamma4 = Parameter(torch.zeros(1))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),nn.BatchNorm2d(in_dim),  nn.PReLU()  #如果batch=1 ，进行batchnorm会有问题
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * in_dim, in_dim, kernel_size=1), nn.BatchNorm2d(in_dim), nn.PReLU()
        )
        self.softmax = Softmax(dim=-1)
    def forward(self, x):
        x = self.down_conv(x)
        spatial_attn_weights = self.spatial_attn(x)
        conv1 = spatial_attn_weights*x
        conv2 = self.conv2(conv1)
        m_batchsize, C, height, width = conv2.size()
        proj_query2 = self.query_conv2(conv2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key2 = self.key_conv2(conv2).view(m_batchsize, -1, width * height)
        energy2 = torch.bmm(proj_query2, proj_key2)
        attention2 = self.softmax(energy2)
        proj_value2 = self.value_conv2(conv2).view(m_batchsize, -1, width * height)
        out2 = torch.bmm(proj_value2, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)
        out2 = self.gamma2* out2 + conv2
        out2 = torch.cat([out2 * torch.cos(out2), out2 * torch.sin(out2)], dim=1)
        conv3 = self.conv3(conv1)
        m_batchsize, C, height, width = conv3.size()
        proj_query3 = self.query_conv3(conv3).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key3 = self.key_conv3(conv3).view(m_batchsize, -1, width * height)
        energy3 = torch.bmm(proj_query3, proj_key3)
        attention3 = self.softmax(energy3)
        proj_value3 = self.value_conv3(conv3).view(m_batchsize, -1, width * height)
        out3 = torch.bmm(proj_value3, attention3.permute(0, 2, 1))
        out3 = out3.view(m_batchsize, C, height, width)
        out3 = self.gamma3 * out3 + conv3
        out3 = torch.cat([out3 * torch.cos(out3), out3 * torch.sin(out3)], dim=1)
        conv4 = self.conv4(conv1)
        m_batchsize, C, height, width = conv4.size()
        proj_query4 = self.query_conv4(conv4).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key4 = self.key_conv4(conv4).view(m_batchsize, -1, width * height)
        energy4 = torch.bmm(proj_query4, proj_key4)
        attention4 = self.softmax(energy4)
        proj_value4 = self.value_conv4(conv4).view(m_batchsize, -1, width * height)
        out4 = torch.bmm(proj_value4, attention4.permute(0, 2, 1))
        out4 = out4.view(m_batchsize, C, height, width)
        out4 = self.gamma4 * out4 + conv4
        out4 = torch.cat([out4 * torch.cos(out4), out4 * torch.sin(out4)], dim=1)
        conv5 = nn.functional.interpolate(self.conv5(F.adaptive_max_pool2d(x, 1)), size=x.size()[2:], mode='bilinear',align_corners=True)  # 如果batch设为1，这里就会有问题。
        return self.fuse(torch.cat((conv1,out2,out3,out4,conv5), 1))




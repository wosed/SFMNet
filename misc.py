import numpy as np
import os
import pydensecrf.densecrf as dcrf

# 定义一个用于计算平均值的类
class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 # 当前值
        self.avg = 0
        self.sum = 0
        self.count = 0  # 计数

    def update(self, val, n=1):
        self.val = val # 更新当前值
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):# 检查目录是否存在，如果不存在则创建该目录
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def cal_precision_recall_mae(prediction, gt):# 计算预测结果的精度、召回率、平均绝对误差（MAE）、F-measure和IoU
    # 断言预测结果的数据类型为uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8# 断言真实标签的数据类型为uint8
    print(prediction.shape,gt.shape)
    assert prediction.shape == gt.shape# 断言预测结果和真实标签的形状相同
    eps = 1e-4# 定义一个极小值，用于避免除零错误
    gt = gt / 255# 将真实标签归一化到[0, 1]范围
    # 对预测结果进行归一化处理
    prediction = (prediction-prediction.min())/(prediction.max()-prediction.min()+ eps)
    gt[gt>0.5] = 1# 将真实标签二值化
    gt[gt!=1] = 0
    mae = np.mean(np.abs(prediction - gt))# 计算平均绝对误差
    # 创建一个和预测结果形状相同的数组，用于存储二值化后的真实标签
    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)# 计算真实标签中前景像素的数量
    precision, recall, iou = [], [], []# 初始化精度、召回率和IoU列表
    # 创建一个和预测结果形状相同的二值数组
    binary = np.zeros(gt.shape)
    th = 2 * prediction.mean()# 计算阈值
    if th > 1:
        th = 1
    binary[prediction >= th] = 1# 根据阈值将预测结果二值化
    sb = (binary * gt).sum()# 计算真阳性的数量
    pre_th = (sb+eps) / (binary.sum() + eps)# 计算基于阈值的精度
    rec_th = (sb+eps) / (gt.sum() + eps)# 计算基于阈值的召回率
    thfm = 1.3 * pre_th * rec_th / (0.3*pre_th + rec_th + eps)# 计算基于阈值的F-measure

    # 遍历256个不同的阈值
    for threshold in range(256):
        threshold = threshold / 255.# 将阈值归一化到[0, 1]范围
        # 创建一个和预测结果形状相同的数组，用于存储基于当前阈值的二值化预测结果
        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1
        # 计算真阳性的数量
        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction) # 计算预测结果中前景像素的数量
        iou.append((tp + eps) / (p+t-tp + eps))# 计算IoU并添加到列表中
        precision.append((tp + eps) / (p + eps))# 计算精度并添加到列表中
        recall.append((tp + eps) / (t + eps))# 计算召回率并添加到列表中


    return precision, recall, iou,mae,thfm


# 计算最大F-measure和最大IoU
def cal_fmeasure(precision, recall,iou): #iou
    beta_square = 0.3# 定义beta的平方
    # 计算所有精度和召回率组合下的F-measure，并找出最大值
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])
    #loc = [(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)]# 存储所有F-measure的值
    #a = loc.index(max(loc))# 找出最大F-measure对应的索引
    max_iou = max(iou)# 找出最大IoU

    return max_fmeasure,max_iou



# 使用条件随机场（CRF）对预测结果进行细化
def crf_refine(img, annos):
    def _sigmoid(x):# 定义sigmoid函数
        return 1 / (1 + np.exp(-x))

    # 断言输入图像的数据类型为uint8
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8# 断言预测结果的数据类型为uint8
    print(img.shape[:2],annos.shape)
    assert img.shape[:2] == annos.shape# 断言输入图像和预测结果的形状相同

    # img and annos should be np array with data type uint8
    # 输入图像和预测结果应该是数据类型为uint8的numpy数组
    EPSILON = 1e-8
    # 类别数量，这里为2（前景和背景）
    M = 2  # salient or not
    tau = 1.05# 一个超参数
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)# 初始化一个二维的密集条件随机场
    # 将预测结果归一化到[0, 1]范围
    anno_norm = annos / 255.
    # 计算背景能量
    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))# 计算前景能量
    # 创建一个大小为(M, 图像像素数量)的数组，用于存储一元能量
    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32') # 创建和输入图片同样大小的U
    U[0, :] = n_energy.flatten()# 将背景能量展平后存储在U的第一行
    U[1, :] = p_energy.flatten()# 将前景能量展平后存储在U的第二行
    # 设置一元能量
    d.setUnaryEnergy(U)
    # 添加高斯成对势
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)# 添加双边成对势

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')# 进行推理
    res = infer[1, :]# 提取前景的推理结果
    # 将推理结果缩放回[0, 255]范围
    res = res * 255
    res = res.reshape(img.shape[:2])  # 和输入图片同样大小
    return res.astype('uint8')

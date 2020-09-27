"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
from Model.config import args
import torch.nn.functional as F


def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
    dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

    if (penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    return d / 3.0


def mse_loss(x, y):
    return torch.mean((x - y) ** 2)


def DSC(pred, target):
    smooth = 1e-5
    m1 = pred.flatten()
    m2 = target.flatten()
    intersection = (m1 * m2).sum()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


def ncc_loss(I, J, win=None):
    '''
    输入大小是[B,C,D,W,H]格式的，在计算ncc时用卷积来实现指定窗口内求和
    '''
    ndims = len(list(I.size())) - 2
    assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims
    if win is None:
        win = [9] * ndims
    sum_filt = torch.ones([1, 1, *win]).to("cuda:{}".format(args.gpu))
    pad_no = math.floor(win[0] / 2)
    stride = [1] * ndims
    padding = [pad_no] * ndims
    I_var, J_var, cross = compute_local_sums(I, J, sum_filt, stride, padding, win)
    cc = cross * cross / (I_var * J_var + 1e-5)
    return -1 * torch.mean(cc)


def compute_local_sums(I, J, filt, stride, padding, win):
    I2, J2, IJ = I * I, J * J, I * J
    I_sum = F.conv3d(I, filt, stride=stride, padding=padding)
    J_sum = F.conv3d(J, filt, stride=stride, padding=padding)
    I2_sum = F.conv3d(I2, filt, stride=stride, padding=padding)
    J2_sum = F.conv3d(J2, filt, stride=stride, padding=padding)
    IJ_sum = F.conv3d(IJ, filt, stride=stride, padding=padding)
    win_size = np.prod(win)
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    return I_var, J_var, cross


def cc_loss(x, y):
    # 根据互相关公式进行计算
    dim = [2, 3, 4]
    mean_x = torch.mean(x, dim, keepdim=True)
    mean_y = torch.mean(y, dim, keepdim=True)
    mean_x2 = torch.mean(x ** 2, dim, keepdim=True)
    mean_y2 = torch.mean(y ** 2, dim, keepdim=True)
    stddev_x = torch.sum(torch.sqrt(mean_x2 - mean_x ** 2), dim, keepdim=True)
    stddev_y = torch.sum(torch.sqrt(mean_y2 - mean_y ** 2), dim, keepdim=True)
    return -torch.mean((x - mean_x) * (y - mean_y) / (stddev_x * stddev_y))


def Get_Ja(flow):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    D_y = (flow[:, 1:, :-1, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_x = (flow[:, :-1, 1:, :-1, :] - flow[:, :-1, :-1, :-1, :])
    D_z = (flow[:, :-1, :-1, 1:, :] - flow[:, :-1, :-1, :-1, :])
    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) * (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])
    D2 = (D_x[..., 1]) * (D_y[..., 0] * (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])
    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] - (D_y[..., 1] + 1) * D_z[..., 0])
    return D1 - D2 + D3


def NJ_loss(ypred):
    '''
    Penalizing locations where Jacobian has negative determinants
    '''
    Neg_Jac = 0.5 * (torch.abs(Get_Ja(ypred)) - Get_Ja(ypred))
    return torch.sum(Neg_Jac)

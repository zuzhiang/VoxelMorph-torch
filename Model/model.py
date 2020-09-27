import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class U_Network(nn.Module):
    def __init__(self, dim, enc_nf, dec_nf, bn=None, full_size=True):
        super(U_Network, self).__init__()
        self.bn = bn
        self.dim = dim
        self.enc_nf = enc_nf
        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7
        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(self.conv_block(dim, prev_nf, enc_nf[i], 4, 2, batchnorm=bn))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(self.conv_block(dim, enc_nf[-1], dec_nf[0], batchnorm=bn))  # 1
        self.dec.append(self.conv_block(dim, dec_nf[0] * 2, dec_nf[1], batchnorm=bn))  # 2
        self.dec.append(self.conv_block(dim, dec_nf[1] * 2, dec_nf[2], batchnorm=bn))  # 3
        self.dec.append(self.conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3], batchnorm=bn))  # 4
        self.dec.append(self.conv_block(dim, dec_nf[3], dec_nf[4], batchnorm=bn))  # 5

        if self.full_size:
            self.dec.append(self.conv_block(dim, dec_nf[4] + 2, dec_nf[5], batchnorm=bn))
        if self.vm2:
            self.vm2_conv = self.conv_block(dim, dec_nf[5], dec_nf[6], batchnorm=bn)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # One conv to get the flow field
        conv_fn = getattr(nn, 'Conv%dd' % dim)
        self.flow = conv_fn(dec_nf[-1], dim, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.batch_norm = getattr(nn, "BatchNorm{0}d".format(dim))(3)

    def conv_block(self, dim, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batchnorm=False):
        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        bn_fn = getattr(nn, "BatchNorm{0}d".format(dim))
        if batchnorm:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                bn_fn(out_channels),
                nn.LeakyReLU(0.2))
        else:
            layer = nn.Sequential(
                conv_fn(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.LeakyReLU(0.2))
        return layer

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], dim=1)
        # Get encoder activations
        x_enc = [x]
        for i, l in enumerate(self.enc):
            x = l(x_enc[-1])
            x_enc.append(x)
        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)
        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)
        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)
        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)
        flow = self.flow(y)
        if self.bn:
            flow = self.batch_norm(flow)
        return flow


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode)

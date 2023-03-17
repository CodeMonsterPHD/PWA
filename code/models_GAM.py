import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch
import numpy as np
import trilinear


def nxor_for_gtmask(gtmask):
    gtmask = gtmask.cuda()
    img_left_top = torch.zeros(gtmask.shape)
    img_left_top = img_left_top.cuda()
    img_left_top[:, :, 1:, 1:] = gtmask[:, :, :-1, :-1]

    channel1 = torch.logical_xor(gtmask, img_left_top).cuda()

    channel1 = torch.logical_not(channel1).cuda()
    channel1 = channel1.float()

    img_top = torch.zeros(gtmask.shape).cuda()
    img_top[:, :, 1:, :] = gtmask[:, :, :-1, :]

    channel2 = torch.logical_xor(gtmask, img_top).cuda()

    channel2 = torch.logical_not(channel2).cuda()
    channel2 = channel2.float()

    img_right_top = torch.zeros(gtmask.shape).cuda()
    img_right_top[:, :, 1:, :-1] = gtmask[:, :, :-1, 1:]

    channel3 = torch.logical_xor(gtmask, img_right_top).cuda()

    channel3 = torch.logical_not(channel3).cuda()
    channel3 = channel3.float()

    img_left = torch.zeros(gtmask.shape).cuda()
    img_left[:, :, :, 1:] = gtmask[:, :, :, :-1]

    channel4 = torch.logical_xor(gtmask, img_left).cuda()

    channel4 = torch.logical_not(channel4).cuda()
    channel4 = channel4.float()

    img_right = torch.zeros(gtmask.shape).cuda()
    img_right[:, :, :, :-1] = gtmask[:, :, :, 1:]

    channel6 = torch.logical_xor(gtmask, img_right).cuda()

    channel6 = torch.logical_not(channel6).cuda()
    channel6 = channel6.float()

    img_left_down = torch.zeros(gtmask.shape).cuda()
    img_left_down[:, :, :-1, 1:] = gtmask[:, :, 1:, :-1]

    channel7 = torch.logical_xor(gtmask, img_left_down).cuda()

    channel7 = torch.logical_not(channel7).cuda()
    channel7 = channel7.float()

    img_down = torch.zeros(gtmask.shape).cuda()
    img_down[:, :, :-1, :] = gtmask[:, :, 1:, :]

    channel8 = torch.logical_xor(gtmask, img_down).cuda()

    channel8 = torch.logical_not(channel8).cuda()
    channel8 = channel8.float()

    img_right_down = torch.zeros(gtmask.shape).cuda()
    img_right_down[:, :, :-1, :-1] = gtmask[:, :, 1:, 1:]

    channel9 = torch.logical_xor(gtmask, img_right_down).cuda()

    channel9 = torch.logical_not(channel9).cuda()
    channel9 = channel9.float()

    mask = torch.cat([channel1, channel2, channel3, channel4, gtmask, channel6, channel7, channel8, channel9], dim=1)

    return mask


def Img_X_Mask(img, mask):
    x = img
    size = x.size()

    mask = mask.reshape(size[0], 1, size[2] * size[3], 3 * 3)

    x = F.unfold(x, kernel_size=[3, 3], padding=1)

    x = x.reshape(size[0], size[1], size[2] * size[3], -1)

    x = torch.mul(x, mask)
    x = torch.sum(x, dim=3)
    x = x.reshape(size[0], size[1], size[2], size[3])

    return x


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__

    if classname.find("Conv") != -1:

        torch.nn.init.xavier_normal_(m.weight.data)


    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:

        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class MaskContext(nn.Module):

    def __init__(self):
        super(MaskContext, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(64, 9, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, transform_img):
        out1 = self.relu(self.conv_1(transform_img))
        out2 = self.sigmoid(self.conv_2(out1))

        return out2


class ContextModulate(nn.Module):

    def __init__(self):
        super(ContextModulate, self).__init__()
        self.conv_1 = nn.Conv2d(27, 128, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(128, 5, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        nn_Unfold = nn.Unfold(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        output_img = nn_Unfold(x)
        transform_img = output_img.view(x.shape[0], 27, x.shape[2], x.shape[3])
        out1 = self.relu(self.conv_1(transform_img))
        out2 = self.conv_2(out1)

        return out2


class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test

        net = models.resnet18(pretrained=True)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')

        net.fc = nn.Linear(512, out_dim)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):

        super(Generator3DLUT_identity, self).__init__()

        if dim == 33:
            file = open("/data1/user12/PPR10K_LCZ/code_3DLUT/IdentityLUT33.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k

                    x = lines[n].split()

                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])

        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = nn.init.kaiming_normal_(torch.zeros(3, dim, dim, dim, dtype=torch.float), mode="fan_in",
                                           nonlinearity="relu")

        self.LUT = nn.Parameter(torch.tensor(self.LUT))

        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):

        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        if batch == 1:
            assert 1 == trilinear.forward(lut, x, output, dim, shift, binsize, W, H, batch)
        elif batch > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(lut, x.permute(1, 0, 2, 3).contiguous(), output, dim, shift, binsize, W, H,
                                          batch)

            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        if batch == 1:
            assert 1 == trilinear.backward(x, x_grad, lut_grad, dim, shift, binsize, W, H, batch)
        elif batch > 1:
            assert 1 == trilinear.backward(x.permute(1, 0, 2, 3).contiguous(), x_grad.permute(1, 0, 2, 3).contiguous(),
                                           lut_grad, dim, shift, binsize, W, H, batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn


def Ff(X):
    FX = 7.787 * X + 0.137931
    index = X > 0.008856
    FX[index] = torch.pow(X[index], 1.0 / 3.0)
    return FX


def myRGB2Lab(img):
    X = (0.412453 * img[:, 0, :, :] + 0.357580 * img[:, 1, :, :] + 0.180423 * img[:, 2, :, :]) / 0.950456
    Y = (0.212671 * img[:, 0, :, :] + 0.715160 * img[:, 1, :, :] + 0.072169 * img[:, 2, :, :])
    Z = (0.019334 * img[:, 0, :, :] + 0.119193 * img[:, 1, :, :] + 0.950227 * img[:, 2, :, :]) / 1.088754

    F_X = Ff(X)
    F_Y = Ff(Y)
    F_Z = Ff(Z)

    L = 903.3 * Y
    index = Y > 0.008856
    L[index] = 116 * F_Y[index] - 16
    a = 500 * (F_X - F_Y)
    b = 200 * (F_Y - F_Z)

    L = L
    a = (a + 128.0)
    b = (b + 128.0)
    return torch.stack([L, a, b], dim=1)


def calculate_Lab_ab(img):
    Lab_img = myRGB2Lab(img)
    a = Lab_img[:, 1, :, :]
    b = Lab_img[:, 2, :, :]
    a = torch.mean(torch.flatten(a, 1), dim=1)
    b = torch.mean(torch.flatten(b, 1), dim=1)

    return a.float(), b.float()

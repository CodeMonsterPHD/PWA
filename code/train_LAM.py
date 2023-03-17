import argparse
import math
import itertools
import time
import datetime
import sys
from torch.backends import cudnn
from torch.cuda.amp import GradScaler, autocast

from torch.utils.data import DataLoader
from torch.autograd import Variable
from models_LAM import *
from datasets import *

cudnn.benchmark = True
cudnn.deterministic = True
cudnn.enabled = True

parser = argparse.ArgumentParser()

parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=200)
parser.add_argument("--data_path", type=str, default="PPR10K_dataset")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--b1", type=float, default=0.9)
parser.add_argument("--b2", type=float, default=0.999)
parser.add_argument("--lambda_smooth", type=float, default=0.0001)
parser.add_argument("--lambda_monotonicity", type=float, default=10.0)
parser.add_argument("--use_mask", type=bool, default=True)
parser.add_argument("--lut_dim", type=int, default=49)
parser.add_argument("--n_cpu", type=int, default=1)
parser.add_argument("--checkpoint_interval", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="Local_context_aware")
opt = parser.parse_args()

print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs("%s" % opt.output_dir, exist_ok=True)

criterion_pixelwise = torch.nn.MSELoss()


def criterion_maskwise(a, b):
    loss = F.binary_cross_entropy_with_logits(a, b)
    return loss


MaskContext = MaskContext()
Context = ContextModulate()
LUT1 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT2 = Generator3DLUT_zero(dim=opt.lut_dim)
LUT3 = Generator3DLUT_zero(dim=opt.lut_dim)
LUT4 = Generator3DLUT_zero(dim=opt.lut_dim)
LUT5 = Generator3DLUT_zero(dim=opt.lut_dim)

classifier = resnet18_224(out_dim=5)

TV3 = TV_3D(dim=opt.lut_dim)

trilinear_ = TrilinearInterpolation()

if cuda:
    MaskContext = MaskContext.cuda()
    Context = Context.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    LUT3 = LUT3.cuda()
    LUT4 = LUT4.cuda()
    LUT5 = LUT5.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

if opt.epoch != 0:
    LUTs = torch.load("%s/LUTs_%d.pth" % (opt.output_dir, opt.epoch))
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    LUT3.load_state_dict(LUTs["3"])
    LUT4.load_state_dict(LUTs["4"])
    LUT5.load_state_dict(LUTs["5"])
    classifier.load_state_dict(torch.load("%s/classifier_%d.pth" % (opt.output_dir, opt.epoch)))

optimizer_G = torch.optim.Adam(
    itertools.chain(MaskContext.parameters(), Context.parameters(), classifier.parameters(), LUT1.parameters(),
                    LUT2.parameters(),
                    LUT3.parameters(), LUT4.parameters(), LUT5.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

dataloader = DataLoader(
    ImageDataset_paper(opt.data_path, mode="train", use_mask=opt.use_mask),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

psnr_dataloader = DataLoader(
    ImageDataset_paper(opt.data_path, mode="test", use_mask=opt.use_mask),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


def get_index(mask):
    x = F.avg_pool2d(mask, kernel_size=[3, 3], padding=1, stride=1)

    zeros = torch.zeros_like(x)
    x = torch.where(x == 1, zeros, x).squeeze()
    x = x.reshape(448 * 448)

    index = torch.nonzero(x).squeeze()
    index = index.unsqueeze(0).unsqueeze(0)
    index = torch.cat([index, index, index, index, index, index, index, index, index], dim=1)
    index = index.long()

    return index


def get_edge_loss(gtmask, mymask):
    bce = 0
    size = gtmask.size()
    gtmask_9 = nxor_for_gtmask(gtmask)
    gtmask_9 = gtmask_9.reshape(size[0], 9, 448 * 448)
    mymask_9 = mymask.reshape(size[0], 9, 448 * 448)
    for i in range(0, size[0]):
        gtmask_index = get_index(gtmask[i].unsqueeze(0))
        if gtmask_index.shape[2] == 0:
            continue
        output_gt = torch.gather(gtmask_9, dim=2, index=gtmask_index)
        output_my = torch.gather(mymask_9, dim=2, index=gtmask_index)

        bce += criterion_maskwise(output_my, output_gt)

    return bce


def generator0(img):
    pred_mask = MaskContext(img)
    imgs = Img_X_Mask(img, pred_mask)
    Mask = Context(imgs).squeeze()
    pred = classifier(img).squeeze()
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)
    gen_A3 = LUT3(img)
    gen_A4 = LUT4(img)
    gen_A5 = LUT5(img)

    combine_A = img.new(img.size())
    combine_A[0, :, :, :] = (
            torch.mul(gen_A1, Mask[0]) * pred[0] + torch.mul(gen_A2, Mask[1]) * pred[1] +
            torch.mul(gen_A3, Mask[2]) * pred[2] + torch.mul(gen_A4, Mask[3]) * pred[3] +
            torch.mul(gen_A5, Mask[4]) * pred[4])

    weights_norm = torch.mean(pred ** 2)

    return combine_A, weights_norm


def generator_batch(img):
    pred_mask = MaskContext(img)
    imgs = Img_X_Mask(img, pred_mask)
    Mask = Context(imgs).squeeze()
    pred = classifier(img).squeeze()
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)
    gen_A3 = LUT3(img)
    gen_A4 = LUT4(img)
    gen_A5 = LUT5(img)

    combine_A = img.new(img.size())
    for i in range(img.size(0)):
        combine_A[i, :, :, :] = (
                torch.mul(gen_A1[i, :, :, :], Mask[i, 0, :, :]) * pred[i, 0] +
                torch.mul(gen_A2[i, :, :, :], Mask[i, 1, :, :]) * pred[i, 1] +
                torch.mul(gen_A3[i, :, :, :], Mask[i, 2, :, :]) * pred[i, 2] +
                torch.mul(gen_A4[i, :, :, :], Mask[i, 3, :, :]) * pred[i, 3] +
                torch.mul(gen_A5[i, :, :, :], Mask[i, 4, :, :]) * pred[i, 4])

    weights_norm = torch.mean(pred ** 2)

    return combine_A, weights_norm, pred_mask


def calculate_psnr():
    classifier.eval()
    Context.eval()
    MaskContext.eval()
    sum_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        fake_B, x = generator0(real_A)
        fake_B = torch.round(fake_B * 255)
        real_B = torch.round(real_B * 255)
        try:
            mse = criterion_pixelwise(fake_B, real_B)
        except:
            print(batch["input_name"])
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        sum_psnr += psnr

    return sum_psnr / len(psnr_dataloader)


time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

prev_time = time.time()

max_psnr = 0
max_epoch = 0
scaler = GradScaler()

for epoch in range(opt.epoch, opt.n_epochs):

    mse_sum = 0
    psnr_sum = 0

    classifier.train()
    Context.train()
    MaskContext.train()

    for i, batch in enumerate(dataloader):

        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        if opt.use_mask:
            mask = Variable(batch["mask"].type(Tensor))
            mask = torch.sum(mask, 1).unsqueeze(1)
            a = torch.full(mask.shape, 1., dtype=float).cuda()
            a = a.float()
            b = torch.zeros_like(mask).cuda().float()
            b = b.float()
            mask = torch.where(mask > 0, a, b)
            mask = mask.float()
            weights = torch.ones_like(mask)
            weights[mask > 0] = 5

        optimizer_G.zero_grad()
        with autocast():

            fake_B, weights_norm, pred_mask = generator_batch(real_A)

            if opt.use_mask:

                mse = criterion_pixelwise(fake_B * weights, real_B * weights)
                bce = get_edge_loss(mask, pred_mask)

                mse = mse + bce

            else:
                mse = criterion_pixelwise(fake_B, real_B)

            tv1, mn1 = TV3(LUT1)
            tv2, mn2 = TV3(LUT2)
            tv3, mn3 = TV3(LUT3)
            tv4, mn4 = TV3(LUT4)
            tv5, mn5 = TV3(LUT5)
            tv_cons = tv1 + tv2 + tv3 + tv4 + tv5
            mn_cons = mn1 + mn2 + mn3 + mn4 + mn5

            loss = mse + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons

            psnr_sum += 10 * math.log10(1 / mse.item())

            mse_sum += mse.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer_G)
        scaler.update()

        batches_done = epoch * len(dataloader) + i

        batches_left = opt.n_epochs * len(dataloader) - batches_done

        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

        prev_time = time.time()

        sys.stdout.write(
            "\r%s, [Epoch %d/%d] [Batch %d/%d] ETA: %s"
            % (
                opt.output_dir, epoch, opt.n_epochs, i, len(dataloader),
                time_left,
            )
        )

    avg_psnr = calculate_psnr()

    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch
    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))

    if epoch % opt.checkpoint_interval == 0:
        LUTs = {"1": LUT1.state_dict(), "2": LUT2.state_dict(), "3": LUT3.state_dict(), "4": LUT4.state_dict(),
                "5": LUT5.state_dict()}

        torch.save(LUTs, "saved_models/%s/LUTs_%d.pth" % (opt.output_dir, epoch))

        torch.save(classifier.state_dict(),
                   "saved_models/%s/classifier_%d.pth" % (opt.output_dir, epoch))
        torch.save(Context.state_dict(),
                   "saved_models/%s/Context_%d.pth" % (opt.output_dir, epoch))
        torch.save(MaskContext.state_dict(),
                   "saved_models/%s/MaskContext_%d.pth" % (opt.output_dir, epoch))

        file = open('saved_models/%s/result.txt' % opt.output_dir, 'a')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))

        file.close()

print(time_string)

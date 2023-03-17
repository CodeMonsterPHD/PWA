import argparse
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models_LAM import *
from datasets_evaluation import *
import time

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="PPR10K_dataset", help="root of the datasets")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
parser.add_argument("--epoch", type=int, default=0, help="epoch to load")
parser.add_argument("--model_dir", type=str, default="Checkpoints_LAM_a", help="path to save model")
parser.add_argument("--lut_dim", type=int, default=33, help="dimension of lut")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
cuda = True if torch.cuda.is_available() else False
criterion_pixelwise = torch.nn.MSELoss()

MaskContext = MaskContext()
Context = ContextModulate()
LUT1 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT2 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT3 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT4 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT5 = Generator3DLUT_identity(dim=opt.lut_dim)
classifier = resnet18_224(out_dim=5)
trilinear_ = TrilinearInterpolation()

if cuda:
    Context = Context.cuda()
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    LUT3 = LUT3.cuda()
    LUT4 = LUT4.cuda()
    LUT5 = LUT5.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()
    MaskContext = MaskContext.cuda()

LUTs = torch.load("saved_models/%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
LUT3.load_state_dict(LUTs["3"])
LUT4.load_state_dict(LUTs["4"])
LUT5.load_state_dict(LUTs["5"])
Context.load_state_dict(
    torch.load("saved_models/%s/Context_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.load_state_dict(
    torch.load("saved_models/%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
MaskContext.load_state_dict(
    torch.load("saved_models/%s/MaskContext_%d.pth" % (opt.model_dir, opt.epoch)))

Context.eval()
classifier.eval()
MaskContext.eval()

# upsample = nn.Upsample(size=(360, 540), mode='bilinear')

dataloader = DataLoader(
    ImageDataset_paper(opt.data_path),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def generator(img):
    imgs = Img_X_Mask(img, MaskContext(img))
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

    return combine_A


def visualize_result():
    """Saves a generated sample from the validation set"""
    out_dir = "results/%s_%d" % (opt.model_dir, opt.epoch)
    os.makedirs(out_dir, exist_ok=True)
    sum_time = 0
    img_number = 0
    for i, batch in enumerate(dataloader):
        input_A = Variable(batch["A_input"].type(Tensor))
        # input_A = upsample(input_A)
        img_name = batch["input_name"]
        # print(input_A.shape)
        start_time = time.time()

        result_B = generator(input_A)

        end_time = time.time()
        time_difference = end_time - start_time
        sum_time += time_difference
        img_number += 1
        print(img_number, sum_time)
        save_image(result_B, os.path.join(out_dir, "%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)

    ave_time = sum_time / img_number
    print(ave_time)


with torch.no_grad():
    visualize_result()

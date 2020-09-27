# python imports
import os
import glob
# external imports
import torch
import numpy as np
import torchsnooper
import SimpleITK as sitk
# internal imports
from Model import losses
from Model.config import args
from Model.model import U_Network, SpatialTransformer


def make_dirs():
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)


def save_image(img, ref_img, name):
    img = sitk.GetImageFromArray(img[0, 0, ...].cpu().detach().numpy())
    img.SetOrigin(ref_img.GetOrigin())
    img.SetDirection(ref_img.GetDirection())
    img.SetSpacing(ref_img.GetSpacing())
    sitk.WriteImage(img, os.path.join(args.result_dir, name))


def compute_label_dice(gt, pred):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    cls_lst = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 61, 62,
               63, 64, 65, 66, 67, 68, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 101, 102, 121, 122, 161, 162,
               163, 164, 165, 166]
    dice_lst = []
    for cls in cls_lst:
        dice = losses.DSC(gt == cls, pred == cls)
        dice_lst.append(dice)
    return np.mean(dice_lst)


# @torchsnooper.snoop()
def test():
    make_dirs()
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    print(args.checkpoint_path)

    f_img = sitk.ReadImage(args.atlas_file)
    input_fixed = sitk.GetArrayFromImage(f_img)[np.newaxis, np.newaxis, ...]
    vol_size = input_fixed.shape[2:]
    # set up atlas tensor
    input_fixed = torch.from_numpy(input_fixed).to(device).float()

    # Test file and anatomical labels we want to evaluate
    test_file_lst = glob.glob(os.path.join(args.test_dir, "*.nii.gz"))
    print("The number of test data: ", len(test_file_lst))

    # Prepare the vm1 or vm2 model and send to device
    nf_enc = [16, 32, 32, 32]
    if args.model == "vm1":
        nf_dec = [32, 32, 32, 32, 8, 8]
    else:
        nf_dec = [32, 32, 32, 32, 32, 16, 16]
    # Set up model
    UNet = U_Network(len(vol_size), nf_enc, nf_dec).to(device)
    UNet.load_state_dict(torch.load(args.checkpoint_path))
    STN_img = SpatialTransformer(vol_size).to(device)
    STN_label = SpatialTransformer(vol_size, mode="nearest").to(device)
    UNet.eval()
    STN_img.eval()
    STN_label.eval()

    DSC = []
    # fixed图像对应的label
    fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.label_dir, "S01.delineation.structure.label.nii.gz")))
    for file in test_file_lst:
        name = os.path.split(file)[1]
        # 读入moving图像
        input_moving = sitk.GetArrayFromImage(sitk.ReadImage(file))[np.newaxis, np.newaxis, ...]
        input_moving = torch.from_numpy(input_moving).to(device).float()
        # 读入moving图像对应的label
        label_file = glob.glob(os.path.join(args.label_dir, name[:3] + "*"))[0]
        input_label = sitk.GetArrayFromImage(sitk.ReadImage(label_file))[np.newaxis, np.newaxis, ...]
        input_label = torch.from_numpy(input_label).to(device).float()

        # 获得配准后的图像和label
        pred_flow = UNet(input_moving, input_fixed)
        pred_img = STN_img(input_moving, pred_flow)
        pred_label = STN_label(input_label, pred_flow)

        # 计算DSC
        dice = compute_label_dice(fixed_label, pred_label[0, 0, ...].cpu().detach().numpy())
        print("dice: ", dice)
        DSC.append(dice)

        if '7' in file:
            save_image(pred_img, f_img, "7_warped.nii.gz")
            save_image(pred_flow.permute(0, 2, 3, 4, 1)[np.newaxis, ...], f_img, "7_flow.nii.gz")
            save_image(pred_label, f_img, "7_label.nii.gz")
        del pred_flow, pred_img, pred_label

    print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC))    


if __name__ == "__main__":
    test()

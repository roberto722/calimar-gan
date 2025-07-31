import os
import random
import copy
import numpy as np
import scipy
import scipy.io as sio
from PIL import Image
import cv2
import glob
from data.base_dataset import BaseDataset, get_transform, to_tensor
from data.image_folder import make_dataset
from data.build_gemotry import initialization, build_gemotry


param = initialization(0.3)
ray_trafo, fbp = build_gemotry(param)

def linear_ct_window(img, window=[3600, 700], rescale=True):
    # window = (window width, window level)
    img_min = window[1] - window[0]//2
    img_max = window[1] + window[0]//2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    img_rescaled = copy.deepcopy(img).astype(np.float32)
    if rescale:
        img_rescaled[img_rescaled <= 1000] = 0.0004286*img_rescaled[img_rescaled <= 1000] + 0.471429
        img_rescaled[img_rescaled > 1000] = 0.0000909*img_rescaled[img_rescaled > 1000] + 0.8090909
        img_rescaled = img_rescaled * 255
        img_rescaled[img_rescaled > 255] = 255
    return img_rescaled

def lin_interp_bins(proj, metalTrace):
    num_bins, num_views = proj.shape
    Pinterp = np.zeros((num_bins, num_views), dtype=np.float32)
    proj_ = copy.deepcopy(proj)

    for i in range(0, num_bins):
        mslice = metalTrace[i, :]
        pslice = proj_[i, :]
        metalpos = np.where(mslice > 0)
        nonmetalpos = np.where(mslice == 0)

        pnonmetal = pslice[nonmetalpos]
        interp_func = scipy.interpolate.interp1d(nonmetalpos[0], pnonmetal, fill_value="extrapolate")
        pslice[metalpos] = interp_func(metalpos)
        Pinterp[i, :] = pslice
    return Pinterp


def lin_interp_views(proj, metalTrace):
    num_bins, num_views = proj.shape
    Pinterp = np.zeros((num_bins, num_views), dtype=np.float32)
    proj_ = copy.deepcopy(proj)

    for i in range(0, num_views):
        mslice = metalTrace[:, i]
        pslice = proj_[:, i]
        metalpos = np.where(mslice > 0)
        nonmetalpos = np.where(mslice == 0)

        pnonmetal = pslice[nonmetalpos]
        interp_func = scipy.interpolate.interp1d(nonmetalpos[0], pnonmetal, fill_value="extrapolate")
        pslice[metalpos] = interp_func(metalpos)
        Pinterp[:, i] = pslice
    return Pinterp


def get_LI(img):
    metal = np.zeros(img.shape)
    metal[img == 255] = 255

    metalTr = np.asarray(ray_trafo(metal))
    imgTr = np.asarray(ray_trafo(img))

    interpTr = lin_interp_bins(imgTr, metalTr)
    interp = np.asarray(fbp(interpTr))

    interp_norm = (((interp - np.min(img)) / (np.max(img) - np.min(img))) / 2) * 255
    interp_norm[interp_norm < 0] = 0
    interp_norm[interp_norm > 255] = 255

    return interp_norm

def get_attention(img, interp_norm):

    attention = interp_norm - img
    attention[attention < 0] = 0

    # plt.subplot(1, 3, 1)
    # plt.imshow(img, cmap='gray')
    # plt.subplot(1, 3, 2)
    # plt.imshow(attention_filt, cmap='gray')
    # plt.subplot(1, 3, 3)
    # plt.imshow(attention, cmap='gray')
    # plt.show()

    return attention

class UnalignedCalimarganDataset(BaseDataset):
    def __init__(self, opt):
        super().__init__(opt)

        self.isTrain = opt.isTrain
        self.circle_mask = cv2.resize(
            sio.loadmat('./data/circle_mask_512.mat')['circle_mask'],
            dsize=(256, 256),
            interpolation=cv2.INTER_AREA
        )

        self.transform_att = to_tensor()

        self.dir_A = os.path.join(opt.dataroot, 'A') if self.isTrain else opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.A_size = len(self.A_paths)

        btoA = opt.direction == 'BtoA'
        input_nc = opt.output_nc if btoA else opt.input_nc
        self.transform_A = get_transform(opt, grayscale=(input_nc == 1))

        if self.isTrain:
            self.dir_B = os.path.join(opt.dataroot, 'B')
            self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
            self.B_size = len(self.B_paths)
            output_nc = opt.input_nc if btoA else opt.output_nc
            self.transform_B = get_transform(opt, grayscale=(output_nc == 1))

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        random_mask = random.randint(1, 25)

        if self.isTrain:
            index_B = index % self.B_size if self.opt.serial_batches else random.randint(0, self.B_size - 1)
            B_path = self.B_paths[index_B]

        # Load A
        if self.isTrain:
            A_file = f"{A_path}{random_mask}_HU.mat"
            if not os.path.exists(A_file):
                A_file = f"{A_path}art_HU.mat"
        else:
            A_file = glob.glob(f"{A_path}/*_HU.mat")
            if len(A_file) == 0:
                raise FileNotFoundError(f"No file *_HU.mat found in {A_path}")
            A_file = A_file[0]

        A_img = sio.loadmat(A_file)['sim_hu_int'] if 'sim_hu_int' in sio.loadmat(A_file) else sio.loadmat(A_file)['slice_metal_HU']

        # Resize + Preprocess A
        A_img_lin = cv2.resize(linear_ct_window(A_img), (256, 256))
        interp_norm = get_LI(A_img_lin)
        interp_norm[self.circle_mask == 1] = A_img_lin[self.circle_mask == 1]
        A_img_lin_2 = copy.deepcopy(A_img_lin)
        A_img_lin_2[(interp_norm - A_img_lin) < -10] = interp_norm[(interp_norm - A_img_lin) < -10]
        A_img_lin_2[A_img_lin == 255] = A_img_lin[A_img_lin == 255]
        A_stack = np.stack((A_img_lin_2, interp_norm, interp_norm)).transpose(1, 2, 0).astype(np.uint8)
        A_img_norm = Image.fromarray(A_stack, 'RGB')
        A_tensor = self.transform_A(A_img_norm)

        # Compute attention
        attention = get_attention(A_img_lin, interp_norm)
        attention_tensor = self.transform_att(attention / 255.0)

        # Inference only
        if not self.isTrain:
            return {'A': A_tensor, 'B': A_tensor, 'A_paths': A_path, 'B_paths': A_path, 'attention': attention_tensor}

        # Load and preprocess B
        B_img = sio.loadmat(B_path + "gt.mat")['slice']
        B_img_lin = linear_ct_window(B_img)
        B_img_norm = Image.fromarray(B_img_lin).convert('RGB')
        B_tensor = self.transform_B(B_img_norm)

        return {
            'A': A_tensor,
            'B': B_tensor,
            'A_paths': A_path,
            'B_paths': B_path,
            'attention': attention_tensor
        }

    def __len__(self):
        return max(self.A_size, self.B_size) if self.isTrain else self.A_size

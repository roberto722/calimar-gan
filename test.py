"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from --checkpoints_dir and save the results to --results_dir.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for --num_test images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util
import scipy.io as sio
from PIL import Image
import glob
from tqdm import tqdm
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt


def save_LI(data, mask, save_path):
    LI = data[:, 2, :, :].cpu().float().numpy()
    im = (np.transpose(LI, (1, 2, 0)) + 1) / 2.0 * 255.0
    im = im.astype(np.uint8).squeeze()
    if mask is not None:
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_AREA)
        im[mask == 1] = 255  # mask == 255 per true
    util.util.save_image(im, save_path)


def do_test(opt):
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.

    circle_mask = sio.loadmat('./data/circle_mask_512.mat')['circle_mask']

    #Get GT and Save
    # GT_save_path_CN = os.path.join(webpage.get_image_dir(), "GT_custom_norm")  # Custom Normalization
    # if not os.path.exists(GT_save_path_CN):
    #     os.makedirs(GT_save_path_CN)
    #     print('Creating GT directory for custom normalization', GT_save_path_CN)
    # GT_save_path_LN = os.path.join(webpage.get_image_dir(), "GT_linear_norm")  # Linear Normalization
    # if not os.path.exists(GT_save_path_LN):
    #     os.makedirs(GT_save_path_LN)
    #     print('Creating GT directory for linear normalization', GT_save_path_LN)
    # LI_save_path = os.path.join(webpage.get_image_dir(), "Outputs_LI_custom")  # Custom Normalization
    # if not os.path.exists(LI_save_path):
    #     os.makedirs(LI_save_path)
    #     print('Creating LI directory for custom normalization', LI_save_path)

    times = []
    if opt.eval:
        model.eval()
    for i, data in enumerate(tqdm(dataset)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        start_time = time.time()
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        end_time = time.time()
        times.append(end_time - start_time)

        # mask = np.zeros((512, 512), dtype=np.uint8)
        # art = sio.loadmat(data['A_paths'][0] + "art_HU.mat")['slice_metal_HU']
        # mask[art >= 2500] = 1

        # plt.imshow(mask, cmap='gray')
        # plt.show()

        # GT_img = sio.loadmat(data['A_paths'][0] + "gt.mat")['slice']
        mask_name = glob.glob(f"{data['A_paths'][0]}/*_mask.mat")
        mask = sio.loadmat(mask_name[0])['metal_translated']
        # GT_img[mask == 1] = 2500
        # GT_linear = util.util.linear_normalization(GT_img)
        # # GT_linear[circle_mask == 1] = 0
        # GT_custom = util.util.custom_normalization(GT_img)
        # # GT_linear[circle_mask == 1] = 10
        # GT_linear = cv2.resize(GT_linear, (256, 256), interpolation=cv2.INTER_LINEAR)
        # GT_custom = cv2.resize(GT_custom, (256, 256), interpolation=cv2.INTER_LINEAR)

        # plt.imshow(test_custom, cmap='gray')
        # plt.show()

        # GT_linear_img = Image.fromarray(GT_linear)
        # GT_linear_img.save(os.path.join(GT_save_path_LN, f"GT_linear_{i}.png"))
        # GT_custom_img = Image.fromarray(GT_custom)
        # GT_custom_img.save(os.path.join(GT_save_path_CN, f"GT_custom_{i}.png"))

        # LI_save_path_ = os.path.join(LI_save_path, f"LI_custom_{i}.png")
        # save_LI(data['A'], mask, LI_save_path_)

        visuals = model.get_current_visuals_test()  # get image results
        img_path = model.get_image_paths()  # get image paths
        # if i % 100 == 0:  # save images to an HTML file
        #     print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, img_number=i, mask=mask, circle_mask=circle_mask)
    webpage.save()  # save the HTML

    # Risultato
    mean_time = np.mean(times)
    std_time = np.std(times)

    print(f"Inference time over {len(times)} runs:")
    print(f"Mean: {mean_time * 1000:.3f} ms")
    print(f"Std Dev: {std_time * 1000:.3f} ms")

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    do_test(opt)
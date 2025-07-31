from pathlib import Path
from PIL import Image
import os
import random
import nibabel as nib
import numpy as np
import cv2
import copy
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from glob import glob

def train_dataset_setup(insert_metal: bool):
    root = Path("datasets_raw/SpineWeb/")
    train_folder = "train_wo_hp_metal_mask_sim"
    metal_masks_dir = "datasets/train_wo_hp_x3/masks_templates/*.png"
    metal_masks = glob(metal_masks_dir)

    # Selecting slices with only vertebrae
    df = pd.read_csv('UWSpineCT-meta-data.csv', header=0, delimiter=';')

    folders = os.listdir(root)
    for i, folder in enumerate(folders):
        fold_path = root / folder
        for patient in os.listdir(fold_path):
            patient_path = fold_path / patient
            patient_dirs = os.listdir(patient_path)
            for j, file in enumerate(patient_dirs):
                nifti_path = patient_path / file
                nifti_dirs = os.listdir(nifti_path)
                for nifti_file in nifti_dirs:
                    if nifti_file.endswith('.nii.gz'):
                        single_image = nib.load(nifti_path / nifti_file)
                        image_data = single_image.get_fdata()
                        nifti_id, ext = nifti_file.split('.n')
                        nifti_details = df.loc[df['Scan ID'] == int(nifti_id)]
                        if not nifti_details.empty:
                            print('Patient ID: ' + nifti_id)
                            better_image_data = image_data[:, :, nifti_details.iloc[0]['Min Slice']:nifti_details.iloc[0]['Max Slice']]
                            for idx in range(1, better_image_data.shape[2]-1):
                                metal_threshold = 2500  # Threshold to check if artifact is inside the single image
                                #########################################################
                                ### Code for getting 3 different images
                                # windowed_image = np.empty((512, 512))
                                # mask = np.empty((512, 512))
                                # windowed_image, mask = linear_ct_window(better_image_data[:, :, idx-1], [3600, 700], metal_threshold)

                                # temp_img, temp_mask = linear_ct_window(better_image_data[:, :, idx], [3600, 700], metal_threshold)
                                # windowed_image = np.dstack((windowed_image, temp_img))
                                # mask = np.dstack((mask, temp_mask))

                                # temp_img, temp_mask = linear_ct_window(better_image_data[:, :, idx+1], [3600, 700], metal_threshold)
                                # windowed_image = np.dstack((windowed_image, temp_img))
                                # mask = np.dstack((mask, temp_mask))
                                #########################################################

                                #########################################################
                                # Code for only the same image
                                windowed_image, mask = linear_ct_window(better_image_data[:, :, idx], [3600, 700],
                                                                        metal_threshold)
                                #########################################################
                                #if np.amax(better_image_data[:, :, idx]) >= metal_threshold:
                                if np.amax(mask) >= 255:
                                    #mask_areas = connected_components(mask)
                                    #mk = Image.fromarray(mask[:, :, 1])
                                    #mk.show()
                                    mask_areas = connected_components(mask)
                                    if len(mask_areas) > 0:
                                        if max(mask_areas) >= 200:
                                            saving_folder = train_folder + "/trainA"

                                            w_image = np.array([windowed_image, windowed_image, mask])
                                            #mask_img_test = Image.fromarray(mask[:, :, 1])
                                            #mask_img_test.save("datasets/" + train_folder + "/masks/" + nifti_file + "_" + str(idx) + ".png", format="PNG")
                                        else:
                                            saving_folder = train_folder + "/not_considered"
                                            w_image = np.array([windowed_image, windowed_image, mask])

                                        # if max(mask_areas) >= 1000:
                                        #     good_mask = Image.fromarray(mask[:, :, 1])
                                        #     good_mask.save("datasets/train_wo_hp_x3/masks_templates/" + nifti_file + "_" + str(idx) + ".png", format="PNG")
                                    else:
                                        saving_folder = train_folder + "/not_considered"
                                        w_image = windowed_image
                                else:
                                    saving_folder = train_folder + "/trainB_real"
                                    w_image = np.array([windowed_image, windowed_image, mask])

                                    if insert_metal:
                                        # Inserting metal inside image
                                        metal_mask = cv2.imread(np.random.choice(metal_masks), cv2.IMREAD_GRAYSCALE)
                                        windowed_image = metal_insertion(windowed_image, metal_mask)

                                final_image = Image.fromarray(w_image.transpose(1, 2, 0))
                                saving_path = f"datasets/{saving_folder}/" + nifti_file + "_" + str(idx) + ".png"
                                final_image.save(saving_path, format="PNG")


def metal_insertion(img, metal_mask):
    # start
    # img_blur = cv2.GaussianBlur(img[:, :, 1], (3, 3), sigmaX=0, sigmaY=0)
    # img_canny = cv2.Canny(image=img_blur, threshold1=255, threshold2=100)
    # # Black circle to remove border of the CT confused as edge
    # img_wo_circle = cv2.circle(img_canny, (round(img_canny.shape[0] / 2), round(img_canny.shape[1] / 2)),
    #                            round(img_canny.shape[0] / 2), color=0, thickness=50, lineType=8, shift=0)

    # Closing edges
    # kernel = np.ones((5, 5), np.uint8)
    # img_closing = cv2.morphologyEx(img_wo_circle, cv2.MORPH_CLOSE, kernel)

    # conn_comps = cv2.connectedComponentsWithStats(img_closing, 8, cv2.CV_32S)
    # (numLabels, labels, stats, centroids) = conn_comps

    # Show centroids... only for debug purpose
    # img_w_centroids = copy.deepcopy(img_closing)
    # for coords in centroids:
    #     img_w_centroids = cv2.circle(img_w_centroids, center=np.round(coords).astype(int), radius=10, color=(255, 0, 0), thickness=-1)


    # Acting on masks
    # empty_mask = np.zeros(img.shape)
    # centroid = centroids[0]  # Choosing the first centroid recognized

    # Scaling metal
    # Define the scaling factor
    # scale_percent = np.random.randint(30, high=100)  # 50% scaling

    # Calculate the new image dimensions
    # new_width = int(metal_mask.shape[1] * scale_percent / 100)
    # new_height = int(metal_mask.shape[0] * scale_percent / 100)
    # dim = (new_width, new_height)

    # Resize the image
    # scaled_img = cv2.resize(metal_mask, dim)

    # Rotating metal
    random_rotations = [cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
    chosen_rotation = np.random.choice(len(random_rotations), size=1, replace=False)
    img_rotated = cv2.rotate(metal_mask, chosen_rotation[0])
    final_mask = np.array([img_rotated, img_rotated, img_rotated]).transpose((1, 2, 0))

    # Adding the metal in all the 3 channels
    img_with_metal_x3 = cv2.addWeighted(img, 1.0, final_mask, 1.0, 0.0)
    img_with_metal_x1 = np.array([img_with_metal_x3[:, :, 1], img_with_metal_x3[:, :, 1], final_mask[:, :, 1]]).transpose((2, 1, 0))

    # Show original image and the resulting one... only for debug purpose
    # fig, ax = plt.subplots(1, 3, figsize=(10, 10))
    # ax[0].imshow(img, cmap='gray')
    # ax[0].set_title('image')
    # ax[1].imshow(img_rotated, cmap='gray')
    # ax[1].set_title('metal mask')
    # ax[2].imshow(img_with_metal, cmap='gray')
    # ax[2].set_title('image with metal')
    # plt.show()
    return img_with_metal_x1  # final_mask is the metal mask


def linear_ct_window(img, window, metal_threshold, rescale=True):
    # window = (window width, window level)

    img_min = window[1] - window[0]//2
    img_max = window[1] + window[0]//2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    img_rescaled = copy.deepcopy(img)
    if rescale:
        img_rescaled[img_rescaled <= 1000] = 0.0004286*img_rescaled[img_rescaled <= 1000] + 0.471429
        img_rescaled[img_rescaled > 1000] = 0.0000909*img_rescaled[img_rescaled > 1000] + 0.8090909

        img_rescaled = img_rescaled * 255
        img_rescaled[img_rescaled > 255] = 255

    # Finding metal mask
    if np.amax(img) > metal_threshold-1:
        mask = copy.deepcopy(img)
        mask[img >= metal_threshold] = 255
        mask[img < metal_threshold] = 0
    else:
        mask = np.zeros((512, 512))
    return img_rescaled.astype('uint8'), mask.astype('uint8')




def get_images_for_testing(source_path, destination_path):
    files = os.listdir(source_path)
    random_test_files = list(set(random.choices(files, k=round(len(files)*0.03))))  # Set is needed since choices can return duplicates
    for file in random_test_files:
        os.rename(source_path / file, destination_path / file)


if __name__ == "__main__":
    print("Inizio conversione dati.\n")
    #train_dataset_setup(insert_metal=False)
    print("Conversione dati di training terminata.\n")
    print("Prendo alcune immagini da A per test.\n")
    get_images_for_testing(source_path=Path('datasets/train_wo_hp_metal_mask_sim/trainA'), destination_path=Path('datasets/test_wo_hp_metal_mask_sim/testA'))
    #print("Prendo alcune immagini da B per test.\n")
    #get_images_for_testing(source_path=Path("datasets/train/trainB"), dest_path=Path("datasets/test/testB"))
    print("Programma terminato\n")


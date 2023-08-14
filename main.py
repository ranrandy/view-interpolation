import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import glob
from tqdm import tqdm

from utils import TwoViewData

from PIL import Image


def saveCV2Images(images, new_idx, output_path):
    images_rgb = []
    for img in images:
        images_rgb.append(cv2.cvtColor(img.astype(np.float32) / 255, cv2.COLOR_BGR2RGB))

    fig, axs = plt.subplots(1, len(images_rgb), figsize=(12 * len(images_rgb), 8))
    for i in range(len(images_rgb)):
        axs[i].imshow(images_rgb[i])
        axs[i].axis('off')
        if i == new_idx:
            axs[i].set_title("Synthesized View", fontsize=30)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_path}.png')


if __name__ == '__main__':
    gif = True

    # Load the middlebury stereo dataset
    datasets = {}
    for folder_name in tqdm(glob.glob('dataset_middlebury/*-perfect')):
        datasets[folder_name.split('/')[-1].split('-')[0]] = TwoViewData(folder_name)

    np.seterr(divide='ignore', invalid='ignore')
    for scene_name, v in tqdm(datasets.items()):
        camera_center_left = np.array([0, 0, 0])
        camera_center_right = np.array([v.cam.baseline, 0, 0])
        
        if gif:
            # Define GIF image step and intermediate camera centers
            step = v.cam.baseline / 20
            camera_center_Xs = np.arange(0 + step, v.cam.baseline, step)

            # Compose a sequence of images
            img_root = f"outputs/{scene_name}_cameraX_"
            gif_seq = [img_root+"0.png"]
            plt.imsave(gif_seq[0], cv2.cvtColor(v.img0.astype(np.float32) / 255, cv2.COLOR_BGR2RGB))
        else:
            # Output a static image
            camera_center_Xs = np.array([v.cam.baseline / 2])
        
        ######## 1. Synthesize a new view which is half-way between the two original views
        for x in tqdm(camera_center_Xs):
            camera_center_middle = np.array([x, 0, 0])

            #### 1.1 Initialize the new view with all black pixels
            # New view image warped from the left/right image
            im2_left = np.zeros((v.cam.img_height, v.cam.img_width, 3)) - 1
            im2_right = np.zeros((v.cam.img_height, v.cam.img_width, 3)) - 1


            #### 1.2 Use ordered forward mapping to improve visibility continuity
            ## 1.2.1 Warp the left image to the middle          
            # X_middle_center > X_left_center, Y_middle_center > Y_left_center
            # Compute the distance to move pixels. Let the new view principal point be the same as the left image
            disp_middle_left = np.array([v.cam.focal * np.divide(d, v.depth0) for d in np.abs(camera_center_middle - camera_center_left)]).astype(int)
            # Compute the corresponding x, y values in the new view
            new_x_left = np.arange(v.cam.img_width).reshape(1, -1) - disp_middle_left[0]
            new_y_left = np.arange(v.cam.img_height).reshape(-1, 1) - disp_middle_left[1]
            # Check if the x, y values are within the image boundary
            valid_locations_left = (0 <= new_x_left) * (new_x_left <= v.cam.img_width - 1) * (0 <= new_y_left) * (new_y_left <= v.cam.img_height - 1)
            # Iterate the left image pixels from "left to right", and "bottom to top"
            y_order_left = np.flipud(np.indices((v.cam.img_height, v.cam.img_width))[0])[np.flipud(valid_locations_left)]
            x_order_left = np.flipud(np.indices((v.cam.img_height, v.cam.img_width))[1])[np.flipud(valid_locations_left)]
            # Move left image pixels to the new view
            im2_left[new_y_left[y_order_left, x_order_left], new_x_left[y_order_left, x_order_left]] = v.img0[y_order_left, x_order_left]

            ## 1.2.2 Right
            # X_middle_center < X_right_center, Y_middle_center > Y_left_center
            disp_middle_right = np.array([v.cam.focal * np.divide(d, v.depth1) for d in np.abs(camera_center_right - camera_center_middle)])
            # Since the new view principal point is the same as the left image, we need to substract the doffs to get the real disparity value
            disp_middle_right[0] -= v.cam.doffs
            disp_middle_right = disp_middle_right.astype(int)
            new_x_right = np.arange(v.cam.img_width).reshape(1, -1) + disp_middle_right[0]
            new_y_right = np.arange(v.cam.img_height).reshape(-1, 1) - disp_middle_right[1]
            valid_locations_right = (0 <= new_x_right) * (new_x_right <= v.cam.img_width - 1) * (0 <= new_y_right) * (new_y_right <= v.cam.img_height - 1)
            y_order_right = np.fliplr(np.flipud(np.indices((v.cam.img_height, v.cam.img_width))[0]))[np.fliplr(np.flipud(valid_locations_right))]
            x_order_right = np.fliplr(np.flipud(np.indices((v.cam.img_height, v.cam.img_width))[1]))[np.fliplr(np.flipud(valid_locations_right))]
            im2_right[new_y_right[y_order_right, x_order_right], new_x_right[y_order_right, x_order_right]] = v.img1[y_order_right, x_order_right]


            #### 1.3 Fill in single holes, which is not a hole in a certain view
            single_hole_left = ((im2_left == -1) * (im2_right != -1))[:, :, 0]
            single_hole_right = ((im2_left != -1) * (im2_right == -1))[:, :, 0]

            single_hole_y_left = np.indices((v.cam.img_height, v.cam.img_width))[0][single_hole_left]
            single_hole_x_left = np.indices((v.cam.img_height, v.cam.img_width))[1][single_hole_left]
            im2_left[single_hole_y_left, single_hole_x_left] = im2_right[single_hole_y_left, single_hole_x_left]

            single_hole_y_right = np.indices((v.cam.img_height, v.cam.img_width))[0][single_hole_right]
            single_hole_x_right = np.indices((v.cam.img_height, v.cam.img_width))[1][single_hole_right]
            im2_right[single_hole_y_right, single_hole_x_right] = im2_left[single_hole_y_right, single_hole_x_right]


            #### 1.4 Adjust the warped image intensities and blend the images 
            ## Regression coefficients
            im2_left_non_holes = im2_left != -1
            im2_right_non_holes = im2_right != -1
            S = np.sum(im2_left_non_holes * im2_right_non_holes) / 3
            S_l = np.sum(im2_left * im2_left_non_holes, axis=(0, 1))
            S_r = np.sum(im2_right * im2_right_non_holes, axis=(0, 1))
            S_lr = np.sum(im2_left * im2_right * im2_left_non_holes * im2_right_non_holes, axis=(0, 1))
            S_ll = np.sum(im2_left * im2_left * im2_left_non_holes, axis=(0, 1))
            S_rr = np.sum(im2_right * im2_right * im2_right_non_holes, axis=(0, 1))
            a = (S_ll * S_r - S_l * S_lr) / (S * S_ll - S_l * S_l)
            b = (S * S_lr - S_l * S_r) / (S * S_ll - S_l * S_l)

            ## Alpha, blending weight
            dist_left = np.linalg.norm(camera_center_middle-camera_center_left)
            dist_right = np.linalg.norm(camera_center_right-camera_center_middle)
            alpha = dist_right / (dist_right + dist_left)

            ## Gamma, intensity weight
            gamma = alpha # Change to 0.5 if we need to synthesize many views

            ## Blend
            # im2 = alpha * im2_left + (1 - alpha) * im2_right
            im2 = alpha * (
                gamma * im2_left + (1 - gamma) * (a + b * im2_left)) + (1 - alpha) * (
                    gamma * ((im2_right - a) / b) + (1 - gamma) * im2_right)


            #### 1.5 Fill in the remaining holes
            ## Fill in the other holes. We actually can't do this. But to avoid as much artifact as possible, we implement this by mirroring the pixel intensities on the scanline
            other_hole_locations = ((im2_left == -1) * (im2_right == -1))[:, :, 0]
            for i in range(v.cam.img_height):
                start_hole_idx = -2
                for j in range(v.cam.img_width):
                    if other_hole_locations[i, j]:
                        if start_hole_idx == -2:
                            start_hole_idx = j - 1
                    else:
                        if start_hole_idx != -2:
                            if start_hole_idx == -1:
                                im2[i, start_hole_idx] = np.array([0, 0, 0])
                            for k in range(start_hole_idx, j):
                                im2[i, k] = (im2[i, start_hole_idx] * (j - k) + im2[i, j] * (k - start_hole_idx)) / (j - start_hole_idx)
                            start_hole_idx = -2
                if start_hole_idx != -2:
                    j = v.cam.img_width
                    for k in range(start_hole_idx, j):
                        im2[i, k] = (im2[i, start_hole_idx] * (j - k)) / (j - start_hole_idx)
                    start_hole_idx = -2

            im2 = np.clip(im2, 0, 255)

            if gif:
                gif_seq.append(img_root+f"{camera_center_middle[0].astype(int)}.png")
                plt.imsave(gif_seq[-1], cv2.cvtColor(im2.astype(np.float32) / 255, cv2.COLOR_BGR2RGB))
            else:
                # Save all the views
                saveCV2Images([v.img0, im2, v.img1], 1, f"outputs/static_images/{scene_name}_middle")
        
        ######## TODO: 2. Synthesize a new view which is beyond either camera
        ######## TODO: 3. Backward mapping instead of forward mapping
        
        if gif:
            gif_seq.append(img_root+f"{camera_center_right[0].astype(int)}.png")
            plt.imsave(gif_seq[-1], cv2.cvtColor(v.img1.astype(np.float32) / 255, cv2.COLOR_BGR2RGB))
            
            # Compress the GIF
            image_list = [Image.open(file) for file in gif_seq]
            for i in range(len(image_list)):
                image_list[i] = image_list[i].resize((int(image_list[i].width / 2), int(image_list[i].height / 2)), Image.Resampling.LANCZOS)
            
            # Make the GIF
            image_list[0].save(
                f"outputs/{scene_name}.gif",
                save_all=True, optimize=True,
                append_images=image_list[1:], # append rest of the images
                duration=100, # in milliseconds
                loop=0)
            
            # Delete the sequence of images
            for img_path in gif_seq:
                os.remove(img_path)

        # break
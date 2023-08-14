import numpy as np
import cv2
import os
import re


class Camera:
    def __init__(self, calib_param_list):
        # Unit: mm
        self.baseline = float(calib_param_list[3][9:]) # Camera optical center physical distance in milimeters

        # Unit: pixel
        self.cam0 = self.stringToNumpy2DArray(calib_param_list[0][6:-1]) # left camera matrix
        self.cam1 = self.stringToNumpy2DArray(calib_param_list[1][6:-1]) # right camera matrix
        self.focal = self.cam0[0, 0] # camera focal length

        self.doffs = float(calib_param_list[2][6:]) # principal point difference
        
        self.img_width = int(calib_param_list[4][6:]) # image width
        self.img_height = int(calib_param_list[5][7:]) # image height
        
        # Others
        self.ndisp = int(calib_param_list[6][6:])
        self.vmin = int(calib_param_list[8][5:])
        self.vmax = int(calib_param_list[9][5:])

    def stringToNumpy2DArray(self, str2Darr):
        rows_str = str2Darr.split(";")
        rows = []
        for row in rows_str:
            rows.append([float(c) for c in row.strip().split(" ")])
        return np.array(rows)

class TwoViewData:
    def __init__(self, folder_name):
        # Disparity maps
        self.disp0, self.disp0_scale = self.read_pfm(os.path.join(folder_name, "disp0.pfm")) # left
        self.disp1, self.disp1_scale = self.read_pfm(os.path.join(folder_name, "disp1.pfm")) # right
        self.disp0[self.disp0 == np.inf] = np.NaN
        self.disp1[self.disp1 == np.inf] = np.NaN

        # Images
        self.img0 = cv2.imread(os.path.join(folder_name, "im0.png"), cv2.IMREAD_COLOR)
        self.img1 = cv2.imread(os.path.join(folder_name, "im1.png"), cv2.IMREAD_COLOR)

        # Camera calibrations
        self.cam = Camera(open(os.path.join(folder_name, "calib.txt"),"r").read().split('\n'))

        # Calculate depth map
        self.depth0 = self.get_depth(self.disp0)
        self.depth1 = self.get_depth(self.disp1)

    def read_pfm(self, pfm_file_path):
        with open(pfm_file_path, 'rb') as pfm_file:
            header = pfm_file.readline().decode().rstrip()
            channels = 3 if header == 'PF' else 1
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
            if dim_match:
                width, height = map(int, dim_match.groups())
            else:
                raise Exception("Malformed PFM header.")

            scale = float(pfm_file.readline().decode().rstrip())
            sign = 0
            if scale < 0:
                endian = '<'  # littel endian
                sign = -1
                # scale = -scale
            else:
                endian = '>'  # big endian
                sign = 1

            dispariy = np.fromfile(pfm_file, endian + 'f')
        if channels == 1:
            img = np.reshape(dispariy, newshape=(height, width))
        else:
            img = np.reshape(dispariy, newshape=(height, width, channels))
        img = np.flipud(img)#.astype('uint8')
        return img, scale
    
    # If disparity does not exist (disp = np.NaN), 
    # then depth = np.NaN -> new disp = f*b/depth = np.NaN -> hole -> ok
    def get_depth(self, disp):
        return self.cam.cam0[0, 0] * self.cam.baseline / (disp + self.cam.doffs)

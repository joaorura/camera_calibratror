import cv2
import numpy as np
import matplotlib.pylab as plt
from controller.DepthMapImage import DepthMapImage
from controller.CameraCalibratorController import CameraCalibratorController

class Stereo3dReconstruction:
    def __init__(self, img1, img2, focal_length):        
        self.img1 = img1
        self.img2 = img2

        self.q = np.float32([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,focal_length*0.05,0],
                    [0,0,0,1]])

        self.h, self.w = img1.shape[:2]

    def create_output(self, vertices, colors, filename):
        colors = colors.reshape(-1,3)
        vertices = np.hstack([vertices.reshape(-1,3),colors])

        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            '''
        with open(filename, 'w') as f:
            f.write(ply_header %dict(vert_num=len(vertices)))
            np.savetxt(f,vertices,'%f %f %f %d %d %d')
  
    def reconstruct3d(self):
        dmi = DepthMapImage(self.img1, self.img2)
        dp = dmi.get_ratification_disparity_map()

        points_3D = cv2.reprojectImageTo3D(dp, self.q)
        colors = cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB)
        mask_map = dp > dp.min()
        output_points = points_3D[mask_map]
        output_colors = colors[mask_map]

        self.create_output(output_points, output_colors, 'reconstructed.ply')

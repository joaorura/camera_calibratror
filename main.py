import cv2
import numpy as np
import matplotlib.pylab as plt
from sys import argv
from controller.BlurBackground import BlurBackground
from controller.Stereo3dReconstruction import Stereo3dReconstruction
from view.ImageView import ImageView
from controller.CameraCalibratorController import CameraCalibratorController

def first_question():
    image_view = ImageView()

    camera_controller = CameraCalibratorController(image_view)
    camera_controller.webcam_loop()

def second_question():
    for i in range(1, 4, 1):
        img_left = cv2.imread(f'./imgs/web_left_{i}.png')
        img_right = cv2.imread(f'./imgs/web_right_{i}.png')

        bb = BlurBackground(img_left, img_right)

        plt.imshow(img_left)
        plt.show()

        plt.imshow(bb.get_dephth_map(), 'gray')
        plt.show()

        plt.imshow(bb.make_blur_image())
        plt.show()

def third_question():
    img_left = cv2.imread('./imgs/img1.jpg')
    img_right = cv2.imread('./imgs/img2.jpg')

    ccm = CameraCalibratorController(ImageView())
    fd = ccm.get_focal_len('./imgs/img1.jpg')

    sdr = Stereo3dReconstruction(img_left, img_right, fd)

    sdr.reconstruct3d()

def save_calibration_matrix():
    image_view = ImageView()
    camera_controller = CameraCalibratorController(image_view)
    camera_controller.save_calibration_matrix('./imgs/chess_table.jpg')


QUESTIONS = {
    '1': first_question,
    '2': second_question,
    '3': third_question,
    '4': save_calibration_matrix
}

def main():
    QUESTIONS[argv[1]]()
    return


if __name__ == '__main__':
    main()

import cv2

from view import ImageView
from controller import CameraCalibratorController

image_view = ImageView.ImageView()

camera_controller = CameraCalibratorController.CameraCalibratorController(image_view)

#camera_controller.webcam_loop() 
img_left = cv2.imread('./web2_left.png')
img_right = cv2.imread('./web2_right.png')
camera_controller.blur_backgrond(img_left, img_right)
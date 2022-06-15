from view import ImageView
from controller import CameraCalibratorController

image_view = ImageView.ImageView()

camera_controller = CameraCalibratorController.CameraCalibratorController(image_view)
camera_controller.webcam_loop()
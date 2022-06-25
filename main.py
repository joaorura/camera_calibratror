import cv2
from sys import argv
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

        image_view = ImageView()

        camera_controller = CameraCalibratorController(image_view)
        camera_controller.blur_backgrond(img_left, img_right)


def third_question():
    return

def save_calibration_matrix():
    image_view = ImageView()

    img = cv2.imread(f'./imgs/chess_table.jpg')
    camera_controller = CameraCalibratorController(image_view)
    camera_controller.save_calibration_matrix(img)


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

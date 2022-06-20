import cv2
from sys import argv
from view.ImageView import ImageView
from controller.CameraCalibratorController import CameraCalibratorController


def first_question():
    image_view = ImageView()

    camera_controller = CameraCalibratorController(image_view)
    camera_controller.webcam_loop()

def second_question():
    for i in range(1, 3, 1):
        img_left = cv2.imread(f'./imgs/web_left_{i}.png')
        img_right = cv2.imread(f'./imgs/web_right_{i}.png')

        image_view = ImageView()

        camera_controller = CameraCalibratorController(image_view)
        camera_controller.blur_backgrond(img_left, img_right)


def third_question():
    return


QUESTIONS = {
    '1': first_question,
    '2': second_question,
    '3': third_question
}

def main():
    QUESTIONS[argv[1]]()
    return


if __name__ == '__main__':
    main()

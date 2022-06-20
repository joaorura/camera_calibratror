import cv2
import matplotlib.pylab as plt

class ImageView:
    def __init__(self):
        return 

    def show_webcam_image_in_window(self, window_title, img):
        cv2.imshow(window_title, img)

    def show_image_in_window(self, window_title, img):
        plt.imshow(img)
        plt.show()

    def show_images(self, imgs):
        for img in imgs:
            plt.imshow(img)

        plt.show()

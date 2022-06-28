import cv2
from controller.DepthMapImage import DepthMapImage

class BlurBackground:
    def __init__(self, img1, img2):
        self.img1 = img1
        self.img2 = img2
        self.dmi = DepthMapImage(img1, img2)
        
    
    def make_blur_image(self):
        disparity_img = self.get_dephth_map()

        out_img = cv2.GaussianBlur(self.img1, (5,5), 3)
        out_img[disparity_img > 128] = self.img1[disparity_img > 128]

        return out_img

    def get_dephth_map(self):
        return self.dmi.get_ratification_disparity_map()

import numpy as np
import cv2
import matplotlib.pylab as plt

class CameraCalibratorController:
    def __init__(self, image_view):
        self.image_view = image_view

    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        frame_with_chess = np.zeros((240, 320, 3))

        while True:
            ret, frame = cap.read()
            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
            self.image_view.show_webcam_image_in_window('Webcam', frame_with_chess)
            img = self.calibrate_camera_with_chess_function(frame)
            frame_with_chess = img

            c = cv2.waitKey(1)
            if c == 27: # press Esc to exit
                break 

        cap.release()
        cv2.destroyAllWindows()

    def calibrate_camera_with_chess_function(self, img):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
        # If found, add object points, image points (after refining them)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (7,6), corners2, ret)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # Printing results
            print('Rotation Vectors: ')
            print(rvecs)

            print('Translation Vectors: ')
            print(tvecs)


        return img

    def convert_bgr_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def blur_backgrond(self, img_left, img_right):
        img_left_gray = self.convert_bgr_to_gray(img_left)
        img_right_gray = self.convert_bgr_to_gray(img_right)
        out_img = cv2.cvtColor(img_left_gray, cv2.COLOR_GRAY2RGB)
        cv2.imshow('img', out_img)
        plt.imshow(out_img)
        plt.show()
        
        stereo = cv2.StereoBM_create(numDisparities=0, blockSize=21)
        disparity_img = stereo.compute(img_left_gray, img_right_gray) 

        
        mask = disparity_img.copy()
        blurry_img = cv2.GaussianBlur(disparity_img, (5,5), 0)
        print(mask > 225)
        out_img = blurry_img.copy()
      
        out_img[mask > 225] = disparity_img[mask > 225]
        plt.imshow(out_img, 'gray')
        plt.show()

        img_short16 = np.float32(out_img)
        
        img_short16 = ((img_short16 / 16) + 1)
        print(img_short16)

        plt.imshow(img_short16, 'gray')
        plt.show()

        out_img = cv2.cvtColor(img_short16, cv2.COLOR_GRAY2BGR)
        cv2.imshow('img', out_img)
        plt.imshow(out_img)
        plt.show()
      


        imgs_to_show = [img_left_gray, img_right_gray, disparity_img, out_img]
        self.image_view.show_images(imgs_to_show)
        

    

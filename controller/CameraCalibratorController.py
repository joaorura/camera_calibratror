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
        mtx = []

        while True:
            ret, frame = cap.read()
            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
            self.image_view.show_webcam_image_in_window('Webcam', frame_with_chess)
            img, mtx = self.calibrate_camera_with_chess_function(frame)
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

        mtx = []

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


        return img, mtx

    def save_calibration_matrix(self, img):
        mtx = []
        img, mtx = self.calibrate_camera_with_chess_function(img)
        
        mtx_out = np.asarray(mtx)
        np.save('./content/calibration_mtx.npy', mtx_out)


    def convert_bgr_to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def blur_backgrond(self, img_left, img_right):
        img_left_gray = self.convert_bgr_to_gray(img_left)
        img_right_gray = self.convert_bgr_to_gray(img_right)

        stereo = cv2.StereoBM_create(numDisparities=0, blockSize=21)
        disparity_img = stereo.compute(img_left_gray, img_right_gray) 

        out_img = cv2.GaussianBlur(img_left, (5,5), 3)
        out_img[disparity_img > 128] = img_left[disparity_img > 128]
        
        plt.imshow(img_left)
        plt.show()

        plt.imshow(disparity_img, 'gray')
        plt.show()

        plt.imshow(out_img)
        plt.show()

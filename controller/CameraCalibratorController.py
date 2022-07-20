import cv2
import PIL.ExifTags
import PIL.Image
import numpy as np


class CameraCalibratorController:
    def __init__(self, image_view):
        self.image_view = image_view

    def webcam_loop(self):
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        frame_with_chess = np.zeros((240, 320, 3))
        ret = []
        mtx = []
        dist = []

        while True:
            ret, frame = cap.read()
            #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    
            self.image_view.show_webcam_image_in_window('Webcam', frame_with_chess)
            img, ret, mtx, dist = self.calibrate_camera_with_chess_function(frame)
            frame_with_chess = img

            c = cv2.waitKey(1)
            if c == 27: # press Esc to exit
                break 

        cap.release()
        cv2.destroyAllWindows()

    def calibrate_camera_with_chess_function(self, img):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        chessboard_size = (7,5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((np.prod(chessboard_size),3),dtype=np.float32)
        objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d point in real world space
        imgpoints = [] # 2d points in image plane.

        ret = []
        mtx = []
        dist = []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret,corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # If found, add object points, image points (after refining them)

        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (5,5), (-1,-1), criteria)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

            # Printing results
            print('Rotation Vectors: ')
            print(rvecs)

            print('Translation Vectors: ')
            print(tvecs)


        return img, ret, mtx, dist

    def save_calibration_matrix(self, img_path):
        img = cv2.imread(img_path)

        mtx = []
        img, ret, mtx, dist = self.calibrate_camera_with_chess_function(img)
        
        print(ret, mtx, dist)
        ret_out = np.asarray(ret)
        mtx_out = np.asarray(mtx)
        dist_out = np.asarray(dist)

        np.save('./content/calibration_ret.npy', ret_out)
        np.save('./content/calibration_mtx.npy', mtx_out)
        np.save('./content/calibration_dist.npy', dist_out)

        print(ret_out, )
        self.save_focal_distance(img_path)

    def get_focal_len(self, img_path):
        exif_img = PIL.Image.open(img_path)
        exif_data = {
        PIL.ExifTags.TAGS[k]:v
        for k, v in exif_img._getexif().items()
        if k in PIL.ExifTags.TAGS}

        focal_length_exif = exif_data['FocalLength']

        return focal_length_exif

    def save_focal_distance(self, img_path):
        np.save("./content/focal_lenght.npy", self.get_focal_len(img_path))

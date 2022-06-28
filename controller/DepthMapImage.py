import numpy as np
import cv2 as cv


class DepthMapImage:
    def __init__(self, img1, img2):        
        self.img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
        self.img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    def _get_key_points(self):
        sift = cv.SIFT_create()
        
        points1 = sift.detectAndCompute(self.img1, None)
        points2 = sift.detectAndCompute(self.img2, None)

        return points1, points2

    def _get_good_matches(self, points):
        kp1, des1 = points[0]
        kp2, des2 = points[1]

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        matchesMask = [[0, 0] for _ in range(len(matches))]
        good = []
        pts1 = []
        pts2 = []

        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        return pts1, pts2
    
    def _stereo_ratification(self, data):
        pts1, pts2 = data

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)
        
        fundamental_matrix, inliers = cv.findFundamentalMat(pts1, pts2, cv.FM_RANSAC)

        pts1 = pts1[inliers.ravel() == 1]
        pts2 = pts2[inliers.ravel() == 1]

        h1, w1 = self.img1.shape
        h2, w2 = self.img2.shape
        _, H1, H2 = cv.stereoRectifyUncalibrated(
            np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
        )

        img1_rectified = cv.warpPerspective(self.img1, H1, (w1, h1))
        img2_rectified = cv.warpPerspective(self.img2, H2, (w2, h2))

        block_size = 11
        min_disp = -128
        max_disp = 128
        num_disp = max_disp - min_disp
        uniquenessRatio = 5
        speckleWindowSize = 200
        speckleRange = 2
        disp12MaxDiff = 0

        stereo = cv.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )
        
        disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)

        disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                    beta=0, norm_type=cv.NORM_MINMAX)
        disparity_SGBM = np.uint8(disparity_SGBM)

        return disparity_SGBM

    def get_ratification_disparity_map(self):
        points = self._get_key_points()
        matchs = self._get_good_matches(points)
        return self._stereo_ratification(matchs)

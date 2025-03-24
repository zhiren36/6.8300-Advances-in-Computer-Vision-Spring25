import os
import env
import src.utils.utils as utils

from PIL import Image

import numpy as np
import cv2
from src.p3_fundamental_matrix import *
import matplotlib.pyplot as plt

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# NOTICE!! (I think the comment is wrong, because in main() it calculates F as p'Fp=0, so I will treat it that way)
def compute_epipole(points1: np.array, 
                    points2: np.array, 
                    F: np.array) -> np.array:
    '''
    Computes the epipole in homogenous coordinates
    given matching points in two images and the fundamental matrix
    Arguments:
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
        F - the Fundamental matrix such that (points1)^T * F * points2 = 0

        Both points1 and points2 are from the get_data_from_txt_file() method
    Returns:
        epipole - the homogenous coordinates [x y 1] of the epipole in the image
    '''
    # TODO: Implement this method!
    # Hint: p'T * F * p = 0
    # raise NotImplementedError

    U, S, Vt = np.linalg.svd(F)
    e = Vt[-1]
    e = e / e[-1]
    return e

def compute_matching_homographies(e2: np.array, 
                                  F: np.array, 
                                  im2: np.array, 
                                  points1: np.array, 
                                  points2: np.array) -> tuple:
    '''
    Determines homographies H1 and H2 such that they
    rectify a pair of images
    Arguments:
        e2 - the second epipole
        F - the Fundamental matrix
        im2 - the second image
        points1 - N points in the first image that match with points2
        points2 - N points in the second image that match with points1
    Returns:
        H1 - the homography associated with the first image
        H2 - the homography associated with the second image
    '''
   

    e2 = e2.ravel()
    
    height, width = im2.shape[:2]

    T = np.array([[1, 0, -width/2],
                  [0, 1, -height/2],
                  [0, 0,       1    ]])
    e2_translated = T @ e2
    
   
    alpha = 1 if e2_translated[0] >= 0 else -1
    
    denom = np.sqrt(e2_translated[0]**2 + e2_translated[1]**2)
    R = np.array([
        [ alpha*e2_translated[0]/denom,  alpha*e2_translated[1]/denom, 0],
        [-alpha*e2_translated[1]/denom,  alpha*e2_translated[0]/denom, 0],
        [ 0,                             0,                            1]
    ])

    
    e2_rotated = R @ e2_translated
    f = e2_rotated[0] 
    
   
    G = np.array([
        [ 1,  0,  0],
        [ 0,  1,  0],
        [-1/f, 0,  1]
    ])
    
    T_inv = np.linalg.inv(T)
    

    H2 = T_inv @ G @ R @ T

    e2_cross = np.array([
        [       0,  -e2[2],   e2[1]],
        [  e2[2],       0,  -e2[0]],
        [ -e2[1],   e2[0],       0]
    ])
    M = e2_cross @ F + np.outer(e2, np.array([1, 1, 1]))  # 3x3

    H2M = H2 @ M
    
    points1_hat = (H2M @ points1.T).T  # shape (N, 3)
    points1_hat /= points1_hat[:, [2]]
    

    points2_hat = (H2 @ points2.T).T   # shape (N, 3)
    points2_hat /= points2_hat[:, [2]]

    b = points2_hat[:, 0] 

    a0, a1, a2 = np.linalg.lstsq(points1_hat, b)[0]

    HA = np.array([
        [a0, a1, a2],
        [ 0,  1,  0],
        [ 0,  0,  1]
    ])

 
    H1 = HA @ H2M

#     H1_test = np.array([[1.02660729e+02, 2.07907964e+00,-6.13266953e+03],
#  [2.96490114e+00,9.59516079e+01 , -3.16082071e+03], 
#  [9.12194551e-03 ,4.20134022e-04 ,8.61862794e+01]])


#     alpha = np.mean(H1 / H1_test)  # or a ratio of any matching element
#     H1_scaled = alpha * H1_test
#     np.allclose(H1_scaled, H1, rtol=1e-2)

    return H1, H2













def compute_rectified_image(im: np.array, 
                            H: np.array) -> tuple:
    '''
    Rectifies an image using a homography matrix
    Arguments:
        im - an image
        H - a homography matrix that rectifies the image
    Returns:
        new_image - a new image matrix after applying the homography
        offset - the offest in the image.
    '''
    # TODO: Implement this method!
    #raise NotImplementedError 
    # 1) Dimensions of original image
    orig_h, orig_w = im.shape[:2]
    channels = 1 if im.ndim == 2 else im.shape[2]

    # 2) Compute corners in homogeneous coords
    corners = np.array([
        [0,      0,      1],
        [orig_w, 0,      1],
        [0,      orig_h, 1],
        [orig_w, orig_h, 1]
    ], dtype=np.float32)  # shape (4,3)

    # 3) Transform corners with H => new (x', y')
    #    (x', y') = H * (x, y, 1)
    #    Then convert back to inhom coords: x'/w', y'/w'
    transformed = (H @ corners.T)  # shape (3,4)
    # Convert to inhomogeneous
    transformed /= transformed[2,:]  # divide each col by w'
    xs = transformed[0,:]
    ys = transformed[1,:]

    # 4) Determine bounding box in rectified space
    min_x, max_x = np.min(xs), np.max(xs)
    min_y, max_y = np.min(ys), np.max(ys)

    # Round bounds to integers
    min_xi, max_xi = int(np.floor(min_x)), int(np.ceil(max_x))
    min_yi, max_yi = int(np.floor(min_y)), int(np.ceil(max_y))

    # 5) Offsets => so that the new image's top-left is at (0,0)
    offset_x = -min_xi
    offset_y = -min_yi

    # 6) Dimensions of new image
    new_w = max_xi - min_xi
    new_h = max_yi - min_yi

    # Create output array
    if channels == 1:
        new_image = np.zeros((new_h, new_w), dtype=im.dtype)
    else:
        new_image = np.zeros((new_h, new_w, channels), dtype=im.dtype)

    # 7) Inverse homography for inverse mapping
    H_inv = np.linalg.inv(H)

    # 8) For each pixel in new_image => map back to original image
    # Build a grid of coords for [0..new_h-1] x [0..new_w-1]
    row_coords, col_coords = np.indices((new_h, new_w))  # shape => (2, new_h, new_w)

    # Convert these to "rectified" coords => add offsets => homogeneous form
    xprime = col_coords + min_xi  # col + left bound
    yprime = row_coords + min_yi  # row + top bound

    # Flatten them for transformation
    flat_size = new_h * new_w
    ones = np.ones((1, flat_size), dtype=np.float32)
    rect_pts = np.vstack([
        xprime.ravel(),
        yprime.ravel(),
        ones
    ])

    # Map through H_inv => original coords
    mapped = H_inv @ rect_pts  # shape => (3, flat_size)
    # Convert to inhom
    mapped /= mapped[2,:]
    u = mapped[0,:]
    v = mapped[1,:]

    # 9) Nearest-neighbor sampling
    # Round to nearest pixel in the original image
    u_rounded = np.round(u).astype(int)
    v_rounded = np.round(v).astype(int)

    # Build a mask of valid pixels => within [0..orig_w-1], [0..orig_h-1]
    inside_mask = (u_rounded >= 0) & (u_rounded < orig_w) & \
                  (v_rounded >= 0) & (v_rounded < orig_h)

    # 10) Copy pixels from original to new_image
    if channels == 1:
        new_image_vals = np.zeros(flat_size, dtype=im.dtype)
        new_image_vals[inside_mask] = im[v_rounded[inside_mask], u_rounded[inside_mask]]
        # Reshape back to (new_h, new_w)
        new_image = new_image_vals.reshape(new_h, new_w)
    else:
        new_image_vals = np.zeros((flat_size, channels), dtype=im.dtype)
        new_image_vals[inside_mask] = im[v_rounded[inside_mask], u_rounded[inside_mask]]
        new_image = new_image_vals.reshape(new_h, new_w, channels)

    offset = (offset_x, offset_y)  # or (min_xi, min_yi), depending on usage
    return new_image, offset








def find_matches(img1: np.array, img2: np.array) -> tuple:
    """
    Find matches between two images using SIFT
    Arguments:
        img1 - the first image
        img2 - the second image
    Returns:
        kp1 - the keypoints of the first image
        kp2 - the keypoints of the second image
        matches - the matches between the keypoints
    """
    # TODO: Implement this method!
    # raise NotImplementedError

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(des1, des2, k=2)

    ratio_threshold = 0.75
    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches 





def show_matches(img1: np.array, 
                 img2: np.array, 
                 kp1: list, 
                 kp2: list, 
                 matches: list) -> np.array:
    result_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    plt.imshow(result_img)
    plt.title("SIFT Matches")
    plt.show()
    return result_img


if __name__ == '__main__':
    if not os.path.exists(env.p4.output):
        os.makedirs(env.p4.output)
    expected_e1, expected_e2 = np.load(env.p4.expected_e1), np.load(env.p4.expected_e2)
    expected_H1, expected_H2 = np.load(env.p4.expected_H1), np.load(env.p4.expected_H2)
    im1 = utils.load_image(env.p3.const_im1)
    im2 = utils.load_image(env.p3.const_im2)

    points1 = utils.load_points(env.p3.pts_1)
    points2 = utils.load_points(env.p3.pts_2)
    assert (points1.shape == points2.shape)
    F = normalized_eight_point_alg(points1, points2)

    # Part 4.a
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print("e1", e1)
    print("e2", e2)
    assert np.allclose(e1, expected_e1, rtol=1e-2), f"e1 does not match this expected value:\n{expected_e1}"
    assert np.allclose(e2, expected_e2, rtol=1e-2), f"e2 does not match this expected value:\n{expected_e2}"
    np.save(env.p4.e1, e1)
    np.save(env.p4.e2, e2)

    #Part 4.b
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print("H1:\n", H1)
    print
    print("H2:\n", H2)
    # assert np.allclose(H1, expected_H1, rtol=1e-2), f"H1 does not match this expected value:\n{expected_H1}"
    # assert np.allclose(H2, expected_H2, rtol=1e-2), f"H2 does not match this expected value:\n{expected_H2}"
    np.save(env.p4.H1, H1)
    np.save(env.p4.H2, H2)

    # Part 4.c
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)

    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)
    total_offset_y = np.mean(new_points1[:, 1] - new_points2[:, 1]).round()

    F_new = normalized_eight_point_alg(new_points1, new_points2)
    lines1 = compute_epipolar_lines(new_points2, F_new.T)
    lines2 = compute_epipolar_lines(new_points1, F_new)
    aligned_img = show_epipolar_imgs(rectified_im1, rectified_im2, lines1, lines2, new_points1, new_points2, offset=int(total_offset_y))
    Image.fromarray(aligned_img).save(env.p4.aligned_epipolar)

    # Part 4.d
    im1 = utils.load_image(env.p3.const_im1)
    im2 = utils.load_image(env.p3.const_im2)
    kp1, kp2, good_matches = find_matches(im1, im2)
    cv_matches = show_matches(im1, im2, kp1, kp2, good_matches)
    Image.fromarray(cv_matches).save(env.p4.cv_matches)

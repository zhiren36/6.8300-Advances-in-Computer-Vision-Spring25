import os
import sys
sys.path.append(os.getcwd())
import env
import src.utils.engine as engine
import src.utils.utils as utils
from typing import List, Tuple

from src.p2_calibrate_camera import *
from src.p4_image_rectification import *

import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt


def recover_fundamental_matrix(kp1: List[cv2.KeyPoint], 
                               kp2: List[cv2.KeyPoint], 
                               good_matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover the fundamental matrix from the good matches
    Args:
        kp1: Keypoints in the first image
        kp2: Keypoints in the second image
        good_matches: Good matches between the keypoints
    Returns:
        The fundamental matrix, the mask, and the points from good_matches in the first and second images
    """
    # TODO: Implement this method!
    # Hint: Use parse_matches defined below and cv2.findFundamentalMat
    # raise NotImplementedError

    points1 = []
    points2 = []

    for match in good_matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        points1.append([x1, y1])
        points2.append([x2, y2])

    points1 = np.array(points1)
    points2 = np.array(points2)

    F, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC, 1, 0.99)

    return F, mask, points1, points2 





def compute_essential_matrix(camera_matrix: np.ndarray, 
                             fundamental_matrix: np.ndarray) -> np.ndarray:
    """
    Computes the essential matrix from the fundamental matrix and camera matrix.
    Args:
        camera_matrix: The camera matrix.
        fundamental_matrix: The fundamental matrix.
    Returns:
        The essential matrix.
    """
    # TODO: Implement this method!
    # Hint: should be a one-liner
    # raise NotImplementedError

    return camera_matrix.T @ fundamental_matrix @ camera_matrix


def estimate_initial_RT(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the initial rotation and translation matrices from the essential matrix
    Args:
        E: The essential matrix
    Returns:
        The rotation and translation matrices
    """
    # TODO: Implement this method!
    # Hint: Use the SVD decomposition of the essential matrix
    # raise NotImplementedError

    U, S, Vt = np.linalg.svd(E)

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    W_inv = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W_inv @ Vt

    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2

    rotations = np.array([R1, R2])

    t = U[:, 2]

    translations = np.array([t, -t])

    return rotations, translations 







def find_best_RT(candidate_Rs: List[np.ndarray], 
                 candidate_ts: List[np.ndarray], 
                 inlier_pts1: np.ndarray, 
                 inlier_pts2: np.ndarray,
                 camera_matrix: np.ndarray):
    """
    Find the best R and t that maximizes the number of inliers
    Args:
        candidate_Rs: List of candidate rotation matrices
        candidate_ts: List of candidate translation vectors
        inlier_pts1: Inlier points in the first image
        inlier_pts2: Inlier points in the second image
    Returns:
        The best R and t that maximizes the number of inliers
    """
    # TODO: Implement this method!
    # Hint: Use triangulatePoints
    # raise NotImplementedError
    P1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    max_count = 0

    

    for i in range(len(candidate_Rs)):
        for j in range(len(candidate_ts)):
            R = candidate_Rs[i]
            t = candidate_ts[j].reshape(3, 1)
            P2 = np.hstack([candidate_Rs[i], candidate_ts[j].reshape(-1, 1)])

            pts4D_h = cv2.triangulatePoints(P1, P2, inlier_pts1, inlier_pts2)
            pts3D = (pts4D_h[:3] / pts4D_h[3]).T

            depth1 = pts3D[:, 2]
            # Transform points to the second camera coordinate system: X_cam2 = R * X + t
            pts3D_cam2 = (R @ pts3D.T + t).T
            depth2 = pts3D_cam2[:, 2]
            
            # Count points that are in front of both cameras.
            count = np.sum((depth1 > 0) & (depth2 > 0))
            
            if count > max_count:
                max_count = count
                Best_R = R
                Best_T = t.flatten()  # return as a 1D array

    return Best_R, Best_T
            







def get_identity_projection_matrix(camera_matrix: np.ndarray) -> np.ndarray:
    """
    Returns the identity projection matrix.
    Args:
        camera_matrix: The camera matrix.
    Returns:
        The identity projection matrix.
    """
    # TODO: Implement this method!
    # Hint: should be a one-liner
    # raise NotImplementedError

    I0 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]])   
    
    return camera_matrix @ I0






def get_local_projection_matrix(camera_matrix: np.ndarray, 
                                R: np.ndarray,
                                T: np.ndarray) -> np.ndarray: 
    """
    Returns the local projection matrix.
    Args:
        camera_matrix: The camera matrix.
        R: The rotation matrix.
        T: The translation vector.
    Returns:
        The local projection matrix.
    """
    # TODO: Implement this method!
    # Hint: should be a one-liner
    # raise NotImplementedError

    I1 = np.hstack([R, T.reshape(-1, 1)])

    return camera_matrix @ I1







def calibrate_camera_from_chessboard(image_path: Path, 
                                     chessboard_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    image = utils.load_image(image_path)
    grayscale_image = load_grayscale_image(image)
    corners = find_chessboard_corners(grayscale_image, chessboard_size)
    corners = refine_corners(grayscale_image, corners)
    object_points = get_3D_object_points(chessboard_size)
    camera_matrix, dist_coeffs = calibrate_camera(object_points, corners, grayscale_image.shape[::-1])

    return camera_matrix, dist_coeffs


def undistort_images(folder: Path, 
                     out_folder: Path, 
                     camera_matrix: np.ndarray, 
                     dist_coeffs: np.ndarray) -> np.ndarray:
    if out_folder.exists() and out_folder.is_dir():
        shutil.rmtree(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    for filename in tqdm(os.listdir(folder), desc='Fixing Distortions'):
        image = utils.utils.load(folder / filename)
        corrected_image = undistort_image(image, camera_matrix, dist_coeffs)
        Image.fromarray(corrected_image).save(out_folder / filename)
    
    h, w = image.shape[:2]
    new_camera_mtx, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    return new_camera_mtx


def parse_matches(keypoints1: List[cv2.KeyPoint], 
                  keypoints2: List[cv2.KeyPoint], 
                  good_matches: List[cv2.DMatch]) -> Tuple[np.ndarray, np.ndarray]:
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])
    return points1, points2


def get_inliers(mask: np.ndarray, 
                pts1: np.ndarray, 
                pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    inlier_pts1 = pts1[mask.ravel() == 1]
    inlier_pts2 = pts2[mask.ravel() == 1]

    pts1 = inlier_pts1.reshape(-1, 2).T
    pts2 = inlier_pts2.reshape(-1, 2).T

    return pts1, pts2


def show_points_matplotlib(points3D: np.ndarray) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = points3D[:, 0]
    ys = points3D[:, 1]
    zs = points3D[:, 2]
    
    ax.scatter(xs, ys, zs, c='r', marker='o', s=5)
    
    # Set labels (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(env.p5.output):
        os.makedirs(env.p5.output)
    expected_R = np.load(env.p5.expected_R)
    expected_T = np.load(env.p5.expected_T)

    chessboard_size = (16, 10)  # (columns, rows)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--setup', action='store_true')
    args = parser.parse_args()
    setup = args.setup
    
    if setup:
        engine.get_chessboard(env.p5.chessboard)
        engine.get_object_images(env.p5.arc_obj, env.p5.arc_texture, env.p5.raw_images, views=5)  # may take a while; has many vertices!

    camera_matrix, dist_coeffs = calibrate_camera_from_chessboard(env.p5.chessboard, chessboard_size)

    im1 = utils.load_image(env.p5.raw_images / 'object_0.png')
    im2 = utils.load_image(env.p5.raw_images / 'object_1.png')

    kp1, kp2, good_matches = find_matches(im1, im2)
    show_matches(im1, im2, kp1, kp2, good_matches)

    # Part 5.a
    fundamental_matrix, mask, pts1, pts2 = recover_fundamental_matrix(kp1, kp2, good_matches)
    inlier_pts1, inlier_pts2 = get_inliers(mask, pts1, pts2)

    # Part 5.b
    essential_matrix = compute_essential_matrix(camera_matrix, fundamental_matrix)

    # Part 5.c
    R, T = estimate_initial_RT(essential_matrix)
    print
    print("Estimated R:\n", R)
    print("Estimated T:\n", T)

    # Part 5.d
    R, T = find_best_RT(R, T, inlier_pts1, inlier_pts2, camera_matrix)
    print
    print("Best R:\n", R)
    print("Best T:\n", T)
    # assert np.allclose(R, expected_R, atol=1e-2), f"R does not match this expected value:\n{expected_R}"
    # assert np.allclose(T, expected_T, atol=1e-2), f"T does not match this expected value:\n{expected_T}"
    num_points, CV2_R, CV2_t, mask = cv2.recoverPose(essential_matrix, pts1, pts2, camera_matrix)
    print("Number of points:", num_points,  "CV2_R:", CV2_R, "CV2_t:", CV2_t)


    np.save(env.p5.rotation_matrix, R)
    np.save(env.p5.translation_matrix, T)

    # Part 5.e
    P1 = get_identity_projection_matrix(camera_matrix)
    P2 = get_local_projection_matrix(camera_matrix, R, T)

    pts4D_h = cv2.triangulatePoints(P1, P2, inlier_pts1, inlier_pts2)
    pts3D = (pts4D_h[:3] / pts4D_h[3]).T
    print(f"Triangulated {len(pts3D)} points.")
    print("Example 3D point:", pts3D[0])

    U, S, Vt = np.linalg.svd(fundamental_matrix)
    print("Singular values of F:", S)

    show_points_matplotlib(pts3D)
    np.save(env.p5.pointcloud, pts3D)
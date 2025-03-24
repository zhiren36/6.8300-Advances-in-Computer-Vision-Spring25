from pathlib import Path
import os

PROJECT_DIR = Path(__file__).resolve().parent
project_data = PROJECT_DIR / 'data'
src = PROJECT_DIR / 'src'
project_output = PROJECT_DIR / 'outputs'

class p1:
    data = project_data / 'p1_edge_identification'
    output = project_output / 'p1_edge_identification'
    chessboard_path = data / 'chessboard.png'
    contours_path = output / 'contours.png'

class p2:
    data = project_data / 'p2_calibrate_camera'
    output = project_output / 'p2_calibrate_camera'
    chessboard_corners = output / 'corners.png'
    camera_matrix = output / 'camera_matrix.npy'
    dist_coeff = output / 'dist_coeff.npy'
    undistorted_image = output / 'undistorted_image.png'
    expected_camera_matrix = data / 'expected_camera_matrix.npy'
    expected_dist_coeffs = data / 'expected_dist_coeffs.npy'

class p3:
    data = project_data / 'p3_fundamental_matrix'
    output = project_output / 'p3_fundamental_matrix'
    test_obj = data / 'test.obj'
    test_texture = data / 'test.mtl'
    im1 = output / 'im1.png'
    im2 = output / 'im2.png'
    pts_1 = data / 'pts_1.txt'
    pts_2 = data / 'pts_2.txt'

    const_im1 = data / 'const_im1.png'
    const_im2 = data / 'const_im2.png'

    lls_img = output / 'lls_img.png'
    norm_img = output / 'norm_img.png'
    
    expected_F_LLS = data / 'expected_F_LLS.npy'
    expected_dist_LLS = data / 'expected_dist_LLS.npy'

    expected_F_normalized = data / 'expected_F_normalized.npy'
    expected_dist_normalized = data / 'expected_dist_normalized.npy'

    F_LLS = output / 'F_LLS.npy'
    dist_LLS = output / 'dist_LLS.npy'

    F_normalized = output / 'F_normalized.npy'
    dist_normalized = output / 'dist_normalized.npy'

class p4:
    data = project_data / 'p4_image_rectification'
    output = project_output / 'p4_image_rectification'
    aligned_epipolar = output / 'aligned_epipolar.png'
    cv_matches = output / 'cv_matches.png'

    expected_e1 = data / 'expected_e1.npy'
    expected_e2 = data / 'expected_e2.npy'
    expected_H1 = data / 'expected_H1.npy'
    expected_H2 = data / 'expected_H2.npy'

    e1 = output / 'e1.npy'
    e2 = output / 'e2.npy'
    H1 = output / 'H1.npy'
    H2 = output / 'H2.npy'

class p5:
    data = project_data / 'p5_3D_reconstruction'
    output = project_output / 'p5_3D_reconstruction'
    arc_obj = data / 'arc_de_triomphe' / 'model.obj'
    arc_texture = data / 'arc_de_triomphe' / 'model.mtl'
    chessboard = data / 'chessboard.png'
    raw_images = data / 'raw_images'
    undistorted_images = output / 'undistorted_images'
    pointcloud = output / 'pointcloud.npy'

    rotation_matrix = output / 'rotation_matrix.npy'
    translation_matrix = output / 'translation_matrix.npy'

    expected_R = data / 'expected_R.npy'
    expected_T = data / 'expected_T.npy'

class p6:
    data = project_data / 'p6_SfM_pipeline'
    output = project_output / 'p6_SfM_pipeline'
    statue_images = data / 'statue'
    pointcloud = output / 'pointcloud.npy'
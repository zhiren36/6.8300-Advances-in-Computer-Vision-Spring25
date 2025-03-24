import numpy as np
from numpy import pi


def magnify_change(
    im1: np.ndarray, im2: np.ndarray, magnification_factor: int
) -> np.ndarray:
    """Magnify the motion between two images."""
    # raise NotImplementedError("This is your homework.")

    # 2d discrete fourier transform

    DFT_im1 = np.fft.fftn(im1)
    DFT_im2 = np.fft.fftn(im2)

    # get phases for the two images
    phase_im1 = np.angle(DFT_im1)
    phase_im2 = np.angle(DFT_im2)

    # compute phase shifts
    phase_shifts = phase_im2 - phase_im1

    magified_phase_shifts = magnification_factor * phase_shifts

    new_phase_im2 = phase_im1 + magified_phase_shifts

    # reconstruct the complex numbers using magnitude
    # and new phase
    magnified_DFT_im2 = np.abs(DFT_im2) * np.exp(1j * new_phase_im2)

    # get back the image using inverse fourier transform

    magnified_im2 = np.fft.ifftn(magnified_DFT_im2).real

    return magnified_im2


def magnify_motion_global_question() -> np.ndarray:
    """
    Given two 9x9 images with the following point movements:
    1. A point moves from position (0,0) to (0,1)
    2. A point moves from position (8,8) to (7,8)

    If we magnify this motion by 4x using phase-based motion magnification,
    what positions would you expect the points to move to?

    Fill in the expected matrix to show where the points should appear
    after 4x magnification.
    """
    im_size = 9
    im1 = np.zeros([im_size, im_size], dtype=np.float32)
    im2 = np.zeros([im_size, im_size], dtype=np.float32)

    # Initial positions
    im1[0, 0] = 1.0
    im2[0, 1] = 1.0

    magnified_point1 = magnify_change(im1, im2, 4)

    im1_point2 = np.zeros([im_size, im_size], dtype=np.float32)
    im2_point2 = np.zeros([im_size, im_size], dtype=np.float32)
    im1_point2[8, 8] = 1.0
    im2_point2[7, 8] = 1.0

    magnified_point2 = magnify_change(im1_point2, im2_point2, 4)

    # Fill in your expected output matrix
    expected = np.zeros([im_size, im_size], dtype=np.float32)
    expected = magnified_point1 + magnified_point2
    # print(expected)
    return expected

    # raise NotImplementedError("This is your homework.")


def magnify_motion_local(
    im1: np.ndarray, im2: np.ndarray, magnification_factor: int, sigma: int
) -> np.ndarray:
    """Magnify motion using localized processing with Gaussian windows."""
    im_size = im1.shape[0]
    magnified = np.zeros([im_size, im_size])

    X, Y = np.meshgrid(np.arange(im_size), np.arange(im_size))
    for y in range(0, im_size, 2 * sigma):
        for x in range(0, im_size, 2 * sigma):
            # TODO: Create a Gaussian mask that covers the whole image and apply
            # it to the images

            smoothed_factors = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma**2))

            smoothed_im1 = smoothed_factors * im1
            smoothed_im2 = smoothed_factors * im2

            # TODO: Magnify the phase changes

            local_magnified = magnify_change(
                smoothed_im1, smoothed_im2, magnification_factor
            )

            magnified += local_magnified
    return magnified

    # raise NotImplementedError("This is your homework.")


def process_phase_shift(
    current_phase: np.ndarray, reference_phase: np.ndarray
) -> np.ndarray:
    """Computes phase shift and constrains it to [-π, π]"""
    shift = current_phase - reference_phase
    shift[shift > pi] -= 2 * pi
    shift[shift < -pi] += 2 * pi
    return shift


def update_moving_average(
    prev_average: np.ndarray, new_value: np.ndarray, alpha: float
) -> np.ndarray:
    """Updates the moving average of phase with temporal smoothing"""
    return alpha * prev_average + (1 - alpha) * new_value


def magnify_motion_video(
    frames: np.ndarray, magnification_factor: int, sigma: int, alpha: float
) -> np.ndarray:
    """Magnifies subtle motions in the video frames using phase-based magnification."""
    num_frames, height, width, num_channels = frames.shape
    magnified = np.zeros_like(frames)

    # raise NotImplementedError("This is your homework.")

    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    for y in range(0, height, 2 * sigma):
        for x in range(0, width, 2 * sigma):
            # TODO: Create a Gaussian mask that covers the whole image
            smoothed_factors = np.exp(-((X - x) ** 2 + (Y - y) ** 2) / (2 * sigma**2))

            for channel_index in range(num_channels):
                # initialize a moving average
                window_avg_phase = None

                for frame_index in range(num_frames):
                    # TODO: Apply gaussian mask to frame

                    # TODO: Perform magnification

                    # TODO: # Aggregate this window's contribution.

                    cur_frame = frames[frame_index, :, :, channel_index]
                    smoothed_cur_frame = smoothed_factors * cur_frame

                    DFT_cur_frame = np.fft.fftn(smoothed_cur_frame)
                    cur_phase = np.angle(DFT_cur_frame)

                    if window_avg_phase is None:
                        window_avg_phase = cur_phase
                    else:
                        window_avg_phase = update_moving_average(
                            window_avg_phase, cur_phase, alpha
                        )

                    shifts = process_phase_shift(cur_phase, window_avg_phase)
                    magnified_phase = magnification_factor * shifts
                    new_phase = cur_phase + magnified_phase
                    new_DFT = np.abs(DFT_cur_frame) * np.exp(1j * new_phase)
                    magnified_frame = np.fft.ifftn(new_DFT).real
                    magnified[frame_index, :, :, channel_index] += magnified_frame

    return magnified

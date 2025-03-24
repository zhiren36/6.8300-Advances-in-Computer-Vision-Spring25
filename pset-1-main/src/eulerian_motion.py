import numpy as np
from beartype.typing import List
import cv2
import scipy.signal as signal


def create_gaussian_pyramid(video: np.ndarray, num_levels: int = 4) -> List[np.ndarray]:
    """Return a list with Gaussian pyramid of the video. You may find cv2.pyrDown useful."""

    ##raise NotImplementedError("This is your homework.")

    gaussian_pyramid = [video]
    for _ in range(1, num_levels):

        prev_video = gaussian_pyramid[-1]
        cur_video = []
        for frame in prev_video:
            downsampled_frame = cv2.pyrDown(frame)
            cur_video.append(downsampled_frame)
        gaussian_pyramid.append(np.array(cur_video))

    return gaussian_pyramid


def create_laplacian_pyramid(gaussian_pyramid: List[np.ndarray]) -> List[np.ndarray]:
    """Return a list with Laplacian pyramid of the video. You may find cv2.pyrUp useful."""

    ## raise NotImplementedError("This is your homework.")

    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):

        laplacian_curr_level = []

        for frame_ind in range(gaussian_pyramid[i].shape[0]):
            curr_gaussian = gaussian_pyramid[i][frame_ind]
            next_gaussian = gaussian_pyramid[i + 1][frame_ind]
            laplacian_curr_frame = curr_gaussian - cv2.pyrUp(next_gaussian)
            laplacian_curr_level.append(laplacian_curr_frame)
        laplacian_pyramid.append(np.array(laplacian_curr_level))

    return laplacian_pyramid


def butter_bandpass_filter(
    laplace_video: np.ndarray,
    low_freq: float = 0.4,
    high_freq: float = 3.0,
    fs: float = 30.0,
    filter_order: int = 5,
) -> np.ndarray:
    """Filter video using a bandpass filter."""
    ## raise NotImplementedError("This is your homework.")

    b, a = signal.butter(filter_order, [low_freq, high_freq], btype="bandpass", fs=fs)

    filtered_video = signal.lfilter(b, a, laplace_video, axis=0)

    return filtered_video


def filter_laplacian_pyramid(
    laplacian_pyramid: List[np.ndarray],
    fs: float = 30.0,
    low: float = 0.4,
    high: float = 3.0,
    amplification: float = 20.0,
) -> List[np.ndarray]:
    """Filter each level of a Laplacian pyramid using a bandpass filter
    and amplify the result."""

    ## raise NotImplementedError("This is your homework.")

    amplified_pyramid = []

    for frames in laplacian_pyramid:
        frames = butter_bandpass_filter(frames, low, high, fs)
        frames *= amplification
        amplified_pyramid.append(frames)
    return amplified_pyramid


def create_euler_magnified_video(
    video: np.ndarray, bandpass_filtered: List[np.ndarray]
) -> np.ndarray:
    """Combine all the bandpassed filtered signals to one matrix which is the same
    dimensions as the input video.
    Hint: start from the lowest resolution of the amplified filtered signal,
    upsample that using cv2.pyrUp and add it to the amplified filtered signal
    at the next higher resolution.
    The output video, 'euler_magnified_video', will be the
    input video frames + combined magnified signal."""

    ## raise NotImplementedError("This is your homework.")

    # for i in range(len(bandpass_filtered) - 1, -1, -1):
    #     print("i, bandpass_filtered[i].shape", i, bandpass_filtered[i].shape)

    for i in range(len(bandpass_filtered) - 1, 0, -1):

        cur_frames = bandpass_filtered[i]
        # print("i, cur_frames.shape", i, cur_frames.shape)

        for frame_ind in range(cur_frames.shape[0]):
            upsampled = cv2.pyrUp(cur_frames[frame_ind])
            # print("i, upsampled.shape", i, upsampled.shape)
            bandpass_filtered[i - 1][frame_ind] += upsampled
    return video + bandpass_filtered[0]

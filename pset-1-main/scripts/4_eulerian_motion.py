import sys
from pathlib import Path
from jaxtyping import install_import_hook

# Add runtime type checking to all imports
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.eulerian_motion import (
        create_gaussian_pyramid,
        create_laplacian_pyramid,
        filter_laplacian_pyramid,
        create_euler_magnified_video,
    )
    from src.provided_code import (
        download_file,
        load_video,
        preprocess_frames,
        save_video,
        save_image_subplots,
        scale_intensity,
    )

OUTPUT_DIR = Path("outputs/4_eulerian_motion")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def run_create_gaussian_pyramid(
    video_path: Path = Path("data/baby.mp4"),
    download_url: str = "http://people.csail.mit.edu/mrub/evm/video/baby.mp4",
):
    """
    Create Gaussian Pyramid.
    """
    print("Running: create_gaussian_pyramid")

    download_file(download_url, video_path)
    raw_frames = load_video(video_path)
    frames = preprocess_frames(raw_frames)
    print("Loaded frames:", frames.shape)

    gaussian_pyramid = create_gaussian_pyramid(frames)
    pyramid_images = [
        scale_intensity(pyramid_level[0, ..., ::-1])
        for pyramid_level in gaussian_pyramid
    ]
    pyramid_titles = [f"Level {i}" for i in range(len(pyramid_images))]

    out_path = OUTPUT_DIR / "gaussian_pyramid.png"

    save_image_subplots(pyramid_images, pyramid_titles, out_path)

    print(f"Saved Gaussian pyramid plot in {out_path}")


def run_create_laplacian_pyramid(
    video_path: Path = Path("data/baby.mp4"),
    download_url: str = "http://people.csail.mit.edu/mrub/evm/video/baby.mp4",
):
    """
    Create Laplacian Pyramid.
    """
    print("Running: create_laplacian_pyramid")

    download_file(download_url, video_path)
    raw_frames = load_video(video_path)
    frames = preprocess_frames(raw_frames)
    print("Loaded frames:", frames.shape)

    gaussian_pyramid = create_gaussian_pyramid(frames)
    laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)
    pyramid_images = [
        scale_intensity(pyramid_level[0, ..., ::-1])
        for pyramid_level in laplacian_pyramid
    ]
    pyramid_titles = [f"Level {i}" for i in range(len(pyramid_images))]

    out_path = OUTPUT_DIR / "laplacian_pyramid.png"

    save_image_subplots(pyramid_images, pyramid_titles, out_path)

    print(f"Saved Laplacian pyramid plot in {out_path}")


def run_create_euler_magnified_video(
    video_path: Path = Path("data/baby.mp4"),
    download_url: str = "http://people.csail.mit.edu/mrub/evm/video/baby.mp4",
):
    download_file(download_url, video_path)
    raw_frames = load_video(video_path)
    frames = preprocess_frames(raw_frames)
    print("Loaded frames:", frames.shape)

    gaussian_pyramid = create_gaussian_pyramid(frames)
    laplacian_pyramid = create_laplacian_pyramid(gaussian_pyramid)

    bandpass_filtered = filter_laplacian_pyramid(laplacian_pyramid)

    euler_magnified_video = create_euler_magnified_video(frames, bandpass_filtered)

    out_path = OUTPUT_DIR / "baby_euler_magnified.avi"
    save_video(euler_magnified_video, out_path)

    # Optional: If the AVI output is not opening on your computer,
    # you can convert it to MP4 by uncommenting the following lines.

    # mp4_out_path = OUTPUT_DIR / "baby_euler_magnified.mp4"
    # subprocess.run(["ffmpeg", "-y", "-i", str(out_path), str(mp4_out_path)], check=True)
    # print(f"Converted video saved to {mp4_out_path}")
    # print(f"Saved video to {out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]

    valid_options = [
        "create_gaussian_pyramid",
        "create_laplacian_pyramid",
        "create_euler_magnified_video",
        "all",
    ]
    if len(args) == 0:
        print("No task specified. Options:", valid_options)
        sys.exit(0)

    task = args[0].lower()
    if task not in valid_options:
        print(f"Invalid task '{task}'. Choose from {valid_options}.")
        sys.exit(1)

    if task == "all":
        run_create_gaussian_pyramid()
        run_create_laplacian_pyramid()
        run_create_euler_magnified_video()
    elif task == "create_gaussian_pyramid":
        run_create_gaussian_pyramid()
    elif task == "create_laplacian_pyramid":
        run_create_laplacian_pyramid()
    elif task == "create_euler_magnified_video":
        run_create_euler_magnified_video()

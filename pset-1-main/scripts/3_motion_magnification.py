import sys
from pathlib import Path
from jaxtyping import install_import_hook

# Add runtime type checking to all imports
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.magnify import (
        magnify_change,
        magnify_motion_global_question,
        magnify_motion_local,
        magnify_motion_video,
    )
    from src.provided_code import (
        save_image_subplots,
        download_file,
        load_video,
        save_video,
        match_brightness_and_contrast,
    )
import numpy as np

OUTPUT_DIR = Path("outputs/3_motion_magnification")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def run_single_motion():
    print("Running: single_motion")

    im_size = 9
    magnification_factor = 4

    im1 = np.zeros((im_size, im_size), dtype=np.float32)
    im2 = np.zeros((im_size, im_size), dtype=np.float32)
    im1[0, 0] = 1.0
    im2[0, 1] = 1.0

    magnified = magnify_change(im1, im2, magnification_factor)

    out_path = OUTPUT_DIR / "single_motion.png"
    save_image_subplots([im1, im2, magnified], ["im1", "im2", "magnified"], out_path)

    print(f"Saved {out_path}")


def run_multiple_motions_global():
    print("Running: multiple_motions_global")

    im_size = 9
    magnification_factor = 4

    im1 = np.zeros((im_size, im_size), dtype=np.float32)
    im2 = np.zeros((im_size, im_size), dtype=np.float32)
    im1[0, 0] = im1[8, 8] = 1.0
    im2[0, 1] = 1.0
    im2[7, 8] = 1.0

    magnified = magnify_change(im1, im2, magnification_factor)
    expected = magnify_motion_global_question()

    out_path = OUTPUT_DIR / "multiple_motions_global.png"
    save_image_subplots(
        [im1, im2, expected, magnified],
        ["im1", "im2", "expected", "magnified"],
        out_path,
    )

    print(f"Saved {out_path}")


def run_multiple_motions_local():
    print("Running: multiple_motions_local")

    im_size = 9
    magnification_factor = 4
    sigma = 2

    im1 = np.zeros((im_size, im_size), dtype=np.float32)
    im2 = np.zeros((im_size, im_size), dtype=np.float32)
    im1[0, 0] = im1[8, 8] = 1.0
    im2[0, 1] = 1.0
    im2[7, 8] = 1.0

    magnified = magnify_motion_local(im1, im2, magnification_factor, sigma)

    out_path = OUTPUT_DIR / "multiple_motions_local.png"
    save_image_subplots([im1, im2, magnified], ["im1", "im2", "magnified"], out_path)

    print(f"Saved {out_path}")


def run_motion_magnified_video(
    download_url: str = "http://6.869.csail.mit.edu/sp21/pset3_data/bill.avi",
    video_path: Path = Path("data/bill.avi"),
):
    print("Running: motion_magnified_video")

    download_file(download_url, video_path)
    frames = load_video(video_path)
    print("Loaded frames:", frames.shape)

    magnification_factor = 10
    sigma = 13
    alpha = 0.5

    magnified = magnify_motion_video(frames, magnification_factor, sigma, alpha)
    magnified_aligned = match_brightness_and_contrast(magnified, frames)

    out_path = OUTPUT_DIR / "bill_magnified.avi"
    save_video(magnified_aligned, out_path)
    print(f"Saved video to {out_path}")

    # Optional: If the AVI output is not opening on your computer,
    # you can convert it to MP4 by uncommenting the following lines.

    # mp4_out_path = OUTPUT_DIR / "bill_magnified.mp4"
    # subprocess.run(["ffmpeg", "-y", "-i", str(out_path), str(mp4_out_path)], check=True)
    # print(f"Converted video saved to {mp4_out_path}")


if __name__ == "__main__":
    args = sys.argv[1:]

    valid_options = [
        "single_motion",
        "multiple_motions_global",
        "multiple_motions_local",
        "motion_magnified_video",
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
        run_single_motion()
        run_multiple_motions_global()
        run_multiple_motions_local()
        run_motion_magnified_video()
    elif task == "single_motion":
        run_single_motion()
    elif task == "multiple_motions_global":
        run_multiple_motions_global()
    elif task == "multiple_motions_local":
        run_multiple_motions_local()
    elif task == "motion_magnified_video":
        run_motion_magnified_video()

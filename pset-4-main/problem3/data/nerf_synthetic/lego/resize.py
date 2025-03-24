import os
import fire
from PIL import Image

def resize_images(src_folder, dst_folder, width=200, height=200):
    """
    Resize all images in the src_folder to the given width and height,
    and save them in the dst_folder with the same image names.

    Args:
        src_folder (str): Path to the folder containing the original images.
        dst_folder (str): Path to the folder where resized images will be saved.
        width (int, optional): The target width. Defaults to 200.
        height (int, optional): The target height. Defaults to 200.
    """
    # Create the destination folder if it does not exist
    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)

    # Process each file in the source folder
    for file_name in os.listdir(src_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            src_path = os.path.join(src_folder, file_name)
            dst_path = os.path.join(dst_folder, file_name)
            try:
                with Image.open(src_path) as img:
                    # Resize the image using LANCZOS resampling
                    resized_img = img.resize((width, height), resample=Image.Resampling.LANCZOS)
                    resized_img.save(dst_path)
                print(f"Resized {file_name} and saved to {dst_path}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

if __name__ == '__main__':
    fire.Fire(resize_images)


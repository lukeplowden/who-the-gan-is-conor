import cv2
import os
import numpy as np

def images_to_video(image_folder, output_folder, fps=25):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(image_folder):
        for dir_name in dirs:
            current_dir = os.path.join(root, dir_name)
            images = [img for img in os.listdir(current_dir) if img.endswith(".jpg")]
            images.sort()  # Ensure the images are sorted

            if images:
                # Get the dimensions of the first image
                frame = cv2.imread(os.path.join(current_dir, images[0]))
                height, width, layers = frame.shape

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
                output_video_path = os.path.join(output_folder, f"{dir_name}.mp4")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

                for image in images:
                    img_path = os.path.join(current_dir, image)
                    frame = cv2.imread(img_path)
                    out.write(frame)  # Write out frame to video

                out.release()  # Release the VideoWriter

base_dir = r'D:\OPEN_ME\coding3\generated_frames'
output_base_dir = r'D:\OPEN_ME\coding3\generated_videos2'
images_to_video(base_dir, output_base_dir)
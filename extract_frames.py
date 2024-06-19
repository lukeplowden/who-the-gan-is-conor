import cv2
import os

def extract_frames(video_path, frame_dir):
    """Extract frames from a video file, one frame per second."""
    if not os.path.exists(frame_dir):
        os.makedirs(frame_dir)
    
    vidcap = cv2.VideoCapture(video_path)
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))  # Get frames per second of the video
    success, image = vidcap.read()
    count = 0
    frame_count = 0
    last_frame = None

    while success:
        if count % fps == 0:  # Extract one frame every 'fps' frames
            cv2.imwrite(os.path.join(frame_dir, "frame{0:05d}.png".format(frame_count)), image)
            frame_count += 1
        last_frame = image
        success, image = vidcap.read()
        count += 1

    # After the loop, write the last frame if it wasn't written as part of the regular extraction
    if last_frame is not None:
        cv2.imwrite(os.path.join(frame_dir, "frame{0:05d}.png".format(frame_count)), last_frame)

def process_videos_in_directory(base_dir, output_base_dir):
    """Process all video files in the given directory and its subdirectories, outputting frames to a different directory."""
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(('.mp4', '.avi', '.mov')):  # Add or remove video formats as needed
                video_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, base_dir)  # Get the relative path of the current directory to the base directory
                frame_dir = os.path.join(output_base_dir, relative_path, file.split('.')[0] + "_frames")
                extract_frames(video_path, frame_dir)

base_dir = r'missing_titles/'
output_base_dir = r'extracted_frames/'  # New base directory for extracted frames
process_videos_in_directory(base_dir, output_base_dir)
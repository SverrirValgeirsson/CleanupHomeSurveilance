import os
import cv2
import torch
import shutil
from pathlib import Path

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l')  # You can also use 'yolov5m', 'yolov5l', 'yolov5x'

# Define the classes we are interested in (person and cat)
classes_of_interest = ['person', 'cat']

def check_for_people_or_cats(video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Perform object detection
        results = model(frame_rgb)
        # Get labels of detected objects
        labels = results.xyxyn[0][:, -1].cpu().numpy()
        names = results.names

        for label in labels:
            if names[int(label)] in classes_of_interest:
                cap.release()
                return True
    cap.release()
    return False

def process_directory(src_directory, remove_directory):
    for root, _, files in os.walk(src_directory):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                print(f"Checking {video_path}")
                if not check_for_people_or_cats(video_path):
                    # Create the remove directory if it does not exist
                    Path(remove_directory).mkdir(parents=True, exist_ok=True)
                    # Move the file to the remove directory
                    
                    os.rename(video_path, remove_directory+"/"+video_path.replace("/mnt/extra/arlo_video/","").replace("/","--"))
                    #shutil.move(video_path, os.path.join(remove_directory, file))
                    print(f"Moved {video_path} to {remove_directory}")

# Define the source and remove directories
src_directory = '/mnt/extra/arlo_video'
remove_directory = '/mnt/extra/remove'

# Process the directory
process_directory(src_directory, remove_directory)

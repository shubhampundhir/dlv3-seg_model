import os
import cv2
import numpy as np
root = "/home/alive/Desktop/himanshi/mord/DeepLabV3Plus-Pytorch-mord/result_emarg_videos_1/Jamulni-Adumber"
#change root, fps, vdo
def create_video(folder_path, root, fps=30):

    images = sorted(os.path.join(folder_path, img) for img in os.listdir(folder_path) if img.endswith('.png'))

    # Check if images exist
    if not images:
        print("No images found in folder:", folder_path)
        return

    # Read first image to determine video dimensions
    first_image = cv2.imread(images[0])
    height, width, channels = first_image.shape

    # Create video writer
    video_writer = cv2.VideoWriter(f"{folder_path}.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Add images to video
    for image_path in images:
        print(image_path)
        image = cv2.imread(image_path)
        video_writer.write(image)

    # Release video writer
    video_writer.release()
    print(f"Video created: {folder_path}.mp4")
    
    run(root, f"{folder_path}.mp4", "/home/alive/Desktop/himanshi/mord/DeepLabV3Plus-Pytorch-mord/datasets/video_data/videos/Videos_from_the_emarg_portal/Jamulni - Adumber.mp4")


def run(results_folder, video_path, og_video_path):
    video1 = cv2.VideoCapture(video_path)
    video2 = cv2.VideoCapture(og_video_path)

    if not (video1.isOpened() and video2.isOpened()):
        print("Error: One or more video files could not be opened.")
        exit()

    # Get dimensions from the original video
    width = int(video2.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video2.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a folder to save the results
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    matched_obj_path = os.path.join(results_folder, 'matched_video.mp4')
    print(matched_obj_path)
    video_writer_matched = cv2.VideoWriter(matched_obj_path, cv2.VideoWriter_fourcc(*'mp4v'), 15, (width*2, height))

    for i in range(int(video1.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame1 = video1.read()
        ret, frame2 = video2.read()

        # Check if frames are read successfully
        if not (ret and frame1 is not None and frame2 is not None):
            print("Error: Failed to read one or more frames.")
            break

        # Resize frame1 to the desired dimensions
        frame1 = cv2.resize(frame1, (width, height))

        # Concatenate frames horizontally
        matched_img = np.hstack((frame1, frame2))

        # Write the result to the output video
        video_writer_matched.write(matched_img)

    # Release video capture and writer objects
    video1.release()
    video2.release()
    video_writer_matched.release()
    cv2.destroyAllWindows()


create_video(root+"/overlay", root)
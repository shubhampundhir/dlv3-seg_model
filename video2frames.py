# w:1920, h:1080
# # 
import cv2
import time
import os

def video_to_frames(input_loc, output_loc):

    try:
        os.mkdir(output_loc)
    except OSError:
        pass
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        if not ret:
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        print(count)
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds forconversion." % (time_end-time_start))
            break

if __name__=="__main__": 
    videos_folder= "/home/alive/Desktop/himanshi/mord/DeepLabV3Plus-Pytorch-mord/datasets/video_data/videos/Videos_from_the_emarg_portal/"
    output_folder = "/home/alive/Desktop/himanshi/mord/DeepLabV3Plus-Pytorch-mord/datasets/video_data/frames/Videos_from_the_emarg_portal/"
 
    for video_file in os.listdir(videos_folder):
        print(video_file)
        if video_file.endswith(".mp4"):
            video_path = os.path.join(videos_folder, video_file)

            video_name = os.path.splitext(video_file)[0]
            output_directory = os.path.join(output_folder, video_name)

            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

            # Extract frames and save them in the created directory
            output_frames_path = os.path.join(output_directory, f"")
            video_to_frames(video_path, output_frames_path)
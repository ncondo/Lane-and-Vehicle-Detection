from moviepy.editor import VideoFileClip
from lane_tracker import LaneTracker
from models.small_fcn import small_fcn
from vehicle_tracker import add_heat, apply_threshold
from collections import deque
from scipy.ndimage.measurements import label
import numpy as np
import cv2


def process_video(img):
    # Find the lane lines first
    img_lanes = lane_tracker.apply_lines(img)

    # Find the cars
    cropped = img[400:660, 0:1280]
    heat = model.predict(cropped.reshape(1,cropped.shape[0],cropped.shape[1],cropped.shape[2]))
    # This finds us rectangles that are interesting
    xx, yy = np.meshgrid(np.arange(heat.shape[2]),np.arange(heat.shape[1]))
    x = (xx[heat[0,:,:,0]>0.9999999])
    y = (yy[heat[0,:,:,0]>0.9999999])
    hot_windows = []
    # We save those rects in a list
    for i,j in zip(x,y):
        hot_windows.append(((i*8,400 + j*8), (i*8+64,400 +j*8+64)))

    # Create image for the heat similar to one shown above
    heat = np.zeros_like(img[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,3)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    boxes = label(heatmap)

    # Iterate through all detected cars
    for car_number in range(1, boxes[1]+1):
        # Find pixels with each car_number label value
        nonzero = (boxes[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Append current boxe to history
        history.append([np.min(nonzerox),np.min(nonzeroy),np.max(nonzerox),np.max(nonzeroy)])

    # Get recent boxes for the last 30 fps
    recent_boxes = np.array(history).tolist()

    # Groups the object candidate rectangles with difference of 10%
    boxes = cv2.groupRectangles(recent_boxes, 10, .1)

    # Draw rectangles if found
    if len(boxes[0]) != 0:
        for box in boxes[0]:
            cv2.rectangle(img_lanes, (box[0], box[1]), (box[2],box[3]), (0,255,0), 6)

    # Return image with found cars and lanes
    return img_lanes


if __name__=='__main__':

    # Create LaneTracker object
    lane_tracker = LaneTracker()
    # Create model for vehicle detection
    model = small_fcn(input_shape=(260, 1280, 3))
    model.load_weights('models/model.h5')

    # Create history for 30 frames
    history = deque(maxlen=30)

    # Name of output video after applying lane lines
    output_video = 'output_videos/project_video_out.mp4'
    # Apply lane lines to each frame of input video and save as new video file
    clip1 = VideoFileClip('test_videos/project_video.mp4')
    video_clip = clip1.fl_image(process_video)
    video_clip.write_videofile(output_video, audio=False)

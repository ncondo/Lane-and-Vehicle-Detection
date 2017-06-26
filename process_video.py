from moviepy.editor import VideoFileClip
from lane_tracker import LaneTracker


def process_video(image):
    # Create LaneTracker object with matrix and distortion coefficients
    lane_tracker = LaneTracker()

    # Apply display of detected lane
    img = lane_tracker.apply_lines(image)

    return img


if __name__=='__main__':

    # Name of output video after applying lane lines
    output_video = 'output_videos/project_video_out.mp4'
    # Apply lane lines to each frame of input video and save as new video file
    clip1 = VideoFileClip('test_videos/project_video.mp4')
    video_clip = clip1.fl_image(process_video)
    video_clip.write_videofile(output_video, audio=False)

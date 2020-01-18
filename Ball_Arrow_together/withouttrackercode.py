## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np

import time
from imutils.video import FPS
import imutils
import cv2

def tracker(frames_for_optical_flow) :
    greenLower = (29, 86, 6)
    greenUpper = (86, 255, 255)

    pipeline = rs.pipeline()
    config = rs.config()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    #clipping_distance_in_meters = 1 #1 meter
    #clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()

            # frames.get_depth_frame() is a 640x360 depth image
            
            # Align the depth frame to color frame
            aligned_frames = align.process(frames)
            
            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()
            image = np.asanyarray(color_frame.get_data())
            frame = imutils.resize(image, width=600)

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue
            
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
            
            #depth 3d distance
            depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            
        
            #gaussian blur
            blurred = cv2.GaussianBlur(color_image, (11, 11), 0)
            
            #hsv mapping
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            
            #masks (threshold, erode, dilate)
            mask = cv2.inRange(hsv, greenLower, greenUpper)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            #contour detection
            cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            center = None
            
            # only proceed if at least one contour was found
            if len(cnts) > 0:

                # find the largest contour in the mask, then use
                # it to compute the minimum enclosing circle and
                # centroid
                c = max(cnts, key=cv2.contourArea)
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                M = cv2.moments(c)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                
                #minimum radius
                if radius > 10:
                    # draw the circle and centroid on the frame,
                    # then update the list of tracked points
                    xx = int(M["m10"] / M["m00"])
                    yy = int(M["m01"] / M["m00"])
                    dist = aligned_depth_frame.get_distance(xx,yy)
                    print(dist)

                    height = 4 
                    width  = 4
                   
     #               detected_bounding_box = cv2.rectangle(xx,yy,width,height)
     #               init_bounding_box = detected_bounding_box
                    #3d camera coordinates
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [yy, xx], dist)

                    print( "x,y,z")
                    print(depth_point[0] )
                    print(depth_point[1] )
                    print(depth_point[2] )

                    #drawing circle and centroid
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
             

            # show the frame to our screen
            cv2.imshow("Frame", frame)
            #text="Fps : "+fps.fps()==
            
            #cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.namedWindow('windoww', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('IR Example', image)

            # end if q pressed
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()


tracker(5)
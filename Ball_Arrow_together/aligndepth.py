## Copyright(c) 2020 ARES Corporation. All Rights Reserved.

#####################################################
##       Align Depth to Color find green ball      ##
#####################################################

import pyrealsense2 as rs
import numpy as np

import time
from imutils.video import FPS
import imutils
import cv2



def euclidDist( x,y) :
    x1 = x[0]
    y1 = x[1]
    x2 = y[0]
    y2 = y[1]
    dist = abs((x1-x2)*(x1*x2) + (y1-y2)*(y1-y2))
    return dist

def slopes(x,y):
    if( y[0] == x[0]) :
        if y[1]>x[1] :
            return 90
        else :
            return -90
    sloper = (y[1]-x[1])/(y[0]-x[0])
    angle = np.rad2deg(np.arctan2(y[1] - x[1], y[0] - x[0]))
    return angle

def quadrant(slope):
    if slope >0 and slope <=45 :
        decision =  8
    elif slope>45 and slope <=90 :
        decision=7
    elif slope>90 and slope<=135:
        decision =6
    elif slope>135 and slope<180 :
        decision =5
    elif slope <0 and slope >=-45 :
        decision =  1
    elif slope <-45 and slope >=-90 :
        decision=2
    elif slope <-90 and slope>=-135:
        decision =3
    else:
        decision =4
    return decision


def arrow_Detector(imgg):
    #Reading image 
    img2 = imgg
       
    # Reading same image in another variable and  
    # converting to gray scale. 
    img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY) 
       
    # Converting image to a binary image  
    # (black and white only image). 
    _,threshold = cv2.threshold(img, 110, 255,cv2.THRESH_BINARY) 
       
    # Detecting shapes in image by selecting region  
    # with same colors or intensity. 
    contours,_=cv2.findContours(threshold, cv2.RETR_TREE, 
                                cv2.CHAIN_APPROX_SIMPLE) 
       
    # Searching through every region selected to  
    # find the required polygon. 
    for cnt in contours : 
        area = cv2.contourArea(cnt) 
       
        # Shortlisting the regions based on there area. 
        if area > 400:  
            approx = cv2.approxPolyDP(cnt,  
                                      0.009 * cv2.arcLength(cnt, True), True) 
            left_right = None
            decision=None
            # Checking if the no. of sides of the selected region is 7. 
            if(len(approx) == 7):
                cv2.drawContours(img2, [approx], 0, (0, 255, 0), 5) 
                #rect = cv2.minAreaRect(approx)
                ellipse = cv2.fitEllipse(approx)
                cv2.ellipse(img2,ellipse,(0,255,0),2)

                M = cv2.moments(approx)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                print(ellipse)
               # G = cv2.moments(ellipse)

                cxx = int(ellipse[0][0])
                cyy = int(ellipse[0][1])
                print((cx,cy))
                print((cxx,cyy))
                slope = slopes((cxx,cyy),(cx,cy))
                cv2.line(img2,(cxx,cyy),(cx,cy),(255,0,0),2)
                
                print(slope)
                decision = quadrant(slope)
                print(decision)
                

    # Showing the image along with outlined arrow. 
    cv2.imshow('image2', img2)  
       
    # Exiting the window if 'q' is pressed on the keyboard. 
    if cv2.waitKey(0) & 0xFF == ord('q'):  
        cv2.destroyAllWindows() 

def green_ball(frames_for_optical_flow) :
    greenLower = (29, 86, 6)
    greenUpper = (86, 255, 255)

    pipeline = rs.pipeline()
    config   = rs.config()

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
    
    #tracker
    '''success=0
    tracker = cv2.TrackerTLD_create()
    init_bounding_box = None
    fps = None
    '''
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
            
            #for tracking

            #if init_bounding_box is not None:
            #    (success,bounding_box) = tracker.update(frame)
            #    if success : 
            #        (x, y, w, h) = [int(v) for v in bounding_box]
            #        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #else:
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
                   
                    detected_bounding_box = cv2.rectangle(xx,yy,width,height)
                    init_bounding_box = detected_bounding_box
                    #3d camera coordinates
                    depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [yy, xx], dist)

                    print( "x,y,z")
                    print(depth_point[0] )
                    print(depth_point[1] )
                    print(depth_point[2] )

                    #drawing circle and centroid
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                    
                        #tracker.init(image, detected_bounding_box)
                        #(x, y, w, h) = [int(v) for v in bounding_box]
                        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
     #                  fps = FPS().start()
                        #tracking starts
                        #success=1

                # show the frame to our screen
                cv2.imshow("Frame", frame)
                #text="Fps : "+fps.fps()
                #cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.namedWindow('windoww', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('IR Example', image)

            #arrow detection -->
            arrow_Detector(image)

            # end if q pressed
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

green_ball(5)
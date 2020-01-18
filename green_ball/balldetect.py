# USAGE
# python ball_tracking.py --video ball_tracking_example.mp4
# python ball_tracking.py

# import the necessary packages
from collections import deque
import numpy as np

import imutils
import argparse
import cv2
import pyrealsense2 as rs
import time



# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points

points = rs.points()
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)


greenLower = (29, 86, 6)
greenUpper = (86, 255, 255)

try:
	while True:
		frames = pipeline.wait_for_frames()
		col_frame = frames.get_color_frame()
		depth = frames.get_depth_frame()
		image = np.asanyarray(col_frame.get_data())
		# horizontal stack
		frame = imutils.resize(image, width=600)

		blurred = cv2.GaussianBlur(frame, (11, 11), 0)

		hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

		mask = cv2.inRange(hsv, greenLower, greenUpper)
		#cv2.namedWindow('windowws', cv2.WINDOW_AUTOSIZE)
		#cv2.imshow('threshold',mask)
		#key=cv2.waitKey(1)
		#if key & 0xFF == ord('q') or key == 27:
		#	cv2.destroyAllWindows()

		mask = cv2.erode(mask, None, iterations=2)
		mask = cv2.dilate(mask, None, iterations=2)

		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,	cv2.CHAIN_APPROX_SIMPLE)
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
			xx = int(M["m10"] / M["m00"])
			yy = int(M["m01"] / M["m00"])
			if not depth:
				print("wtfff")
			else:
				dist = depth.get_distance(xx,yy)
				print(dist)
			if radius > 10:
				# draw the circle and centroid on the frame,
				# then update the list of tracked points
				cv2.circle(frame, (int(x), int(y)), int(radius),
					(0, 255, 255), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
			# show the frame to our screen
			cv2.imshow("Frame", frame)
			cv2.namedWindow('windoww', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('IR Example', image)



			

			key = cv2.waitKey(1)
			# Press esc or 'q' to close the image window
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				pipeline.stop()
				break

finally:
	pipeline.stop()



import cv2 
import pyrealsense2 as rs
import numpy as np
pipeline = rs.pipeline()
config = rs.config()


pipeline.start(config)

i=1
frame_no=1
try:
	while True: 
		#cv2.imwrite('frame_'+str(frame_no)+'.jpg',image)
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q') or key == 27:
			break
		frame_no = frame_no +1
finally:
	pipeline.stop()
	print("over bruh")
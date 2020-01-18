
# Python code to detect an arrow (seven-sided shape) from an image. 
import numpy as np 
import cv2 
import math


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
    elif slope<-45 and slope >=-90 :
        decision=2
    elif slope<-90 and slope>=-135:
        decision =3
    else:
        decision =4
    return decision


def arrow_Detector():
    #Reading image 
    img2 = cv2.imread('arrow5.png', cv2.IMREAD_COLOR) 
       
    # Reading same image in another variable and  
    # converting to gray scale. 
    img = cv2.imread('arrow5.png', cv2.IMREAD_GRAYSCALE) 
       
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

arrow_Detector()
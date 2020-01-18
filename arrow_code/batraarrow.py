import numpy as np
import cv2
cap = cv2.VideoCapture(-1)

while(True):
    # Capture frame-by-frame
    ret, img = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((4, 4), np.uint8)
#     dilation = cv2.dilate(gray, kernel, iterations=1)
#     blur = cv2.GaussianBlur(dilation, (5, 5), 0)

    cv2.imshow("blur.png", gray)
#     ret, thresh = cv2.threshold(gray,70,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 20)
    cv2.imshow("thresh", thresh)
    # Now finding Contours         ###################
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []
    for cnt in contours:
            # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(
            cnt, 0.07 * cv2.arcLength(cnt, True), True)
        if len(approx) == 2 and cv2.contourArea(cnt) > 600 :
            coordinates.append([cnt])
            cv2.drawContours(img, [cnt], 0, (0, 0, 255), 3)
        if len(approx) >= 7 and cv2.contourArea(cnt) >200 :
            cv2.drawContours(img, [cnt], 0, (0, 255, 0), 3)
        print(len(approx))
    out = np.zeros_like(img)

    cv2.imshow("result.png", img)



    # cv2.imshow('sample image',img)

    # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
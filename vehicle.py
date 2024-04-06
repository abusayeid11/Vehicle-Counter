import cv2
import numpy as np

#web Camera
cap = cv2.VideoCapture('video2.mp4')

minWidthRect = 80
minHeightRect = 80
countLinePos = 550

#initialize substractor

algo = cv2.createBackgroundSubtractorKNN()

def centerHandle(x, y, w, h):
      x1 = int(w/2)
      y1 = int(h/2)
      cx = x + x1
      cy = y + y1
      return cx, cy

detect = []
offset = 6 #allowable error
counter = 0

while True:
    ret, frame1 = cap.read()
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey,(3,3), 5)

    #applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilatData = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatData = cv2.morphologyEx(dilatData, cv2.MORPH_CLOSE, kernel)
    counterShape, h = cv2.findContours(dilatData, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25, countLinePos), (1200, countLinePos), (255, 127, 0), 3)

    for (i, c) in enumerate(counterShape):
          (x, y, w, h) = cv2.boundingRect(c)
          validateCounter = (w>=minWidthRect) and (h>=minHeightRect)
          if not validateCounter:
                continue
          
          cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
          center = centerHandle(x, y, w, h)
          detect.append(center)
          cv2.circle(frame1, center,4, (0, 0, 255), -1)


    for (x, y) in detect:
          if y < (countLinePos + offset) and y > (countLinePos-offset):
                counter+=1  
          cv2.line(frame1, (25, countLinePos), (1200, countLinePos), (0, 127, 255), 3)
          detect.remove((x, y))

          print("Vehicle Counter :" + str(counter))

    #cv2.imshow('Detector', dilatData)

    cv2.putText(frame1, "Vehicle Counter:"+ str(counter), (450, 70), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 5)
    cv2.imshow('Video Original', frame1)

    if cv2.waitKey(1) == 13:
            break
    

cv2.destroyAllWindows()
cap.release()


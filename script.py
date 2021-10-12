import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import time

fc_path = "./haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(fc_path)

def detect_and_blur_face(img):
    face_img = img.copy()
    roi = img.copy() # region of interest
      
    # get coord of detected fact
    face_rects = face_cascade.detectMultiScale(face_img,
                                            scaleFactor=1.3,
                                            minNeighbors=3)
    for (x,y,w,h) in face_rects:    
        roi = roi[y:y+h,x:x+w]
        # blurred_roi = cv2.medianBlur(roi,9) # doesn't blur to anonymity
        # blur coordinates of roi
        blurred_roi = cv2.GaussianBlur(roi, (23, 23), 30)      
        face_img[y:y+h,x:x+w] = blurred_roi
        
    return face_img

if __name__ == "__main__":
    vid_path = "./video.mp4"
    cap = cv2.VideoCapture(vid_path)

    # check if the video is there
    if cap.isOpened()== False: 
        print("Error opening the video file.")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

    while cap.isOpened():        
        ret, frame = cap.read() 

        try:
            frame = detect_and_blur_face(frame)
        except Exception as e:
            print(f"{e}")
            frame = frame

        cv2.imshow('Video Face Blur', frame) 
        output.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):    
            break  

    cap.release()
    cv2.destroyAllWindows()
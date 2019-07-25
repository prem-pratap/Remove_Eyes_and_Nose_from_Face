#!/usr/bin/env python3
import cv2
import numpy as np
import dlib


cap=cv2.VideoCapture(0)
detector=dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")	
while cap.isOpened():
    status,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=detector(gray)
    for face in faces:
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        landmarks=predictor(gray,face)
        for i in range(0,68):
            if i == 15:
                x_eye=landmarks.part(i).x
                y_eye=landmarks.part(i).y
                #cv2.circle(frame,(x_eye,y_eye),3,(0,0,255),-1)
            if i ==39:
                nose_l_top_x=landmarks.part(i).x
                nose_l_top_y=landmarks.part(i).y
            
            if i ==35:
                nose_r_bot_x=landmarks.part(i).x
                nose_r_bot_y=landmarks.part(i).y

        #cv2.rectangle(frame,(nose_l_top_x,nose_l_top_y),(nose_r_bot_x,nose_r_bot_y),(0,255,0),2)
        eyes=frame[y1:y_eye,x1:x_eye]
        nose=frame[nose_l_top_y:nose_r_bot_y,nose_l_top_x:nose_r_bot_x]
    cv2.imshow("Detect",frame)
    cv2.imshow("Eyes",eyes)
    cv2.imshow("nose",nose)
    if cv2.waitKey(30) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
cap.release()

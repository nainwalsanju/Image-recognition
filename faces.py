import numpy as np 
import cv2
import pickle


face_cascade=cv2.CascadeClassifier('C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')        #Cascade classifier
eye_cascade=cv2.CascadeClassifier('C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

labels={"person_name":1}
with open("face-labels.pickle", 'rb') as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items() }

cap=cv2.VideoCapture(0)        ##capture live video from webcam


while True:                    ##if capturing from camera installed in laptop
    ret, frame=cap.read()      ##collect frame from the video 
    frame=cv2.flip(frame,1,0)   #mirror the frame
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)       #convert frame into grayscale
    faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5) 
    for(x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray=gray[y:y+h,x:x+w]     #(YCORD_START ,YCORD_END)    reigon of intrest for gray
        roi_color=frame[y:y+h,x:x+w]

        #reconize the face-deep learnedmodel project keras,tenserflow,pytorch,scikit,leran
        id_, conf=recognizer.predict(roi_gray)       #conf=confidence
        if conf>=45:
            print(id_)
            print(labels[id_])
            font=cv2.FONT_HERSHEY_SIMPLEX
            name=labels[id_]
            color=(255,255,255)
            stroke=1
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)

        img_item="user.png"
        cv2.imwrite(img_item,roi_color)
        #draw a rectangle
        color=(0,0,255)  #BGR 0-255
        stroke=2        #thickness of rectangle
        end_cord_x=x+w+10
        end_cord_y=y+h+10
        cv2.rectangle(frame,(x,y),(end_cord_x,end_cord_y),color,stroke)
        eyes=eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('frame', frame)   #imshow -  shows the collected frame   in frame window
    if cv2.waitKey(20) & 0xFF == ord('q'):          ## if  we presses q it quits
        break

cap.release()                   ##release the capture
cv2.destroyAllWindows()         ##destroy the window



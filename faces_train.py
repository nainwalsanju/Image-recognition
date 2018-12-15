import os
from PIL import Image  #PIL=python image library
import numpy as np
import cv2
import pickle


BASE_DIR=os.path.dirname(os.path.abspath(__file__))  #gives absolute path of file  faces_train.py
image_dir=os.path.join(BASE_DIR,"images")               #gives path of image directory
recognizer = cv2.face.LBPHFaceRecognizer_create()

face_cascade=cv2.CascadeClassifier('C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml')        #cascade classifier


current_id=0
label_ids={}
y_labels=[]
x_train=[]


for root,dirs,files in os.walk(image_dir):        #iterate through folders
    for file in files:          #iterate through files
        if file.endswith("png") or file.endswith("jpg"):   #
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).replace(" ","-").lower()      #gives name of directory in lower case of file   sanjay for sanjay images 1,2,3,4
            print(label,path)    #print path of single file and label   : label = actual directory
            
            if not label in label_ids:
                label_ids[label]=current_id
                current_id=current_id+1
            id_=label_ids[label]
            print(label_ids)
            

            #y_labels.append(label)  #some number
            #x_train.append(path)    #verify this image and convert into NUMPY array,GRAY
            
            pil_image=Image.open(path).convert("L")  #gives /an image at this path in grayscale
            size=(500,550)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array=np.array(pil_image,"uint8")  #convert pixel of image into numpy array uint8=type
            print(image_array) 
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.1,minNeighbors=5)
            for(x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
    
#print(y_labels)
#print(x_train)

with open("face-labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("face-trainner.yml")
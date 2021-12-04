import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
# Rpi
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
init_button_pin = 24
exit_button_pin = 25
GPIO.setup(init_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(exit_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#ReadNetfromCaffe
#Config path
protoPath = os.path.join(os.getcwd(),'face_detector','deploy.prototxt.txt')
#Net path
modelPath = os.path.join(os.getcwd(),'face_detector','res10_300x300_ssd_iter_140000.caffemodel')
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#CNN
embedder = cv2.dnn.readNetFromTorch(os.path.join(os.getcwd(),'openface.nn4.small2.v1.t7'))

def get_embedding(img):
    blob = img_blob(img)
    detector.setInput(blob)
    detections = detector.forward()
    h,w ,c= img.shape
    if len(detections)>0:
        #Assume only one face
        i  = np.argmax(detections[0, 0, :, 2])
        
        box = detections[0,0,i,3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")
        
        id_face = img[startY:endY, startX:endX]
        id_face_blur = cv2.GaussianBlur(id_face, (5,5), 1)
        sharp_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        id_face = cv2.filter2D(id_face_blur,-1,sharp_kernel)

        # plt.imshow(id_face)
        # plt.show()

        id_face2 = face_blob(id_face)
        # embedded = vgg_features.predict(id_face2)
        embedder.setInput(id_face2)
        embedded = embedder.forward()
        return embedded[0]
        
def img_blob(img):
    img = img.astype(np.float32)
    resized_img = cv2.resize(img,(300,300))
    # subtract image with RGB mean
    resized_img[:,:,0] -= 104.0
    resized_img[:,:,1] -= 177.0
    resized_img[:,:,2] -= 123.0

    resized_img = np.transpose(resized_img, [2, 0, 1])
    resized_img = np.array([resized_img])
    
    return resized_img

def face_blob(img):
    img = img*(1/255.0)
    swapRB = img[:,:,::-1]
    # resized_img = cv2.resize(swapRB,(224,224))
    # resized_img[:,:,0] += 93.5940 
    # resized_img[:,:,1] += 103.8827
    # resized_img[:,:,2] += 129.1863
    # resized_img = np.asarray((resized_img,))
    resized_img = cv2.resize(swapRB, (96,96))
    resized_img = np.transpose(resized_img, [2, 0, 1])
    resized_img = np.array([resized_img])
    return resized_img
def selfie():
    #Number of pictures taken
    NUMBER = 5
    embedded_selfie = []
    frames = []
    
    cap = cv2.VideoCapture(2)
    if cap.isOpened() == False:
        print("Can't connect to camera")
        return None
    
    #Is it capturing
    capturing = False
    #Time capturing started
    starttime = 0
    sec = 0
    while(cap.isOpened()):
        

        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame', frame)
            cv2.waitKey(1)
            if capturing and len(frames)<NUMBER:
                delta = time.time()-starttime
                if sec >= 3:
                    sec=0
                    starttime = starttime+3
                    print('Capturing')
                    frames.append(frame)
                    
                elif delta>sec:
                    print(str(3-sec)+'...')
                    sec += 1
                if len(frames) == 5:
                    break
            #Selfie on y key
            
            elif GPIO.input(init_button_pin)==0:
                capturing = True
                starttime=time.time()
                print('Y')
            #Exit on q key
            elif GPIO.input(exit_button_pin)==0:
                break
    cap.release()
    cv2.destroyAllWindows()
    if len(frames) > 0:
        for img in frames:
            embedded_selfie.append(get_embedding(img))
            
    return embedded_selfie

#Return the face that has the lowest euclidean distance
def compare_face(face_features,embedded_selfies,threshold=0.8):
    if len(face_features) == 0 or len(embedded_selfies) == 0:
        euclidean_distance = np.empty((0))
    names = list(face_features.keys())
    closeness = {name:1 for name in names}
    max = names[0]
    for name in names:
        euclidean_distances = []
        for embedded_img in embedded_selfies: #(1,128)
            euclidean_distances.append(np.linalg.norm(embedded_img-face_features[name]))
        euclidean_distance = np.mean(euclidean_distances)
        closeness[name] = euclidean_distance
        print("Comparing with {}, Closeness: {}".format(name,euclidean_distance))
        if euclidean_distance < closeness[max]:
            max = name
    if closeness[max] < threshold:
        return max
    else:
        return ""

#For preparing message to be sent
def items_message(items):
    items_str = str(items.to_dict())[1:-1]
    items_str = items_str.split('\'')
    items_str = ''.join(items_str)
    items_str = items_str.split(',')
    items_str = ''.join(items_str)
    items_str = items_str.split()
    if 'Juice:' in items_str:
        items_str.remove('Juice:')
    message = []
    for i in range(len(items_str)-1):
        if items_str[i] == 'Orange':
            items_str[i] = 'Orange_Juice:'
        items_str[i] = items_str[i].replace(items_str[i][0],items_str[i][0].upper())
        if i%2==0:
            message.append(''.join([items_str[i],items_str[i+1]]))
    print(message)
            
    message = ' '.join(message)
    return message
def user_message(user, adds, diffs):
    adds = ','.join(items_message(adds).split(' '))
    diffs =','.join(items_message(diffs).split(' '))
    message = user+' '+adds+' '+diffs
    return message
    
            

    

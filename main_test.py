import numpy as np
import time
import cv2
import os
import pandas as pd
import embedded
import pickle
import paho.mqtt.client as paho

# R Pi
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
#stop_button_pin =
init_button_pin = 24
exit_button_pin = 25
relay_pin = 18
#GPIO.setup(stop_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(relay_pin, GPIO.OUT)
GPIO.setup(init_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(exit_button_pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

#Call back function for mqtt client
def on_publish(client,userdata,mid):   #create function for callback
   print("data published mid=",mid, "\n")
   pass
def on_disconnect(client, userdata, rc):
   print("client disconnected ok")

#Initiate mqtt client connection to server
ITEMS_TOPIC = "ISE/mecha/items"
USER_TOPIC = "ISE/mecha/user"
broker="test.mosquitto.org"
port= 8080
print("connecting to broker ",broker,"on port ",port)
client= paho.Client("client-socks",transport='websockets')
client.on_publish = on_publish
client.on_disconnect = on_disconnect
client.connect(broker,port)           #establish connection


#Variable for face_recognition
REGISTER_LISTS = pd.read_csv('names.csv')['Name'].tolist()
starttime = 0
occupied = False
selfie_frames = []
embedded_selfies = []
NUM_SELFIES = 5
SELFIE_DURATION = 2.0
person_in = ""
capturing = False
exiting = False

#Checking for object count before and after the inventory is occupied

#Load face encoded features from the list of registered names
face_features = {}
for name in REGISTER_LISTS:
    print("Loading {}'s features".format(name))
    with open(os.path.join(name,"encoded"),'rb') as f:
        face_features[name] = pickle.load(f)

#Set minimum confidence
CONFIDENCE = 0.6
NMS_THRESHOLD =0.3

labelsPath = os.path.join('yolov4-tiny','obj.names')
LABELS = open(labelsPath).read().strip().split("\n")

# #Set series for counting objects (get only same object with at least 5 frames)
last_count = pd.Series(np.zeros(len(LABELS)))
last_count.index = LABELS
now_count = pd.Series(np.zeros(len(LABELS)))
now_count.index = LABELS
true_count = pd.Series(np.zeros(len(LABELS)))
true_count.index = LABELS
first = True
CONSECUTIVE_FRAME = 5
count = 0


# #Initialize a list of colors to represent each class
# np.random.seed(42)
# COLORS = np.random.randint(0, 255, size=(len(LABELS),3), dtype='uint8')

# #Derive the paths to the YOLO weights and model config
# weightsPath = os.path.join('yolov4-tiny','custom-yolov4-tiny-detector_best.weights')
# configPath = os.path.join('yolov4-tiny','custom-yolov4-tiny-detector.cfg')

# #Initialize the DarkNet
# net = cv2.dnn.readNetFromDarknet(configPath,weightsPath)
# ln = net.getLayerNames()
# ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# print(ln)

#Read from webcam for object detection
#vid = cv2.VideoCapture(0)

#Read from another webcam for face recognition
face_vid = cv2.VideoCapture(0)


while True:

    #ret, image = vid.read()
    face_ret, face_image = face_vid.read()
    if not (face_ret):
        break

    # (H, W) = image.shape[:2]

    # #Preprocess image
    # blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), swapRB=True, crop=False)
    # net.setInput(blob)
    # start = time.time()
    # layerOutputs = net.forward(ln)
    # end = time.time()

    #Show timing information on YOLO
    # print("YOLO took {:.6f} seconds".format(end-start))

    boxes = []
    confidences = []
    classIDs = []

    # for output in layerOutputs:
    #     for detection in output:
    #         scores = detection[5:]
    #         classID = np.argmax(scores)
    #         confidence = scores[classID]

    #         if confidence > CONFIDENCE:

    #             box = detection[0:4] * np.array([W, H, W, H])
    #             (centerX, centerY, width, height) = box.astype('int')

    #             x = int(centerX-(width/2))
    #             y = int(centerY - (height/2))

    #             boxes.append([x, y, int(width), int(height)])
    #             confidences.append(float(confidence))
    #             classIDs.append(classID)
    # #Applying non-maxima suppression
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, NMS_THRESHOLD)
    # now_count = pd.Series(np.zeros(len(LABELS)))
    # now_count.index = LABELS
    # if len(idxs) > 0:
    #     for i in idxs.flatten():

    #         (x,y) = (boxes[i][0], boxes[i][1])
    #         (w,h) = (boxes[i][2], boxes[i][3])

    #         color = [int(c) for c in COLORS[classIDs[i]]]
    #         cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
    #         text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    #         cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    #         now_count.loc[LABELS[classIDs[i]]] += 1
    # #Check for same object in consecutive FRAMES
    # if not first:
    #     if now_count.equals(last_count):
    #         count += 1
    #     else:
    #         count = 0
    #     if count >= CONSECUTIVE_FRAME:
    #         true_count = now_count.copy()
    # else:
    #     first = False
    # last_count = now_count.copy()
    # print(true_count[true_count>0])

    #If capturing is True, capture 5 selfies within 2 seconds, then fed it in compare_face
    if capturing and count>=5:
        now = time.time()
        delta = now-starttime
        if delta>=SELFIE_DURATION/NUM_SELFIES:
            print('Capturing')
            selfie_frames.append(face_image)
            startime = time.time()
        if len(selfie_frames) == NUM_SELFIES:
            capturing=False
            embedded_selfies = [embedded.get_embedding(selfie_frame) for selfie_frame in selfie_frames]
            result = embedded.compare_face(face_features,embedded_selfies)
            if result == "":
                print("Unrecognized face")
            else:
                #-----Insert magnetic lock code here!------#
                print("Welcome! {}".format(result))
                GPIO.output(relay_pin, 1)
                #------------------------------------------#
                occupied = True
                before_count = true_count.copy()
                person_in = result
                print(before_count[before_count>0])
            capturing = False

    elif capturing and count<5:
        print('Waiting for camera to settle...')
        starttime = time.time()

    if count==5 and not capturing:
        message = embedded.items_messages(true_count)
        client.publish(ITEMS_TOPIC,message)
        time.sleep(0.1)


    #Clean up if the person exits
    if exiting and count>=5:
        person_in = ""
        after_count = true_count.copy()
        # print(after_count.loc['clock'])
        # print(before_count.loc['clock'])
        diff = before_count-after_count
        diff = diff[diff>0]
        add = after_count-before_count
        add = add[add>0]
        message = embedded.user_message(person_in,add,diff)
        client.publish(USER_TOPIC,message)
        exiting = False
        occupied = False
        time.sleep(0.5)
    elif exiting and count<5:
        print('Waiting for camera to settle...')




   # cv2.imshow('Object',image)
    cv2.imshow('Face',face_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    elif GPIO.input(init_button_pin)==0 and not capturing and not occupied:
        capturing = True
        starttime=time.time()
        print('Performing face recognition...')
    elif GPIO.input(exit_button_pin)==0 and occupied:
        exiting = True
        time.sleep(10)
        GPIO.output(relay_pin, 0)
        print('{} exiting...'.format(person_in))
#vid.release()
face_vid.release()
cv2.destroyAllWindows()
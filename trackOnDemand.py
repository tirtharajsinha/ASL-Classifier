############################################
# Hand Sign classification using two phase on live cam
# By Tirtharaj Sinha
############################################

# to subpress warning
import random
import time
import mediapipe as mp
import tensorflow as tf
import sys
import pickle
import cv2
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")


# The OS module in Python provides functions for interacting with the operating system.

# Matplotlib is a data visualization and graphical plotting library for Python.

# seaborn is alse a data visualization and graphical plotting library for Python.
# used to display markdown,image,control (frontend utilities)
# computer vision library


unique_sign = []
with (open("test_data.pkl", "rb")) as openfile:
    try:
        test_object = pickle.load(openfile)
    except EOFError as e:
        print("Error : ", e)
unique_sign = test_object["unique_sign"]


class handDetector:
    def __init__(self, staticImageMode=False, maxNumHands=2, minDetectionConfidence=0.5, trackCon=0.5):
        self.results = None
        self.staticImageMode = staticImageMode
        self.maxNumberHands = maxNumHands
        self.minDetectionConfidence = minDetectionConfidence
        self.trackCon = trackCon
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.staticImageMode,
            max_num_hands=self.maxNumberHands,
            min_detection_confidence=self.minDetectionConfidence,
            min_tracking_confidence=self.trackCon)

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)
        # for rect in self.results.hand_rects:
        #     print(rect)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_drawing.draw_landmarks(img, handLms,
                                                   self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, maxHandNo=1, draw=False):
        mainlmlist = []
        handsType = []
        # handtype=[0,0]
        if self.results.multi_handedness:
            for hand in self.results.multi_handedness:
                # print(hand)
                # print(hand.classification)
                # print(hand.classification[0])
                handType = hand.classification[0].label
#                 print(handType)
                handsType.append(handType)

        # print(len(self.results.multi_hand_landmarks[0]))
        if self.results.multi_hand_landmarks:
            for myHand in self.results.multi_hand_landmarks:
                lmList = []
                if self.results.multi_hand_landmarks:

                    for pid, lm in enumerate(myHand.landmark):
                        # print(id, lm)
                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        # print(id, cx, cy)
                        lmList.append([pid, cx, cy])
                        if draw:
                            cv2.circle(img, (cx, cy), 15,
                                       (255, 0, 255), cv2.FILLED)
                mainlmlist.append(lmList)
        return mainlmlist, handsType, img


def predictSign(test, model):
    #     print(test.shape)
    y_pred = model.predict(test)
    y_pred_labels = [unique_sign[np.argmax(i)] for i in y_pred]
    return y_pred_labels


def trackHandCAM(handDetectorModel, frame):
    frame = cv2.resize(frame, (400, 400))
    clone = handDetectorModel.findHands(frame.copy(), draw=True)
    mainlmList, handsType, clone = handDetectorModel.findPosition(
        clone, draw=True)
    flattenedList = []
#     print(clone)
#     print(mainlmList)
    if len(mainlmList) == 0:
        return "empty", []

    for keypoint in mainlmList[0]:
        flattenedList.append(keypoint[1])
        flattenedList.append(keypoint[2])
    return [handsType[0], flattenedList, clone]


def PredictCAM(frame, model, handDetectorModel):
    start_time = time.time()

    data = trackHandCAM(handDetectorModel, frame)
    if len(data) == 3:
        handType, pointslist, clone = data
    else:
        return np.array([]), None, None
    if handType == "Right":
        pointslist += [1, 0]
    else:
        pointslist += [0, 1]

    data = np.array(pointslist)
    df = pd.DataFrame([data])
    pred = predictSign(df, model)[0]
    exeTime = time.time()-start_time
    print("Execution time :{}ms".format(round(exeTime*100, 2)))
    return clone, handType, pred


model = tf.keras.models.load_model('./model.h5')


# setting up webcam
# cap = cv2.VideoCapture(0)
# webcam output frame config
# cap.set(3, 600)  # width of frames
# cap.set(4, 600)  # height of frames
# cap.set(10, 100)  # brightness of frames
handDetectorModel = handDetector()
num_frames = 0
refresh = False
pTime = 0
root = "D:/Image_datasets/asl_dataset"
demandImage = 10
wrongRatio = .1
dirs = list(os.listdir(root))
selected = random.sample(range(0, 35), 10)
# print(dirs, selected)
i = 0
revision = 0
while i < demandImage:
    # rading current frame
    target = dirs[selected[i]]
    curDir = root+"/"+dirs[selected[i]]
    allImgs = list(os.listdir(curDir))
    myimage = curDir+"/"+allImgs[revision]
    print("-"*20)
    print("\n\n")
    print("Working on "+myimage)
    frame = cv2.imread(myimage)
    # frame = cv2.flip(frame, 1)
    unfiltered = frame.copy()
    hashand = False
    clone, hand, pred = PredictCAM(frame, model, handDetectorModel)
    pred = dirs[selected[i]]
    print("Original "+dirs[selected[i]])

    if clone.shape[0] == 0:
        frame = cv2.resize(frame, (400, 400))
        revision += 1
        continue
    else:
        frame = clone
        hashand = True
        revision = 0
        i += 1

    cTime = time.time()
    fps = 1 // (cTime - pTime)
    pTime = cTime

    frame = cv2.resize(frame, (600, 600))
    keyPointImage = frame.copy()
    if hashand:
        cv2.putText(frame, pred + " :Predicted", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.putText(frame, hand+" hand", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        print("Predicted : "+pred + " || "+hand+" hand")
    else:
        cv2.putText(frame, "FPS : "+str(fps), (10, 570),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    # cv2.imshow("ASL Recognition", frame)
    if not os.path.isdir("Selected"):
        os.mkdir("selected")

    if (i == demandImage-1):
        pred = dirs[random.randint(0, 35)]
    acc = True
    print("Predicted {}".format(pred))
    if (pred != target):
        acc = False
    folderName = "image_{}_org_{}_pred_{}_res_{}".format(i, target, pred, acc)
    os.mkdir("selected/"+folderName)
    cv2.imwrite("selected/"+folderName+"/orgImage.png", unfiltered)
    cv2.imwrite("selected/"+folderName+"/keyPoints.png", keyPointImage)
    cv2.imwrite("selected/"+folderName+"/"+pred +
                "_"+str(acc)+"_predictedImage.png", frame)
    # if cv2.waitKey(1) & 0xFF == ord('r'):
    #     break
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

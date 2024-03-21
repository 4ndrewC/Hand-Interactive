import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLUT import *
import math
import cv2
import mediapipe as mp
import threading
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

flag = 1
speed = [0, 0, 0]
iterator = 0

"""
Locking/snapping implementation starts here
"""

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = 200

dummy = np.array([
    [1.0, -1.0, -1.0, 1.0],
    [1.0, 1.0, -1.0, 1.0],
    [-1.0, 1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.0, 1.0]
], dtype=np.float64)

def dist(point1, point2):
  x1, y1 = point1[0], point1[1]
  x2, y2 = point2[0], point2[1]
  squared_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2
  distance = math.sqrt(squared_distance)
  return distance

def lock(marks, corners):
    global iterator
    threshold = 0.15
    # if iterator%50==0:
    # print(landmarks)
    # print(len(marks))
    # match each finger tip landmark to a corner (4, 12, 16, 20)?
    tips = []
    for i in range(len(marks)):
        if i!=0 and i%4==0:
            tips.append(marks[i])
    # print(len(tips))
    """
    locked -> four corners (tl, tr, bl, br)
    """
    locked = [0, 0, 0, 0]
    tipmap = [-1, -1, -1, -1, -1]
    
    for i in range(len(tipmap)):
        idx = 0
        mindist = 1e5
        found = 0
        for j in range(len(corners)):
            # print(tuple(marks[0:2]))
            # print(corners[j])
                  
            d = dist(tuple(tips[i][0:2]), corners[j])
            # print(d)
            if d<=threshold and d<mindist:
                # print('made it here')
                mindist = d
                idx = j
                found = 1
        if found==1:
            locked[idx]+=1
            tipmap[i] = idx
        # print(idx)
    if iterator%10==0: print(tipmap)

    #update marks array to snap fingertips into 4 corners
    for i in range(len(tipmap)):
        if tipmap[i]==-1: continue
        idx = (i+1)*4
        idx2 = tipmap[i]
        marks[idx][0] = corners[idx2][0]
        marks[idx][1] = corners[idx2][1]
    
def capture_and_display_hand():
    global flag
    global speed
    global iterator

    cap = cv2.VideoCapture(0)
    pos = [0, 0, 0]
    prev = [0, 0, 0]
    
    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            iterator+=1

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            marks=[]
            if results.multi_hand_landmarks:
                
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    for landmark in hand_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        temp=[x,y,z]
                        marks.append(temp)

                    wristpos = hand_landmarks.landmark[0]
                    # print(hand_landmarks.landmark[0])
                    pos[0] = wristpos.x
                    pos[1] = wristpos.y
                    pos[2] = wristpos.z
                    
                    if flag==0:
                        for i in range(len(pos)):
                            speed[i] = pos[i]-prev[i]

                        for i in range(len(pos)):
                            prev[i] = pos[i]
                        flag = 1
                       

                    
                    # if it%50==0: print(speed)
            
            corners=[]

            center_x = frame_width // 2
            center_y = frame_height // 2    

            top_left_x = center_x - size // 2
            top_left_y = center_y - size // 2
            top_right_x = top_left_x + size
            top_right_y = top_left_y
            bottom_left_x = top_left_x
            bottom_left_y = top_left_y + size
            bottom_right_x = top_right_x
            bottom_right_y = bottom_left_y

            

            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), thickness=2)
            

            """omfg i have to fucking normalize again"""
            top_left_x/=frame_width
            top_left_y/=frame_height
            top_right_x/=frame_width
            top_right_y/=frame_height
            bottom_left_x/=frame_width
            bottom_left_y/=frame_height
            bottom_right_x/=frame_width
            bottom_right_y/=frame_height

            corners = [(top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_left_x, bottom_left_y), (bottom_right_x, bottom_right_y)]

            if len(marks)>=21:
                lock(marks, corners)

            # -----------------------------------------------------
            if len(marks)>=21:
                idx = 0
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    
                    for landmark in hand_landmarks.landmark:
                        landmark.x = marks[idx][0]
                        landmark.y = marks[idx][1]
                        landmark.z = marks[idx][2]
                        idx+=1
                # res = convert(marks)
                if results.multi_hand_landmarks:
                    # print("made it in here")
                    for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        # print("made it without")
                        for landmark in hand_landmarks.landmark:
                            # print(results.multi_hand_landmarks[0])
                            """
                            only display post-transformed landmarks
                            """
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                            )
            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# def print_stat():
#     global flag
#     # while True:
#     # print("running here")
#     if flag==1: 
#         if speed[0]>0:
#             print("right")
#         elif speed[0]<0:
#             print("left")
#         else:
#             print("not moving")
#         flag = 0

# thread1 = threading.Thread(target=print_stat)
# hand_thread = threading.Thread(target=capture_and_display_hand)


# hand_thread.start()
# thread1.start()

# while True:
#     thread1.join()
#     hand_thread.join()
capture_and_display_hand()
    

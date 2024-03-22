import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import glMatrixMode, glOrtho
import math
import cv2
import mediapipe as mp
import threading
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)

# -----------------------------------------------
"""Shared mem variables"""
flag = 1
iterator = 0
iterator2 = 0
outputang = 0

rotation_m = np.array([
    [0, 0],
    [0, 0]
], dtype=np.float64)

prevstate = np.array([
    [-1.0, 1.0],
    [1.0, 1.0],
    [-1.0, -1.0],
    [1.0, -1.0]    
], dtype=np.float64)

# ------------------------------------------------


"""OpenGL Part"""

verticesT = np.array([
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0]    
], dtype=np.float64)

#scale that boi down
verticesT = np.multiply(verticesT, 0.25)


translation_matrix = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float64)


edges = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 3)
)


#constants
rotation_speed = 1
friction = 0.0001
rotation_friction = 0.01
rotation = [0, 0]
angle = 0
translation = [0, 0]

keyon = [0,0]

def rotate_cube(key):
    global rotation
    if key == pygame.K_LEFT:
        keyon[0] = 1
        rotation[0] += rotation_speed
    elif key == pygame.K_RIGHT:
        keyon[0] = 1
        rotation[0] -= rotation_speed
    elif key == pygame.K_UP:
        keyon[1]=1
        rotation[1] += rotation_speed
    elif key == pygame.K_DOWN:
        keyon[1]=1
        rotation[1] -= rotation_speed
    elif key == pygame.K_SPACE:
        rotation[0] = 0
        rotation[1] = 0

def translate_cube(key):
    global translation
    speed = 0.01
    if key == pygame.K_a:
        translation[0] -= speed
    elif key == pygame.K_d:
        translation[0] += speed
    elif key == pygame.K_w:
        translation[1] += speed
    elif key == pygame.K_s:
        translation[1] -= speed
    print("key pressed")


def rotate(a):
    global rotation_m
    theta = np.radians(a)
    # reset(zRotation, 'z')
    rotation_m[0,0]=np.cos(theta)
    rotation_m[0,1]=np.sin(theta)
    rotation_m[1,0]=-np.sin(theta)
    rotation_m[1,1]=np.cos(theta)
    

def Cube():
    glBegin(GL_LINES)
    global iterator
    global flag
    global prevstate

    # rotate(angle)
    if flag==0:
        translated_vertices = np.dot(verticesT, translation_matrix.T)

        result = []
        for arr in translated_vertices:
            result.append(tuple(arr)[0:2])

        #-------------------------------------------------------------------
        # print(rotation_m)
        # print(result)
        
        final = np.dot(np.array(result), rotation_m)
        #-------------------------------------------------------------------

        if iterator2%50==0: 
            print("final")
            print(final)
        
        for i in range(len(final)):
            for j in range(len(final[i])):
                prevstate[i][j] = final[i][j]

        for edge in edges:
            for vertex in edge:
                glVertex2fv(final[vertex])
        
        flag = 1
    else:
        for edge in edges:
            for vertex in edge:
                glVertex2fv(prevstate[vertex])
    glEnd()


def print_coords():
    print("Transformed Vertices: ")
    print("x:", translation_matrix[0,2], "y:", translation_matrix[1,2])
    # print("y angle", angle[1])

def glrun():
    global iterator2
    global angle
    global flag
    global rotation_m
    # global translation_matrix
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glOrtho(-1, 1, -1, 1, 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5)
    

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                rotate_cube(event.key)
                translate_cube(event.key)
        
        # Actions:
        # if flag==0:
        #     # print("MADE IT HERE!!!!!!")
        #     translation_matrix[0,2] += translation[0]
        #     translation_matrix[1,2] += translation[1]
        #     # if iterator%50==0: 
        #         # print("after")
        #         # print(translation_matrix)

        #     #add friction to rotation
        #     if rotation[0]>0: rotation[0] = max(rotation[0]-rotation_friction, 0)
        #     if rotation[0]<0: rotation[0] = min(rotation[0]+rotation_friction, 0)
            
        #     #translation friction
        #     if translation[0]>0: translation[0] = max(translation[0]-friction, 0)
        #     if translation[1]>0: translation[1] = max(translation[1]-friction, 0)
        #     if translation[0]<0: translation[0] = min(translation[0]+friction, 0)
        #     if translation[1]<0: translation[1] = min(translation[1]+friction, 0)

        #     # angle += rotation[0]
        #     # angle += outputang
        #     angle%=360

        #     angle = max(angle-rotation_friction, 0)
        #     if iterator2%50==0: print('rotation out:', rotation_m)
        #     flag = 1

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity() #unit-ize everything
        # if flag==0:
        Cube()
            # flag = 1
        # if iterator2%50 == 0: 
            # print_coords()
            # print("angle:", angle)
        iterator2+=1
        pygame.display.flip()
        pygame.time.wait(10)

# ----------------------------------------------------------------
"""Hands Part"""

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = 150

rex = 12

def convert(point):
    return (point[0]/2 + 1/2, point[1]/2 + 1/2)

def dist(point1, point2):
  x1, y1 = point1[0], point1[1]
  x2, y2 = point2[0], point2[1]
  squared_distance = (x2 - x1) ** 2 + (y2 - y1) ** 2
  distance = math.sqrt(squared_distance)
  return distance

def lock(marks, corners):
    global iterator

    threshold = 0.15

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
            locked[idx]=1
            tipmap[i] = idx
        # print(idx)
    # if iterator%10==0: print(tipmap)


    #update marks array to snap fingertips into 4 corners
    for i in range(len(tipmap)):
        if tipmap[i]==-1: continue
        idx = (i+1)*4
        idx2 = tipmap[i]
        marks[idx][0] = corners[idx2][0]
        marks[idx][1] = corners[idx2][1]
    
    return sum(locked)

"""
for rotations, get initial angle of 0-9
for as long as 'locked' only contains >=3, then the change in angle of 0-9 
should be tracked and emulated in a rotation matrix   
"""

def calculate_angle(point1, point2, point3, point4):
    vector1 = (point3[0] - point4[0], point3[1] - point4[1])
    vector2 = (point1[0] - point2[0], point1[1] - point2[1])
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    angle_radians = math.atan2(cross_product, dot_product)
    # angle_degrees = math.degrees(angle_radians)

    return angle_radians

def rotation_matrix(angle):
    cos_theta = math.cos(angle)
    sin_theta = math.sin(angle)

    rotation_matrix = [
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ]

    return np.array(rotation_matrix)

def handrun():
    global flag
    global speed
    global iterator
    global rotation_m
    global prevstate

    cap = cv2.VideoCapture(0)

    pos = [0, 0, 0]
    prev = [0, 0, 0]
    locked = []
    p = -1

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

    """preset to spawn the box"""
    top_left_x/=frame_width
    top_left_y/=frame_height
    top_right_x/=frame_width
    top_right_y/=frame_height
    bottom_left_x/=frame_width
    bottom_left_y/=frame_height
    bottom_right_x/=frame_width
    bottom_right_y/=frame_height

    started = 0
    
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
            
            cur = 0
            if results.multi_hand_landmarks:
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    for landmark in hand_landmarks.landmark:
                        x, y, z = landmark.x, landmark.y, landmark.z
                        temp=[x,y,z]
                        marks.append(temp)
            
            corners=[]
            
            
            # ----------------------------------------------------------
            """track corners dynamically"""
            if started:
                top_left_x, top_left_y = convert(prevstate[0])
                top_right_x, top_right_y = convert(prevstate[1])
                bottom_left_x, bottom_left_y = convert(prevstate[2])
                bottom_right_x, bottom_right_y = convert(prevstate[3])

            # cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), thickness=2)

            corners = [(top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_left_x, bottom_left_y), (bottom_right_x, bottom_right_y)]


            # -----------------------------------------------------------
            """update and show updated landmarks"""
            if len(marks)>=21:
                cur = lock(marks, corners)
                idx = 0
                for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    for landmark in hand_landmarks.landmark:
                        landmark.x = marks[idx][0]
                        landmark.y = marks[idx][1]
                        landmark.z = marks[idx][2]
                        idx+=1

                if results.multi_hand_landmarks:
                    for num, hand_landmarks in enumerate(results.multi_hand_landmarks):
                        for landmark in hand_landmarks.landmark:
                            """only display post-transformed landmarks"""
                            mp_drawing.draw_landmarks(
                                image,
                                hand_landmarks,
                                mp_hands.HAND_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                            )

            # ----------------------------------------------------------
            """movement requirements: locked array must not contain <3"""
            move = 0
            if flag==1:
                locked.append(cur)
                if cur<3:
                    p = 4
                if len(locked)>=5:
                    locked.pop(0)
                if p>=0:
                    # if iterator%10==0: print('not moving')
                    move = 0
                else: 
                    # if iterator%10==0: print('moving')
                    move = 1
                    started = 1
                

                p = max(p-1, -1)
                
                """get 2 points for each line"""
                if len(marks)>=21 and move==1:
                    wrist = tuple(marks[0][0:2])
                    midjoint = tuple(marks[9][0:2])
                    botmid = (0.5, 1)
                    topmid = (0.5, 0)

                    ang = calculate_angle(wrist, midjoint, botmid, topmid)
                    rot = rotation_matrix(ang)
                    rotation_m = rot
                    outputang = ang
                    if iterator%50==0:
                        print('rotation in:', rotation_m)
                        # print(rot)
                        

                # switch to gl thread
                flag = 0

                



            cv2.imshow('Hand Tracking', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


glthread = threading.Thread(target=glrun)
handthread = threading.Thread(target=handrun)


handthread.start()
glthread.start()

while True:
    glthread.join()
    handthread.join()
    

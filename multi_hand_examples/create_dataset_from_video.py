import cv2
import sys, os
import mediapipe as mp
import numpy as np
import modules.holistic_module as hm
from modules.utils import createDirectory
import json
import time

# 저장할 파일 이름
save_file_name = "train"

# 시퀀스의 길이(30 -> 10)
seq_length = 10

actions = ['yes', 'no', 'like', 'heart']

dataset = dict()

for i in range(len(actions)):
    dataset[i] = []

# MediaPipe holistic model
detector = hm.HolisticDetector(min_detection_confidence=0.1)

videoFolderPath = "dataset/output_video"
videoTestList = os.listdir(videoFolderPath)

testTargetList =[]

created_time = int(time.time())

for videoPath in videoTestList:
    actionVideoPath = f'{videoFolderPath}/{videoPath}'
    actionVideoList = os.listdir(actionVideoPath)
    for actionVideo in actionVideoList:
        fullVideoPath = f'{actionVideoPath}/{actionVideo}'
        testTargetList.append(fullVideoPath)

print("---------- Start Video List ----------")
testTargetList = sorted(testTargetList, key=lambda x:x[x.find("/", 9)+1], reverse=True)
print(testTargetList)
print("----------  End Video List  ----------\n")

for target in testTargetList:

    data = []
    idx = actions.index(target[target.find("/", 10)+1:target.find("/", 21)])

    print("Now Streaming :", target)
    cap = cv2.VideoCapture(target)

    # 열렸는지 확인
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    # 웹캠의 속성 값을 받아오기
    # 정수 형태로 변환하기 위해 round
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

    if fps != 0:
        delay = round(1000/fps)
    else:
        delay = round(1000/30)

    # 프레임을 받아와서 저장하기
    while True:
        ret, img = cap.read()

        if not ret:
            break

        img = detector.findHolistic(img, draw=True)
        _, left_hand_lmList = detector.findLefthandLandmark(img)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        
        # if len(left_hand_lmList) != 0 and len(right_hand_lmList) != 0:
        if left_hand_lmList is not None and right_hand_lmList is not None:
            
            joint = np.zeros((42, 2))
            # 왼손 랜드마크 리스트
            for j, lm in enumerate(left_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]
            
            # 오른손 랜드마크 리스트
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j+21] = [lm.x, lm.y]

            # Compute angles between joints
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19] + [i+21 for i in [0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19]], :2] # Parent joint
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] + [i+21 for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]], :2] # Child joint
            v = v2 - v1 
            # Normalize v
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            # Get angle using arcos of dot product
            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18] + [i+20 for i in [0,1,2,4,5,6,8,9,10,12,13,14,16,17,18]] ,:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19] + [i+20 for i in [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]],:])) 

            angle = np.degrees(angle) # Convert radian to degree

            angle_label = np.array([angle], dtype=np.float32)

            # 정답 라벨링
            angle_label = np.append(angle_label, idx)

            # 위치 종속성을 가지는 데이터 저장
            # d = np.concatenate([joint.flatten(), angle_label])
        
            # 정규화 벡터를 활용한 위치 종속성 제거
            d = np.concatenate([v.flatten(), angle_label.flatten()])
        
            data.append(d)

        

        # draw box
        cv2.rectangle(img, (0,0), (w, 30), (245, 117, 16), -1)

        # draw text target name
        cv2.putText(img, target
                    , (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('img', img)
        cv2.waitKey(delay)

        # esc를 누르면 강제 종료
        if cv2.waitKey(delay) == 27: 
            break

    print("\n---------- Finish Video Streaming ----------")

    data = np.array(data)

    # Create sequence data
    for seq in range(len(data) - seq_length):
        dataset[idx].append(data[seq:seq + seq_length])    

for i in range(len(actions)):
    save_data = np.array(dataset[i])
    np.save(os.path.join('dataset', f'seq_{actions[i]}_{created_time}'), save_data)


print("\n---------- Finish Save Dataset ----------")





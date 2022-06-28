import sys
# sys.path.append('pingpong')
# from pingpong.pingpongthread import PingPongThread
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import modules.holistic_module as hm
from tensorflow.keras.models import load_model
import math
from modules.utils import Coordinate_Normalization, Vector_Normalization

actions = ['yes', 'no', 'like', 'heart']
seq_length = 10

# MediaPipe holistic model
detector = hm.HolisticDetector(min_detection_confidence=0.1)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/normalization_classifier.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    img = detector.findHolistic(img, draw=True)
    _, left_hand_lmList = detector.findLefthandLandmark(img)
    _, right_hand_lmList = detector.findRighthandLandmark(img)

    if left_hand_lmList is not None and right_hand_lmList is not None:
        joint = np.zeros((42, 2))
        # 왼손 랜드마크 리스트
        for j, lm in enumerate(left_hand_lmList.landmark):
            joint[j] = [lm.x, lm.y]
        
        # 오른손 랜드마크 리스트
        for j, lm in enumerate(right_hand_lmList.landmark):
            joint[j+21] = [lm.x, lm.y]

        # 좌표 정규화
        full_scale = Coordinate_Normalization(joint)

        # 벡터 정규화
        vector, angle_label = Vector_Normalization(joint)

        # 위치 종속성을 가지는 데이터 저장
        # d = np.concatenate([joint.flatten(), angle_label])
    
        # 정규화 벡터를 활용한 위치 종속성 제거
        # d = np.concatenate([vector.flatten(), angle_label.flatten()])

        # 정규화 좌표를 활용한 위치 종속성 제거 
        d = np.concatenate([full_scale, angle_label.flatten()])
        

        seq.append(d)

        if len(seq) < seq_length:
            continue

        # Test model on random input data.
        # input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        
        # 시퀀스 데이터와 넘파이화
        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        input_data = np.array(input_data, dtype=np.float32)

        # tflite 모델을 활용한 예측
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        y_pred = interpreter.get_tensor(output_details[0]['index'])
        i_pred = int(np.argmax(y_pred[0]))
        conf = y_pred[0][i_pred]

        if conf < 0.9:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

            if last_action != this_action:
                last_action = this_action
            

        cv2.putText(img, f'{this_action.upper()}', org=(int(left_hand_lmList.landmark[0].x * img.shape[1]), int(left_hand_lmList.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)


    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == 27:
        break


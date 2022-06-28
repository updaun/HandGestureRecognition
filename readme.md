## Hand Gesture Recognition
---
- 비대면 화상 회의 중 양손 제스처를 활용하여 의사를 표현하는 방법에 대한 아이디어
- 표현 
    - 긍정(⭕표시)
    - 부정(❌표시)
    - 좋아요(엄지척👍👍)
    - 하트(손하트🤞🤞)
- 왜 양손🙌 인가요?
    - 한 손으로 의사표현시 발생할 수 있는 문제 방지, 얼굴을 만지거나 다른 행동에서 인식될 수 있는 확률을 줄임
    - 양 손을 활용한 공손함 표현

![image](https://user-images.githubusercontent.com/82289435/176146728-c599914e-2a87-4a13-a492-687fb85019cd.png)

### 데이터 수집
---
- 학습 및 테스트 데이터 촬영

![image](https://user-images.githubusercontent.com/82289435/176146966-b63bf315-0bc6-4727-9a49-fc67d77be8c9.png)

### 과적합 방지 대책(Dropout, L2 규제)
---
- 첫 번째 게시글
- https://dacon.io/codeshare/4956
- 자세한 설명은 링크를 참고해주세요.

![image](https://user-images.githubusercontent.com/82289435/176147477-6aa649cf-1f5d-45fd-9b70-79964a6cccb2.png)

### 정규화 적용 전 좌표계
---
![image](https://user-images.githubusercontent.com/82289435/176146519-3fa81a5b-3091-474a-8fd3-e3e3dcbdb3bf.png)

### 정규화 적용 후 좌표계
--- 
- 두 번째 게시글
- https://dacon.io/codeshare/5006
- 자세한 설명은 링크를 참고해주세요.

![image](https://user-images.githubusercontent.com/82289435/176146601-2f3c84ff-3598-49d2-ac39-7341a9c3cb44.png)

### 위치종속성 제거 결과물
---
![image](https://user-images.githubusercontent.com/82289435/176146676-e065a76d-0b40-4318-ab9e-bf57a83eeca3.png)

</br>

### Pipeline
- create_dataset
    - mediapipe를 활용하여 웹캠이미지로부터 손동작 좌표를 추출하여 데이터셋을 직접 생성합니다.
- train_hand_gesture
    - tensorflow를 활용하여 LSTM 모델을 학습시킵니다.
    - 학습 metric을 시각화하여 학습 상태를 확인합니다.
    - tensorflow lite 모델로 변환합니다.
- test_hand_gesture
    - keras h5 model을 동작 테스트합니다.
- test_model_tflite
    - tensorflow lite 모델을 테스트합니다.

</br>

### Setting Develop Enviorments
- conda env 생성
```
conda create -n mp python=3.8
```
- conda env activate
```
conda actiavate mp
```
- python lib install(requirements.txt가 있는 디렉토리로 이동)
```
pip install -r requirements.txt
```

</br>

### Examples Execution
1. examples/create_dataset.py : 데이터 촬영 및 생성
2. examples/train_hand_gesture.ipynb : 모델 학습
2. examples/test_hand_gesture.py : LSTM 모델 테스트
2. examples/test_model_tflite.py : LSTM TFlite 모델 테스트
 
</br>

### Directory Structure
```
.
├─dataset(git 미포함)
│      gesture_1.npy
│      gesture_2.npy
│      gesture_3.npy
│      gesture_4.npy
│
├─examples
│      create_dataset.py
│      test_hand_gesture.py
│      test_model_tflite.py
│      train_hand_gesture.ipynb
│
└─models(git 미포함)
        hand_gesture_classifier.h5
        hand_gesture_classifier.tflite
```

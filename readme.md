## Hand Gesture Recognition

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

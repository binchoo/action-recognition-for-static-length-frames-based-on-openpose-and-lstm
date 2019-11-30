# Action Recognition for Static Length Frames Based on OpenPose and LSTM

## Ideas

1. 정적 길이의 영상에 등장하는 사람의 행동을 레이블링 하려고 한다. (실시간 영상 스트림에 대한 처리가 아님)
2. 행동은 연속된 움직임으로부터 판단하는 것이 직관적
3. 그러므로 영상 프레임을 시계열 분석한다 (LSTM)
4. End to End로 프레임에서 특성을 추출하도록 할까?
5. 그러지 말고, Open Pose를 사용하면 사람의 관절의 변화로 행동을 인지할 수 있지 않을까?
6. 이미 있는 모델들을 재사용할 수 있을 것 같다!



## Data

- 다양한 각도에서 촬영한 영상
- 이 영상에 오픈포즈를 적용하여 관절점을 예측
- 원본 데이터는 공개할 수 없음.

- 데이터로부터 추출한 관절점 자료는 pkl 파일로 저장되어 있다.
  - `./data_pkl`
    - `ours_data_five.pkl` : FHD 화질에 대하여 5가지 행동이 레이블링.
    - `image_data_five.pkl` : 512*512 화질에 대하여 5가지 행동이 레이블링.
    - `image_data_binary.pkl` : 위 자료를 fall detection에 특화시킨 자료.
- 관절점 데이터는 약 2500개가 존재. 좌우반전 어규멘테이션을 적용함.

## 문서 요약

1. VideoAnalyst
   1. 영상에 대해 포즈를 예측하여 리스트에 저장한다. 
   2. 이 때, 하나의 영상에 대해 (frames, 17, 2) 형상의 NDArray가 만들어지며 
   3. 타겟 폴더에 100개의 영상이 있었다면 이런 NDArray 100개가 리스트를 구성하게 된다.
2. ImageAnalyst
   1. 영상을 이미 전처리하여 프레임들을 이미지 파일로 준비해 놓았다면, 이미지들에 대해 포즈를 예측하고 리스트에 저장한다.
   2. 당연하게도 이미지들의 이름이 특정 형식을 지켜야 분석을 진행할 수 있다.
      1. `image<id><action><sequence>.[jpg|png|otherimageformat]`
      2. id는 몇 번째 영상에서 나온 프레임인지를 명시한다.
      3. action는 해당 영상이 어떤 행동을 위한 영상이었는지 명시한다.
      4. sequence는 이 이미지가 몇 번째 프레임인지를 명시한다.
3. NBatchSkeleton
   1. 1번과 2번 문서에서 제작된 관절점 리스트로 학습을 진행한다.
   2. 학습하는 모델은 `fall|walk|sleep|sit|bend`를 구분하는 5중 분류기이다.
4. BinaryBatchSkeleton
   1. 3번 문서에서 제작된 5중 분류기의 output layer를 변형한다
   2. 학습하는 모델은 `fall|no_fall`으을 구분하는 바이너리 분류기이다.
   3. 그러므로 문서 속에, 5중 레이블을 바이너리 레이블로 처리하는 과정이 포함되어 있다.


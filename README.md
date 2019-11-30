# Action Recognition for Static Length Frames Based on OpenPose and LSTM

## Ideas

1. 정적 길이의 영상에 등장하는 사람의 행동을 레이블링 하려고 한다. (실시간 영상 스트림에 대한 처리가 아님)
2. 행동은 연속된 움직임으로부터 판단하는 것이 직관적
3. 그러므로 영상 프레임을 시계열 분석한다 (LSTM)
4. End to End로 프레임에서 특성을 추출하도록 할까?
5. 혹시, Open Pose를 사용하면 사람의 관절의 변화로 행동을 인지할 수 있지 않을까?
6. 이미 있는 모델들을 재사용하자!



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
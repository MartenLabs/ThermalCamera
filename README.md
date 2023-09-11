
# MLX90640을 사용한 Object Detection 

<br/>
<br/>
<br/>
<br/>

# 1차 센서 테스트 및 후처리

<br/>
<br/>

## 1. 개요
라즈베리파이 4B에 MLX90640를 사용하여 Object Detection을 수행하는 프로젝트 수행 중 후처리를 통한 딥러닝 학습 가능 여부 판단 을 위한 테스트 프로젝트

## 2. 대표 라이브러리
- **adafruit_mlx90640**: mlx90640을 사용하기 위한 라이브러리

- **tensorflow**: object detection을 위한 라이브러리

- **cv2**: 영상 녹화 및 노이즈 필터링을 위한 라이브러리

- **scipy**: 가우시안 필터 및 픽셀 보간을 위한 라이브러리

## 3. 하드웨어
- **RaspberryPi4B 8GB**

## 4. 알고리즘

1. MLX90640 센서에서 데이터 프레임 get

2. 인체 온도 범위 외의 픽셀을 마스킹
3. 데이터 정규화: 데이터 값을 0-255 범위로 변환
4. 노이즈 제거: NLmeans 필터를 사용하여 이미지에서 노이즈 제거
5. 이미지를 가우시안 필터를 사용해 부드러운 쉐입으로 이미지 생성
6. 데이터에서 유효한 값을 추출
7. 유효한 값을 기반으로 빈 픽셀을 보간하여 완전한 이미지를 생성
8. 인체 온도 범위 외의 픽셀을 마스킹 
9.  이미지를 0-1 범위로 정규화 (전처리)
10. 모델에 이미지 입력
11. 녹화를 위해 이미지를 3 채널로 확장
12. 예측 결과에 따라 텍스트 표시

<br/>

```python
# MLX90640 센서에서 데이터 프레임 get
self.mlx.getFrame(self.frame)

# 인체 온도 범위 외의 픽셀을 마스킹
self.frame[(self.frame <= 23) | (self.frame > 40)] = 0

# 데이터 shape 변환
data_array_raw = np.fliplr(np.reshape(self.frame, self.mlx_shape))

# 데이터 정규화: 데이터 값을 0-255 범위로 변환
data_array_raw_normalized_uint8 = ((data_array_raw - np.min(data_array_raw)) / (np.max(data_array_raw) - np.min(data_array_raw)) * 255).astype(np.uint8)

# 노이즈 제거: NLmeans 필터를 사용하여 이미지에서 노이즈 제거
img_array_uint8_denoised = cv2.fastNlMeansDenoising(data_array_raw_normalized_uint8, None, h=23, searchWindowSize=33)

# 이미지를 다시 실수 값 범위로 변환
img_float64_denoised = (img_array_uint8_denoised.astype(float) / 255) * (np.max(data_array_raw) - np.min(data_array_raw)) + np.min(data_array_raw)

# 이미지를 가우시안 필터를 사용해 부드러운 쉐입으로 이미지 생성
data_array_smoothed = ndimage.gaussian_filter(img_float64_denoised, sigma=0.3)

# 데이터에서 유효한 값을 추출
x_idx_valid_values, y_idx_valid_values = np.where(data_array_smoothed > 0)
valid_values = data_array_smoothed[x_idx_valid_values, y_idx_valid_values]

# 보간: 유효한 값을 기반으로 빈 픽셀을 보간하여 완전한 이미지를 생성
x_idx_all, y_idx_all = np.indices(data_array_smoothed.shape)
interpolated_data_array = interpolate.griddata((x_idx_valid_values, y_idx_valid_values), valid_values, (x_idx_all, y_idx_all), 'cubic')

# 인체 온도 범위 외의 픽셀을 마스킹 (온도 가정)
interpolated_data_array[(interpolated_data_array <= 25) | (interpolated_data_array > 40)] = 0

# 이미지를 0-1 범위로 정규화 (전처리)
image = (interpolated_data_array - np.min(interpolated_data_array)) / (np.max(interpolated_data_array) - np.min(interpolated_data_array))

# 모델에 이미지 입력
predictions = self.model.predict(np.expand_dims(image, axis=0))  # 이미지를 배치 차원을 추가하여 모델에 전달
        
# 이미지를 0-255 범위로 변환
img_array = (interpolated_data_array - np.min(interpolated_data_array)) / (np.max(interpolated_data_array) - np.min(interpolated_data_array)) * 255
        
# 녹화를 위해 이미지를 3 채널로 확장
expanded_img_array = np.expand_dims(img_array, axis=2)
expanded_img_array = np.tile(expanded_img_array, (1, 1, 3))

# 데이터 타입을 uint8로 변환
expanded_img_array = expanded_img_array.astype(np.uint8)
        
# 예측 결과에 따라 텍스트 표시
text = 'Person Detected' if predictions > 0.5 else 'No Person'
```

<br/>

###  1.5M 기준 후처리 비교
| 원본 | 후처리 | 
|:---:|:---:|
|<img src="Data/Readme/비교_original.png" alt="drawing" width="300"/>| <img src="Data/Readme/비교_filters.png" alt="drawing" width="300"/>|

<br/>
<br/>
<br/>

| 1M | 1.5M | 2M | 3M |
| :---: | :---: | :---: | :---: |
| <img src="Data/Readme/1M.png" alt="drawing" width="300"/> | <img src="Data/Readme/비교_filters.png" alt="drawing" width="300"/>| <img src="Data/Readme/2M.png" alt="drawing" width="300"/> | <img src="Data/Readme/3M.png" alt="drawing" width="300"/>|


## 5. 사용한 데이터셋

with human data (24 x 32)

<img src="Model/dataset/train/with_human/2023-09-11%2005-33-46.mp4_132.png" alt="drawing" width="400"/>

<br/>
<br/>
<br/>

without human data (24 x 32)

<img src="Model/dataset/train/without_human/2023-09-11 05-35-40.mp4_213.png" alt="drawing" width="400"/>


## 6. 사용한 학습 모델

### CNN Model 아키텍쳐 정의 

``` python
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 1)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

### Model Training

``` bash
85/85 [==============================] - 14s 106ms/step - loss: 0.1039 - accuracy: 0.9670 - val_loss: 0.0079 - val_accuracy: 0.9985
Epoch 2/5
85/85 [==============================] - 1s 16ms/step - loss: 0.0090 - accuracy: 0.9974 - val_loss: 0.0026 - val_accuracy: 0.9985
Epoch 3/5
85/85 [==============================] - 1s 16ms/step - loss: 0.0043 - accuracy: 0.9981 - val_loss: 0.0062 - val_accuracy: 0.9985
Epoch 4/5
85/85 [==============================] - 1s 16ms/step - loss: 0.0187 - accuracy: 0.9944 - val_loss: 0.0086 - val_accuracy: 0.9956
Epoch 5/5
85/85 [==============================] - 1s 16ms/step - loss: 0.0078 - accuracy: 0.9978 - val_loss: 0.0027 - val_accuracy: 1.0000
```

### 모델 평가

``` bash
22/22 [==============================] - 0s 9ms/step - loss: 0.0027 - accuracy: 1.0000
Test accuracy: 1.0
```

## 7. 1차 학습 결과

![](Result/v1_CNN/1차_CNN.gif)


<br/>
<br/>
<br/>
<br/>
<br/>

# 2차 SSD를 사용한 Object Detection

<br/>
<br/>

## 1. 개요
라즈베리파이 4B에 MLX90640를 사용하여 SSD Object Detection을 수행하는 프로젝트 수행

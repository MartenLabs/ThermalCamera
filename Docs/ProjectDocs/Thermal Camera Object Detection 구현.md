

# 0. npz 데이터 셋에서 데이터 분리 


### 1. 기존 npz파일 로드 및 데이터 재배치
``` python
import numpy as np

dataset1_origin = np.load('npz/dataset1_origin.npz', allow_pickle=True)
dataset1_horizon = np.load('npz/dataset1_horizon.npz', allow_pickle=True)
dataset1_vertical = np.load('npz/dataset1_vertical.npz', allow_pickle=True)
dataset1_vh = np.load('npz/dataset1_vh.npz', allow_pickle=True)
dataset2_origin = np.load('npz/dataset2_origin.npz', allow_pickle=True)
dataset2_horizon = np.load('npz/dataset2_horizon.npz', allow_pickle=True)
dataset2_vertical = np.load('npz/dataset2_vertical.npz', allow_pickle=True)
dataset2_vh = np.load('npz/dataset2_vh.npz', allow_pickle=True)

d1o_origin_images, d1o_target_images, d1o_number_labels, d1o_coordinates = dataset1_origin['images'], dataset1_origin['filters'], dataset1_origin['numbers'],  dataset1_origin['coordinates']
d1h_origin_images, d1h_target_images, d1h_number_labels, d1h_coordinates = dataset1_horizon['images'], dataset1_horizon['filters'], dataset1_horizon['numbers'],  dataset1_horizon['coordinates']
d1v_origin_images, d1v_target_images, d1v_number_labels, d1v_coordinates = dataset1_vertical['images'], dataset1_vertical['filters'], dataset1_vertical['numbers'],  dataset1_vertical['coordinates']
d1vh_origin_images, d1vh_target_images, d1vh_number_labels, d1vh_coordinates = dataset1_vh['images'], dataset1_vh['filters'], dataset1_vh['numbers'],  dataset1_vh['coordinates']
d2o_origin_images, d2o_target_images, d2o_number_labels, d2o_coordinates = dataset2_origin['images'], dataset2_origin['filters'], dataset2_origin['numbers'],  dataset2_origin['coordinates']
d2h_origin_images, d2h_target_images, d2h_number_labels, d2h_coordinates = dataset2_horizon['images'], dataset2_horizon['filters'], dataset2_horizon['numbers'],  dataset2_horizon['coordinates']
d2v_origin_images, d2v_target_images, d2v_number_labels, d2v_coordinates = dataset2_vertical['images'], dataset2_vertical['filters'], dataset2_vertical['numbers'],  dataset2_vertical['coordinates']
d2vh_origin_images, d2vh_target_images, d2vh_number_labels, d2vh_coordinates = dataset2_vh['images'], dataset2_vh['filters'], dataset2_vh['numbers'],  dataset2_vh['coordinates']

origin_images = np.concatenate([d1o_origin_images, d2o_origin_images, d1h_origin_images, d2h_origin_images, d1v_origin_images, d2v_origin_images, d1vh_origin_images, d2vh_origin_images], axis = 0)
target_images = np.concatenate([d1o_target_images, d2o_target_images, d1h_target_images, d2h_target_images, d1v_target_images, d2v_target_images, d1vh_target_images, d2vh_target_images], axis = 0)
numbers_labels = np.concatenate([d1o_number_labels, d2o_number_labels, d1h_number_labels, d2h_number_labels, d1v_number_labels, d2v_number_labels, d1vh_number_labels, d2vh_number_labels], axis = 0)
coordinates = np.concatenate([d1o_coordinates, d2o_coordinates, d1h_coordinates, d2h_coordinates, d1v_coordinates, d2v_coordinates, d1vh_coordinates, d2vh_coordinates], axis = 0)
```



### 2. 데이터 확인
``` python
origin_images = origin_images.reshape(origin_images.shape[0], 24, 32, 1)

copy_coord = coordinates[13000]

new_coords = []
changed_coords = []
for i in range(copy_coord.shape[0]):
    copy_coord[i]
    x_coords = copy_coord[i][0::2] / origin_images.shape[2]
    y_coords = copy_coord[i][1::2] / origin_images.shape[1]

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    new_coords.append([xmin, ymin, xmax, ymax])

boxes = np.array(new_coords)
print(boxes)

"""
[[21 9 31 16] 
 [ 0 19 4 24] 
 [ 0 12 7 19] 
 [ 0 0 0 0]]
"""
```

``` python
plt.axis('off')
plt.imshow(origin_images[13000])
ax = plt.gca()
boxes = tf.stack(
	[
	 boxes[:, 0] * origin_images.shape[2],
	 boxes[:, 1] * origin_images.shape[1],
	 boxes[:, 2] * origin_images.shape[2],
	 boxes[:, 3] * origin_images.shape[1]
	], axis = -1
)
for box in boxes:
	xmin, ymin = box[:2]
	w, h = box[2:] - box[:2]
	patch = plt.Rectangle(
		[xmin, ymin], w, h, fill = False, edgecolor = [1, 0, 0], linewidth = 2
	)
	ax.add_patch(patch)
plt.show()
```
![](Data/imgs/ObjectDetection/1.png)



$$\text{ratio} = {\text{width} \over \text{height}}$$
$$\text{width} = \text{height} \times \text{ratio}$$
$$\text{area} = \text{width} \times \text{height} => \text{area} = \text{height}^2 \times \text{ratio}$$
$$\text{anchor height} = \sqrt{\frac{\text{area}}{\text{ratio}}}$$
$$\text{anchor width} = \frac{\text{area}}{\text{anchor height}}$$

### 3. 데이터 전처리
``` python
bboxes = []
for coords in coordinates:
    new_coords = []  # 각 coords 집합에 대해 new_coords를 초기화
    for i in range(coords.shape[0]):
        x_coords = coords[i][0::2] / origin_images.shape[2]
        y_coords = coords[i][1::2] / origin_images.shape[1]

        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        new_coords.append([xmin, ymin, xmax, ymax])
    
    bboxes.append(new_coords)  # 변환된 좌표 추가

bboxes = np.array(bboxes)
print(coordinates.shape)
print(bboxes.shape)


"""
(13276, 4, 8) 
(13276, 4, 4)
"""
```



### 4. 데이터 확인 
``` python
for img, boxes in zip(origin_images[-10:], bboxes[-10:]):
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(img)  # 여기를 수정
    ax = plt.gca()
    boxes = tf.stack(
    	[
    	 boxes[:, 0] * origin_images.shape[2],
    	 boxes[:, 1] * origin_images.shape[1],
    	 boxes[:, 2] * origin_images.shape[2],
    	 boxes[:, 3] * origin_images.shape[1]
    	], axis = -1
    )
    
    for box in boxes:
        xmin, ymin = box[:2]
        w, h = box[2:] - box[:2]
        patch = plt.Rectangle(
            [xmin, ymin], w, h, fill=False, edgecolor=[1, 0, 0], linewidth=2
        )
        ax.add_patch(patch)
    plt.show()
```


### 5. 데이터 저장
``` python 
np.savez('npz/ObjectDetection.npz', images=origin_images, numbers=numbers_labels, coordinates = bboxes)
```


---



# 1. 데이터 전처리

### 1. 데이터 로드 및 데이터 추가
``` python
import numpy as np

datasets = np.load('npz/ObjectDetection.npz', allow_pickle=True)
images, numbers, bboxes = datasets['images'], datasets['numbers'], datasets['bboxes']

cls = []
for i in numbers:
	label = 1 if i != 0 else 0
	cls.append(label)
cls = np.array(cls)

dataset = {
	'images' : images,
	'numbers' : numbers,
	'bboxes' : bboxes,
	'cls' : cls
}

print(dataset['images'].shape)
print(dataset['numbers'].shape)
print(dataset['bboxes'].shape)
print(dataset['cls'].shape)


"""
(13276, 24, 32, 1) 
(13276,) 
(13276, 4, 4) 
(13276,)
"""
```


### 2. 데이터 확인
``` python
import matplotlib.pyplot as plt
import tensorflow as tf
images = dataset['images']
numbers =dataset['numbers']
bboxes = dataset['bboxes']
cls = dataset['cls']

boxes = bboxes[9000]
plt.figure(figsize = (8, 8))
plt.axis('off')
plt.imshow(images[9000])
ax = plt.gca()
boxes = tf.stack([
	boxes[:, 0] * images.shape[2],
	boxes[:, 1] * images.shape[1],
	boxes[:, 2] * images.shape[2],
	boxes[:, 3] * images.shape[1]], axis = -1
)
    

for box in boxes:
	xmin, ymin = box[:2]
	w, h = box[2:] - box[:2]
	patch = plt.Rectangle(
		[xmin, ymin], w, h, fill = False, edgecolor = [1, 0, 0], linewidth = 2
	)
	ax.add_patch(patch)
plt.show()
```
![](Data/imgs/ObjectDetection/2.png)



### 3. 데이터 변환
``` python
import os
import random
import tensorflow as tf

IMG_SIZE_WIDTH = images.shape[2]
IMG_SIZE_HEIGHT = images.shape[1]
N_DATA = images.shape[0]
N_TRAIN = int(images.shape[0] * 0.7)
N_VAL = images.shape[0] - N_TRAIN
LOG_DIR = 'ObjectDetectionLog'

cur_dir = os.getcwd()
tfr_dir = os.path.join(cur_dir, 'tfrecord/ObjectDetection')
os.makedirs(tfr_dir, exist_ok=True)

print("IMG_SIZE_WIDTH:  ", IMG_SIZE_WIDTH)
print("IMG_SIZE_HEIGHT: ", IMG_SIZE_HEIGHT)
print("N_DATA:          ", N_DATA)
print("N_TRAIN:         ", N_TRAIN)
print("N_VAL:           ", N_VAL)
```


---


# Data Augmentation

### 1. 이미지 크기 변경
``` python
def resize_and_pad_image(image, ratio = 3, stride = 8):
    image_shape = tf.cast(tf.shape(image)[:2], dtype = tf.float32)
    
    image_shape = ratio * image_shape
    image = tf.image.resize(image, tf.cast(image_shape, dtype = tf.int32))

    padded_image_shape = tf.cast(
        tf.math.ceil(image_shape / stride) * stride, dtype = tf.int32
    )

    image = tf.image.pad_to_bounding_box(
        image, 0, 0, padded_image_shape[0], padded_image_shape[1]
    )

    return image, image_shape, ratio
```



### 2. 데이터 전처리 
``` python
def preprocess_data(sample):
    image = sample["images"]
    bbox = sample["bboxes"]
    class_id = tf.cast(sample["cls"], dtype = tf.int32)

    image, image_shape, _ = resize_and_pad_image(image)
    bbox = tf.stack([
        bbox[:, 0] * image_shape[1],
        bbox[:, 1] * image_shape[0],
        bbox[:, 2] * image_shape[1],
        bbox[:, 3] * image_shape[0]],
        axis = -1
    )

    return image, bbox, class_id
```



### 3. stride 확인 
``` python
import numpy as np
import matplotlib.pyplot as plt

img = images[9000]
num = numbers[9000]
bb = bboxes[9000]
cs = cls[9000]
sample = {
    'images' : img,
    'numbers' : num,
    'bboxes' : bb,
    'cls' : cs
}

img, _, _ = preprocess_data(sample)
print(img.shape) # (72, 96, 1)

anchor_img = np.zeros((*img.shape[:2], 3), dtype=np.uint8)
print(anchor_img.shape)

strides = [2, 4, 8]
colors = {
    2: [0, 0, 255],  # 파란색
    4: [255, 0, 0],  # 빨간색
    8:[255, 255, 0]  # 노란색
}

for stride in strides:
    color = colors[stride]
    for y in range(0, anchor_img.shape[0], stride):
        for x in range(0, anchor_img.shape[1], stride):
            anchor_img[y, x, :] = color

plt.imshow(img, alpha=1)  
plt.imshow(anchor_img, alpha=0.5) 
plt.axis('off')
plt.show()
```
![](Data/imgs/ObjectDetection/4.png)


### 4. Area, Ratio, Height, Width 확인
``` python
import math
ratios = [0.5, 1.0, 2.0]
areas = [16, 64, 256]

for area in areas:
    for ratio in ratios:
        H = math.sqrt(area / ratio)
        W = area / H
        print(f'Area: {area}')
        print(f'Ratio: {ratio}')
        print(f'Height: {H}\nWidth: {W}\n')
    

"""
Area: 16
Ratio: 0.5
Height: 5.656854249492381
Width: 2.82842712474619

Area: 16
Ratio: 1.0
Height: 4.0
Width: 4.0

Area: 16
Ratio: 2.0
Height: 2.8284271247461903
Width: 5.65685424949238

Area: 64
Ratio: 0.5
Height: 11.313708498984761
Width: 5.65685424949238

Area: 64
Ratio: 1.0
Height: 8.0
Width: 8.0

Area: 64
Ratio: 2.0
Height: 5.656854249492381
Width: 11.31370849898476

Area: 256
Ratio: 0.5
Height: 22.627416997969522
Width: 11.31370849898476

Area: 256
Ratio: 1.0
Height: 16.0
Width: 16.0

Area: 256
Ratio: 2.0
Height: 11.313708498984761
Width: 22.62741699796952
"""
```



### 5. Center 확인 
``` python


```


### . AnchorBox 생성
``` python
class AnchorBox:
    def __init__(self):
        self.aspect_ratios = [0.5, 1.0, 2.0]
        self.scales = [2** x for x in [0, 1/3, 2/3]]
        self._num_anchors = len(self.aspect_ratios) * len(self.scales)
        self._strides = [2 ** i for i in range(1, 4)]
        self._areas = [x ** 2 for x in [4, 8, 16]]
        self._anchor_dims = self._compute_dims()

    def _compute_dims(self):
        anchor_dims_all = []

        for area in self._areas:
            anchor_dims = []

            for ratio in self.aspect_ratios: 
                anchor_height = tf.math.squrt(area / ratio)
                anchor_width = area / anchor_height

                dims = tf.reshape(
                    tf.stack([anchor_width, anchor_height], axis = -1)
                )

                for scale in self.scales: 
                    anchor_dims.append(scale * dims) 
        
            anchor_dims_all.append(tf.stack(anchor_dims), axis = -2)
        return anchor_dims_all 
    
    def _get_anchors(self, feature_height, feature_widht, level):
        rx = tf.range(feature_widht, dtype = tf.float32) + 0.5
        ry = tf.range(feature_height, dtype = tf.foat32) + 0.5

        centers = tf.stack(tf.meshgrid(rx, ry), axis = -1) * self._strides[level -3]

        centers = tf.expand_dims(centers, axis = -2)
        centers = tf.tile(centers, [1, 1, self._num_anchors, 1])
```


# 데이터셋 설계 

### training 
- input 
	- image
	- anchor boxes
	- target anchor box
	- target anchor box move loc
	- labels


- output  
	- target anchor boxes loc
	- labels






이 `LabelEncoder` 클래스는 객체 검출 모델을 위한 레이블 인코딩을 수행합니다. 주요 기능은 앵커 박스와 실제 그라운드 트루스 박스(ground truth boxes) 간의 매칭을 수행하고, 이에 따라 레이블을 생성하는 것입니다. 

### 1. 앵커 박스와 Ground Truth 박스 매칭

#### a. IoU 계산
$[ \text{IoU} = \text{compute\_iou}(\text{anchor\_boxes}, \text{gt\_boxes}) ]$

#### b. 최대 IoU 및 매칭 인덱스
$[ \text{max\_iou} = \text{tf.reduce\_max}(\text{IoU}, \text{axis}=1) ]$
$[ \text{matched\_gt\_idx} = \text{tf.argmax}(\text{IoU}, \text{axis}=1) ]$

#### c. 마스크 계산
- **Positive Mask**: 
$[ \text{positive\_mask} = \text{tf.greater\_equal}(\text{max\_iou}, \text{match\_iou}) ]$
- **Negative Mask**: 
  $[ \text{negative\_mask} = \text{tf.less}(\text{max\_iou}, \text{ignore\_iou}) ]$
- **Ignore Mask**: 
  $[ \text{ignore\_mask} = \text{tf.logical\_not}(\text{tf.logical\_or}(\text{positive\_mask}, \text{negative\_mask})) ]$

### 2. 박스 타겟 계산

$[ \text{box\_target} = \frac{(\text{matched\_gt\_boxes}[:,:2] - \text{anchor\_boxes}[:,:2])}{\text{anchor\_boxes}[:,2:]} \oplus \text{tf.math.log}(\frac{\text{matched\_gt\_boxes}[:,2:]}{\text{anchor\_boxes}[:,2:]}) ]$
$[ \text{box\_target} = \frac{\text{box target}}{\text{self.box variance}} ]$

$\oplus$ = concatenation

### 3. 클래스 타겟 및 최종 레이블 계산

#### a. 클래스 타겟
- **클래스 타겟**:
  $[ \text{cls\_target} = \text{tf.where}(\text{tf.not\_equal}(\text{positive\_mask}, 1.0), -1.0, \text{matched\_gt\_cls\_ids}) ]$
  $[ \text{cls\_target} = \text{tf.where}(\text{tf.equal}(\text{ignore\_mask}, 1.0), -2.0, \text{cls\_target}) ]$

#### b. 최종 레이블
$[ \text{label} = \text{tf.concat}([\text{box\_target}, \text{tf.expand\_dims}(\text{cls\_target}, \text{axis}=-1)], \text{axis}=-1) ]$

### 4. 배치 처리

각 이미지에 대해 위 과정을 반복하여 배치 레이블을 생성합니다.

이 클래스와 메서드는 TensorFlow를 사용하여 객체 검출 모델의 학습을 위한 레이블을 준비하는 데 필요한 계산과정을 수행합니다. 여기서 사용된 TensorFlow 함수들(tf.reduce_max, tf.argmax, tf.where 등)은 TensorFlow의 계산 그래프를 통해 효율적으로 실행됩니다.





### CustomNetBoxLoss 수식

$$L(y, \hat{y}) = \sum_{i} \begin{cases} 
  0.5 \times (y_i - \hat{y}_i)^2 & \text{if } |y_i - \hat{y}_i| < \delta \\
  |y_i - \hat{y}_i| - 0.5 & \text{otherwise}
\end{cases}$$


- $y$는 실제 값(y_true)
- $\hat{y}$는 예측 값(y_pred)
- δ는 손실 계산에서 사용되는 임계값(self._delta)
- $|y_i - \hat{y}_i|$는 예측 값과 실제 값의 절대 차이(absolute_difference)
- $(y_i - \hat{y}_i)^2$는 제곱 차이(squared_difference)





$$L(y, \hat{y}) = - \sum_{i} \alpha_i \times (1 - p_t)^{\gamma} \times \text{CE}(y_i, \hat{y}_i)$$
- $y$는 실제 값(y_true)
- $\hat{y}$​는 예측 값(y_pred)
- $\alpha$ 는 양성 예시에 대한 가중치(self._alpha)
- $\gamma$ 는 모델이 잘못 분류된 예시에 얼마나 집중할지 조절하는 파라미터(self._gamma)
- $p_t$ ​는 모델의 예측 확률(probs)
- $CE$ 는 크로스 엔트로피 손실(cross_entropy)






$$L(y, \hat{y}) = L_{cls}(y, \hat{y}) + L_{box}(y, \hat{y})$$
- $L_{cls}​ \text{는 클래스 분류 손실(ClassificationLoss)}$
- $L_{box} \text{​는 박스 손실(BoxLoss)}$
- $y$ 는 실제 값(y_true)
- $\hat{y}$ ​는 예측 값(y_pred)





$$L_{cls}(y, \hat{y}) = \frac{\sum \text{ClassificationLoss}(y_{cls}, \hat{y}_{cls})}{N}$$

$$L_{box}(y, \hat{y}) = \frac{\sum \text{BoxLoss}(y_{box}, \hat{y}_{box})}{N}$$


- $y_{cls}$ ​와 $\hat{y}_{cls}$ ​는 각각 실제 클래스 레이블과 예측된 클래스 레이블
- $y_{box}$​와 $\hat{y}_{box}$​는 각각 실제 박스 레이블과 예측된 박스 레이블
- $N$은 손실을 정규화하기 위한 양성 예시의 총 수
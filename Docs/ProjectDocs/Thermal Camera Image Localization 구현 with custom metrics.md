

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


### 2. 데이터 저장
``` python 
# 사람이 한사람인 것들만 필터링
one_indices = np.where(numbers_labels == 1)[0]

one_origin_images = origin_images[one_indices]
one_number_labels = numbers_labels[one_indices]
one_coordinates = coordinates[one_indices][:, 0, :]

# (8, ) -> xmin, ymin, xmax, ymax 로 재배치
new_coordinates = []
for idx in one_coordinates:
    x_coords = idx[0::2]
    y_coords = idx[1::2]

    xmin, xmax = min(x_coords), max(x_coords)
    ymin, ymax = min(y_coords), max(y_coords)
    new_coordinates.append([xmin, ymin, xmax, ymax])

new_coordinates = np.array(new_coordinates)

np.savez('npz/localization.npz', images=one_origin_images, numbers=one_number_labels, coordinates = new_coordinates)

one_origin_images.shape, one_number_labels.shape, one_coordinates.shape
```


### 3. 데이터 확인 
``` python
import numpy as np
datasets = np.load('npz/localization.npz', allow_pickle=True)
images, numbers, coordinates = datasets['images'], datasets['numbers'], datasets['coordinates']

print(images.max(), images.min())
print(images.shape, numbers.shape, coordinates.shape)

print(coordinates[2500])


import matplotlib.pyplot as plt

def draw_bounding_box_from_vertices(image, vertices):
    xmin, xmax = vertices[0::2]
    ymin, ymax = vertices[1::2]
    
    rect_x = xmin
    rect_y = ymin
    rect_w = xmax - xmin
    rect_h = ymax - ymin

    rect = plt.Rectangle((rect_x, rect_y), rect_w, rect_h, fill = False, color = 'red')
    plt.axes().add_patch(rect)
    plt.imshow(image)
    plt.show()


draw_bounding_box_from_vertices(images[700], coordinates[700])	
```

---



# 1. npz to tfr


### 1. 데이터 로드 및 확인
``` python
import numpy as np

datasets = np.load('npz/localization.npz', allow_pickle=True)
images, numbers, coordinates = datasets['images'], datasets['numbers'], datasets['coordinates']

print(images.shape, numbers.shape, coordinates.shape)
print(images.max(), images.min())
print(coordinates[2500])

"""
(2588, 24, 32) (2588,) (2588, 4) 
255 0 
[ 8 10 20 22]
"""
```

``` python
import matplotlib.pyplot as plt

def draw_bounding_box_from_vertices(image, vertices):
	xmin, xmax = vertices[0::2]
	ymin, ymax = vertices[1::2]

	rect_x = xmin
	rect_y = ymin
	rect_w = xmax - xmin
	rect_h = ymax - ymin

	rect = plt.Rectangle((rect_x, rect_y), rect_w, rect_h, fill = False, color = 'red')

	plt.axes().add_patch(rect)
	plt.imshow(image)
	plt.show()

draw_bounding_box_from_vertices(images[700], coordinates[700])
```
![](Data/imgs/ImageLocalization/1.png)


### 2. tfrecord 파일 만들기 
``` python
import os
import random
import tensorflow as tf

IMG_SIZE_WIDTH = images.shape[2]
IMG_SIZE_HEIGHT = images.shape[1]
N_DATA = images.shape[0]
N_TRAIN = int(images.shape[0] * 0.7)
N_VAL = images.shape[0] - N_TRAIN
LOG_DIR = 'LocalizationLog'

print("IMG_SIZE_WIDTH:  ", IMG_SIZE_WIDTH)
print("IMG_SIZE_HEIGHT: ", IMG_SIZE_HEIGHT)
print("N_DATA:          ", N_DATA)
print("N_TRAIN:         ", N_TRAIN)
print("N_VAL:           ", N_VAL)

shuffle_list = list(range(N_DATA))
random.shuffle(shuffle_list)

train_idx_list = shuffle_list[:N_TRAIN]
val_idx_list = shuffle_list[N_TRAIN:]

cur_dir = os.getcwd()
tfr_dir = os.path.join(cur_dir, 'tfrecord/localization')
os.makedirs(tfr_dir, exist_ok = True)

tfr_train_dir = os.path.join(tfr_dir, 'loc_train.tfr')
tfr_val_dir = os.path.join(tfr_dir, 'loc_val.tfr')

writer_train = tf.io.TFRecordWriter(tfr_train_dir)
writer_val = tf.io.TFRecordWriter(tfr_val_dir)

'''
IMG_SIZE_WIDTH:  32 
IMG_SIZE_HEIGHT: 24 
N_DATA:          2588 
N_TRAIN:         1811 
N_VAL:           777
'''
```

``` python
def _bytes_feature(value):
	if isinstance(value, type(tf.constant(0))):
		value = value.numpy()
	return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
	return tf.train.Feature(float_list = tf.train.FloatList(value = [value]))

def _int64_feature(value):
	return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
```

``` python
for idx in train_idx_list: # val_idx_list도 동일하게 수행
	bbox = coordinates[idx]
	xmin = bbox[0]
	ymin = bbox[1]
	xmax = bbox[2]
	ymax = bbox[3]

	xc = (xmin + xmax) / 2
	yc = (ymin + ymax) / 2

	x = xc / IMG_SIZE_WIDTH
	y = yc / IMG_SIZE_HEIGHT

	w = (xmax - xmin) / IMG_SIZE_WIDTH
	h = (ymax - ymin) / IMG_SIZE_HEIGHT

	image = images[idx]
	bimage = image.tobytes()

	example = tf.train.Example(features = tf.train.Features(feature = {
		'image': _bytes_feature(bimage),
		'x': _float_feature(x),
		'y': _float_feature(y),
		'w': _float_feature(w),
		'h': _float_feature(h)
	}))

	  writer_train.write(example.SerializeToString())
writer_train.close()
```

---




# 2. Image Localization 모델 작성 


### 1. tfr 데이터 로드
``` python
AUTOTUNE = tf.data.AUTOTUNE

RES_HEIGHT = 24
RES_WIDTH = 32
N_EPOCHS = 100
N_BATCH = 7
LR = 0.0005

def _parse_function(tfrecord_serialized):
	features = {
		'image': tf.io.FixedLenFeature([], tf.string),
		'x': tf.io.FixedLenFeature([], tf.float32),
		'y': tf.io.FixedLenFeature([], tf.float32),
		'w': tf.io.FixedLenFeature([], tf.float32),
		'h': tf.io.FixedLenFeature([], tf.float32)
	}

	parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

	image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
	image = tf.reshape(image, [RES_HEIGHT, RES_WIDTH, 1])
	image = tf.cast(image, tf.float32) / 255.

	x = tf.cast(parsed_features['x'], tf.float32)
	y = tf.cast(parsed_features['y'], tf.float32)
	w = tf.cast(parsed_features['w'], tf.float32)
	h = tf.cast(parsed_features['h'], tf.float32)
	gt = tf.stack([x, y, w, h], -1)

	return image, gt




train_dataset = tf.data.TFRecordDataset(tfr_train_dir)
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(AUTOTUNE).batch(N_BATCH, drop_remainder=True)

val_dataset = tf.data.TFRecordDataset(tfr_val_dir)
val_dataset = val_dataset.map(_parse_function,num_parallel_calls=AUTOTUNE).batch(N_BATCH, drop_remainder=True)
```


### 2. 데이터 확인 
``` python
for image, gt in val_dataset.take(10):
	x = gt[:, 0]
	y = gt[:, 1]
	w = gt[:, 2]
	h = gt[:, 3]
	
	xmin = x[0].numpy() - w[0].numpy() / 2.
	ymin = y[0].numpy() - h[0].numpy() / 2.

	rect_x = int(xmin * IMG_SIZE_WIDTH)
	rect_y = int(ymin * IMG_SIZE_HEIGHT)
	rect_w = int(w[0].numpy() * IMG_SIZE_WIDTH)
	rect_h = int(h[0].numpy() * IMG_SIZE_HEIGHT)

	rect = plt.Rectangle((rect_x, rect_y), rect_w, rect_h, fill = False, color = 'red')
	
	plt.axes().add_patch(rect)
	plt.imshow(image[0])
	plt.show()
```
![](Data/imgs/ImageLocalization/2.png)


### 3. 모델링
``` python
from keras.layers import Dense, Input, Conv2D, Conv2DTranspose, Flatten, Activation
from keras.layers import BatchNormalization, Dropout, ZeroPadding2D
from keras.models import Model
from keras.layers import ZeroPadding2D
from keras.regularizers import l2
from keras.layers import Add

class BackBone:
    def __init__(self):
        self.l2_regularizer = l2(0.001)

    def residual_layer(self, feature_map, latent, name:str):
        add_layer = Add(name = name+'_output')([feature_map, latent])
        return add_layer

    def feature_extraction_block(self, feature_map, filters_conv1:int, filters_conv2:int, name:str):
        feature_map = Conv2D(filters=filters_conv1, kernel_size = 3, strides = 1, padding = 'same', 
                        kernel_regularizer=self.l2_regularizer,
                        name = name)(feature_map)
        feature_map = BatchNormalization()(feature_map)
        feature_map = Activation('relu')(feature_map)
        feature_map = Dropout(0.3)(feature_map)

        feature_map = ZeroPadding2D(padding=((0, 1), (0, 1)), name=name+'_pad')(feature_map)
        feature_map = Conv2D(filters=filters_conv2, kernel_size = 3, strides = 2, padding = 'valid', 
                        kernel_regularizer=self.l2_regularizer,
                        name = name+'_2')(feature_map)
        feature_map = BatchNormalization()(feature_map)
        return feature_map

    def convolutional_residual_block(self, feature_map, filters_conv1:int, filters_conv2:int, name:str):
        latent = Conv2D(filters=filters_conv1, kernel_size = 3, strides = 1, padding = 'same', 
                        kernel_regularizer=self.l2_regularizer,
                        name = name)(feature_map)
        latent =  BatchNormalization()(latent)
        latent = Activation('relu')(latent)
        feature_map = Dropout(0.3)(feature_map)

        latent = Conv2D(filters=filters_conv2, kernel_size = 3, strides = 1, padding = 'same', 
                        kernel_regularizer=self.l2_regularizer,
                        name = name+'_2')(latent)
        latent = BatchNormalization()(latent)
        residual_block = self.residual_layer(feature_map, latent, name)
        return residual_block
    
    def __call__(self, input_shape=(24, 32, 1)):
        inputs_image = Input(shape=input_shape)
        upsample_layer = Conv2DTranspose(filters = 6, kernel_size = 3, strides = (3, 3), padding = 'same')(inputs_image)
        block_1 = self.feature_extraction_block(upsample_layer, 3, 5,'block_1')
        block_1_output = self.convolutional_residual_block(block_1, 3, 5,'block_2')
        block_2 = self.feature_extraction_block(block_1_output, 3, 5, 'block_3')
        block_2_output = self.convolutional_residual_block(block_2, 3, 5, 'block_4')
        block_3_output = self.feature_extraction_block(block_2_output, 3, 3,'block_5')



        latent = Flatten()(block_3_output)
        latent = Dense(64, activation='relu', dtype='float32')(latent)
        latent = Dropout(0.2)(latent)
        number_output = Dense(4, activation='sigmoid', name='number_output', dtype='float32')(latent)

        model = Model(inputs_image, number_output)
        return model
```


### 4. IoU Metric 설계 
``` python
class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, img_size_width, img_size_height, name='iou_metric', **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs) # 상위 클래스의 생성자를 호출해 초기화
        self.img_size_width = img_size_width
        self.img_size_height = img_size_height # 이미지 높이와 너비를 인스턴스 변수로 저장
        self.iou_sum = self.add_weight(name="iou_sum", initializer="zeros") # IoU의 합과 샘플의 총 개수를 추적하기 위해 변수 초기화
				        # 메트릭 클래스 내부에서 사용할 가중치를 생성하고 초기화
        self.total_samples = self.add_weight(name="total_samples", initializer="zeros") 


	# Tensorflow 훈련 함수 오버라이드
	"""
	- update_state : 각 배치의 데이터가 처리될 때(여기선 1배치 7장) 호출되며(스탭마다 호출), 메트릭의 상태 (ex. 누적된 값)를 업데이트

	- result : 현재까지 누적된 메트릭의 결과를 계산해 반환. 일반적으로 epoch가 끝날 때나 평가 시점에 호출

	- reset_states : 메트릭의 상태를 초기화. 주로 새로운 epoch나 새로운 평가 시작 전에 호출
	"""
    def update_state(self, y_true, y_pred, sample_weight=None):
	# y_true와 y_pred를 받아 IoU를 계산하고, iou_sum과 total_samples 를 업데이트
        x, y, w, h = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
        pred_x, pred_y, pred_w, pred_h = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

        xmin = (x - w / 2.) * self.img_size_width
        ymin = (y - h / 2.) * self.img_size_height
        xmax = (x + w / 2.) * self.img_size_width
        ymax = (y + h / 2.) * self.img_size_height

        pred_xmin = (pred_x - pred_w / 2.) * self.img_size_width
        pred_ymin = (pred_y - pred_h / 2.) * self.img_size_height
        pred_xmax = (pred_x + pred_w / 2.) * self.img_size_width
        pred_ymax = (pred_y + pred_h / 2.) * self.img_size_height

        w_inter = tf.maximum(tf.minimum(xmax, pred_xmax) - tf.maximum(xmin, pred_xmin), 0)
        h_inter = tf.maximum(tf.minimum(ymax, pred_ymax) - tf.maximum(ymin, pred_ymin), 0)

        inter = w_inter * h_inter
        union = (xmax - xmin) * (ymax - ymin) + (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin) - inter

        iou = tf.math.divide_no_nan(inter, union)
        iou_sum_batch = tf.reduce_sum(iou)

        self.iou_sum.assign_add(iou_sum_batch)
        self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))
        

    def result(self):
    # epoch의 끝이나 평가 시점에 호출되며 현재까지 누적된 IoU값을 반환
        return tf.math.divide_no_nan(self.iou_sum, self.total_samples) 
		        # tf.math.divide_no_nan : 분모가 0일 때 NaN 대신 0 반환

    def reset_states(self):
    # epoch의 시작 또는 평가 전에 호출되며 iou_sum과 total_samples 를 초기화
        self.iou_sum.assign(0.)
        self.total_samples.assign(0.)
```


### 5. 모델 컴파일 
``` python
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.metrics import Precision, Recall

backbone = BackBone()
model = backbone()

initial_learning_rate = LR
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=5000,
    decay_rate=0.90,
    staircase=True)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(optimizer=optimizer, 
              loss=['mse'], 
              metrics=['accuracy', IoUMetric(IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT), Precision(), Recall()])

checkpoint = ModelCheckpoint('v4_backbone_localization_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

model.summary()
```


### 6. tensorboard log callback 정의 
``` python
from tensorflow.keras.callbacks import TensorBoard

log_dir = LOG_DIR
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
```



### 7. 모델 학습
``` python
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=N_EPOCHS,
    verbose = 1,
    callbacks=[checkpoint, tensorboard_callback]
)

'''
tensorboard --logdir=LocalizationLog --bind_all
'''


"""
Epoch 100/100
256/258 [============================>.] - ETA: 0s - loss: 0.0021 - accuracy: 0.8990 - iou_metric: 0.6459 - precision: 1.0000 - recall: 0.2637
Epoch 100: val_accuracy did not improve from 0.91506
258/258 [==============================] - 5s 19ms/step - loss: 0.0021 - accuracy: 0.8992 - iou_metric: 0.6457 - precision: 1.0000 - recall: 0.2633 - val_loss: 0.0020 - val_accuracy: 0.9048 - val_iou_metric: 0.6961 - val_precision: 1.0000 - val_recall: 0.2574
"""
```

---



# 3. 학습 모델 확인 

### 1. model.predict
``` python
idx = 1
for val_data, val_gt in val_dataset.take(10):
    x = val_gt[:, 0]
    y = val_gt[:, 1]
    w = val_gt[:, 2]
    h = val_gt[:, 3]

    xmin = x[idx].numpy() - w[idx].numpy() / 2.
    ymin = y[idx].numpy() - h[idx].numpy() / 2.

    rect_x = int(xmin * IMG_SIZE_WIDTH)
    rect_y = int(ymin * IMG_SIZE_HEIGHT)
    rect_w = int(w[idx].numpy() * IMG_SIZE_WIDTH)
    rect_h = int(h[idx].numpy() * IMG_SIZE_HEIGHT)

    rect = plt.Rectangle((rect_x, rect_y), rect_w, rect_h, fill = False, color = 'red')
    ax = plt.gca()
    ax.add_patch(rect)
    prediction = model.predict(val_data)
    pred_x = prediction[:, 0]
    pred_y = prediction[:, 1]
    pred_w = prediction[:, 2]
    pred_h = prediction[:, 3]



    pred_xmin = pred_x[idx] - pred_w[idx] / 2.
    pred_ymin = pred_y[idx] - pred_h[idx] / 2.

    pred_rect_x = int(pred_xmin * IMG_SIZE_WIDTH)
    pred_rect_y = int(pred_ymin * IMG_SIZE_HEIGHT)
    pred_rect_w = int(pred_w[idx] * IMG_SIZE_WIDTH)
    pred_rect_h = int(pred_h[idx] * IMG_SIZE_HEIGHT)

    pred_rect = plt.Rectangle((pred_rect_x, pred_rect_y), pred_rect_w, pred_rect_h, fill = False, color = 'blue')
    ax.add_patch(pred_rect)
    plt.imshow(val_data[idx], cmap = 'gray')
    plt.show()
```
![](Data/imgs/ImageLocalization/3.png)
![](Data/imgs/ImageLocalization/4.png)
![](Data/imgs/ImageLocalization/5.png)
![](Data/imgs/ImageLocalization/6.png)


### 2. IOU 계산 
``` python
avg_iou = 0

for val_data, val_gt in val_dataset:
    x = val_gt[:, 0]
    y = val_gt[:, 1]
    w = val_gt[:, 2]
    h = val_gt[:, 3]

    prediction = model.predict(val_data)
    pred_x = prediction[:, 0]
    pred_y = prediction[:, 1]
    pred_w = prediction[:, 2]
    pred_h = prediction[:, 3]


    for idx in range(N_BATCH):
        xmin = int((x[idx].numpy() - w[idx].numpy() / 2.) * IMG_SIZE_WIDTH)
        ymin = int((y[idx].numpy() - h[idx].numpy() / 2.) * IMG_SIZE_HEIGHT)
        xmax = int((x[idx].numpy() + w[idx].numpy() / 2.) * IMG_SIZE_WIDTH)
        ymax = int((y[idx].numpy() + h[idx].numpy() / 2.) * IMG_SIZE_HEIGHT)

        W = xmax - xmin
        H = ymax - ymin

        pred_xmin = int((pred_x[idx] - pred_w[idx] / 2.) * IMG_SIZE_WIDTH)
        pred_ymin = int((pred_y[idx] - pred_h[idx] / 2.) * IMG_SIZE_HEIGHT)
        pred_xmax = int((pred_x[idx] + pred_w[idx] / 2.) * IMG_SIZE_WIDTH)
        pred_ymax = int((pred_y[idx] + pred_h[idx] / 2.) * IMG_SIZE_HEIGHT)

        pred_W = pred_xmax - pred_xmin
        pred_H = pred_ymax - pred_ymin

        if xmin > pred_xmax or xmax < pred_xmin:
            continue
        if ymin > pred_ymax or ymax < pred_ymin:
            continue


        w_inter = np.min((xmax, pred_xmax)) - np.max((xmin, pred_xmin))
        h_inter = np.min((ymax, pred_ymax)) - np.max((ymin, pred_ymin))

        inter = w_inter * h_inter

        union = (W * H + pred_W * pred_H) - inter

        iou = inter / union
        avg_iou += iou / N_VAL

print(avg_iou)


"""
0.6829128161199601
"""
```



`compute_iou` 함수는 두 세트의 박스(boxes1과 boxes2) 간의 교차 영역(IoU, Intersection over Union)를 계산합니다. 각 박스는 `[x_center, y_center, width, height]` 형식으로 제공됩니다. 이 함수의 주요 단계는 다음과 같은 수식으로 표현될 수 있습니다:

1. **박스 변환**: 각 박스를 `[x_center, y_center, width, height]`에서 `[x_min, y_min, x_max, y_max]` 형식으로 변환. 이 변환은 `convert_to_corners` 함수에 의해 수행

1. **교차 영역 계산**:
   - **왼쪽 상단 좌표(LU, Left Upper) 계산**: 
     $[ LU = \max(boxes1\_corners[:2], boxes2\_corners[:2])]$
   - **오른쪽 하단 좌표(RD, Right Down) 계산**:
     $[ RD = \min(boxes1\_corners[2:], boxes2\_corners[2:])]$

2. **교차 면적(Intersection Area) 계산**:
   - **교차 영역의 너비와 높이 계산**:
     $[ intersection = \max(RD - LU, 0.0) ]$
   - **교차 면적 계산**:
     $[ intersection\_area = intersection[:,:,0] \times intersection[:,:,1] ]$

3. **각 박스의 면적 계산**:
   - **boxes1의 면적**:
     $[ boxes1\_area = boxes1[:, 2] \times boxes1[:, 3]]$
   - **boxes2의 면적**:
     $[ boxes2\_area = boxes2[:, 2] \times boxes2[:, 3] ]$

4. **합집합 면적(Union Area) 계산**:
   $[ union\_area = \max(boxes1\_area[:, None] + boxes2\_area - intersection\_area, 1e-8) ]$

5. **IoU 계산**:
   $[ IoU = \frac{intersection\_area}{union\_area} ]$
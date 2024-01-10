

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
copy_coord = coordinates[13000]

new_coords = []
changed_coords = []
for i in range(copy_coord.shape[0]):
    copy_coord[i]
    x_coords = copy_coord[i][0::2]
    y_coords = copy_coord[i][1::2]

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
plt.figure(figsize = (8, 8))
plt.axis('off')
plt.imshow(origin_images[13000])
ax = plt.gca()
boxes = tf.stack(
	[
	 boxes[:, 0],
	 boxes[:, 1],
	 boxes[:, 2],
	 boxes[:, 3]
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



### 3. 데이터 변환 
``` python
bboxes = []
for coords in coordinates:
    new_coords = []  # 각 coords 집합에 대해 new_coords를 초기화
    for i in range(coords.shape[0]):
        x_coords = coords[i][0::2]
        y_coords = coords[i][1::2]

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
    	 boxes[:, 0],
    	 boxes[:, 1],
    	 boxes[:, 2],
    	 boxes[:, 3]
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



# 1. npz to tfr

### 1. 데이터 로드
``` python
import numpy as np

datasets = np.load('npz/ObjectDetection.npz', allow_pickle=True)
images, numbers, bboxes = datasets['images'], datasets['numbers'], datasets['bboxes']
```


### 2. 데이터 확인
``` python
import matplotlib.pyplot as plt

boxes = bboxes[9000]
plt.figure(figsize = (8, 8))
plt.axis('off')
plt.imshow(images[9000])
ax = plt.gca()

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



### 3. 데이터 변환 및 저장
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
tfr_dir = os.path.join(cur_dir, 'tfrecord/ObjectDetection')
os.makedirs(tfr_dir, exist_ok=True)

tfr_train_dir = os.path.join(tfr_dir, 'od_train.tfr')
tfr_val_dir = os.path.join(tfr_dir, 'od_val.tfr')

writer_train = tf.io.TFRecordWriter(tfr_train_dir)
writer_val = tf.io.TFRecordWriter(tfr_val_dir)
```

``` python
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))
```

``` python
for idx in train_idx_list: # val_idx_list도 동일
    bbox = bboxes[idx]
    xmin, ymin, xmax, ymax = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]

    xmin = xmin / IMG_SIZE_WIDTH
    ymin = ymin / IMG_SIZE_HEIGHT
    xmax = xmax / IMG_SIZE_WIDTH
    ymax = ymax / IMG_SIZE_HEIGHT

    bbox = np.stack([xmin, ymin, xmax, ymax], axis=-1).flatten()

    image = images[idx]
    bimage = image.tobytes()

    number = numbers[idx]
    label = 1 if number != 0 else 0    
    
    example = tf.train.Example(features = tf.train.Features(feature = {
        'image': _bytes_feature(bimage),
        'bbox': _float_feature(bbox),
        'count': _int64_feature(number),
        'label':_int64_feature(label)
    }))
    
    writer_train.write(example.SerializeToString())
writer_train.close()
```



### 4. 저장 데이터 확인
``` python
AUTOTUNE = tf.data.AUTOTUNE

RES_HEIGHT =24
RES_WIDTH = 32
N_EPOCHS = 100
N_BATCH = 8
LR = 0.0005


def _parse_function(tfrecord_serialized):
    features = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'bbox': tf.io.VarLenFeature(tf.float32),  # VarLenFeature로 변경
        'count': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)

    image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.reshape(image, [RES_HEIGHT, RES_WIDTH, 1])
    image = tf.cast(image, tf.float32) / 255.0

    bbox = tf.sparse.to_dense(parsed_features['bbox'])  # 변환 필요
    bbox = tf.cast(bbox, tf.float32)
    num_boxes = tf.shape(bbox)[0] // 4
    bbox = tf.reshape(bbox, [num_boxes, 4])

    count = tf.cast(parsed_features['count'], tf.int64)
    label = tf.cast(parsed_features['label'], tf.int64)

    return image, bbox, count, label


train_dataset = tf.data.TFRecordDataset(tfr_train_dir)
train_dataset = train_dataset.map(_parse_function, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=N_TRAIN).prefetch(AUTOTUNE).batch(N_BATCH, drop_remainder=True)

val_dataset = tf.data.TFRecordDataset(tfr_val_dir)
val_dataset = val_dataset.map(_parse_function, num_parallel_calls=AUTOTUNE).batch(N_BATCH, drop_remainder=True)
```


### 5. 데이터 확인
``` python
import matplotlib.pyplot as plt

for image, bbox, count, label in val_dataset.take(5):
    image = image[0].numpy()
    bbox = bbox[0]
    count = count[0]
    label = label[0]

    plt.figure(figsize = (8, 8))
    plt.axis('off')
    plt.imshow(image)
    ax = plt.gca()  

    image_h = image.shape[0]    
    image_w = image.shape[1]

    boxes = tf.stack(
    	[
    	 bbox[:, 0] * image_w,
    	 bbox[:, 1] * image_h,
    	 bbox[:, 2] * image_w,
    	 bbox[:, 3] * image_h
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
    print(count, label)
    
    """
    tf.Tensor(3, shape=(), dtype=int64) tf.Tensor(1, shape=(), dtype=int64)
    """
```
![](Data/imgs/ObjectDetection/3.png)


---


# Data Augmentation


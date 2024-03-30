import board
import busio
import numpy as np
from scipy import ndimage, interpolate
from flask import Flask, Response, render_template, make_response
import cv2
import adafruit_mlx90640
import datetime
import logging
import time
from tensorflow.keras.models import load_model
from loss import ssim_loss  
from keras.utils import custom_object_scope

from sklearn.cluster import DBSCAN
from flask_cors import CORS
import threading
import gc

app = Flask(__name__)
CORS(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

model_path = 'model/filter_model_backup.h5'
filter_path = 'model/93.h5'

current_people_count = 0
frame_lock = threading.Lock()

model = load_model(model_path)
filter_model = load_model(filter_path)


class ThermalCamera:
    def __init__(self, frame_shape=(24, 32), fps=7, max_retries=3):
        self.frame_shape = frame_shape
        self.fps = fps
        self.max_retries = max_retries
        self.mlx = None
        self.initialize_sensor()

    def initialize_sensor(self):
        self.release_resources() 
        for attempt in range(self.max_retries):
            try:
                i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
                self.mlx = adafruit_mlx90640.MLX90640(i2c)
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ
                self.frame = np.zeros(self.frame_shape[0] * self.frame_shape[1])
                print("MLX90640 센서 초기화 성공")
                break
            except RuntimeError as e:
                print(f"Retry initializing MLX90640: Attempt {attempt + 1}/{self.max_retries} - {e}")
                time.sleep(1)
                self.frame = np.zeros(self.frame_shape[0] * self.frame_shape[1])
                if attempt == self.max_retries - 1:
                    raise RuntimeError("센서 초기화에 실패했습니다.")

    def release_resources(self):
        if self.mlx is not None:
            self.mlx = None
        if 'i2c' in globals():
            global i2c
            del i2c
            i2c = None
        gc.collect()  
        print('gc 강제실행')

    def capture_frame(self):
        self.mlx.getFrame(self.frame)
        
        data_array_raw = np.fliplr(np.reshape(self.frame, self.frame_shape))
        valid_min_temp = 20
        data_array_raw[data_array_raw < valid_min_temp] = valid_min_temp
        data_array_raw_normalized_uint8 = ((data_array_raw - np.min(data_array_raw)) / (np.max(data_array_raw) - np.min(data_array_raw)) * 255).astype(np.uint8)
        expanded_img_array = np.expand_dims(data_array_raw_normalized_uint8, axis= 0)
        return expanded_img_array.astype(np.uint8)


def draw_bounding_boxes(image, number):
    try:
        _, thresholded = cv2.threshold(image, 137, 255, cv2.THRESH_BINARY)

        points = np.column_stack(np.where(thresholded.transpose() > 0))
        if len(points) < 1:
            return image  
        
        clustering = DBSCAN(eps=1, min_samples=1).fit(points)
        labels = clustering.labels_

        bboxes = []
        for label in np.unique(labels):
            if label == -1:
                continue  
            label_points = points[labels == label]
            if len(label_points) == 0:
                return image
            x, y, w, h = cv2.boundingRect(label_points.astype(np.int32))
            bboxes.append((x, y, w, h))

        bboxes = sorted(bboxes, key=lambda b: b[2]*b[3], reverse=True)[:number]

        for (x, y, w, h) in bboxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        return image
    except Exception as e:
        print(f"Error in draw_bounding_boxes: {e}")
        return image  # 오류 발생 시 원본 이미지를 반환


def preprocessing(frame):
    preprocessed_image = ((frame - np.min(frame)) / (np.max(frame) - np.min(frame)) * 255).astype(np.uint8) / 255
    return preprocessed_image


def predict(preprocessed_image):
    noise_removed_image, _ = filter_model.predict(preprocessed_image)
    _, people_count = model.predict(preprocessed_image)
    noise_removed_image = ((noise_removed_image - np.min(noise_removed_image)) / (np.max(noise_removed_image) - np.min(noise_removed_image)) * 255).astype(np.uint8)
    counted_person = np.array(people_count[0]).argmax()
    image_with_boxes = draw_bounding_boxes(np.array(noise_removed_image[0]), counted_person)
    image_with_boxes = np.tile(image_with_boxes, (1, 1, 3))
    return image_with_boxes, counted_person


def generate_frames():
    global current_people_count  
    while True:
        try:
            frame = camera.capture_frame()
            preprocessed_image = preprocessing(frame)
            noise_removed_image, people_count = predict(preprocessed_image)
            ret, buffer = cv2.imencode('.jpg', noise_removed_image)  
            streaming_frame = buffer.tobytes()
            with frame_lock:
                current_people_count = people_count
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + streaming_frame + b'\r\n')
        except Exception as e:
            print(f"Error generating frames: {e}")
            time.sleep(1)  


def start_data_capture_thread():
    data_thread = threading.Thread(target=generate_frames, daemon=True)
    data_thread.start()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/no-cache')
def no_cache():
    response = make_response("This response is not cached")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/people_count')
def people_count():
    global current_people_count  
    return {"count": int(current_people_count)}

camera = ThermalCamera() 

if __name__ == '__main__':
    start_data_capture_thread()
    app.run(host='0.0.0.0', port=5000, threaded=True)
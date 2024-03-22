import board
import busio
import numpy as np
from scipy import ndimage, interpolate
from flask import Flask, Response, render_template, make_response
import cv2
import adafruit_mlx90640
import cv2
import datetime
import logging
import time
from tensorflow.keras.models import load_model
from loss import ssim_loss  
from keras.utils import custom_object_scope

from sklearn.cluster import DBSCAN
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)


model_path = 'model/filter_model_backup.h5'
filter_path = 'model/93.h5'
current_people_count = 0  

with custom_object_scope({'ssim_loss': ssim_loss}):
    model = load_model(model_path)
    filter_model = load_model(filter_path)

class ThermalCamera:
    def __init__(self, frame_shape=(24, 32), fps=7, max_retries=3):
        self.current_datetime = datetime.datetime.now()

        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        
        for attempt in range(max_retries):
            try:
                self.mlx = adafruit_mlx90640.MLX90640(i2c)
                self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_4_HZ
                break
            except RuntimeError as e:
                if "Adjacent outlier pixels" in str(e) and attempt < max_retries - 1:
                    print(f"Retry initializing MLX90640: Attempt {attempt + 1}")
                    time.sleep(1) 
                else:
                    raise 

        self.mlx_shape = frame_shape
        self.frame = np.zeros(self.mlx_shape[0] * self.mlx_shape[1])  # 768


    def capture_frame(self):
        self.mlx.getFrame(self.frame)
        
        data_array_raw = np.fliplr(np.reshape(self.frame, self.mlx_shape))
        
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
    camera = ThermalCamera() 
    while True:
        frame = camera.capture_frame()  
        preprocessed_image = preprocessing(frame)
        noise_removed_image, people_count = predict(preprocessed_image)
        ret, buffer = cv2.imencode('.jpg', noise_removed_image)  
        streaming_frame = buffer.tobytes()
        current_people_count = people_count
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + streaming_frame + b'\r\n')


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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
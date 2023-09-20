import time
import board
import busio
import numpy as np
from scipy import ndimage, interpolate
import adafruit_mlx90640
import cv2
import datetime
import tensorflow as tf
from tensorflow import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
import matplotlib.pyplot as plt

class ThermalCamera:
    def __init__(self, output_filename, model_filename, frame_shape=(24, 32), fps=3):
        # 현재 시간을 가져와서 출력 파일 이름 생성
        self.current_datetime = datetime.datetime.now()
        self.output_filename = output_filename
        self.labels_to_names = {0: 'person'}

        # I2C 통신 설정
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        
        # MLX90640 센서 초기화
        self.mlx = adafruit_mlx90640.MLX90640(i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.mlx_shape = frame_shape

        # 비디오 녹화를 위한 설정
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_filename, fourcc, fps, self.mlx_shape[::-1], isColor=True)  # 가로와 세로를 반전

        # 센서에서 읽어온 데이터를 저장하기 위한 배열 초기화
        self.frame = np.zeros(self.mlx_shape[0] * self.mlx_shape[1])  # 총 픽셀 수: 768개

        # 딥러닝 모델 로드
        self.model = models.load_model(model_filename, backbone_name='resnet50')
        print(self.model.summary())

    def capture_frame(self):
        # MLX90640 센서에서 데이터 프레임 get
        self.mlx.getFrame(self.frame)
        
        # 인체 온도 범위 외의 픽셀을 마스킹
        self.frame[(self.frame <= 23) | (self.frame > 40)] = 0
        
        # 데이터를 올바른 형태로 변환
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
    
        return image  # 수정: draw 변수를 이미지로 반환

    def detect_objects(self, image):
        
        # 이미지 데이터 타입 변환 (CV_64F에서 CV_8U로)
        image = (image * 255).astype(np.uint8)
        
        # 이미지를 3개의 채널로 확장
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # preprocess image for network
        image_rgb = preprocess_image(image_rgb)
        image_rgb, scale = resize_image(image_rgb)
        
        # process image
        image_rgb = image_rgb.astype(np.uint8)  # 추가: 이미지를 np.uint8로 변환
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image_rgb, axis=0))

        # correct for image scale
        boxes /= scale

        # adjust box coordinates for the original image size
        boxes[:, :4] /= 10  # 확대 비율로 좌표 조정

        # create a copy of the image to draw on
        draw = image_rgb.copy()

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            if score < 0.5:
                continue

            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            draw_caption(draw, b, caption)

        # resize the drawn image back to the original size
        draw = cv2.resize(draw, (self.mlx_shape[1], self.mlx_shape[0]))

        return cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)



    def record_video(self):
        try:
            t_start = time.monotonic()
            frame_count = 0

            while True:
                try:
                    t_frame_start = time.monotonic()
                    img_array = self.capture_frame()

                    # Detect objects in the captured frame
                    detected_frame = self.detect_objects(img_array)

                    frame_count += 1
                    self.out.write(detected_frame)
                    t_frame = time.monotonic() - t_frame_start
                    print('Frame Rate: {0:2.1f}fps'.format(1 / t_frame))

                except Exception as e:
                    print(f"An error occurred: {str(e)}")
                    continue

        except KeyboardInterrupt:
            self.out.release()
            exit(1)

        finally:
            t_total = time.monotonic() - t_start
            print('Average Frame Rate: {0:2.1f}fps'.format(frame_count / t_total))
            self.out.release()
            exit(1)
        print("Recording finished.")


if __name__ == "__main__":
    output_filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".mp4"
    model_filename = "main.h5"
    thermal_camera = ThermalCamera(output_filename, model_filename)
    thermal_camera.record_video()

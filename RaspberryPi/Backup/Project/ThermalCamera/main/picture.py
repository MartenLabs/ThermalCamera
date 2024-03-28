import cv2
import numpy as np
import os
import time
import datetime
import board
import busio
from scipy import ndimage, interpolate
import adafruit_mlx90640
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# 로드할 객체 검출 모델
model = models.load_model('main.h5', backbone_name='resnet50')

class ThermalCamera:
    def __init__(self, output_folder, frame_shape=(24, 32)):
        # 현재 시간을 가져와서 출력 파일 이름 생성
        self.current_datetime = datetime.datetime.now()
        self.output_folder = output_folder
        self.labels_to_names = {0: 'person'}

        # I2C 통신 설정
        i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)
        
        # MLX90640 센서 초기화
        self.mlx = adafruit_mlx90640.MLX90640(i2c)
        self.mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
        self.mlx_shape = frame_shape

        # 센서에서 읽어온 데이터를 저장하기 위한 배열 초기화
        self.frame = np.zeros(self.mlx_shape[0] * self.mlx_shape[1])  # 총 픽셀 수: 768개

    def capture_frame(self):
        # MLX90640 센서에서 데이터 프레임 가져오기
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
    
        return image

    def detect_objects(self, image):
        # 이미지 데이터 타입 변환 (CV_64F에서 CV_8U로)
        image = (image * 255).astype(np.uint8)
        
        # 이미지를 3개의 채널로 확장
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 10배 업스케일링
        image_rgb_scaled = cv2.resize(image_rgb, None, fx=10, fy=10, interpolation=cv2.INTER_LINEAR)
        
        # preprocess image for network
        image_rgb_scaled = preprocess_image(image_rgb_scaled)
        image_rgb_scaled, scale = resize_image(image_rgb_scaled)
        
        # process image
        image_rgb_scaled = image_rgb_scaled.astype(np.uint8)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image_rgb_scaled, axis=0))

        # correct for image scale
        boxes /= scale

        # adjust box coordinates for the original image size
        boxes[:, :4] /= 10  # 확대 비율로 좌표 조정

        # create a copy of the image to draw on
        draw = image_rgb_scaled.copy()

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

    def save_image_with_detection(self, image):
        # 이미지 파일로 저장
        output_image_filename = os.path.join(self.output_folder, 'output_image.jpg')
        cv2.imwrite(output_image_filename, image)

    def run(self):
        try:
            img_array = self.capture_frame()

            # Detect objects in the captured frame
            detected_image = self.detect_objects(img_array)

            # Save the image with detections
            self.save_image_with_detection(detected_image)

        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    output_folder = "img"  # 저장할 이미지 폴더
    os.makedirs(output_folder, exist_ok=True)

    thermal_camera = ThermalCamera(output_folder)
    thermal_camera.run()

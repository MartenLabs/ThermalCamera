import time
import board
import busio
import numpy as np
from scipy import ndimage, interpolate
import adafruit_mlx90640
import cv2
import datetime

class ThermalCamera:
    def __init__(self, output_filename, frame_shape=(24, 32), fps=3):
        # 현재 시간을 가져와서 출력 파일 이름 생성
        self.current_datetime = datetime.datetime.now()
        self.output_filename = output_filename

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
        interpolated_data_arrary = interpolate.griddata((x_idx_valid_values, y_idx_valid_values), valid_values, (x_idx_all, y_idx_all), 'cubic')

        # 인체 온도 범위 외의 픽셀을 마스킹 (온도 가정)
        interpolated_data_arrary[(interpolated_data_arrary <= 25) | (interpolated_data_arrary > 40)] = 0

        # 이미지를 0-255 범위로 변환
        img_array = (interpolated_data_arrary - np.min(interpolated_data_arrary)) / (np.max(interpolated_data_arrary) - np.min(interpolated_data_arrary)) * 255

        # 녹화를 위해 이미지를 3 채널로 확장
        expanded_img_array = np.expand_dims(img_array, axis=2)
        expanded_img_array = np.tile(expanded_img_array, (1, 1, 3))

        # 데이터 타입을 uint8로 변환
        expanded_img_array = expanded_img_array.astype(np.uint8)

        return expanded_img_array

    def record_video(self):
        try:
            t_start = time.monotonic()
            frame_count = 0

            while True:
                try:
                    t_frame_start = time.monotonic()
                    img_array = self.capture_frame()
                    frame_count += 1
                    self.out.write(img_array)
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
            print("Recording finished.")
            self.out.release()
            exit(1)

if __name__ == "__main__":
    output_filename = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + ".mp4"
    thermal_camera = ThermalCamera(output_filename)
    thermal_camera.record_video()

import time
import board
import busio
import numpy as np
from scipy import ndimage, interpolate
import adafruit_mlx90640
import cv2
import datetime

class ThermalCamera:
    def __init__(self, output_filename, frame_shape=(24, 32), fps=7):
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

        # 데이터를 올바른 형태로 변환
        data_array_raw = np.fliplr(np.reshape(self.frame, self.mlx_shape))
        
        # 데이터 정규화: 데이터 값을 0-255 범위로 변환
        data_array_raw_normalized_uint8 = ((data_array_raw - np.min(data_array_raw)) / (np.max(data_array_raw) - np.min(data_array_raw)) * 255).astype(np.uint8)

        # return expanded_img_array
        expanded_img_array = np.expand_dims(data_array_raw_normalized_uint8, axis=2)
        expanded_img_array = np.tile(expanded_img_array, (1, 1, 3))
        return expanded_img_array.astype(np.uint8)

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

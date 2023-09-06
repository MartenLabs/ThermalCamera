import time
import board
import busio
import numpy as np
from scipy import ndimage, interpolate
import adafruit_mlx90640
import matplotlib.pyplot as plt
import cv2
import datetime

current_datetime = datetime.datetime.now()

# I2C 통신 설정.
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# MLX90640 센서 초기화.
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

# 센서 배열의 형태를 정의합니다.
mlx_shape = (24, 32)

# 영상 파일 설정
output_filename = f"{current_datetime.strftime('%Y-%m-%d %H-%M-%S')}.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 3 # 프레임 속도 (프레임/초)
out = cv2.VideoWriter(output_filename, fourcc, fps, mlx_shape[::-1], isColor=False) # 가로와 세로를 반전

frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 총 픽셀 수:768개


def update_frame():
	try:
		mlx.getFrame(frame)  
		data_array_raw = np.fliplr(np.reshape(frame, mlx_shape))  
		# 가우시안 필터를 사용하여 노이즈 감소 및 스무딩 - 픽셀 최적화 & 노이즈 제거 
		data_array_smoothed = ndimage.gaussian_filter(data_array_raw, sigma=1.3)
		# 결측치나 제로값을 채우기 위해 보간법 사용 - 보간 기술 
		x_idx_valid_values,y_idx_valid_values=np.where(data_array_smoothed > 0)
		valid_values=data_array_smoothed[x_idx_valid_values,y_idx_valid_values]
		x_idx_all,y_idx_all=np.indices(data_array_smoothed.shape)
		interpolated_data_array=interpolate.griddata((x_idx_valid_values,y_idx_valid_values),valid_values,(x_idx_all,y_idx_all),'nearest')
		# 인체 온도 범위 외의 픽셀 마스킹 (°C 가정).
		interpolated_data_array[interpolated_data_array < 20] = 0
		interpolated_data_array[interpolated_data_array > 40] = 0
		img_array = (interpolated_data_array - np.min(interpolated_data_array)) / (np.max(interpolated_data_array) - np.min(interpolated_data_array)) * 255
		img_array = img_array.astype(np.uint8)
		img_array = cv2.resize(img_array, (320, 240), interpolation=cv2.INTER_CUBIC) # 인터폴레이션
		out.write(img_array)

	except Exception as e:
		print(f"An error occurred: {str(e)}")
    	# out.release()
		# exit(0)
    

t_start=time.monotonic()
frame_count=0

while True:
	try:
		t_frame_start=time.monotonic()
		try:
			update_frame()  
			frame_count+=1
		except:
			continue
		t_frame=time.monotonic()-t_frame_start
		print('Frame Rate: {0:2.1f}fps'.format(1/t_frame))
  
	except Exception as e:
		t_total=time.monotonic()-t_start
		print('Average Frame Rate: {0:2.1f}fps'.format(frame_count/t_total))
		out.release()
		exit(0)

print("Recording finished.")
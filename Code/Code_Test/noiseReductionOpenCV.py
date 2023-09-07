import numpy as np
import scipy.signal
import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt

# ...

def bilateral_filter(image, d, sigma_color, sigma_space):
    """
    Bilateral 필터링을 적용하는 함수

    :param image: 입력 이미지 (그레이스케일)
    :param d: 공간 거리 가중치
    :param sigma_color: 색상 가중치의 표준 편차
    :param sigma_space: 공간 가중치의 표준 편차
    :return: 필터링된 이미지
    """
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            pixel_value = 0.0
            normalization = 0.0
            for i in range(-d, d+1):
                for j in range(-d, d+1):
                    neighbor_x = x + j
                    neighbor_y = y + i
                    if 0 <= neighbor_x < width and 0 <= neighbor_y < height:
                        color_weight = np.exp(-((image[neighbor_y, neighbor_x] - image[y, x]) ** 2) / (2 * sigma_color ** 2))
                        space_weight = np.exp(-((i ** 2 + j ** 2) / (2 * sigma_space ** 2)))
                        pixel_weight = color_weight * space_weight
                        pixel_value += image[neighbor_y, neighbor_x] * pixel_weight
                        normalization += pixel_weight
            result[y, x] = int(pixel_value / normalization)

    return result

# ...

# Setup I2C communication
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# Initialize MLX90640 sensor
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
mlx_shape = (24, 32)

# Setup the figure for plotting
plt.ion()  # Enable interactive plotting
fig, ax = plt.subplots(figsize=(12, 9))
therm1 = ax.imshow(np.zeros(mlx_shape), cmap='gray', vmin=0, vmax=60)  # Use gray colormap
cbar = fig.colorbar(therm1)  # Setup colorbar for temperatures
cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)  # Colorbar label

frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 768 pts

def plot_update():
    mlx.getFrame(frame)  # Read MLX temperatures into the frame variable
    data_array = np.fliplr(np.reshape(frame, mlx_shape))  # Reshape, flip data

    # Normalize the data_array to 0-255
    data_array = ((data_array - data_array.min()) / (data_array.max() - data_array.min()) * 255).astype(np.uint8)
    
    d = 9  # 공간 거리 가중치
    sigma_color = 75  # 색상 가중치의 표준 편차
    sigma_space = 75  # 공간 가중치의 표준 편차
    data_array = bilateral_filter(data_array, d, sigma_color, sigma_space)
    
    # Histogram Equalization (적용해도 무관)
    # data_array = cv2.equalizeHist(data_array)
    
    # Adaptive Thresholding (임계값을 조절하여 최적화)
    threshold = 100  # 임계값 설정
    data_array[data_array <= threshold] = 0
    data_array[data_array > threshold] = 255
    
    therm1.set_array(data_array)  # Set data
    therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))  # Set bounds

    plt.pause(0.001)  # Required


t_array = []
while True:
    t1 = time.monotonic()  # for determining frame rate
    try:
        plot_update()  # Update plot
    except:
        continue

    # Approximating frame rate
    t_array.append(time.monotonic() - t1)
    if len(t_array) > 10:
        t_array = t_array[1:]  # Recent times for frame rate approximation
    print('Frame Rate: {0:2.1f}fps'.format(len(t_array) / np.sum(t_array)))



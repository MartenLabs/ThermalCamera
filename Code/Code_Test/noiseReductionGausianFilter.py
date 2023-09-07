import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
from scipy import ndimage, ndimage

# Setup I2C communication
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# Initialize MLX90640 sensor
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
mlx_shape = (24, 32)

# Setup the figure for plotting
plt.ion()  # Enable interactive plotting
fig, ax = plt.subplots(figsize=(12, 9))
therm1 = ax.imshow(np.zeros(mlx_shape), cmap='gray', vmin=25, vmax=45)  # Use gray colormap
cbar = fig.colorbar(therm1)  # Setup colorbar for temperatures
cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)  # Colorbar label

frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 768 pts

def plot_update():
    mlx.getFrame(frame)  # Read MLX temperatures into the frame variable
    data_array = np.fliplr(np.reshape(frame, mlx_shape))  # Reshape, flip data

    # Apply Gaussian filter for additional noise reduction and smoothing
    data_array = ndimage.gaussian_filter(data_array, sigma=1.3)

    # Mask pixels outside human body temperature range (assuming °C)
    data_array[data_array < 25] = 0
    data_array[data_array > 38] = 0

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

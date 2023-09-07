import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt

# Setup I2C communication
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# Initialize MLX90640 sensor
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ
mlx_shape = (24, 32)

# Setup the figure for plotting
plt.ion()  # Enable interactive plotting
fig, ax = plt.subplots(figsize=(12, 7))
therm1 = ax.imshow(np.zeros(mlx_shape), cmap='gray', vmin=0, vmax=60)  # Use gray colormap
cbar = fig.colorbar(therm1)  # Setup colorbar for temperatures
cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)  # Colorbar label

frame = np.zeros((24 * 32,))  # Setup array for storing all 768 temperatures
t_array = []

while True:
    t1 = time.monotonic()
    try:
        mlx.getFrame(frame)  # Read MLX temperatures into the frame variable
        data_array = np.reshape(frame, mlx_shape)  # Reshape to 24x32
        therm1.set_data(np.fliplr(data_array))  # Flip left to right
        therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array))  # Set bounds
        plt.pause(0.001)  # Required
        # Save the figure as an image (optional)
        # fig.savefig('mlx90640_test_fliplr.png', dpi=300, facecolor='#FCFCFC', bbox_inches='tight')
        t_array.append(time.monotonic() - t1)
        sample_rate = len(t_array) / np.sum(t_array)
        print(f'Sample Rate: {sample_rate:.1f}fps')
    except ValueError:
        continue  # If error, just read again

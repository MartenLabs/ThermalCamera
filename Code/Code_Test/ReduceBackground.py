import time
import board
import busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
from scipy import ndimage

# Setup I2C communication.
i2c = busio.I2C(board.SCL, board.SDA, frequency=400000)

# Initialize MLX90640 sensor.
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_16_HZ

# Define the shape of the sensor array.
mlx_shape = (24, 32)

# Setup the figure for plotting.
plt.ion()  # Enable interactive plotting.
fig, ax = plt.subplots(figsize=(12, 9))
therm1 = ax.imshow(np.zeros(mlx_shape), cmap='gray', vmin=0, vmax=60)  
cbar = fig.colorbar(therm1)  
cbar.set_label('Temperature [$^{\circ}$C]', fontsize=14)  

frame = np.zeros(mlx_shape[0] * mlx_shape[1]) 
background_frame = None

def plot_update():
    global background_frame

    mlx.getFrame(frame)  
    data_array = np.fliplr(np.reshape(frame.copy(), mlx_shape))  

    if background_frame is None:
        background_frame = data_array.copy()
        
    data_array -= background_frame
    
    # Apply Gaussian filter for additional noise reduction and smoothing.
    data_array_smoothed = ndimage.gaussian_filter(data_array.copy(), sigma=1.3)
    
    threshold_value = np.max(data_array_smoothed)*0.45
       
    mask_pixels_below_threshold_in_original_data=data_array < threshold_value
    
   # Set those pixels to minimum value in original image where smoothed image has pixel values less than threshold value 
   
    data_array[mask_pixels_below_threshold_in_original_data]=np.min(data_array)
   
    therm1.set_data(data_array)
    therm1.set_clim(vmin=np.min(data_array), vmax=np.max(data_array)) 

    plt.pause(0.001)  # Required
while True:
     t_start=time.monotonic()
     try:
         plot_update()  
     except Exception as e:
         print(f"Error occurred: {e}")
         continue 

     t_end=time.monotonic()-t_start
     
     print('Frame Rate: {0:2.1f}fps'.format(1/t_end))

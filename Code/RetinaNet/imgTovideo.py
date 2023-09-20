import os

fps = 3
video_path = '2person.mp4'
output_folder = 'folder'
output_video_path = 'video.mp4'
os.system(f"ffmpeg -framerate {fps} -i {output_folder}/frame_%d_annotated.jpg -c:v libx264 -pix_fmt yuv420p {output_video_path}")
os.system(f"ffmpeg -framerate {fps} -i {output_folder}/frame_%d_annotated.jpg -c:v libx264 -pix_fmt yuv420p {output_video_path}")
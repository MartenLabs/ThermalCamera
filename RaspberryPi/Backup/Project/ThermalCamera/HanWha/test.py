from flask import Flask, Response, render_template_string
import cv2

app = Flask(__name__)

# USB 카메라 설정
camera = cv2.VideoCapture(0)  # 0은 첫 번째 연결된 카메라를 의미합니다.

def generate_frames():
    while True:
        success, frame = camera.read()  # 카메라 프레임 읽기
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    # HTML 페이지를 렌더링합니다.
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>USB Camera Stream</title>
        <style>
            img {
                width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <h1>USB Camera Stream</h1>
        <img src="/video_feed" alt="Video Feed">
    </body>
    </html>
    """
    return render_template_string(html)

@app.route('/video_feed')
def video_feed():
    # 비디오 스트리밍 경로
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

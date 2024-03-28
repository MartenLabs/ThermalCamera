import cv2

# V4L2 백엔드를 사용하여 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite('captured_image.jpg', frame)
        print("이미지가 성공적으로 캡처되어 저장되었습니다.")
    else:
        print("카메라에서 프레임을 캡처하는 데 실패했습니다.")
    cap.release()

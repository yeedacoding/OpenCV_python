# 카메라 프레임 동영상 파일로 저장하기

import cv2

# 0 -> pc의 디폴트 카메라와 연결
# capture.isOpen() -> 카메라가 정상적으로 open 되어있는지 확인 가능
capture = cv2.VideoCapture(0)
if capture.isOpened() == False:
    raise Exception("카메라 연결 안됨")

fps = 20.0  # 초당 프레임 수
delay = round(1000/fps)  # 프레임 간 지연 시간 = 1 프레임과 다음 프레임 사이의 간격
size = (640, 360)   # 동영상 파일 해상도 (정수로 설정)

# 비디오 코덱은 운영체제에 따라 다름
# WINDOW -> *'DIVX'
# MAC OS or Linux -> *'XVID
fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 압축 코덱 설정

# 카메라 속성 실행창에 출력
print("width x height : ", size)
print("VideoWriterfourcc : %s" % fourcc)
print("delay : %2d" % delay)
print("fps: %.2f" % fps)

capture.set(cv2.CAP_PROP_ZOOM, 1)
capture.set(cv2.CAP_PROP_FOCUS, 0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])

# 동영상파일 개방 및 코덱, 해상도 설정
# 영상 저장 메서드
# cv2.VideoWriter(outputFile, fourcc, frame, size) -> retval(ret)
# outputFile (str) – 저장될 파일명 및 경로
# fourcc – Codec정보. cv2.VideoWriter_fourcc()
# frame (float) – 초당 저장될 frame 수 (fps)
# size (list) – 저장될 사이즈(ex : (640, 480)), 튜플

writer = cv2.VideoWriter("images/output.avi", fourcc, fps, size)
if writer.isOpened() == False:
    raise Exception("동영상파일 개방 안됨")

while True:
    # VideoCapture가 보내오는 싱글 이미지들을 frame으로 사용
    ret, frame = capture.read()
    # 싱글 이미지들이 계속 업데이트되면서 영상을 완성시키기 때문에 싱글프레임에 이미지 작업을 수행할 수 있음
    # 예를 들어 grayscale 영상을 만들고 싶다면
    # cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # write() : capture되는 frame을 저장하는 객체(writer)에 써주는 메서드
    writer.write(frame)

    cv2.imshow("View Frame from Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
capture.release()
cv2.destroyAllWindows()

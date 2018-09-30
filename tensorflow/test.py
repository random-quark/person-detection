import cv2
import time
#capture = cv2.VideoCapture("rtsp://admin:mrgooby@192.168.1.71:554/h264Preview_01_main")
#capture = cv2.VideoCapture("rtsp://admin:mrgooby@192.168.1.71:555/h264Preview_01_main")
capture = cv2.VideoCapture(0)
# rtmp://bcs/channel0_main.bcs?token=[TOKEN]&channel=0&stream=0
time.sleep(5)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# capture.set(CV_CAP_PROP_FOURCC)
    
while True:
    r, img = capture.read()
    print("{} x {}".format(capture.get(3), capture.get(4)))

    cv2.imshow("preview", img)
    cv2.moveWindow("preview", 0, 0)

    

    key = cv2.waitKey(1000)
    if key & 0xFF == ord('q'):
        break
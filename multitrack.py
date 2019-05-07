import cv2
import sys
selection = None
track_window = None
track_start = False
tracker1 = cv2.TrackerCSRT_create()
tracker2 = cv2.TrackerCSRT_create()
track_window =[(200,200,50,50),(400,400,50,50)]

#video = cv2.VideoCapture(0)
video = cv2.VideoCapture("F:\FFOutput\绿园树.mp4")
# 摄像头未正确打开则退出
if not video.isOpened():
    print('Could not open video')
    sys.exit()
while True:
    # 读取当前帧
    ok, frame = video.read()
    if not ok:
        print('视频没打开')
        break
    tracker1.init(frame, track_window[0])
    tracker2.init(frame, track_window[1])

    #print(track_)
    if track_window and track_window[0][2] > 0 and track_window[0][3] > 0:#开始跟踪
        track_start = True
        track_ok, box1 = tracker1.update(frame)
        print(box1)
    if track_window and track_window[1][2] > 0 and track_window[1][3] > 0:#开始跟踪
        track_start = True
        track_ok, box2 = tracker2.update(frame)
        print(box2)
    print('--------')
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
video.release()


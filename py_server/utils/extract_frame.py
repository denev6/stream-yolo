import cv2

cap = cv2.VideoCapture("../assets/test_12.mp4")

cap.set(cv2.CAP_PROP_POS_MSEC, 3100)

ret, frame = cap.read()
if ret:
    cv2.imwrite("../assets/test.jpg", frame)

cap.release()

import cv2

#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('D:\\py_lab\\py_lec_git\\03\\face.mp4')
#cap = cv2.VideoCapture("http://moebs.iptime.org:3333/mjpegfeed?640x480")     #  download app store droidcam app

while(cap.isOpened()):
    ret, frame = cap.read()
    if(ret):

        cv2.imshow('frame', frame)

    if cv2.waitKey(5) & 0xFF == 27: # press esc-key to exit
        break

cap.release()
cv2.destroyAllWindows()

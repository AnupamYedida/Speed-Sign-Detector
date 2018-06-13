import cv2
import sys

cascPath = sys.argv[1]
faceCascade = cv2.CascadeClassifier(cascPath)
font = cv2.FONT_HERSHEY_SIMPLEX

video_capture = cv2.VideoCapture('vid.mp4')

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 255, 0), 3)
        cv2.line(frame,(5,100),(x,y),(255,0,0),5)
        cv2.line(frame,(5,110),(700,110),(203,192,255),2)

        #cv2.line(frame,(x,y),(x-50,y),(255,0,0),3)
        #cv2.line(frame,(x,y),(x,y-50),(255,0,0),3)

        cv2.putText(frame,'Speed Sign Detected!',(5,100), font, 2,(25,25,255),4,cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

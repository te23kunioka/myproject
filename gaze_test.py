from gaze_tracking import GazeTracking
import cv2

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()

    gaze.refresh(frame)

    frame = gaze.annotated_frame()

    text = "CENTER"

    if gaze.is_right():
        text = "RIGHT"
    elif gaze.is_left():
        text = "LEFT"
    elif gaze.is_center():
        text = "CENTER"

    cv2.putText(
        frame,
        text,
        (20, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1.5,
        (0, 255, 0),
        2
    )

    cv2.imshow("Gaze Tracking", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()

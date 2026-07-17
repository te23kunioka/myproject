import cv2
import time

face_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

state = "OFF"

hold_start = None
ready = True

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gaze = "CENTER"

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:

            eyes = sorted(eyes, key=lambda e:e[0])

            ex, ey, ew, eh = eyes[1]

            cx = ex + ew/2

            if cx < w * 0.35:
                gaze = "LEFT"

            elif cx > w * 0.65:
                gaze = "RIGHT"

            else:
                gaze = "CENTER"

    now = time.time()

    if ready:

        if gaze == "RIGHT":

            if hold_start is None:
                hold_start = now

            if now - hold_start > 1.5:

                state = "ON"
                ready = False

                print("SWITCH ON")

        elif gaze == "LEFT":

            if hold_start is None:
                hold_start = now

            if now - hold_start > 1.5:

                state = "OFF"
                ready = False

                print("SWITCH OFF")

        else:
            hold_start = None

    if gaze == "CENTER":
        ready = True
        hold_start = None

    cv2.putText(
        frame,
        f"GAZE: {gaze}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"STATE: {state}",
        (20,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    cv2.imshow("Gaze Switch", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
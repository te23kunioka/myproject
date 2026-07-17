import cv2
import time

face_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

look_start = None
triggered = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        1.3,
        5
    )

    gaze = "CENTER"

    for (x,y,w,h) in faces:

        cv2.rectangle(
            frame,
            (x,y),
            (x+w,y+h),
            (0,255,0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:

            eyes = sorted(eyes, key=lambda e: e[0])

            ex,ey,ew,eh = eyes[1]

            center_x = ex + ew/2

            if center_x > w * 0.65:
                gaze = "RIGHT"
            elif center_x < w * 0.35:
                gaze = "LEFT"

            cv2.putText(
                frame,
                gaze,
                (20,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

    if gaze == "RIGHT":

        if look_start is None:
            look_start = time.time()

        elapsed = time.time() - look_start

        if elapsed > 1.0 and not triggered:

            print("SWITCH ON")

            triggered = True

    else:

        look_start = None

        if gaze == "CENTER":
            triggered = False

    cv2.imshow("Eye Switch", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
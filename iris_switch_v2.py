import cv2
import time
import winsound
from collections import deque

face_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

eye_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

center_start = None
switch_on = False

# 黒目位置の履歴
history = deque(maxlen=10)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3
    )

    # 顔が消えたらリセット
    if len(faces) == 0:

        switch_on = False
        center_start = None

        cv2.putText(
            frame,
            "NO FACE",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        cv2.imshow(
            "Eye Switch",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

        continue

    gaze = "UNKNOWN"

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

        eyes = eye_cascade.detectMultiScale(
            roi_gray
        )

        for (ex,ey,ew,eh) in eyes[:2]:

            cv2.rectangle(
                roi_color,
                (ex,ey),
                (ex+ew,ey+eh),
                (255,0,0),
                2
            )

            eye_gray = roi_gray[
                ey:ey+eh,
                ex:ex+ew
            ]

            eye_color = roi_color[
                ey:ey+eh,
                ex:ex+ew
            ]

            _, thresh = cv2.threshold(
                eye_gray,
                70,
                255,
                cv2.THRESH_BINARY_INV
            )

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:

                c = max(
                    contours,
                    key=cv2.contourArea
                )

                if cv2.contourArea(c) > 30:

                    M = cv2.moments(c)

                    if M["m00"] != 0:

                        cx = int(
                            M["m10"] /
                            M["m00"]
                        )

                        cy = int(
                            M["m01"] /
                            M["m00"]
                        )

                        cv2.circle(
                            eye_color,
                            (cx,cy),
                            5,
                            (0,0,255),
                            -1
                        )

                        history.append(
                            cx / ew
                        )

                        ratio = (
                            sum(history)
                            /
                            len(history)
                        )

                        if 0.10 <= ratio <= 0.90:
                            gaze = "CENTER"
                        elif ratio < 0.10:
                            gaze = "LEFT"
                        else:
                            gaze = "RIGHT"

        break

    current_time = time.time()

    if gaze == "CENTER":

        if center_start is None:
            center_start = current_time

        hold = current_time - center_start

        cv2.putText(
            frame,
            f"HOLD {hold:.1f}s",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,0),
            2
        )

        if hold >= 2.0 and not switch_on:

            switch_on = True

            print(
                "SWITCH ON"
            )

            winsound.Beep(
                1500,
                500
            )

    else:

        center_start = None

    state = (
        "ON"
        if switch_on
        else "OFF"
    )

    cv2.putText(
        frame,
        f"GAZE : {gaze}",
        (20,100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"STATE : {state}",
        (20,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    if switch_on:

        cv2.putText(
            frame,
            "SWITCH ON",
            (120,250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0,0,255),
            5
        )

    cv2.imshow(
        "Eye Switch",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import time
import winsound

# 顔検出
face_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

# 目検出
eye_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_eye.xml"
)

cap = cv2.VideoCapture(0)

center_start = None
switch_on = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3
    )

    gaze = "CENTER"

    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes[:2]:

            cv2.rectangle(
                roi_color,
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 0, 0),
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

            if len(contours) > 0:

                c = max(contours, key=cv2.contourArea)

                if cv2.contourArea(c) > 30:

                    M = cv2.moments(c)

                    if M["m00"] != 0:

                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])

                        cv2.circle(
                            eye_color,
                            (cx, cy),
                            5,
                            (0, 0, 255),
                            -1
                        )

                        ratio = cx / ew

                        if 0.15 <= ratio <= 0.85:
                            gaze = "CENTER"
                        elif ratio < 0.15:
                            gaze = "LEFT"
                        else:
                            gaze = "RIGHT"

        break

    current_time = time.time()

    if gaze == "CENTER":

        if center_start is None:
            center_start = current_time

        hold_time = current_time - center_start

        cv2.putText(
            frame,
            f"HOLD {hold_time:.1f}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        if hold_time >= 1.5 and not switch_on:

            switch_on = True

            print("SWITCH ON")

            # ビープ音
            winsound.Beep(1500, 500)

    else:

        center_start = None
        switch_on = False

    state = "ON" if switch_on else "OFF"

    cv2.putText(
        frame,
        f"GAZE : {gaze}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"STATE : {state}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if switch_on:

        cv2.putText(
            frame,
            "SWITCH ON",
            (150, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            5
        )

    cv2.imshow("Eye Switch", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
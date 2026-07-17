import cv2
import time
import pyttsx3
from collections import deque

# -------------------------
# 音声設定
# -------------------------

engine = pyttsx3.init()

engine.setProperty("rate", 220)
engine.setProperty("volume", 1.0)

# -------------------------
# 顔検出
# -------------------------

face_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

# -------------------------
# 目検出
# -------------------------

eye_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_eye.xml"
)

# -------------------------
# カメラ開始
# -------------------------

cap = cv2.VideoCapture(0)

# -------------------------
# 状態変数
# -------------------------

switch_on = False
center_start = None

# 黒目位置平均化
history = deque(maxlen=30)

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

    gaze = "AWAY"

    # -------------------------
    # 顔なし
    # -------------------------

    if len(faces) == 0:

        switch_on = False
        center_start = None
        history.clear()

        cv2.putText(
            frame,
            "NO FACE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            "STATE : OFF",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "PBL Switch",
            frame
        )

        if cv2.waitKey(1) == 27:
            break

        continue

    # -------------------------
    # 顔検出
    # -------------------------

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

        eyes = eye_cascade.detectMultiScale(
            roi_gray
        )

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

            if contours:

                c = max(
                    contours,
                    key=cv2.contourArea
                )

                if cv2.contourArea(c) > 20:

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
                            (cx, cy),
                            5,
                            (0, 0, 255),
                            -1
                        )

                        history.append(
                            cx / ew
                        )

                        ratio = (
                            sum(history) /
                            len(history)
                        )

                        # 黒目中央判定
                        if 0.10 <= ratio <= 0.90:
                            gaze = "CENTER"
                        else:
                            gaze = "AWAY"

        break

    # -------------------------
    # 中央2秒
    # -------------------------

    current_time = time.time()

    if gaze == "CENTER":

        if center_start is None:
            center_start = current_time

        hold_time = (
            current_time -
            center_start
        )

        cv2.putText(
            frame,
            f"HOLD {hold_time:.1f}s",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 0),
            2
        )

        # OFF → ON
        if hold_time >= 2.0 and not switch_on:

            switch_on = True

            print("SWITCH ON")

            engine.say("はい！")
            engine.runAndWait()

    else:

        center_start = None

    # -------------------------
    # 表示
    # -------------------------

    state = "ON" if switch_on else "OFF"

    cv2.putText(
        frame,
        f"GAZE : {gaze}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"STATE : {state}",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    if switch_on:

        cv2.putText(
            frame,
            "SWITCH ON",
            (80, 260),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            5
        )

    cv2.imshow(
        "PBL Switch",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
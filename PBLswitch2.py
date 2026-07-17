import cv2
import time
import pyttsx3

# 音声設定
engine = pyttsx3.init()
engine.setProperty("rate", 220)

# 顔検出
face_cascade = cv2.CascadeClassifier(
    r"venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

switch_on = False
start_time = None

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

    # 顔なし
    if len(faces) == 0:

        switch_on = False
        start_time = None

        state = "OFF"

    else:

        state = "OFF"

        if start_time is None:
            start_time = time.time()

        hold_time = time.time() - start_time

        if hold_time >= 2.0 and not switch_on:

            switch_on = True
            state = "ON"

            print("SWITCH ON")

            engine.say("はい！")
            engine.runAndWait()

        if switch_on:
            state = "ON"

    # 顔枠表示
    for (x, y, w, h) in faces:

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

    cv2.putText(
        frame,
        f"STATE : {state}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "PBLswitch2",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
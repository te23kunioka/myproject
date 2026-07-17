import cv2
import time

cascade_path = r"venv\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)

face_start = None
triggered = False

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) > 0:

        if face_start is None:
            face_start = time.time()

        elapsed = time.time() - face_start

        if elapsed > 1.0 and not triggered:
            print("SWITCH ON")
            triggered = True

        cv2.putText(
            frame,
            f"TIME:{elapsed:.1f}",
            (20,50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    else:
        face_start = None
        triggered = False

    cv2.imshow("Switch Test", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
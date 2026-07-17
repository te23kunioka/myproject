print("プログラム開始")
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1
) as face_mesh:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            face = results.multi_face_landmarks[0]

            iris = face.landmark[468]
            left_corner = face.landmark[33]
            right_corner = face.landmark[133]

            ratio = (
                (iris.x - left_corner.x) /
                (right_corner.x - left_corner.x)
            )

            if ratio < 0.38:
                text = "LEFT"
            elif ratio > 0.62:
                text = "RIGHT"
            else:
                text = "CENTER"

            cv2.putText(
                frame,
                text,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        cv2.imshow("Eye Control", frame)

        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
import cv2
import face_recognition

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera could not be opened.")
    exit()

print("Camera started! Press Q to quit.")

while True:
    ret, frame = camera.read()

    if not ret:
        print("Error: Could not read camera frame.")
        break

    # Convert BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)

    # Draw rectangles around detected faces
    for top, right, bottom, left in face_locations:
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

    cv2.imshow("FaceScan - Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Camera could not be opened")
    exit()

print("✅ Camera started")
print("Press Q to quit")

while True:
    ret, frame = camera.read()

    if not ret:
        print("❌ Failed to read camera")
        break

    cv2.imshow("Face Access System - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
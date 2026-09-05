import cv2
import mediapipe as mp
from deepface import DeepFace
import json
import time
import os
import math
import sys

from database import create_database, add_user


REGISTERED_DIR = "registered_faces"
IMAGES_PER_PERSON = 5
CAPTURE_INTERVAL = 3.0

os.makedirs(REGISTERED_DIR, exist_ok=True)

create_database()


# --------------------------------
# Get person name
# --------------------------------
if len(sys.argv) > 1:
    name = sys.argv[1].strip()
else:
    name = input("Enter person name: ").strip()


if not name:
    raise SystemExit("Name cannot be empty")


# Prevent invalid folder names
invalid_chars = '<>:"/\\|?*'

if any(char in name for char in invalid_chars):
    raise SystemExit(
        "Name contains invalid characters"
    )


person_dir = os.path.join(
    REGISTERED_DIR,
    name
)

os.makedirs(
    person_dir,
    exist_ok=True
)


# --------------------------------
# MediaPipe
# --------------------------------
BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
RunningMode = mp.tasks.vision.RunningMode


options = FaceDetectorOptions(
    base_options=BaseOptions(
        model_asset_path="models/face_detector.tflite"
    ),
    running_mode=RunningMode.VIDEO,
    min_detection_confidence=0.5
)


# --------------------------------
# Camera
# --------------------------------
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise SystemExit(
        "Could not open camera"
    )


start_time = time.time()

captured = 0
embeddings = []

registration_started = False
last_capture_time = None


print()
print(f"Registering: {name}")
print("Press S to start registration.")
print("Press Q to cancel.")
print()


# --------------------------------
# Registration
# --------------------------------
with FaceDetector.create_from_options(
    options
) as detector:

    while True:

        success, frame = camera.read()

        if not success:
            break


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        timestamp_ms = int(
            (time.time() - start_time) * 1000
        )


        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )


        face_crop = None
        face_detected = False


        if result.detections:

            face_detected = True

            detection = result.detections[0]

            bbox = detection.bounding_box


            x = max(
                bbox.origin_x,
                0
            )

            y = max(
                bbox.origin_y,
                0
            )

            x2 = min(
                x + bbox.width,
                frame.shape[1]
            )

            y2 = min(
                y + bbox.height,
                frame.shape[0]
            )


            face_crop = frame[
                y:y2,
                x:x2
            ]


            cv2.rectangle(
                frame,
                (x, y),
                (x2, y2),
                (0, 255, 0),
                2
            )


        # --------------------------------
        # Waiting
        # --------------------------------
        if not registration_started:

            cv2.putText(
                frame,
                f"Register: {name}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                "Press S to start",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


        # --------------------------------
        # Registration started
        # --------------------------------
        else:

            cv2.putText(
                frame,
                f"Captured: {captured}/{IMAGES_PER_PERSON}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            if last_capture_time is None:

                remaining = 0

            else:

                elapsed = (
                    time.time()
                    - last_capture_time
                )

                remaining = max(
                    0,
                    CAPTURE_INTERVAL - elapsed
                )


            if last_capture_time is None:

                message = "Get ready..."

            elif remaining > 0:

                countdown = math.ceil(
                    remaining
                )

                message = (
                    f"Next capture in: "
                    f"{countdown}"
                )

            else:

                message = "Capturing..."


            cv2.putText(
                frame,
                message,
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )


            ready_for_capture = (
                last_capture_time is None
                or
                (
                    time.time()
                    - last_capture_time
                    >= CAPTURE_INTERVAL
                )
            )


            if (
                ready_for_capture
                and face_detected
                and face_crop is not None
                and face_crop.size > 0
                and captured < IMAGES_PER_PERSON
            ):

                image_number = (
                    captured + 1
                )


                image_path = os.path.join(
                    person_dir,
                    f"{image_number}.jpg"
                )


                cv2.imwrite(
                    image_path,
                    face_crop
                )


                print(
                    f"Captured image "
                    f"{image_number}/"
                    f"{IMAGES_PER_PERSON}"
                )


                try:

                    result_embedding = (
                        DeepFace.represent(
                            img_path=face_crop,
                            model_name="Facenet512",
                            detector_backend="skip",
                            enforce_detection=False
                        )
                    )


                    embedding = (
                        result_embedding[0][
                            "embedding"
                        ]
                    )


                    embeddings.append(
                        embedding
                    )


                    captured += 1

                    last_capture_time = (
                        time.time()
                    )


                except Exception as e:

                    print(
                        "Embedding error:",
                        e
                    )


            if captured >= IMAGES_PER_PERSON:

                cv2.putText(
                    frame,
                    "Registration complete!",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )


                cv2.imshow(
                    "Register Face",
                    frame
                )

                cv2.waitKey(1000)

                break


        cv2.imshow(
            "Register Face",
            frame
        )


        key = (
            cv2.waitKey(1)
            & 0xFF
        )


        if key == ord("s"):

            if not registration_started:

                registration_started = True

                print(
                    "Registration started."
                )


        if key == ord("q"):
            break


camera.release()
cv2.destroyAllWindows()


# --------------------------------
# Save
# --------------------------------
if len(embeddings) == IMAGES_PER_PERSON:

    output_path = os.path.join(
        person_dir,
        "embeddings.json"
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            embeddings,
            f
        )


    add_user(name)


    print()
    print("==============================")
    print("Registration successful")
    print("==============================")

    print("Person:", name)
    print("Images:", IMAGES_PER_PERSON)
    print("Embeddings:", len(embeddings))


else:

    print(
        "Registration cancelled."
    )
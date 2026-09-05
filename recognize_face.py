import cv2
import mediapipe as mp

from deepface import DeepFace

import numpy as np
import json
import os
import time

from datetime import datetime

from database import (
    create_database,
    log_access as db_log_access,
    is_user_active
)

from settings import (
    load_settings
)


# ==========================================
# Paths
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

REGISTERED_DIR = os.path.join(
    BASE_DIR,
    "registered_faces"
)

SNAPSHOT_DIR = os.path.join(
    BASE_DIR,
    "access_snapshots"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "face_detector.tflite"
)


os.makedirs(
    SNAPSHOT_DIR,
    exist_ok=True
)


# ==========================================
# Settings
# ==========================================

SYSTEM_SETTINGS = load_settings()


THRESHOLD = float(
    SYSTEM_SETTINGS[
        "threshold"
    ]
)

MIN_MARGIN = float(
    SYSTEM_SETTINGS[
        "min_margin"
    ]
)

REQUIRED_STABLE_TIME = float(
    SYSTEM_SETTINGS[
        "stable_time"
    ]
)

LOG_COOLDOWN = int(
    SYSTEM_SETTINGS[
        "log_cooldown"
    ]
)

RECOGNITION_INTERVAL = int(
    SYSTEM_SETTINGS[
        "recognition_interval"
    ]
)

CAMERA_INDEX = int(
    SYSTEM_SETTINGS[
        "camera_index"
    ]
)


# Require multiple consecutive
# real-face checks before recognition.
LIVENESS_REQUIRED_CHECKS = 2


print()

print(
    "FaceAccess Settings"
)

print(
    "-------------------"
)

print(
    f"Threshold: {THRESHOLD}"
)

print(
    f"Minimum Margin: {MIN_MARGIN}"
)

print(
    "Stable Time: "
    f"{REQUIRED_STABLE_TIME}s"
)

print(
    "Log Cooldown: "
    f"{LOG_COOLDOWN}s"
)

print(
    "Recognition Interval: "
    f"{RECOGNITION_INTERVAL}"
)

print(
    "Camera Index: "
    f"{CAMERA_INDEX}"
)

print(
    "Passive Liveness: Enabled"
)

print(
    "Liveness Confirmations: "
    f"{LIVENESS_REQUIRED_CHECKS}"
)

print()


# ==========================================
# Database
# ==========================================

create_database()


# ==========================================
# Registered Faces
# ==========================================

def load_registered_faces():

    people = {}


    if not os.path.isdir(
        REGISTERED_DIR
    ):

        return people


    for person_name in os.listdir(
        REGISTERED_DIR
    ):

        person_dir = os.path.join(
            REGISTERED_DIR,
            person_name
        )


        if not os.path.isdir(
            person_dir
        ):

            continue


        embeddings_file = os.path.join(
            person_dir,
            "embeddings.json"
        )


        if not os.path.isfile(
            embeddings_file
        ):

            continue


        try:

            with open(
                embeddings_file,
                "r",
                encoding="utf-8"
            ) as file:

                embeddings = json.load(
                    file
                )


            if embeddings:

                people[
                    person_name
                ] = embeddings


        except Exception as error:

            print(
                "Could not load "
                f"{person_name}:",
                error
            )


    return people


registered_faces = (
    load_registered_faces()
)


print(
    "Registered people:",
    list(
        registered_faces.keys()
    )
)


# ==========================================
# Cosine Similarity
# ==========================================

def cosine_similarity(
    embedding1,
    embedding2
):

    vector1 = np.array(
        embedding1,
        dtype=np.float32
    )

    vector2 = np.array(
        embedding2,
        dtype=np.float32
    )


    denominator = (
        np.linalg.norm(
            vector1
        )
        *
        np.linalg.norm(
            vector2
        )
    )


    if denominator == 0:

        return 0.0


    return float(
        np.dot(
            vector1,
            vector2
        )
        /
        denominator
    )


# ==========================================
# Liveness
# ==========================================

def check_liveness(
    face_crop
):

    if (
        face_crop is None
        or
        face_crop.size == 0
    ):

        return (
            False,
            0.0
        )


    results = (
        DeepFace.extract_faces(

            img_path=
                face_crop,

            detector_backend=
                "skip",

            enforce_detection=
                False,

            align=
                False,

            anti_spoofing=
                True
        )
    )


    if not results:

        return (
            False,
            0.0
        )


    result = results[0]


    is_real = bool(
        result.get(
            "is_real",
            False
        )
    )


    try:

        score = float(
            result.get(
                "antispoof_score",
                0.0
            )
        )

    except (
        TypeError,
        ValueError
    ):

        score = 0.0


    return (
        is_real,
        score
    )


# ==========================================
# Snapshot
# ==========================================

def safe_filename(text):

    result = ""


    for character in str(
        text
    ):

        if (
            character.isalnum()
            or
            character in "-_"
        ):

            result += character

        else:

            result += "_"


    return result


def save_face_snapshot(
    face_crop,
    name,
    status
):

    if (
        face_crop is None
        or
        face_crop.size == 0
    ):

        return None


    now = datetime.now()


    date_folder = now.strftime(
        "%Y-%m-%d"
    )


    day_directory = os.path.join(
        SNAPSHOT_DIR,
        date_folder
    )


    os.makedirs(
        day_directory,
        exist_ok=True
    )


    safe_name = safe_filename(
        name
    )

    safe_status = safe_filename(
        status
    )


    filename = (
        now.strftime(
            "%H%M%S_%f"
        )
        +
        f"_{safe_status}_"
        +
        f"{safe_name}.jpg"
    )


    full_path = os.path.join(
        day_directory,
        filename
    )


    saved = cv2.imwrite(
        full_path,
        face_crop
    )


    if not saved:

        return None


    relative_path = os.path.join(
        date_folder,
        filename
    )


    return relative_path.replace(
        "\\",
        "/"
    )


# ==========================================
# Access Logging
# ==========================================

last_logged_key = None

last_log_time = 0.0


def log_event(
    name,
    status,
    similarity,
    face_crop
):

    global last_logged_key
    global last_log_time


    current_time = time.time()


    key = (
        name,
        status
    )


    should_log = False


    if last_logged_key != key:

        should_log = True


    elif (
        current_time
        -
        last_log_time
        >=
        LOG_COOLDOWN
    ):

        should_log = True


    if not should_log:

        return


    image_path = (
        save_face_snapshot(
            face_crop,
            name,
            status
        )
    )


    db_log_access(
        name,
        status,
        similarity,
        image_path
    )


    last_logged_key = key

    last_log_time = current_time


# ==========================================
# MediaPipe
# ==========================================

BaseOptions = (
    mp.tasks.BaseOptions
)

FaceDetector = (
    mp.tasks.vision.FaceDetector
)

FaceDetectorOptions = (
    mp.tasks.vision
    .FaceDetectorOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)


options = FaceDetectorOptions(

    base_options=BaseOptions(
        model_asset_path=
            MODEL_PATH
    ),

    running_mode=
        RunningMode.VIDEO,

    min_detection_confidence=
        0.5
)


# ==========================================
# Camera
# ==========================================

camera = cv2.VideoCapture(
    CAMERA_INDEX
)


if not camera.isOpened():

    raise SystemExit(
        "Could not open camera "
        f"index {CAMERA_INDEX}."
    )


start_time = time.time()

frame_count = 0


# ==========================================
# Recognition State
# ==========================================

candidate_name = None

candidate_start_time = None


liveness_streak = 0

last_liveness_score = 0.0


display_text = (
    "Waiting for face..."
)

display_color = (
    255,
    255,
    255
)


# ==========================================
# Recognition Loop
# ==========================================

with FaceDetector.create_from_options(
    options
) as detector:


    while True:


        success, frame = (
            camera.read()
        )


        if not success:

            break


        frame_count += 1


        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=
                mp.ImageFormat.SRGB,

            data=
                rgb_frame
        )


        timestamp_ms = int(
            (
                time.time()
                -
                start_time
            )
            *
            1000
        )


        detection_result = (
            detector.detect_for_video(
                mp_image,
                timestamp_ms
            )
        )


        face_crop = None

        face_detected = False


        if detection_result.detections:


            face_detected = True


            detection = (
                detection_result
                .detections[0]
            )


            bbox = (
                detection.bounding_box
            )


            x1 = max(
                int(
                    bbox.origin_x
                ),
                0
            )

            y1 = max(
                int(
                    bbox.origin_y
                ),
                0
            )

            x2 = min(
                x1
                +
                int(
                    bbox.width
                ),
                frame.shape[1]
            )

            y2 = min(
                y1
                +
                int(
                    bbox.height
                ),
                frame.shape[0]
            )


            face_crop = frame[
                y1:y2,
                x1:x2
            ].copy()


            cv2.rectangle(
                frame,
                (
                    x1,
                    y1
                ),
                (
                    x2,
                    y2
                ),
                (
                    255,
                    255,
                    255
                ),
                2
            )


        # ==================================
        # Recognition Cycle
        # ==================================

        if (
            face_detected
            and
            face_crop is not None
            and
            face_crop.size > 0
            and
            frame_count
            %
            RECOGNITION_INTERVAL
            ==
            0
        ):


            # ==============================
            # Liveness Check
            # ==============================

            try:

                (
                    is_real,
                    liveness_score
                ) = check_liveness(
                    face_crop
                )


                last_liveness_score = (
                    liveness_score
                )


                if not is_real:


                    liveness_streak = 0

                    candidate_name = None

                    candidate_start_time = None


                    display_text = (
                        "Spoof Detected - "
                        "Access Denied "
                        f"(L:{liveness_score:.3f})"
                    )


                    display_color = (
                        0,
                        0,
                        255
                    )


                    log_event(
                        "Spoof Attempt",
                        "Denied",
                        0.0,
                        face_crop
                    )


                    # Do NOT perform
                    # face recognition
                    # after spoof failure.
                    continue


                liveness_streak += 1


                if (
                    liveness_streak
                    >
                    LIVENESS_REQUIRED_CHECKS
                ):

                    liveness_streak = (
                        LIVENESS_REQUIRED_CHECKS
                    )


                if (
                    liveness_streak
                    <
                    LIVENESS_REQUIRED_CHECKS
                ):


                    candidate_name = None

                    candidate_start_time = None


                    display_text = (
                        "Checking Liveness "
                        f"{liveness_streak}/"
                        f"{LIVENESS_REQUIRED_CHECKS} "
                        f"(L:{liveness_score:.3f})"
                    )


                    display_color = (
                        0,
                        255,
                        255
                    )


                    continue


            except Exception as error:


                print(
                    "Liveness error:",
                    error
                )


                liveness_streak = 0

                candidate_name = None

                candidate_start_time = None


                display_text = (
                    "Liveness Check Failed - "
                    "Access Denied"
                )


                display_color = (
                    0,
                    0,
                    255
                )


                log_event(
                    "Liveness Error",
                    "Denied",
                    0.0,
                    face_crop
                )


                # Fail closed:
                # never grant access if
                # liveness cannot be checked.
                continue


            # ==============================
            # Face Recognition
            # ==============================

            try:


                representation = (
                    DeepFace.represent(

                        img_path=
                            face_crop,

                        model_name=
                            "Facenet512",

                        detector_backend=
                            "skip",

                        enforce_detection=
                            False
                    )
                )


                current_embedding = (
                    representation[0][
                        "embedding"
                    ]
                )


                scores = []


                for (
                    person_name,
                    person_embeddings
                ) in (
                    registered_faces.items()
                ):


                    person_scores = []


                    for saved_embedding in (
                        person_embeddings
                    ):


                        similarity = (
                            cosine_similarity(
                                current_embedding,
                                saved_embedding
                            )
                        )


                        person_scores.append(
                            similarity
                        )


                    if person_scores:


                        person_score = max(
                            person_scores
                        )


                        scores.append(
                            (
                                person_name,
                                person_score
                            )
                        )


                scores.sort(
                    key=lambda item:
                        item[1],
                    reverse=True
                )


                if scores:


                    best_name = (
                        scores[0][0]
                    )


                    best_similarity = (
                        scores[0][1]
                    )


                    if len(scores) > 1:

                        second_similarity = (
                            scores[1][1]
                        )

                    else:

                        second_similarity = 0.0


                    margin = (
                        best_similarity
                        -
                        second_similarity
                    )


                else:


                    best_name = (
                        "Unknown"
                    )

                    best_similarity = 0.0

                    margin = 0.0


                valid_match = (

                    best_similarity
                    >=
                    THRESHOLD

                    and

                    margin
                    >=
                    MIN_MARGIN
                )


                # ==========================
                # Valid Identity
                # ==========================

                if valid_match:


                    if (
                        candidate_name
                        !=
                        best_name
                    ):


                        candidate_name = (
                            best_name
                        )


                        candidate_start_time = (
                            time.time()
                        )


                    stable_time = (
                        time.time()
                        -
                        candidate_start_time
                    )


                    if (
                        stable_time
                        >=
                        REQUIRED_STABLE_TIME
                    ):


                        if is_user_active(
                            best_name
                        ):


                            display_text = (
                                f"{best_name} - "
                                "Access Granted "
                                f"({best_similarity:.3f}) "
                                f"[Live:{last_liveness_score:.3f}]"
                            )


                            display_color = (
                                0,
                                255,
                                0
                            )


                            log_event(
                                best_name,
                                "Granted",
                                best_similarity,
                                face_crop
                            )


                        else:


                            display_text = (
                                f"{best_name} - "
                                "Access Disabled "
                                f"({best_similarity:.3f}) "
                                f"[Live:{last_liveness_score:.3f}]"
                            )


                            display_color = (
                                0,
                                0,
                                255
                            )


                            log_event(
                                best_name,
                                "Disabled",
                                best_similarity,
                                face_crop
                            )


                    else:


                        display_text = (
                            "Verifying "
                            f"{best_name} "
                            f"{stable_time:.1f}/"
                            f"{REQUIRED_STABLE_TIME:.1f}s "
                            f"({best_similarity:.3f}) "
                            f"[Live:{last_liveness_score:.3f}]"
                        )


                        display_color = (
                            0,
                            255,
                            255
                        )


                # ==========================
                # Unknown Person
                # ==========================

                else:


                    candidate_name = None

                    candidate_start_time = None


                    display_text = (
                        "Unknown - "
                        "Access Denied "
                        f"({best_similarity:.3f}) "
                        f"[Live:{last_liveness_score:.3f}]"
                    )


                    display_color = (
                        0,
                        0,
                        255
                    )


                    log_event(
                        "Unknown",
                        "Denied",
                        best_similarity,
                        face_crop
                    )


            except Exception as error:


                print(
                    "Recognition error:",
                    error
                )


                candidate_name = None

                candidate_start_time = None


        # ==================================
        # No Face
        # ==================================

        if not face_detected:


            candidate_name = None

            candidate_start_time = None

            liveness_streak = 0

            last_liveness_score = 0.0


            display_text = (
                "Waiting for face..."
            )


            display_color = (
                255,
                255,
                255
            )


        # ==================================
        # Top Status Bar
        # ==================================

        cv2.rectangle(
            frame,
            (
                0,
                0
            ),
            (
                frame.shape[1],
                70
            ),
            (
                20,
                20,
                20
            ),
            -1
        )


        cv2.putText(
            frame,
            display_text,
            (
                15,
                42
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            display_color,
            2
        )


        cv2.imshow(
            "Face Access System",
            frame
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            break


camera.release()

cv2.destroyAllWindows()
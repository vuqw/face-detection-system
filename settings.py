import json
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

SETTINGS_FILE = os.path.join(
    BASE_DIR,
    "settings.json"
)


DEFAULT_SETTINGS = {
    "threshold": 0.90,
    "min_margin": 0.05,
    "stable_time": 3.0,
    "log_cooldown": 10,
    "recognition_interval": 15,
    "camera_index": 0
}


def get_default_settings():

    return DEFAULT_SETTINGS.copy()


def validate_settings(data):

    settings = {}


    try:

        threshold = float(
            data.get(
                "threshold",
                DEFAULT_SETTINGS["threshold"]
            )
        )

    except (
        TypeError,
        ValueError
    ):

        threshold = (
            DEFAULT_SETTINGS[
                "threshold"
            ]
        )


    try:

        min_margin = float(
            data.get(
                "min_margin",
                DEFAULT_SETTINGS[
                    "min_margin"
                ]
            )
        )

    except (
        TypeError,
        ValueError
    ):

        min_margin = (
            DEFAULT_SETTINGS[
                "min_margin"
            ]
        )


    try:

        stable_time = float(
            data.get(
                "stable_time",
                DEFAULT_SETTINGS[
                    "stable_time"
                ]
            )
        )

    except (
        TypeError,
        ValueError
    ):

        stable_time = (
            DEFAULT_SETTINGS[
                "stable_time"
            ]
        )


    try:

        log_cooldown = int(
            data.get(
                "log_cooldown",
                DEFAULT_SETTINGS[
                    "log_cooldown"
                ]
            )
        )

    except (
        TypeError,
        ValueError
    ):

        log_cooldown = (
            DEFAULT_SETTINGS[
                "log_cooldown"
            ]
        )


    try:

        recognition_interval = int(
            data.get(
                "recognition_interval",
                DEFAULT_SETTINGS[
                    "recognition_interval"
                ]
            )
        )

    except (
        TypeError,
        ValueError
    ):

        recognition_interval = (
            DEFAULT_SETTINGS[
                "recognition_interval"
            ]
        )


    try:

        camera_index = int(
            data.get(
                "camera_index",
                DEFAULT_SETTINGS[
                    "camera_index"
                ]
            )
        )

    except (
        TypeError,
        ValueError
    ):

        camera_index = (
            DEFAULT_SETTINGS[
                "camera_index"
            ]
        )


    threshold = max(
        0.50,
        min(
            threshold,
            1.00
        )
    )


    min_margin = max(
        0.00,
        min(
            min_margin,
            0.50
        )
    )


    stable_time = max(
        0.50,
        min(
            stable_time,
            15.00
        )
    )


    log_cooldown = max(
        1,
        min(
            log_cooldown,
            300
        )
    )


    recognition_interval = max(
        1,
        min(
            recognition_interval,
            120
        )
    )


    camera_index = max(
        0,
        min(
            camera_index,
            10
        )
    )


    settings[
        "threshold"
    ] = round(
        threshold,
        3
    )


    settings[
        "min_margin"
    ] = round(
        min_margin,
        3
    )


    settings[
        "stable_time"
    ] = round(
        stable_time,
        2
    )


    settings[
        "log_cooldown"
    ] = log_cooldown


    settings[
        "recognition_interval"
    ] = recognition_interval


    settings[
        "camera_index"
    ] = camera_index


    return settings


def save_settings(data):

    settings = validate_settings(
        data
    )


    with open(
        SETTINGS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            settings,
            file,
            indent=4
        )


    return settings


def load_settings():

    if not os.path.isfile(
        SETTINGS_FILE
    ):

        return save_settings(
            DEFAULT_SETTINGS
        )


    try:

        with open(
            SETTINGS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        return validate_settings(
            data
        )


    except (
        OSError,
        json.JSONDecodeError
    ):

        return save_settings(
            DEFAULT_SETTINGS
        )


def reset_settings():

    return save_settings(
        DEFAULT_SETTINGS
    )


if __name__ == "__main__":

    print(
        load_settings()
    )
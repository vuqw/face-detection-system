import json
import os
import secrets

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ADMIN_FILE = os.path.join(
    BASE_DIR,
    "admin_account.json"
)

SECRET_FILE = os.path.join(
    BASE_DIR,
    ".faceaccess_secret"
)


# ==========================================
# Secret Key
# ==========================================

def get_secret_key():

    if os.path.isfile(SECRET_FILE):

        try:

            with open(
                SECRET_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                secret = file.read().strip()

            if secret:
                return secret

        except OSError:
            pass


    secret = secrets.token_urlsafe(64)


    with open(
        SECRET_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(secret)


    return secret


# ==========================================
# Admin
# ==========================================

def admin_exists():

    return os.path.isfile(
        ADMIN_FILE
    )


def load_admin():

    if not admin_exists():
        return None


    try:

        with open(
            ADMIN_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)


        username = str(
            data.get(
                "username",
                ""
            )
        ).strip()


        password_hash = str(
            data.get(
                "password_hash",
                ""
            )
        ).strip()


        try:

            auth_version = int(
                data.get(
                    "auth_version",
                    1
                )
            )

        except (
            TypeError,
            ValueError
        ):

            auth_version = 1


        if (
            not username
            or
            not password_hash
        ):

            return None


        return {
            "username":
                username,

            "password_hash":
                password_hash,

            "auth_version":
                auth_version
        }


    except (
        OSError,
        json.JSONDecodeError
    ):

        return None


def save_admin(data):

    with open(
        ADMIN_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


def create_admin(
    username,
    password
):

    username = str(
        username
    ).strip()

    password = str(
        password
    )


    if len(username) < 3:

        raise ValueError(
            "Username must be at least 3 characters."
        )


    if len(password) < 8:

        raise ValueError(
            "Password must be at least 8 characters."
        )


    data = {
        "username":
            username,

        "password_hash":
            generate_password_hash(
                password
            ),

        "auth_version":
            1
    }


    save_admin(data)

    return True


def verify_admin(
    username,
    password
):

    admin = load_admin()


    if admin is None:
        return False


    if (
        str(username).strip()
        !=
        admin["username"]
    ):

        return False


    return check_password_hash(
        admin["password_hash"],
        str(password)
    )


def get_auth_version():

    admin = load_admin()


    if admin is None:
        return None


    return int(
        admin["auth_version"]
    )


def change_admin_password(
    current_password,
    new_password
):

    admin = load_admin()


    if admin is None:

        raise ValueError(
            "Admin account was not found."
        )


    if not check_password_hash(
        admin["password_hash"],
        str(current_password)
    ):

        raise ValueError(
            "Current password is incorrect."
        )


    new_password = str(
        new_password
    )


    if len(new_password) < 8:

        raise ValueError(
            "New password must be at least 8 characters."
        )


    if check_password_hash(
        admin["password_hash"],
        new_password
    ):

        raise ValueError(
            "New password must be different from the current password."
        )


    admin[
        "password_hash"
    ] = generate_password_hash(
        new_password
    )


    admin[
        "auth_version"
    ] = (
        int(
            admin.get(
                "auth_version",
                1
            )
        )
        +
        1
    )


    save_admin(
        admin
    )


    return int(
        admin[
            "auth_version"
        ]
    )
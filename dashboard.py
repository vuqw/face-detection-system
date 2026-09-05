from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    request,
    flash,
    jsonify,
    send_from_directory,
    abort,
    session
)

from database import (
    create_database,
    get_users,
    get_logs,
    get_log,
    delete_log,
    enable_user,
    disable_user,
    delete_user
)

from settings import (
    load_settings,
    save_settings,
    reset_settings
)

from auth import (
    admin_exists,
    load_admin,
    create_admin,
    verify_admin,
    get_secret_key,
    get_auth_version,
    change_admin_password
)

from datetime import (
    datetime,
    timedelta
)

import subprocess
import sys
import os
import shutil


# ==========================================
# Flask
# ==========================================

app = Flask(__name__)

app.secret_key = (
    get_secret_key()
)


app.config.update(

    SESSION_COOKIE_HTTPONLY=True,

    SESSION_COOKIE_SAMESITE="Lax",

    SESSION_COOKIE_SECURE=False,

    PERMANENT_SESSION_LIFETIME=
        timedelta(
            hours=8
        )
)


create_database()


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


os.makedirs(
    SNAPSHOT_DIR,
    exist_ok=True
)


recognition_process = None


# ==========================================
# Authentication
# ==========================================

@app.before_request
def protect_application():

    endpoint = request.endpoint


    if endpoint == "static":

        return None


    # No admin yet
    if not admin_exists():

        if endpoint == "setup":
            return None

        return redirect(
            url_for(
                "setup"
            )
        )


    # Setup is disabled after
    # account creation
    if endpoint == "setup":

        return redirect(
            url_for(
                "login"
            )
        )


    # Login is public
    if endpoint == "login":

        return None


    # Not logged in
    if not session.get(
        "authenticated"
    ):

        if request.path.startswith(
            "/api/"
        ):

            return jsonify({
                "error":
                    "Authentication required"
            }), 401


        return redirect(
            url_for(
                "login"
            )
        )


    # ======================================
    # Session Version Check
    # ======================================

    current_version = (
        get_auth_version()
    )


    session_version = (
        session.get(
            "auth_version"
        )
    )


    if (
        current_version is None
        or
        session_version
        !=
        current_version
    ):

        session.clear()


        if request.path.startswith(
            "/api/"
        ):

            return jsonify({
                "error":
                    "Session expired"
            }), 401


        flash(
            "Your session has expired. Please sign in again."
        )


        return redirect(
            url_for(
                "login"
            )
        )


    return None


# ==========================================
# Setup
# ==========================================

@app.route(
    "/setup",
    methods=[
        "GET",
        "POST"
    ]
)
def setup():

    if admin_exists():

        return redirect(
            url_for(
                "login"
            )
        )


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        if password != confirm_password:

            flash(
                "Passwords do not match."
            )

            return redirect(
                url_for(
                    "setup"
                )
            )


        try:

            create_admin(
                username,
                password
            )

        except ValueError as error:

            flash(
                str(error)
            )

            return redirect(
                url_for(
                    "setup"
                )
            )


        admin = load_admin()


        session.clear()

        session.permanent = True

        session[
            "authenticated"
        ] = True

        session[
            "username"
        ] = admin[
            "username"
        ]

        session[
            "auth_version"
        ] = admin[
            "auth_version"
        ]


        flash(
            "Admin account created successfully."
        )


        return redirect(
            url_for(
                "dashboard"
            )
        )


    return render_template(
        "setup.html"
    )


# ==========================================
# Login
# ==========================================

@app.route(
    "/login",
    methods=[
        "GET",
        "POST"
    ]
)
def login():

    if not admin_exists():

        return redirect(
            url_for(
                "setup"
            )
        )


    if session.get(
        "authenticated"
    ):

        current_version = (
            get_auth_version()
        )

        if (
            session.get(
                "auth_version"
            )
            ==
            current_version
        ):

            return redirect(
                url_for(
                    "dashboard"
                )
            )


        session.clear()


    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        )

        password = request.form.get(
            "password",
            ""
        )


        if verify_admin(
            username,
            password
        ):

            admin = load_admin()


            session.clear()

            session.permanent = True

            session[
                "authenticated"
            ] = True

            session[
                "username"
            ] = admin[
                "username"
            ]

            session[
                "auth_version"
            ] = admin[
                "auth_version"
            ]


            return redirect(
                url_for(
                    "dashboard"
                )
            )


        flash(
            "Incorrect username or password."
        )


    return render_template(
        "login.html"
    )


# ==========================================
# Logout
# ==========================================

@app.route(
    "/logout",
    methods=["POST"]
)
def logout():

    session.clear()


    return redirect(
        url_for(
            "login"
        )
    )


# ==========================================
# Account
# ==========================================

@app.route(
    "/account",
    methods=[
        "GET",
        "POST"
    ]
)
def account_page():

    admin = load_admin()


    if admin is None:

        abort(404)


    if request.method == "POST":

        current_password = (
            request.form.get(
                "current_password",
                ""
            )
        )

        new_password = (
            request.form.get(
                "new_password",
                ""
            )
        )

        confirm_password = (
            request.form.get(
                "confirm_password",
                ""
            )
        )


        if new_password != confirm_password:

            flash(
                "New passwords do not match."
            )

            return redirect(
                url_for(
                    "account_page"
                )
            )


        try:

            change_admin_password(
                current_password,
                new_password
            )

        except ValueError as error:

            flash(
                str(error)
            )

            return redirect(
                url_for(
                    "account_page"
                )
            )


        session.clear()


        flash(
            "Password changed successfully. Please sign in again."
        )


        return redirect(
            url_for(
                "login"
            )
        )


    return render_template(
        "account.html",
        admin_username=
            admin[
                "username"
            ]
    )


# ==========================================
# Recognition Process
# ==========================================

def recognition_is_running():

    global recognition_process


    if recognition_process is None:

        return False


    return (
        recognition_process.poll()
        is None
    )


def start_recognition_process():

    global recognition_process


    recognition_script = os.path.join(
        BASE_DIR,
        "recognize_face.py"
    )


    recognition_process = (
        subprocess.Popen(
            [
                sys.executable,
                recognition_script
            ],
            cwd=BASE_DIR
        )
    )


def stop_recognition_process():

    global recognition_process


    if not recognition_is_running():

        recognition_process = None

        return


    recognition_process.terminate()


    try:

        recognition_process.wait(
            timeout=3
        )

    except subprocess.TimeoutExpired:

        recognition_process.kill()

        recognition_process.wait()


    recognition_process = None


def restart_recognition_process():

    was_running = (
        recognition_is_running()
    )


    if was_running:

        stop_recognition_process()

        start_recognition_process()


    return was_running


# ==========================================
# Helpers
# ==========================================

def get_user_by_name(name):

    for user in get_users():

        if str(
            user[1]
        ) == str(
            name
        ):

            return user


    return None


def get_image_url(log):

    if (
        len(log) < 6
        or
        not log[5]
    ):

        return None


    return url_for(
        "access_snapshot",
        filename=log[5]
    )


def log_to_dict(log):

    return {

        "id":
            log[0],

        "name":
            log[1],

        "timestamp":
            log[2],

        "status":
            log[3],

        "similarity":
            round(
                float(
                    log[4]
                ),
                3
            ),

        "image_url":
            get_image_url(
                log
            )
    }


def get_registered_images(name):

    images = []


    registered_root = os.path.realpath(
        REGISTERED_DIR
    )


    person_dir = os.path.realpath(
        os.path.join(
            REGISTERED_DIR,
            name
        )
    )


    try:

        if os.path.commonpath([
            registered_root,
            person_dir
        ]) != registered_root:

            return []

    except ValueError:

        return []


    for number in range(
        1,
        6
    ):

        filename = (
            f"{number}.jpg"
        )


        image_path = os.path.join(
            person_dir,
            filename
        )


        if os.path.isfile(
            image_path
        ):

            images.append({

                "number":
                    number,

                "url":
                    url_for(
                        "registered_face_image",
                        name=name,
                        filename=filename
                    )
            })


    return images


def delete_snapshot_file(
    relative_path
):

    if not relative_path:

        return


    snapshot_root = os.path.realpath(
        SNAPSHOT_DIR
    )


    full_path = os.path.realpath(
        os.path.join(
            SNAPSHOT_DIR,
            relative_path
        )
    )


    try:

        if os.path.commonpath([
            snapshot_root,
            full_path
        ]) != snapshot_root:

            return

    except ValueError:

        return


    if os.path.isfile(
        full_path
    ):

        try:

            os.remove(
                full_path
            )

        except OSError as error:

            print(
                "Could not delete snapshot:",
                error
            )


# ==========================================
# Images
# ==========================================

@app.route(
    "/access-snapshots/<path:filename>"
)
def access_snapshot(filename):

    return send_from_directory(
        SNAPSHOT_DIR,
        filename
    )


@app.route(
    "/registered-face/<name>/<filename>"
)
def registered_face_image(
    name,
    filename
):

    if filename not in {
        "1.jpg",
        "2.jpg",
        "3.jpg",
        "4.jpg",
        "5.jpg"
    }:

        abort(404)


    if get_user_by_name(
        name
    ) is None:

        abort(404)


    person_dir = os.path.realpath(
        os.path.join(
            REGISTERED_DIR,
            name
        )
    )


    registered_root = os.path.realpath(
        REGISTERED_DIR
    )


    try:

        if os.path.commonpath([
            registered_root,
            person_dir
        ]) != registered_root:

            abort(404)

    except ValueError:

        abort(404)


    return send_from_directory(
        person_dir,
        filename
    )


# ==========================================
# Dashboard Stats
# ==========================================

def build_daily_stats(logs):

    today = datetime.now().date()

    days = []


    for i in range(
        6,
        -1,
        -1
    ):

        day = (
            today
            -
            timedelta(
                days=i
            )
        )


        days.append({

            "date":
                day.strftime(
                    "%Y-%m-%d"
                ),

            "label":
                day.strftime(
                    "%a"
                ),

            "granted":
                0,

            "denied":
                0
        })


    indexed_days = {

        day["date"]:
            day

        for day in days
    }


    for log in logs:

        log_date = str(
            log[2]
        )[:10]


        if log_date not in indexed_days:
            continue


        if log[3] == "Granted":

            indexed_days[
                log_date
            ]["granted"] += 1


        elif log[3] == "Denied":

            indexed_days[
                log_date
            ]["denied"] += 1


    return days


# ==========================================
# Security Alerts
# ==========================================

def build_security_alerts(
    logs,
    last_seen_id
):

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )


    denied_today = [

        log

        for log in logs

        if (
            str(
                log[3]
            )
            ==
            "Denied"

            and

            str(
                log[2]
            ).startswith(
                today
            )
        )
    ]


    unread = [

        log

        for log in denied_today

        if int(
            log[0]
        ) > last_seen_id
    ]


    latest = None


    if denied_today:

        latest = log_to_dict(
            denied_today[0]
        )


    return {

        "today_count":
            len(
                denied_today
            ),

        "unread_count":
            len(
                unread
            ),

        "latest":
            latest,

        "latest_id":
            denied_today[0][0]
            if denied_today
            else 0
    }


# ==========================================
# Dashboard
# ==========================================

@app.route("/")
def dashboard():

    users = get_users()

    logs = get_logs()


    return render_template(

        "dashboard.html",

        total_users=
            len(
                users
            ),

        active_users=
            sum(
                1
                for user in users
                if user[3] == 1
            ),

        granted_count=
            sum(
                1
                for log in logs
                if log[3]
                ==
                "Granted"
            ),

        denied_count=
            sum(
                1
                for log in logs
                if log[3]
                ==
                "Denied"
            ),

        recognition_running=
            recognition_is_running(),

        admin_username=
            session.get(
                "username",
                "Admin"
            )
    )


@app.route(
    "/api/dashboard"
)
def dashboard_api():

    users = get_users()

    logs = get_logs()


    try:

        last_seen_id = int(
            request.args.get(
                "last_seen_denied_id",
                "0"
            )
        )

    except ValueError:

        last_seen_id = 0


    return jsonify({

        "stats": {

            "total_users":
                len(
                    users
                ),

            "active_users":
                sum(
                    1
                    for user in users
                    if user[3] == 1
                ),

            "granted_count":
                sum(
                    1
                    for log in logs
                    if log[3]
                    ==
                    "Granted"
                ),

            "denied_count":
                sum(
                    1
                    for log in logs
                    if log[3]
                    ==
                    "Denied"
                )
        },

        "recognition_running":
            recognition_is_running(),

        "security_alerts":
            build_security_alerts(
                logs,
                last_seen_id
            ),

        "daily_stats":
            build_daily_stats(
                logs
            ),

        "users": [

            {
                "id":
                    user[0],

                "name":
                    user[1],

                "created_at":
                    user[2],

                "active":
                    bool(
                        user[3]
                    )
            }

            for user in users
        ],

        "logs": [

            log_to_dict(
                log
            )

            for log in logs[:20]
        ]
    })


# ==========================================
# Settings
# ==========================================

@app.route(
    "/settings",
    methods=[
        "GET",
        "POST"
    ]
)
def settings_page():

    if request.method == "POST":

        data = {

            "threshold":
                request.form.get(
                    "threshold"
                ),

            "min_margin":
                request.form.get(
                    "min_margin"
                ),

            "stable_time":
                request.form.get(
                    "stable_time"
                ),

            "log_cooldown":
                request.form.get(
                    "log_cooldown"
                ),

            "recognition_interval":
                request.form.get(
                    "recognition_interval"
                ),

            "camera_index":
                request.form.get(
                    "camera_index"
                )
        }


        save_settings(data)


        was_running = (
            restart_recognition_process()
        )


        if was_running:

            flash(
                "Settings saved and recognition restarted."
            )

        else:

            flash(
                "Settings saved successfully."
            )


        return redirect(
            url_for(
                "settings_page"
            )
        )


    settings = load_settings()


    return render_template(
        "settings.html",
        settings=settings,
        recognition_running=
            recognition_is_running()
    )


@app.route(
    "/settings/reset",
    methods=["POST"]
)
def settings_reset():

    reset_settings()


    was_running = (
        restart_recognition_process()
    )


    if was_running:

        flash(
            "Settings reset to defaults and recognition restarted."
        )

    else:

        flash(
            "Settings reset to defaults."
        )


    return redirect(
        url_for(
            "settings_page"
        )
    )


# ==========================================
# Delete Log
# ==========================================

@app.route(
    "/log/<int:log_id>/delete",
    methods=["POST"]
)
def delete_access_log(
    log_id
):

    log = get_log(
        log_id
    )


    if log is None:

        flash(
            "Access log was not found."
        )

        return redirect(
            url_for(
                "access_logs"
            )
        )


    image_path = (
        log[5]
        if len(log) >= 6
        else None
    )


    person_name = str(
        log[1]
    )


    if delete_log(
        log_id
    ):

        delete_snapshot_file(
            image_path
        )

        flash(
            f"Access log #{log_id} deleted."
        )


    return_to = request.form.get(
        "return_to",
        "logs"
    )


    if return_to == "dashboard":

        return redirect(
            url_for(
                "dashboard"
            )
        )


    if (
        return_to == "profile"
        and
        person_name != "Unknown"
        and
        get_user_by_name(
            person_name
        )
        is not None
    ):

        return redirect(
            url_for(
                "user_profile",
                name=person_name
            )
        )


    return redirect(
        url_for(
            "access_logs"
        )
    )


# ==========================================
# Logs
# ==========================================

@app.route(
    "/logs"
)
def access_logs():

    return render_template(
        "access_logs.html"
    )


@app.route(
    "/api/logs"
)
def logs_api():

    logs = get_logs()


    search = request.args.get(
        "search",
        ""
    ).strip().lower()


    status_filter = request.args.get(
        "status",
        "all"
    ).strip()


    date_filter = request.args.get(
        "date",
        ""
    ).strip()


    if status_filter not in {
        "all",
        "Granted",
        "Denied",
        "Disabled"
    }:

        status_filter = "all"


    filtered = []


    for log in logs:

        name = str(
            log[1]
        )

        timestamp = str(
            log[2]
        )

        status = str(
            log[3]
        )


        if (
            search
            and
            search not in name.lower()
        ):

            continue


        if (
            status_filter != "all"
            and
            status != status_filter
        ):

            continue


        if (
            date_filter
            and
            timestamp[:10]
            !=
            date_filter
        ):

            continue


        filtered.append(
            log
        )


    return jsonify({

        "count":
            len(
                filtered
            ),

        "summary": {

            "granted":
                sum(
                    1
                    for log in filtered
                    if log[3]
                    ==
                    "Granted"
                ),

            "denied":
                sum(
                    1
                    for log in filtered
                    if log[3]
                    ==
                    "Denied"
                ),

            "disabled":
                sum(
                    1
                    for log in filtered
                    if log[3]
                    ==
                    "Disabled"
                )
        },

        "logs": [

            log_to_dict(
                log
            )

            for log in filtered
        ]
    })


# ==========================================
# Users
# ==========================================

@app.route(
    "/users"
)
def users_page():

    return render_template(
        "users.html"
    )


@app.route(
    "/api/users"
)
def users_api():

    users = get_users()

    logs = get_logs()


    search = request.args.get(
        "search",
        ""
    ).strip().lower()


    status_filter = request.args.get(
        "status",
        "all"
    ).strip()


    latest_logs = {}


    for log in logs:

        log_name = str(
            log[1]
        )


        if log_name == "Unknown":
            continue


        if (
            log_name
            not in latest_logs
        ):

            latest_logs[
                log_name
            ] = log


    results = []


    for user in users:

        name = str(
            user[1]
        )

        active = bool(
            user[3]
        )


        if (
            search
            and
            search not in name.lower()
        ):

            continue


        if (
            status_filter == "active"
            and
            not active
        ):

            continue


        if (
            status_filter == "disabled"
            and
            active
        ):

            continue


        images = (
            get_registered_images(
                name
            )
        )


        latest = (
            latest_logs.get(
                name
            )
        )


        results.append({

            "id":
                user[0],

            "name":
                name,

            "created_at":
                user[2],

            "active":
                active,

            "profile_image":
                images[0]["url"]
                if images
                else None,

            "last_activity":
                log_to_dict(
                    latest
                )
                if latest
                else None
        })


    active_count = sum(

        1

        for user in users

        if user[3] == 1
    )


    return jsonify({

        "summary": {

            "total":
                len(
                    users
                ),

            "active":
                active_count,

            "disabled":
                len(
                    users
                )
                -
                active_count
        },

        "users":
            results
    })


# ==========================================
# User Profile
# ==========================================

@app.route(
    "/users/<name>"
)
def user_profile(name):

    if get_user_by_name(
        name
    ) is None:

        abort(404)


    return render_template(
        "user_profile.html",
        person_name=name
    )


@app.route(
    "/api/users/<name>"
)
def user_profile_api(name):

    user = get_user_by_name(
        name
    )


    if user is None:

        return jsonify({
            "error":
                "User not found"
        }), 404


    logs = [

        log

        for log in get_logs()

        if str(
            log[1]
        ) == str(
            name
        )
    ]


    similarities = [

        float(
            log[4]
        )

        for log in logs
    ]


    return jsonify({

        "user": {

            "id":
                user[0],

            "name":
                user[1],

            "created_at":
                user[2],

            "active":
                bool(
                    user[3]
                )
        },

        "summary": {

            "total_attempts":
                len(
                    logs
                ),

            "granted":
                sum(
                    1
                    for log in logs
                    if log[3]
                    ==
                    "Granted"
                ),

            "denied":
                sum(
                    1
                    for log in logs
                    if log[3]
                    ==
                    "Denied"
                ),

            "disabled":
                sum(
                    1
                    for log in logs
                    if log[3]
                    ==
                    "Disabled"
                ),

            "average_similarity":
                round(
                    sum(
                        similarities
                    )
                    /
                    len(
                        similarities
                    ),
                    3
                )
                if similarities
                else None,

            "highest_similarity":
                round(
                    max(
                        similarities
                    ),
                    3
                )
                if similarities
                else None
        },

        "last_activity":
            log_to_dict(
                logs[0]
            )
            if logs
            else None,

        "registered_images":
            get_registered_images(
                name
            ),

        "logs": [

            log_to_dict(
                log
            )

            for log in logs[:50]
        ]
    })


# ==========================================
# Recognition
# ==========================================

@app.route(
    "/recognition/start",
    methods=["POST"]
)
def start_recognition():

    if not recognition_is_running():

        start_recognition_process()


    return redirect(
        url_for(
            "dashboard"
        )
    )


@app.route(
    "/recognition/stop",
    methods=["POST"]
)
def stop_recognition():

    stop_recognition_process()


    return redirect(
        url_for(
            "dashboard"
        )
    )


# ==========================================
# Register
# ==========================================

@app.route(
    "/register",
    methods=[
        "GET",
        "POST"
    ]
)
def register():

    if request.method == "POST":

        name = request.form.get(
            "name",
            ""
        ).strip()


        if not name:

            flash(
                "Please enter a name."
            )

            return redirect(
                url_for(
                    "register"
                )
            )


        invalid_chars = (
            '<>:"/\\|?*'
        )


        if any(
            char in name
            for char in invalid_chars
        ):

            flash(
                "Invalid name."
            )

            return redirect(
                url_for(
                    "register"
                )
            )


        subprocess.Popen(

            [
                sys.executable,

                os.path.join(
                    BASE_DIR,
                    "register_face.py"
                ),

                name
            ],

            cwd=BASE_DIR
        )


        return redirect(
            url_for(
                "dashboard"
            )
        )


    return render_template(
        "register.html"
    )


# ==========================================
# User Controls
# ==========================================

@app.route(
    "/user/<name>/enable",
    methods=["POST"]
)
def enable(name):

    enable_user(
        name
    )


    return redirect(
        request.referrer
        or
        url_for(
            "dashboard"
        )
    )


@app.route(
    "/user/<name>/disable",
    methods=["POST"]
)
def disable(name):

    disable_user(
        name
    )


    return redirect(
        request.referrer
        or
        url_for(
            "dashboard"
        )
    )


@app.route(
    "/user/<name>/delete",
    methods=["POST"]
)
def delete(name):

    delete_user(
        name
    )


    person_dir = os.path.join(
        REGISTERED_DIR,
        name
    )


    if os.path.isdir(
        person_dir
    ):

        shutil.rmtree(
            person_dir
        )


    return_to = request.form.get(
        "return_to"
    )


    if return_to == "profile":

        return redirect(
            url_for(
                "users_page"
            )
        )


    return redirect(
        request.referrer
        or
        url_for(
            "dashboard"
        )
    )


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
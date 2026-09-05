import sqlite3
from datetime import datetime
import os


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DB_NAME = os.path.join(
    BASE_DIR,
    "access_system.db"
)


def create_database():

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            similarity REAL NOT NULL,
            image_path TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            created_at TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1
        )
    """)

    cursor.execute(
        "PRAGMA table_info(access_logs)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "image_path" not in columns:

        cursor.execute("""
            ALTER TABLE access_logs
            ADD COLUMN image_path TEXT
        """)

    connection.commit()
    connection.close()

    print("Database ready.")


def add_user(name):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    created_at = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    try:

        cursor.execute("""
            INSERT INTO users (
                name,
                created_at,
                active
            )
            VALUES (?, ?, 1)
        """, (
            name,
            created_at
        ))

        connection.commit()

        print(
            f"USER ADDED: {name}"
        )

    except sqlite3.IntegrityError:

        print(
            f"USER ALREADY EXISTS: {name}"
        )

    connection.close()


def delete_user(name):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM users
        WHERE name = ?
    """, (
        name,
    ))

    connection.commit()

    deleted = (
        cursor.rowcount > 0
    )

    connection.close()

    return deleted


def is_user_active(name):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT active
        FROM users
        WHERE name = ?
    """, (
        name,
    ))

    result = cursor.fetchone()

    connection.close()

    if result is None:
        return False

    return result[0] == 1


def get_users():

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            created_at,
            active
        FROM users
        ORDER BY id ASC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


def enable_user(name):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE users
        SET active = 1
        WHERE name = ?
    """, (
        name,
    ))

    connection.commit()
    connection.close()


def disable_user(name):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        UPDATE users
        SET active = 0
        WHERE name = ?
    """, (
        name,
    ))

    connection.commit()
    connection.close()


def log_access(
    name,
    status,
    similarity,
    image_path=None
):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cursor.execute("""
        INSERT INTO access_logs (
            name,
            timestamp,
            status,
            similarity,
            image_path
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        name,
        timestamp,
        status,
        float(similarity),
        image_path
    ))

    log_id = cursor.lastrowid

    connection.commit()
    connection.close()

    return log_id


def get_logs():

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            timestamp,
            status,
            similarity,
            image_path
        FROM access_logs
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    connection.close()

    return rows


def get_log(log_id):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            id,
            name,
            timestamp,
            status,
            similarity,
            image_path
        FROM access_logs
        WHERE id = ?
    """, (
        int(log_id),
    ))

    row = cursor.fetchone()

    connection.close()

    return row


def delete_log(log_id):

    connection = sqlite3.connect(DB_NAME)
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM access_logs
        WHERE id = ?
    """, (
        int(log_id),
    ))

    connection.commit()

    deleted = (
        cursor.rowcount > 0
    )

    connection.close()

    return deleted


if __name__ == "__main__":

    create_database()
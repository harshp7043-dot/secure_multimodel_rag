
import json
import time
import hmac
import hashlib
import secrets
from config import SESSION_TIMEOUT_SEC
import streamlit as st

from config import (
    USERS_FILE,
    MAX_FAILED_ATTEMPTS,
    LOCKOUT_SEC
)


# Same demo accounts as your original implementation.
# Existing users.json is never overwritten.

DEMO_USERS = {
    "dave": ("Dave (Employee)", "employee", "Dave@2026"),
    "alice": ("Alice (HR Manager)", "hr", "Alice@HR26"),
    "bob": ("Bob (Finance Lead)", "finance", "Bob@Fin26"),
    "meera": ("Meera Shah (CEO)", "executive", "Meera@CEO26")
}


def hash_password(password, salt):

    return hashlib.scrypt(
        password.encode(),
        salt=salt,
        n=2 ** 14,
        r=8,
        p=1,
        dklen=32
    )


def init_users_file():

    if USERS_FILE.exists():
        return

    users = {}

    for username, (name, role, password) in DEMO_USERS.items():

        salt = secrets.token_bytes(16)

        users[username] = {
            "name": name,
            "role": role,
            "salt": salt.hex(),
            "hash": hash_password(password, salt).hex()
        }

    USERS_FILE.write_text(
        json.dumps(users, indent=2)
    )


def load_users():

    init_users_file()

    with open(USERS_FILE, "r") as file:
        return json.load(file)


def initialize_auth_state():

    defaults = {
        "user": None,
        "failed_attempts": {},
        "login_locked_until": {},
        "last_active": time.time()
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


def authenticate_user(username, password):

    username = username.strip().lower()

    users = load_users()

    now = time.time()

    locked_until = st.session_state.login_locked_until.get(
        username, 0
    )

    if now < locked_until:
        return None, "Too many failed attempts. Try again later."

    user = users.get(username)

    dummy_salt = b"secure-dummy-salt"

    salt = (
        bytes.fromhex(user["salt"])
        if user else dummy_salt
    )

    candidate = hash_password(password, salt)

    valid = (
        user is not None
        and hmac.compare_digest(
            candidate,
            bytes.fromhex(user["hash"])
        )
    )

    if valid:

        st.session_state.failed_attempts.pop(
            username, None
        )

        st.session_state.login_locked_until.pop(
            username, None
        )

        st.session_state.user = {
            "username": username,
            "name": user["name"],
            "role": user["role"]
        }

        st.session_state.last_active = now

        return st.session_state.user, None

    failures = st.session_state.failed_attempts.get(
        username, 0
    ) + 1

    st.session_state.failed_attempts[username] = failures

    if failures >= MAX_FAILED_ATTEMPTS:

        st.session_state.login_locked_until[username] = (
            now + LOCKOUT_SEC
        )

        st.session_state.failed_attempts[username] = 0

        return None, "Account temporarily locked. Try again later."

    return None, "Invalid username or password."


def logout_user():

    st.session_state.user = None
    st.session_state.last_active = time.time()
    st.session_state.chat_history = []

    st.session_state.pending_query = None


def get_current_user():

    return st.session_state.get("user")


def is_authenticated():

    return get_current_user() is not None


def check_session_expiry():

    user = get_current_user()

    if user is None:
        return

    last_active = st.session_state.last_active

    if time.time() - last_active > SESSION_TIMEOUT_SEC:

        logout_user()

        st.session_state.session_expired = True

    else:
        st.session_state.last_active = time.time()

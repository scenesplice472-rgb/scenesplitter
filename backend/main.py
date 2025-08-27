"""
Main backend application for the SaaS demo.

This module defines a very simple FastAPI application that exposes a few
endpoints allowing users to register/login, upload a video and obtain an
automatically detected list of scenes. The purpose of this code is to
demonstrate the core logic of the SaaS described by the user. It does not
implement a production-ready service – for example, passwords are
stored as plaintext and authentication tokens are simplistic. In a real
deployment you would use a proper database, password hashing, JWT tokens
and robust video processing using ffmpeg or similar libraries.

Limitations
-----------
Due to the restrictions of the environment where this code runs we do not
depend on external packages beyond FastAPI, Pydantic and OpenCV (cv2). There
is no access to FFmpeg so video splitting is not performed. Instead the
backend analyses the uploaded file frame by frame and returns the time
indices where significant scene changes occur. These start/stop timepoints
can later be used by client‑side or external tooling to cut the video.
"""

import os
import uuid
from typing import List, Optional

import sqlite3
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2


app = FastAPI(title="ClipSplitter API", description="Automatic scene detection")


# ----------------------------------------------------------------------------
# Persistent storage using SQLite
# ----------------------------------------------------------------------------

DB_PATH = os.path.join(os.path.dirname(__file__), "db.sqlite3")



def get_db() -> sqlite3.Connection:
    """Return a connection to the SQLite database.

    This helper ensures that the connection uses a row factory so that
    results behave like dictionaries. Each call returns a new connection
    which will need to be closed by the caller. FastAPI will reuse the
    connection per request implicitly when used within endpoint scopes.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn



def init_db() -> None:
    """Initialize the SQLite database with required tables if absent."""
    conn = get_db()
    with conn:
        # Users table: stores username and plaintext password (for demo only)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
        )
        # Sessions table: maps session tokens to users
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        # Files table: stores uploaded files and metadata
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                path TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
            """
        )
        # Scenes table: stores start/end times for each file
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scenes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                start REAL NOT NULL,
                end REAL NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id)
            )
            """
        )
    conn.close()


# Initialise DB on module import
init_db()


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class Scene(BaseModel):
    start: float  # seconds
    end: float    # seconds

class UploadRequest(BaseModel):
    """Data required to upload a video via JSON."""
    filename: str
    data: str  # base64‑encoded binary data


class UploadResponse(BaseModel):
    file_id: str
    scenes: List[Scene]


class CheckoutResponse(BaseModel):
    checkout_url: str


class YouTubeUploadRequest(BaseModel):
    """
    Data required to schedule an upload to YouTube.

    This object includes the file ID returned from `/upload`, metadata such
    as title and description, an optional ISO‑8601 scheduled time when the
    clip should go live and a YouTube OAuth token. In this demonstration
    environment we do not perform the actual API call; instead the request
    is validated and a placeholder response is returned. When integrating
    with Google APIs you would exchange the provided token for credentials
    and upload the clip via the YouTube Data API.
    """
    file_id: str
    title: str
    description: Optional[str] = None
    scheduled_time: Optional[str] = None
    youtube_token: str


# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------

def detect_scenes(video_path: str, threshold: float = 30.0) -> List[Scene]:
    """
    Naive scene detection using frame differencing.

    This function opens the video located at ``video_path`` using OpenCV,
    computes the absolute difference between consecutive frames and
    identifies boundaries where the mean difference exceeds a configurable
    threshold. Each scene is defined by a start time and an end time (in
    seconds). The algorithm will always include the very first frame and
    final frame.

    Parameters
    ----------
    video_path: str
        Path to the video file on disk.
    threshold: float
        Threshold on the average pixel difference to declare a new scene.

    Returns
    -------
    List[Scene]
        A list of Scene objects describing start and end timepoints.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scenes = []
    prev_frame = None
    current_scene_start = 0.0

    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            # Compute absolute difference between grayscale versions of frames
            diff = cv2.absdiff(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                               cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY))
            mean_diff = diff.mean()
            # If the difference is large, mark end of current scene
            if mean_diff > threshold:
                end_time = i / fps
                scenes.append(Scene(start=current_scene_start, end=end_time))
                current_scene_start = end_time
        prev_frame = frame

    # Append last scene up to video end
    if total_frames > 0:
        scenes.append(Scene(start=current_scene_start, end=total_frames / fps))

    cap.release()
    return scenes


def authenticate(token: str) -> str:
    """Simple helper to check a session token.

    Parameters
    ----------
    token: str
        Session token provided by the client (query parameter or header).

    Returns
    -------
    str
        The username associated with the session.

    Raises
    ------
    HTTPException
        If the token is invalid or missing.
    """
    if not token:
        raise HTTPException(status_code=401, detail="Invalid or missing token")
    conn = get_db()
    try:
        session = conn.execute(
            "SELECT users.username FROM sessions JOIN users ON sessions.user_id = users.id WHERE sessions.token = ?",
            (token,),
        ).fetchone()
        if session is None:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return session["username"]
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------------

@app.post("/register")
def register(request: RegisterRequest):
    """Create a new user account."""
    conn = get_db()
    try:
        with conn:
            # Attempt to insert a new user
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (request.username, request.password),
            )
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="User already exists")
    finally:
        conn.close()


@app.post("/login")
def login(request: LoginRequest):
    """Authenticate and return a session token."""
    conn = get_db()
    try:
        # Look up user
        row = conn.execute(
            "SELECT id, password FROM users WHERE username = ?", (request.username,)
        ).fetchone()
        if row is None or row["password"] != request.password:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        # Generate a token and insert into sessions
        token = str(uuid.uuid4())
        with conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id) VALUES (?, ?)",
                (token, row["id"]),
            )
        return {"token": token}
    finally:
        conn.close()


@app.post("/upload", response_model=UploadResponse)
def upload_video(request: UploadRequest, token: str):
    """
    Upload a video file encoded as base64 in JSON and perform scene detection.

    Because multipart/form‑data handling requires the optional
    ``python‑multipart`` package (not available in this environment), this
    endpoint expects a JSON payload with two fields:

    - ``filename``: the name of the uploaded file (used for storage)
    - ``data``: the base64‑encoded contents of the video file

    The session token must be supplied as a query parameter named ``token``.

    Returns a unique file ID together with the list of detected scenes.
    """
    username = authenticate(token)

    # Decode the base64 data
    import base64
    try:
        binary = base64.b64decode(request.data, validate=True)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 data")

    # Save the uploaded file to a temporary location
    upload_dir = os.path.join(os.path.dirname(__file__), "..", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_id = str(uuid.uuid4())
    sanitized = os.path.basename(request.filename)
    save_path = os.path.join(upload_dir, f"{file_id}_{sanitized}")
    with open(save_path, "wb") as f:
        f.write(binary)

    # Run naive scene detection
    try:
        scenes = detect_scenes(save_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Persist file and scenes
    conn = get_db()
    try:
        # Look up user id
        user_row = conn.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone()
        if user_row is None:
            raise HTTPException(status_code=404, detail="User not found")
        with conn:
            conn.execute(
                "INSERT INTO files (id, user_id, filename, path) VALUES (?, ?, ?, ?)",
                (file_id, user_row["id"], request.filename, save_path),
            )
            for scene in scenes:
                conn.execute(
                    "INSERT INTO scenes (file_id, start, end) VALUES (?, ?, ?)",
                    (file_id, scene.start, scene.end),
                )
    finally:
        conn.close()

    return UploadResponse(file_id=file_id, scenes=scenes)


@app.get("/scenes/{file_id}", response_model=List[Scene])
def get_scenes(file_id: str, token: str):
    """
    Retrieve the scenes for a previously uploaded file.

    The session token must be supplied as a query parameter named ``token``.
    """
    username = authenticate(token)
    conn = get_db()
    try:
        # Find file and verify ownership
        file_row = conn.execute(
            "SELECT files.id FROM files JOIN users ON files.user_id = users.id WHERE files.id = ? AND users.username = ?",
            (file_id, username),
        ).fetchone()
        if file_row is None:
            raise HTTPException(status_code=404, detail="File not found")
        # Fetch all scenes associated with the file
        scene_rows = conn.execute(
            "SELECT start, end FROM scenes WHERE file_id = ? ORDER BY id",
            (file_id,),
        ).fetchall()
        return [Scene(start=row["start"], end=row["end"]) for row in scene_rows]
    finally:
        conn.close()


# ----------------------------------------------------------------------------
# Payment and YouTube integration stubs
# ----------------------------------------------------------------------------


@app.post("/create-checkout-session", response_model=CheckoutResponse)
def create_checkout_session(token: str):
    """
    Create a payment checkout session for the current user.

    This endpoint would normally interact with Stripe or another payment
    processor to generate a checkout page where the user can enter their
    payment details. However, in this restricted environment the `stripe`
    package cannot be installed and network access is limited, so this
    function returns a placeholder URL. When deploying in production you
    should replace this implementation with actual API calls to your
    payment provider using your secret API key.

    The session token must be provided as a query parameter. The response
    contains a `checkout_url` property which the client should redirect the
    user to.
    """
    # Verify user authentication
    username = authenticate(token)
    # Placeholder: generate a dummy checkout URL; you should replace with
    # actual Stripe Checkout Session creation code using `stripe.checkout.Session.create`
    dummy_url = f"https://example.com/pay?user={username}"
    return CheckoutResponse(checkout_url=dummy_url)


@app.post("/youtube/upload")
def youtube_upload(req: YouTubeUploadRequest, token: str):
    """
    Schedule or publish a clip to YouTube.

    Accepts a `file_id` referencing a previously uploaded video, along with
    metadata such as `title`, `description`, an optional `scheduled_time`
    (ISO‑8601 string) and a `youtube_token` representing OAuth credentials
    obtained on the client side. This endpoint validates ownership of the
    file and returns a simple success message. In a real implementation you
    would use Google's `google-api-python-client` to upload the video to
    YouTube and schedule its publication.
    """
    username = authenticate(token)
    # Verify that the file exists and belongs to the user
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT files.id FROM files JOIN users ON files.user_id = users.id WHERE files.id = ? AND users.username = ?",
            (req.file_id, username),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="File not found or not owned by user")
    finally:
        conn.close()
    # Placeholder: normally you would call the YouTube API here
    return {
        "status": "scheduled",
        "message": "Your clip has been scheduled for upload (simulated).",
        "file_id": req.file_id,
        "title": req.title,
        "scheduled_time": req.scheduled_time,
    }

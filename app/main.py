from __future__ import annotations

import os
import json
import time
import base64
import math
import uuid
import shutil
import subprocess
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import stripe

# ============================================================
# CONFIG
# ============================================================

APP_NAME = "motionforge_saas_backend"

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
OUTPUTS_DIR = DATA_DIR / "outputs"
CREDITS_DB_PATH = DATA_DIR / "credits_db.json"
GRANTS_DB_PATH = DATA_DIR / "grants_db.json"  # track which Stripe session_ids already granted (idempotency)

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Public base URL for building download links
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://www.getmotionforge.com").rstrip("/")

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
if not STRIPE_SECRET_KEY:
    raise RuntimeError("STRIPE_SECRET_KEY not set")
stripe.api_key = STRIPE_SECRET_KEY

# Price → credits mapping (LIVE)
PRICE_TO_CREDITS: Dict[str, int] = {
    # Tester pack $5 -> 10 credits (you had two variants at some point; keep both)
    "price_1Sjzy32L998MB1DP0pYyuyTY": 10,
    "price_1Sjzy32L998MB1DPOpYyuyTY": 10,

    # Creator Pack $97 -> 50 credits
    "price_1Sd90A2L998MB1DPzBnPWnTA": 50,

    # Refills
    "price_1Sh7La2L998MB1DPtfvTv31N": 250,
    "price_1Sh7Px2L998MB1DPQcQbGLfR": 500,
}

# CORS
cors_origins = [
    "https://www.getmotionforge.com",
    "https://getmotionforge.com",
]

# ============================================================
# APP
# ============================================================

print("MAIN APP LOADED")
print("ROUTES:", [r.path for r in app.routes])


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated outputs so the frontend can download them
# Backend URL: https://...up.railway.app/outputs/<file>
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ============================================================
# STORAGE (credits + idempotent grants)
# ============================================================

_db_lock = Lock()

def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def _load_credits_db() -> Dict[str, int]:
    data = _read_json(CREDITS_DB_PATH, {})
    if not isinstance(data, dict):
        return {}
    # ensure ints
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k).lower()] = int(v)
        except Exception:
            out[str(k).lower()] = 0
    return out

def _save_credits_db(db: Dict[str, int]) -> None:
    _write_json(CREDITS_DB_PATH, db)

def _load_grants_db() -> Dict[str, Dict[str, Any]]:
    data = _read_json(GRANTS_DB_PATH, {})
    if not isinstance(data, dict):
        return {}
    return data

def _save_grants_db(db: Dict[str, Dict[str, Any]]) -> None:
    _write_json(GRANTS_DB_PATH, db)

def get_credits(email: str) -> int:
    email = email.strip().lower()
    with _db_lock:
        db = _load_credits_db()
        return int(db.get(email, 0))

def add_credits(email: str, amount: int) -> int:
    email = email.strip().lower()
    with _db_lock:
        db = _load_credits_db()
        db[email] = int(db.get(email, 0)) + int(amount)
        _save_credits_db(db)
        return int(db[email])

def spend_credits(email: str, amount: int) -> int:
    email = email.strip().lower()
    with _db_lock:
        db = _load_credits_db()
        cur = int(db.get(email, 0))
        if amount > cur:
            raise HTTPException(status_code=402, detail="Not enough credits")
        db[email] = cur - int(amount)
        _save_credits_db(db)
        return int(db[email])

# ============================================================
# TOKEN (simple, consistent with your current mf_ tokens)
# ============================================================

def issue_token_for_email(email: str) -> str:
    payload = {"email": email, "iat": int(time.time())}
    token = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"mf_{token}"

def decode_token(token: str) -> str:
    """
    Returns email from token. Raises 401 on invalid.
    Token format: mf_<base64url(json)>
    """
    if not token or not token.startswith("mf_"):
        raise HTTPException(status_code=401, detail="Invalid token")
    b64 = token[3:]
    # pad
    pad = "=" * ((4 - (len(b64) % 4)) % 4)
    try:
        raw = base64.urlsafe_b64decode((b64 + pad).encode())
        payload = json.loads(raw.decode())
        email = (payload.get("email") or "").strip().lower()
        if not email:
            raise ValueError("missing email")
        return email
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def get_bearer_token(request: Request) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return auth.split(" ", 1)[1].strip()

# ============================================================
# MEDIA HELPERS (ffprobe + ffmpeg)
# ============================================================

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def probe_duration_seconds(filepath: Path) -> float:
    """
    Uses ffprobe to read duration in seconds.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(filepath),
    ]
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()[:400]}")
    try:
        return float(p.stdout.strip())
    except Exception:
        raise RuntimeError("Could not parse ffprobe output")

def split_uniform(
    input_path: Path,
    out_dir: Path,
    clip_seconds: int,
) -> List[Path]:
    """
    Splits video into uniform clips using ffmpeg segment muxer.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = out_dir / "clip_%03d.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-c", "copy",
        "-map", "0",
        "-f", "segment",
        "-reset_timestamps", "1",
        "-segment_time", str(int(clip_seconds)),
        str(out_pattern),
    ]
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr.strip()[:800]}")

    clips = sorted(out_dir.glob("clip_*.mp4"))
    if not clips:
        raise RuntimeError("No clips produced")
    return clips

# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME, "stripe_mode": "live", "cors_origins": cors_origins}

@app.get("/credits")
def credits(request: Request):
    token = get_bearer_token(request)
    email = decode_token(token)
    return {"ok": True, "email": email, "credits": get_credits(email)}

@app.get("/auth/access-from-session")
def access_from_session(session_id: str):
    """
    Exchanges Stripe Checkout session → token and grants credits based on line items.
    Idempotent: does not double-grant same session_id.
    """
    # Retrieve session (expand line_items.price so we can read price ids)
    try:
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=["line_items.data.price"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Email
    email = None
    if getattr(session, "customer_details", None) and session.customer_details and session.customer_details.email:
        email = session.customer_details.email
    elif getattr(session, "customer_email", None):
        email = session.customer_email

    if not email:
        raise HTTPException(status_code=400, detail="No email found on Stripe session")

    email = email.strip().lower()
    token = issue_token_for_email(email)

    # Determine credits from purchased items (reliable method)
    credits_to_add = 0
    seen_price_ids: List[str] = []

    try:
        items = stripe.checkout.Session.list_line_items(session_id, limit=100)
        for item in items.data:
            price = getattr(item, "price", None)
            price_id = getattr(price, "id", None) if price else None
            qty = int(getattr(item, "quantity", 1) or 1)

            if price_id:
                seen_price_ids.append(price_id)
                per = int(PRICE_TO_CREDITS.get(price_id, 0))
                credits_to_add += per * qty
    except Exception:
        credits_to_add = 0
        seen_price_ids = []

    # Idempotent grant
    with _db_lock:
        grants = _load_grants_db()
        prev = grants.get(session_id)

        if prev is None:
            # first time we see this session_id
            if credits_to_add > 0:
                add_credits(email, credits_to_add)
            grants[session_id] = {
                "email": email,
                "credits_added": credits_to_add,
                "price_ids": seen_price_ids,
                "ts": int(time.time()),
            }
            _save_grants_db(grants)
        else:
            # already granted; do NOT double-add
            credits_to_add = int(prev.get("credits_added", 0))
            seen_price_ids = list(prev.get("price_ids", []))

    course_url = f"{PUBLIC_BASE_URL}/course.html?token={token}&session_id={session_id}"
    tool_url = f"{PUBLIC_BASE_URL}/tool.html?token={token}"

    return {
        "course_url": course_url,
        "tool_url": tool_url,
        "token": token,
        "credits_added": credits_to_add,
        "price_ids": seen_price_ids,
    }

@app.post("/upload")
async def upload_and_process(
    request: Request,
    file: UploadFile = File(...),
    clip_seconds: int = Form(30),
):
    """
    Uploads a video, splits into clips, charges credits (1 credit per clip),
    returns downloadable clip URLs.

    Frontend currently calls POST /upload. This endpoint is the missing piece.
    """
    token = get_bearer_token(request)
    email = decode_token(token)

    if clip_seconds < 10 or clip_seconds > 180:
        raise HTTPException(status_code=400, detail="clip_seconds must be between 10 and 180")

    job_id = uuid.uuid4().hex
    job_upload_dir = UPLOADS_DIR / job_id
    job_output_dir = OUTPUTS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    input_path = job_upload_dir / (file.filename or "input.mp4")

    try:
        # save upload
        with input_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

        # duration -> estimate clips
        duration = probe_duration_seconds(input_path)
        estimated_clips = max(1, int(math.ceil(duration / float(clip_seconds))))
        credits_needed = estimated_clips  # 1 credit = 1 short

        current = get_credits(email)
        if current < credits_needed:
            raise HTTPException(
                status_code=402,
                detail=f"Not enough credits. Need {credits_needed}, have {current}.",
            )

        # process
        clips = split_uniform(input_path, job_output_dir, clip_seconds=clip_seconds)

        # actual cost = number of clips produced
        actual_clips = len(clips)
        credits_needed = actual_clips

        # spend after successful processing
        remaining = spend_credits(email, credits_needed)

        # build URLs
        out = []
        for p in clips:
            url = f"{PUBLIC_BASE_URL}/outputs/{job_id}/{p.name}"
            out.append({"filename": p.name, "url": url})

        return {
            "ok": True,
            "job_id": job_id,
            "email": email,
            "credits_spent": credits_needed,
            "credits_remaining": remaining,
            "clips": out,
        }

    except HTTPException:
        # keep for debugging; do not delete outputs if any created
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try:
            await file.close()
        except Exception:
            pass

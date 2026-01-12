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
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
GRANTS_DB_PATH = DATA_DIR / "grants_db.json"  # session_id -> grant record (idempotent)

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://www.getmotionforge.com").rstrip("/")

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_KEY_TYPE = "restricted" if (STRIPE_SECRET_KEY or "").startswith("rk_") else "standard"
if not STRIPE_SECRET_KEY:
    raise RuntimeError("STRIPE_SECRET_KEY not set")

stripe.api_key = STRIPE_SECRET_KEY

# Keep network behavior predictable
stripe.max_network_retries = 0
STRIPE_TIMEOUT_SECONDS = int(os.getenv("STRIPE_TIMEOUT_SECONDS", "15"))  # used as request_timeout

# LIVE Price → credits mapping
PRICE_TO_CREDITS: Dict[str, int] = {
    # Tester pack $5 -> 10 credits (keep both variants)
    "price_1Sjzy32L998MB1DP0pYyuyTY": 10,
    "price_1Sjzy32L998MB1DPOpYyuyTY": 10,
    # Creator Pack $97 -> 50 credits
    "price_1Sd90A2L998MB1DPzBnPWnTA": 50,
    # Refills
    "price_1Sh7La2L998MB1DPtfvTv31N": 250,
    "price_1Sh7Px2L998MB1DPQcQbGLfR": 500,
}

cors_origins = [
    "https://www.getmotionforge.com",
    "https://getmotionforge.com",
]

# ============================================================
# APP
# ============================================================

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated clips via backend URL:
# https://motionforgesaasbackend-production.up.railway.app/outputs/<job_id>/<filename>
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# ============================================================
# STORAGE (credits + idempotent grants)
# ============================================================

# MUST be re-entrant to avoid deadlocks when calling helpers inside lock
_db_lock = RLock()

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
# TOKEN (mf_ base64url(json))
# ============================================================

def issue_token_for_email(email: str) -> str:
    payload = {"email": email, "iat": int(time.time())}
    token = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"mf_{token}"

def decode_token(token: str) -> str:
    if not token or not token.startswith("mf_"):
        raise HTTPException(status_code=401, detail="Invalid token")
    b64 = token[3:]
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
# VIDEO TOOLS
# ============================================================

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")

def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def probe_duration_seconds(filepath: Path) -> float:
    if not FFPROBE:
        raise RuntimeError("ffprobe not available on this server")
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
    return float(p.stdout.strip())

def split_uniform(input_path: Path, out_dir: Path, clip_seconds: int) -> List[Path]:
    if not FFMPEG:
        raise RuntimeError("ffmpeg not available on this server")

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
# STRIPE HELPERS (timeouts!)
# ============================================================

def _stripe_retrieve_session(session_id: str) -> stripe.checkout.Session:
    """
    IMPORTANT: Stripe Python uses request_timeout (not timeout).
    This prevents hanging requests.
    """
    return stripe.checkout.Session.retrieve(
        session_id,
        expand=["line_items.data.price", "customer_details"],
        request_timeout=STRIPE_TIMEOUT_SECONDS,
    )

def _stripe_list_line_items(session_id: str) -> Any:
    return stripe.checkout.Session.list_line_items(
        session_id,
        limit=100,
        request_timeout=STRIPE_TIMEOUT_SECONDS,
    )

def _extract_email_from_session(session: stripe.checkout.Session) -> Optional[str]:
    email = None
    try:
        cd = getattr(session, "customer_details", None)
        if cd and getattr(cd, "email", None):
            email = cd.email
    except Exception:
        pass
    if not email:
        try:
            ce = getattr(session, "customer_email", None)
            if ce:
                email = ce
        except Exception:
            pass
    return email.strip().lower() if email else None

def _calc_credits_from_expanded_line_items(session: stripe.checkout.Session) -> Tuple[int, List[str]]:
    credits_to_add = 0
    seen_price_ids: List[str] = []

    line_items = getattr(session, "line_items", None)
    data = getattr(line_items, "data", None) if line_items else None
    if not data:
        return 0, []

    for item in data:
        qty = int(getattr(item, "quantity", 1) or 1)
        price = getattr(item, "price", None)
        price_id = getattr(price, "id", None) if price else None
        if price_id:
            seen_price_ids.append(price_id)
            credits_to_add += int(PRICE_TO_CREDITS.get(price_id, 0)) * qty

    return credits_to_add, seen_price_ids

# ============================================================
# ROUTES
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": APP_NAME,
        "stripe_mode": "live",
        "cors_origins": cors_origins,
        "ffmpeg": bool(FFMPEG),
        "ffprobe": bool(FFPROBE),
        "price_ids_loaded": list(PRICE_TO_CREDITS.keys()),
        "stripe_request_timeout_seconds": STRIPE_TIMEOUT_SECONDS,
    }
@app.get("/stripe/account")
def stripe_account():
    try:
        acct = stripe.Account.retrieve()
        return {
            "ok": True,
            "account_id": getattr(acct, "id", None),
            "country": getattr(acct, "country", None),
            "default_currency": getattr(acct, "default_currency", None),
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Stripe account retrieve failed: {str(e)}")

@app.get("/credits")
def credits(request: Request):
    token = get_bearer_token(request)
    email = decode_token(token)
    return {"ok": True, "email": email, "credits": get_credits(email)}

@app.get("/auth/access-from-session")
def access_from_session(session_id: str):
    """
    Stripe Checkout session -> token + credits grant (idempotent per session_id)

    HARD REQUIREMENTS:
    - Must never hang: Stripe calls use request_timeout
    - Must be idempotent per session_id
    - Must never deadlock: use RLock
    """
    if not session_id or not session_id.startswith("cs_"):
        raise HTTPException(status_code=400, detail="Invalid session_id")

    # 1) Retrieve session with hard timeout
    try:
        session = _stripe_retrieve_session(session_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Stripe retrieve failed: {str(e)}")

    # 2) Extract email
    email = _extract_email_from_session(session)
    if not email:
        raise HTTPException(status_code=400, detail="No email found on Stripe session")

    token = issue_token_for_email(email)

    # 3) Calculate credits
    credits_to_add, seen_price_ids = _calc_credits_from_expanded_line_items(session)

    # Fallback (also timed)
    if credits_to_add == 0 and not seen_price_ids:
        try:
            items = _stripe_list_line_items(session_id)
            for item in items.data:
                price = getattr(item, "price", None)
                price_id = getattr(price, "id", None) if price else None
                qty = int(getattr(item, "quantity", 1) or 1)
                if price_id:
                    seen_price_ids.append(price_id)
                    credits_to_add += int(PRICE_TO_CREDITS.get(price_id, 0)) * qty
        except Exception:
            credits_to_add = 0
            seen_price_ids = []

    # 4) Idempotent grant
    with _db_lock:
        grants = _load_grants_db()
        prev = grants.get(session_id)

        if prev is None:
            if credits_to_add > 0:
                # Safe: add_credits uses same RLock (re-entrant)
                add_credits(email, credits_to_add)

            grants[session_id] = {
                "email": email,
                "credits_added": int(credits_to_add),
                "price_ids": list(seen_price_ids),
                "ts": int(time.time()),
            }
            _save_grants_db(grants)
        else:
            credits_to_add = int(prev.get("credits_added", 0))
            seen_price_ids = list(prev.get("price_ids", []))

    course_url = f"{PUBLIC_BASE_URL}/course.html?token={token}&session_id={session_id}"
    tool_url = f"{PUBLIC_BASE_URL}/tool.html?token={token}"

    return {
        "status": "ok",
        "course_url": course_url,
        "tool_url": tool_url,
        "token": token,
        "credits_added": int(credits_to_add),
        "price_ids": seen_price_ids,
    }
@app.post("/checkout/create")
def checkout_create(payload: Dict[str, Any] = Body(...)):
    """
    Create a Stripe Checkout Session and redirect user to access.html
    Expected payload:
      {
        "price_id": "price_....",
        "quantity": 1
      }
    """
    price_id = payload.get("price_id")
    quantity = int(payload.get("quantity", 1) or 1)

    if not price_id or price_id not in PRICE_TO_CREDITS:
        raise HTTPException(status_code=400, detail="Invalid or unsupported price_id")

    try:
        session = stripe.checkout.Session.create(
            mode="payment",
            line_items=[{
                "price": price_id,
                "quantity": quantity,
            }],
            success_url=f"{PUBLIC_BASE_URL}/access.html?session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{PUBLIC_BASE_URL}/",
            
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Stripe create failed: {str(e)}")

    return {
        "ok": True,
        "checkout_url": session.url,
        "session_id": session.id,
    }

# Optional: avoid 404 spam if Stripe hits a webhook URL you configured earlier
@app.post("/stripe/webhook")
async def stripe_webhook(_: Request):
    return {"ok": True}

# ============================================================
# /upload route (registered safely)
#   - If python-multipart missing, we still start the app and return 503 instead of crashing.
# ============================================================

try:
    import multipart  # noqa: F401
    MULTIPART_OK = True
except Exception:
    MULTIPART_OK = False

if not MULTIPART_OK:
    @app.post("/upload")
    async def upload_disabled():
        raise HTTPException(
            status_code=503,
            detail='Upload disabled: missing dependency "python-multipart". Add it to requirements and redeploy.',
        )
else:
    from fastapi import UploadFile, File, Form  # only import when safe

    @app.post("/upload")
    async def upload(
        request: Request,
        file: UploadFile = File(...),
        clip_seconds: int = Form(30),
    ):
        """
        Uploads a video. If ffmpeg/ffprobe are available: splits into clips and deducts credits.
        If ffmpeg/ffprobe are missing: stores file and returns a clear 503 (no crash).
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

        filename = file.filename or "input.mp4"
        input_path = job_upload_dir / filename

        try:
            with input_path.open("wb") as f:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

            if not (FFMPEG and FFPROBE):
                raise HTTPException(
                    status_code=503,
                    detail="Video processing unavailable: ffmpeg/ffprobe not installed on the server.",
                )

            duration = probe_duration_seconds(input_path)
            estimated_clips = max(1, int(math.ceil(duration / float(clip_seconds))))
            current = get_credits(email)
            if current < estimated_clips:
                raise HTTPException(
                    status_code=402,
                    detail=f"Not enough credits. Need {estimated_clips}, have {current}.",
                )

            clips = split_uniform(input_path, job_output_dir, clip_seconds=clip_seconds)

            actual = len(clips)
            remaining = spend_credits(email, actual)

            backend_base = str(request.base_url).rstrip("/")

            out: List[Dict[str, str]] = []
            for p in clips:
                out.append({
                    "filename": p.name,
                    "url": f"{backend_base}/outputs/{job_id}/{p.name}",
                })

            return {
                "ok": True,
                "job_id": job_id,
                "email": email,
                "clip_seconds": clip_seconds,
                "credits_spent": actual,
                "credits_remaining": remaining,
                "clips": out,
            }

        finally:
            try:
                await file.close()
            except Exception:
                pass

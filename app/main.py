from __future__ import annotations

import os
import json
import time
import uuid
import base64
import hashlib
import subprocess
import tempfile
import wave
import struct
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# Optional: Stripe (keeps your existing flow working)
try:
    import stripe  # type: ignore
except Exception:
    stripe = None


APP_NAME = "motionforge_saas_backend"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DEMO_DIR = BASE_DIR / "demo_outputs"
CREDITS_PATH = BASE_DIR / "credits.json"

for p in [DATA_DIR, UPLOADS_DIR, OUTPUTS_DIR, DEMO_DIR]:
    p.mkdir(parents=True, exist_ok=True)

TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "86400"))

# CORS
DEFAULT_ORIGINS = [
    "https://www.getmotionforge.com",
    "https://getmotionforge.com",
]
EXTRA = os.getenv("ALLOWED_ORIGINS", "").strip()
if EXTRA:
    DEFAULT_ORIGINS += [o.strip() for o in EXTRA.split(",") if o.strip()]

# Stripe
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "").strip()
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()  # optional for future
STRIPE_PRICE_ID_TO_CREDITS_RAW = os.getenv("STRIPE_PRICE_ID_TO_CREDITS", "").strip()

PRICE_ID_TO_CREDITS: Dict[str, int] = {}
if STRIPE_PRICE_ID_TO_CREDITS_RAW:
    try:
        PRICE_ID_TO_CREDITS = json.loads(STRIPE_PRICE_ID_TO_CREDITS_RAW)
        PRICE_ID_TO_CREDITS = {k: int(v) for k, v in PRICE_ID_TO_CREDITS.items()}
    except Exception:
        PRICE_ID_TO_CREDITS = {}

# Demo settings
DEMO_MAX_CLIPS = 3
DEMO_MAX_FILE_MB = int(os.getenv("DEMO_MAX_FILE_MB", "250"))
DEMO_MAX_DURATION_SECONDS = int(os.getenv("DEMO_MAX_DURATION_SECONDS", "300"))  # 5 min
DEMO_RATE_LIMIT_PER_HOUR = int(os.getenv("DEMO_RATE_LIMIT_PER_HOUR", "3"))

# Audio analysis settings (used for BOTH demo and paid tool)
AUDIO_ANALYZE_MAX_SECONDS = int(os.getenv("AUDIO_ANALYZE_MAX_SECONDS", "1800"))  # cap analysis to 30 min
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_WINDOW_SECONDS = float(os.getenv("AUDIO_WINDOW_SECONDS", "0.5"))  # energy window
AUDIO_MIN_SEPARATION_SECONDS = float(os.getenv("AUDIO_MIN_SEPARATION_SECONDS", "6"))  # avoid similar clips

# In-memory demo rate limiting (OK for soft-launch; reset on deploy)
_demo_ip_hits: Dict[str, List[float]] = {}


def _now() -> float:
    return time.time()


def _safe_json_read(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default


def _safe_json_write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _load_credits() -> Dict[str, int]:
    data = _safe_json_read(CREDITS_PATH, {})
    if not isinstance(data, dict):
        return {}
    out: Dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k).lower()] = int(v)
        except Exception:
            continue
    return out


def _save_credits(d: Dict[str, int]) -> None:
    _safe_json_write(CREDITS_PATH, d)


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def make_token(email: str) -> str:
    payload = {"email": email.lower().strip(), "iat": int(_now())}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return "mf_" + _b64url_encode(raw)


def parse_token(token: str) -> Dict[str, Any]:
    if not token or not token.startswith("mf_"):
        raise HTTPException(status_code=401, detail="invalid token")
    raw = token[3:]
    try:
        payload = json.loads(_b64url_decode(raw).decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")
    if not isinstance(payload, dict):
        raise HTTPException(status_code=401, detail="invalid token")
    iat = int(payload.get("iat", 0))
    if iat <= 0 or (_now() - iat) > TOKEN_TTL_SECONDS:
        raise HTTPException(status_code=401, detail="token expired")
    email = str(payload.get("email", "")).lower().strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=401, detail="invalid token")
    return payload


def auth_email_from_request(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = auth.split(" ", 1)[1].strip()
    payload = parse_token(token)
    return payload["email"]


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def ffprobe_duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr[:4000]}")
    data = json.loads(p.stdout)
    dur = float(data["format"]["duration"])
    return max(0.0, dur)


def ensure_ffmpeg_present() -> Dict[str, bool]:
    def ok(binname: str) -> bool:
        try:
            p = _run([binname, "-version"])
            return p.returncode == 0
        except Exception:
            return False
    return {"ffmpeg": ok("ffmpeg"), "ffprobe": ok("ffprobe")}


def _public_base(request: Request) -> str:
    return str(request.base_url).rstrip("/")


def _hash_ip(ip: str) -> str:
    return hashlib.sha256(ip.encode("utf-8")).hexdigest()[:16]


def _client_ip(request: Request) -> str:
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _demo_rate_limit_check(ip: str) -> None:
    now = _now()
    hits = _demo_ip_hits.get(ip, [])
    hits = [t for t in hits if (now - t) < 3600]
    if len(hits) >= DEMO_RATE_LIMIT_PER_HOUR:
        raise HTTPException(status_code=429, detail="Demo rate limit reached. Try again later.")
    hits.append(now)
    _demo_ip_hits[ip] = hits


def _validate_demo_file(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > DEMO_MAX_FILE_MB:
        raise HTTPException(status_code=400, detail=f"Demo max file size is {DEMO_MAX_FILE_MB}MB.")


def ffmpeg_make_clip_with_watermark(src: Path, out_path: Path, start: int, length: int) -> None:
    vf = (
        "drawtext=text='MotionForge DEMO':"
        "x=w-tw-18:y=18:"
        "fontsize=24:fontcolor=white@0.75:"
        "box=1:boxcolor=black@0.35:boxborderw=10"
    )

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(length),
        "-i", str(src),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    p = _run(cmd)
    if p.returncode == 0 and out_path.exists():
        return

    # Fallback: no watermark
    cmd2 = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(length),
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    p2 = _run(cmd2)
    if p2.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[:1200]} | fallback: {p2.stderr[:1200]}")


def ffmpeg_make_clip(src: Path, out_path: Path, start: int, length: int) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-t", str(length),
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    p = _run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {p.stderr[:2000]}")


# --------------------------
# SMART START TIME SELECTION
# --------------------------
def _evenly_spaced_starts(duration: float, clip_seconds: int, count: int) -> List[int]:
    if count <= 0:
        return []
    max_start = max(0, int(duration) - clip_seconds)
    if max_start <= 0:
        return [0] * min(count, 1)
    if count == 1:
        return [max_start // 2]
    step = max_start / float(count - 1)
    starts = [int(round(i * step)) for i in range(count)]
    # clamp
    return [max(0, min(max_start, s)) for s in starts]


def _extract_audio_wav(src: Path, wav_path: Path, max_seconds: int) -> None:
    # Extract mono wav at fixed sample rate. Limit duration for speed.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", str(AUDIO_SAMPLE_RATE),
        "-t", str(max_seconds),
        str(wav_path),
    ]
    p = _run(cmd)
    if p.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"audio extract failed: {p.stderr[:1200]}")


def _compute_energy_peaks(wav_path: Path, top_k: int) -> List[Tuple[float, float]]:
    """
    Returns list of (time_seconds, energy) for top energy windows.
    No numpy; uses RMS energy in windows.
    """
    with wave.open(str(wav_path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        fr = w.getframerate()
        n_frames = w.getnframes()

        if n_channels != 1:
            raise RuntimeError("wav not mono")
        if sampwidth not in (2,):
            raise RuntimeError("unexpected sample width")

        window = max(1, int(fr * AUDIO_WINDOW_SECONDS))
        # Read all frames (bounded by max_seconds in extraction)
        raw = w.readframes(n_frames)

    # 16-bit signed little-endian
    total_samples = len(raw) // 2
    if total_samples <= window:
        return []

    fmt = "<" + ("h" * total_samples)
    samples = struct.unpack(fmt, raw)

    peaks: List[Tuple[float, float]] = []
    # Compute RMS per window
    # Stride half-window for smoother picking
    stride = max(1, window // 2)

    for start in range(0, total_samples - window, stride):
        chunk = samples[start:start + window]
        # RMS energy
        acc = 0.0
        for v in chunk:
            acc += float(v) * float(v)
        rms = (acc / float(window)) ** 0.5
        t = start / float(AUDIO_SAMPLE_RATE)
        peaks.append((t, rms))

    # pick top_k by energy
    peaks.sort(key=lambda x: x[1], reverse=True)
    return peaks[: max(50, top_k * 10)]  # keep more for spacing selection


def smart_start_times(
    src: Path,
    duration: float,
    clip_seconds: int,
    count: int,
) -> List[int]:
    """
    Choose clip start times based on audio energy peaks with spacing + fallback.
    Works for BOTH demo and paid tool. If anything fails -> fallback to evenly spaced.
    """
    # guard rails
    if count <= 0:
        return []
    max_start = max(0, int(duration) - clip_seconds)
    if max_start <= 0:
        return [0]

    # Do audio analysis on capped duration for performance
    analyze_seconds = int(min(max(duration, 1.0), float(AUDIO_ANALYZE_MAX_SECONDS)))

    try:
        with tempfile.TemporaryDirectory() as td:
            wav_path = Path(td) / "a.wav"
            _extract_audio_wav(src, wav_path, max_seconds=analyze_seconds)

            candidates = _compute_energy_peaks(wav_path, top_k=count)
            if not candidates:
                raise RuntimeError("no audio candidates")

            # Convert candidate times to start seconds, clamp, then select with min separation
            min_sep = max(float(AUDIO_MIN_SEPARATION_SECONDS), float(clip_seconds) * 0.6)
            selected: List[int] = []
            selected_f: List[float] = []

            for t, _e in candidates:
                # center clip around peak a bit earlier to catch the moment
                s = int(max(0.0, t - (clip_seconds * 0.25)))
                if s > max_start:
                    s = max_start
                # spacing check
                ok = True
                for prev in selected_f:
                    if abs(float(s) - prev) < min_sep:
                        ok = False
                        break
                if ok:
                    selected.append(s)
                    selected_f.append(float(s))
                if len(selected) >= count:
                    break

            # If not enough from peaks, fill with evenly spaced across full duration
            if len(selected) < count:
                fill = _evenly_spaced_starts(duration, clip_seconds, count)
                for s in fill:
                    if len(selected) >= count:
                        break
                    if all(abs(float(s) - prev) >= min_sep for prev in selected_f):
                        selected.append(s)
                        selected_f.append(float(s))

            # Final clamp and sort
            selected = [max(0, min(max_start, int(s))) for s in selected]
            selected = sorted(list(dict.fromkeys(selected)))  # uniq + ordered
            if len(selected) > count:
                selected = selected[:count]

            # If still empty, fallback
            if not selected:
                return _evenly_spaced_starts(duration, clip_seconds, count)

            # If we selected on capped duration but video is longer, we still accept;
            # the "fill" may cover full duration.
            return selected

    except Exception:
        # Hard fallback that cannot break tool
        return _evenly_spaced_starts(duration, clip_seconds, count)


app = FastAPI(title=APP_NAME, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEFAULT_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static outputs
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")
app.mount("/demo_outputs", StaticFiles(directory=str(DEMO_DIR)), name="demo_outputs")


@app.get("/health")
def health() -> Dict[str, Any]:
    ff = ensure_ffmpeg_present()
    stripe_mode = "disabled"
    if stripe and STRIPE_SECRET_KEY:
        stripe_mode = "live" if STRIPE_SECRET_KEY.startswith("sk_live") else "test"
    return {
        "status": "ok",
        "service": APP_NAME,
        "stripe_mode": stripe_mode,
        "refill_price_ids_loaded": sorted(list(PRICE_ID_TO_CREDITS.keys())),
        "cors_origins": DEFAULT_ORIGINS,
        "token_ttl_seconds": TOKEN_TTL_SECONDS,
        "ffmpeg": ff,
        "smart_split": {
            "audio_analyze_max_seconds": AUDIO_ANALYZE_MAX_SECONDS,
            "audio_window_seconds": AUDIO_WINDOW_SECONDS,
            "audio_min_separation_seconds": AUDIO_MIN_SEPARATION_SECONDS,
        },
        "demo": {
            "max_clips": DEMO_MAX_CLIPS,
            "max_file_mb": DEMO_MAX_FILE_MB,
            "max_duration_seconds": DEMO_MAX_DURATION_SECONDS,
            "rate_limit_per_hour": DEMO_RATE_LIMIT_PER_HOUR,
        },
    }


@app.get("/credits")
def credits(request: Request) -> Dict[str, Any]:
    email = auth_email_from_request(request)
    d = _load_credits()
    return {"ok": True, "email": email, "credits": int(d.get(email, 0))}


@app.post("/upload")
async def upload(
    request: Request,
    file: UploadFile = File(...),
    clip_seconds: int = Form(...),
) -> Dict[str, Any]:
    """
    Paid tool upload:
    - spends 1 credit per upload
    - returns clip URLs
    - NOW uses smart_start_times for more varied clips (with fallback)
    """
    email = auth_email_from_request(request)

    clip_seconds = int(clip_seconds)
    if clip_seconds <= 0 or clip_seconds > 120:
        raise HTTPException(status_code=400, detail="clip_seconds must be between 1 and 120")

    d = _load_credits()
    current = int(d.get(email, 0))
    if current <= 0:
        raise HTTPException(status_code=402, detail="insufficient credits")

    job_id = str(uuid.uuid4())
    in_path = UPLOADS_DIR / f"{job_id}_{file.filename}"
    with in_path.open("wb") as f:
        f.write(await file.read())

    # Spend 1 credit per job
    d[email] = current - 1
    _save_credits(d)

    out_dir = OUTPUTS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        duration = ffprobe_duration_seconds(in_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read video: {e}")

    # How many clips? same as before: floor(duration/clip_seconds), capped at 30, min 1
    n = int(duration // clip_seconds)
    if n <= 0:
        n = 1
    n = min(n, 30)

    starts = smart_start_times(in_path, duration=duration, clip_seconds=clip_seconds, count=n)

    clips: List[str] = []
    for i, start in enumerate(starts):
        out_file = out_dir / f"clip_{i:03d}.mp4"
        try:
            ffmpeg_make_clip(in_path, out_file, start=int(start), length=clip_seconds)
        except Exception:
            break
        clips.append(f"{_public_base(request)}/outputs/{job_id}/clip_{i:03d}.mp4")

    return {
        "ok": True,
        "job_id": job_id,
        "credits_spent": 1,
        "credits_remaining": int(d[email]),
        "clips": clips,
        "starts": starts,  # helpful for debugging; remove later if you want
    }


@app.get("/stripe/account")
def stripe_account() -> Dict[str, Any]:
    if not stripe or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=400, detail="stripe not configured")
    stripe.api_key = STRIPE_SECRET_KEY
    acct = stripe.Account.retrieve()
    return {
        "id": acct.get("id"),
        "country": acct.get("country"),
        "default_currency": acct.get("default_currency"),
        "charges_enabled": acct.get("charges_enabled"),
        "payouts_enabled": acct.get("payouts_enabled"),
    }


@app.get("/auth/access-from-session")
def access_from_session(session_id: str) -> Dict[str, Any]:
    if not stripe or not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=400, detail="stripe not configured")

    stripe.api_key = STRIPE_SECRET_KEY

    try:
        sess = stripe.checkout.Session.retrieve(
            session_id,
            expand=["line_items.data.price", "customer_details"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"stripe error: {e}")

    cd = sess.get("customer_details") or {}
    email = (cd.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="missing customer email in session")

    credits_to_add = 0
    li = sess.get("line_items") or {}
    data = li.get("data") or []
    for item in data:
        price = (item.get("price") or {})
        price_id = price.get("id")
        qty = int(item.get("quantity") or 1)
        if price_id and price_id in PRICE_ID_TO_CREDITS:
            credits_to_add += int(PRICE_ID_TO_CREDITS[price_id]) * qty

    d = _load_credits()
    d[email] = int(d.get(email, 0)) + int(credits_to_add)
    _save_credits(d)

    token = make_token(email)
    course_url = f"https://www.getmotionforge.com/course.html?token={token}"
    tool_url = f"https://www.getmotionforge.com/tool.html?token={token}"

    return {
        "ok": True,
        "email": email,
        "credits_added": int(credits_to_add),
        "token": token,
        "course_url": course_url,
        "tool_url": tool_url,
    }


@app.post("/demo/upload")
async def demo_upload(
    request: Request,
    file: UploadFile = File(...),
    clip_seconds: int = Form(...),
) -> Dict[str, Any]:
    """
    Demo mode:
    - No auth
    - Rate-limited per IP
    - Max duration and size caps
    - Generates up to 3 clips (watermarked)
    - NOW uses smart_start_times (with fallback)
    """
    ip = _client_ip(request)
    _demo_rate_limit_check(ip)

    clip_seconds = int(clip_seconds)
    if clip_seconds <= 0 or clip_seconds > 30:
        raise HTTPException(status_code=400, detail="clip_seconds must be between 1 and 30 for demo")

    demo_id = f"demo_{uuid.uuid4().hex[:10]}"
    in_path = DEMO_DIR / f"{demo_id}_{file.filename}"
    with in_path.open("wb") as f:
        f.write(await file.read())

    _validate_demo_file(in_path)

    try:
        duration = ffprobe_duration_seconds(in_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read video: {e}")

    if duration <= 0.5:
        raise HTTPException(status_code=400, detail="Video duration too short.")
    if duration > DEMO_MAX_DURATION_SECONDS:
        raise HTTPException(status_code=400, detail=f"Demo max duration is {DEMO_MAX_DURATION_SECONDS} seconds.")

    out_dir = DEMO_DIR / demo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    starts = smart_start_times(in_path, duration=duration, clip_seconds=clip_seconds, count=DEMO_MAX_CLIPS)
    starts = starts[:DEMO_MAX_CLIPS]

    clips: List[Dict[str, Any]] = []
    for i, start in enumerate(starts):
        out_file = out_dir / f"demo_clip_{i+1:02d}.mp4"
        try:
            ffmpeg_make_clip_with_watermark(in_path, out_file, start=int(start), length=clip_seconds)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Demo rendering failed: {e}")

        clips.append({
            "url": f"{_public_base(request)}/demo_outputs/{demo_id}/demo_clip_{i+1:02d}.mp4",
            "watermarked": True,
            "start": int(start),
        })

    return {
        "ok": True,
        "demo_id": demo_id,
        "ip_hash": _hash_ip(ip),
        "clips": clips,
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

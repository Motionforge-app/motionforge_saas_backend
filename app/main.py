from __future__ import annotations

import os
import uuid
import json
import time
import shutil
import wave
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional deps
try:
    import numpy as np
except Exception:
    np = None

try:
    import cv2  # opencv-python
except Exception:
    cv2 = None

# Stripe (optional)
try:
    import stripe  # pip install stripe
except Exception:
    stripe = None


APP_NAME = "motionforge_saas_backend"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
JOBS_DIR = DATA_DIR / "jobs"
OUTPUTS_DIR = DATA_DIR / "outputs"

for d in [DATA_DIR, JOBS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory stores (MVP)
JOBS: Dict[str, Dict[str, Any]] = {}
USERS: Dict[str, Dict[str, Any]] = {}  # {"email": {"credits": int, "updated_at": float}}

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Stripe defaults (TEST)
# These are YOUR confirmed Stripe TEST price IDs.
# -----------------------------
TEST_PRICE_IDS = {
    "starter": "price_1SdTz22L998MB1DPQbq4fN4b",  # 50 credits + course ($97)
    "creator": "price_1Sdu2g2L998MB1DPg748ZPbd",  # 200 credits ($49)
    "pro": "price_1Sdu3f2L998MB1DPJmndWTRx",      # 500 credits ($99)
}

DEFAULT_PACK_CREDITS = {
    "starter": 50,
    "creator": 200,
    "pro": 500,
}

# -----------------------------
# Helpers / Credits
# -----------------------------
def now_ts() -> float:
    return time.time()

def ensure_user(email: str) -> None:
    if email not in USERS:
        USERS[email] = {"credits": 0, "updated_at": now_ts()}

def get_credits(email: str) -> int:
    ensure_user(email)
    return int(USERS[email]["credits"])

def set_credits(email: str, credits: int) -> None:
    ensure_user(email)
    USERS[email]["credits"] = int(max(0, credits))
    USERS[email]["updated_at"] = now_ts()

def add_credits(email: str, delta: int) -> None:
    ensure_user(email)
    USERS[email]["credits"] = int(max(0, USERS[email]["credits"] + int(delta)))
    USERS[email]["updated_at"] = now_ts()

def deduct_credits(email: str, delta: int) -> None:
    ensure_user(email)
    cur = int(USERS[email]["credits"])
    nxt = cur - int(delta)
    if nxt < 0:
        raise HTTPException(status_code=402, detail=f"insufficient credits: have {cur}, need {delta}")
    USERS[email]["credits"] = nxt
    USERS[email]["updated_at"] = now_ts()

# -----------------------------
# Stripe mode + price lookup (NO double work)
# -----------------------------
def get_stripe_mode() -> Literal["test", "live", "disabled"]:
    key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not key:
        return "disabled"
    if key.startswith("sk_test_"):
        return "test"
    if key.startswith("sk_live_"):
        return "live"
    return "disabled"

def pack_credits(pack_id: str) -> int:
    pack_id = pack_id.lower()
    env_key = f"PACK_CREDITS_{pack_id.upper()}"
    val = os.getenv(env_key)
    if val is not None:
        try:
            return int(val)
        except Exception:
            pass
    return int(DEFAULT_PACK_CREDITS.get(pack_id, 0))

def price_id_for_pack(pack_id: str) -> Optional[str]:
    """
    Priority:
    - If LIVE: read env PRICE_ID_LIVE_*
    - If TEST: read env PRICE_ID_TEST_* else fallback to your confirmed TEST ids above
    """
    mode = get_stripe_mode()
    pack_id = pack_id.lower()

    if mode == "disabled":
        return None

    if mode == "live":
        env_key = f"PRICE_ID_LIVE_{pack_id.upper()}"
        pid = os.getenv(env_key)
        return pid.strip() if pid else None

    # mode == "test"
    env_key = f"PRICE_ID_TEST_{pack_id.upper()}"
    pid = os.getenv(env_key)
    if pid and pid.strip():
        return pid.strip()
    return TEST_PRICE_IDS.get(pack_id)

def stripe_enabled_or_throw() -> None:
    if stripe is None:
        raise HTTPException(status_code=500, detail="stripe package not installed. Run: pip install stripe")
    key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not key:
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY not set")
    mode = get_stripe_mode()
    if mode == "disabled":
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY invalid format (expected sk_test_ or sk_live_)")
    stripe.api_key = key

def stripe_urls() -> Tuple[str, str]:
    success = os.getenv("STRIPE_SUCCESS_URL", "http://localhost:3000/success?session_id={CHECKOUT_SESSION_ID}")
    cancel = os.getenv("STRIPE_CANCEL_URL", "http://localhost:3000/cancel")
    return success, cancel

# -----------------------------
# Models
# -----------------------------
Aspect = Literal["vertical", "square", "horizontal"]

class CreateJobRequest(BaseModel):
    email: str = Field(default="motionforge-tester@example.com")
    video_path: str

    segments: Optional[List[Dict[str, float]]] = None
    top_n: int = Field(default=3, ge=1, le=20)
    export_top_only: bool = True

    seg_len_s: float = Field(default=6.0, ge=2.0, le=60.0)
    seg_step_s: float = Field(default=3.0, ge=0.5, le=60.0)
    max_candidates: int = Field(default=60, ge=5, le=300)

    aspect: Aspect = "vertical"
    include_metadata_json: bool = True

    analysis_cost_per_segment: int = Field(default=1, ge=0, le=100)
    export_cost_per_mp4: int = Field(default=5, ge=0, le=500)

class CreateJobResponse(BaseModel):
    ok: bool
    job_id: str

# -----------------------------
# Utility
# -----------------------------
def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\nSTDERR:\n{p.stderr}")

def ffmpeg_exists() -> bool:
    return shutil.which("ffmpeg") is not None

def ffprobe_exists() -> bool:
    return shutil.which("ffprobe") is not None

def get_video_duration_seconds(video_path: str) -> Optional[float]:
    if not ffprobe_exists():
        return None
    try:
        cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", video_path]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            return None
        data = json.loads(p.stdout)
        dur = float(data["format"]["duration"])
        return dur if dur > 0 else None
    except Exception:
        return None

def normalize_and_validate_segments(
    segments: List[Dict[str, float]],
    duration_s: Optional[float],
    min_len_s: float = 1.8
) -> List[Dict[str, float]]:
    cleaned: List[Dict[str, float]] = []
    for seg in segments:
        s = safe_float(seg.get("start", 0.0))
        e = safe_float(seg.get("end", s))
        if e <= s:
            continue
        if duration_s is not None:
            if s >= duration_s:
                continue
            e = min(e, duration_s)
        if (e - s) < min_len_s:
            continue
        cleaned.append({"start": round(s, 3), "end": round(e, 3)})
    return cleaned

def generate_sliding_segments(duration_s: float, seg_len_s: float, seg_step_s: float, max_candidates: int) -> List[Dict[str, float]]:
    segs: List[Dict[str, float]] = []
    if duration_s <= 0:
        return segs
    if duration_s <= seg_len_s:
        return [{"start": 0.0, "end": round(duration_s, 3)}]
    t = 0.0
    while t < duration_s and len(segs) < max_candidates:
        end = min(duration_s, t + seg_len_s)
        segs.append({"start": round(t, 3), "end": round(end, 3)})
        if end >= duration_s:
            break
        t += seg_step_s
    return segs

# -----------------------------
# Upsell (A): only when exports blocked
# -----------------------------
def build_upsell_payload(missing_credits: int) -> Dict[str, Any]:
    mode = get_stripe_mode()

    packs = []
    for pid in ["starter", "creator", "pro"]:
        packs.append({
            "pack_id": pid,
            "credits": pack_credits(pid),
            "price_id": price_id_for_pack(pid),  # in TEST: always filled (fallback defaults)
            "mode": mode,
        })

    suggested = next((p for p in packs if p["credits"] >= missing_credits), packs[-1])

    return {
        "reason": "insufficient_export_credits",
        "missing_credits": int(max(0, missing_credits)),
        "suggested_pack": suggested,
        "all_packs": packs,
        "message": "You unlocked more high-performing clips. Add credits to export them.",
        "checkout_endpoint": "/stripe/create-checkout-session?email=YOUR_EMAIL&pack_id=starter|creator|pro",
    }

# -----------------------------
# Audio features
# -----------------------------
def extract_audio_envelope_ffmpeg(video_path: str, tmp_wav: Path, target_sr: int = 16000) -> Optional["np.ndarray"]:
    if np is None or not ffmpeg_exists():
        return None

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "wav",
        str(tmp_wav),
    ]
    try:
        run_cmd(cmd)
    except Exception:
        return None

    try:
        with wave.open(str(tmp_wav), "rb") as wf:
            n = wf.getnframes()
            raw = wf.readframes(n)
            if wf.getsampwidth() != 2:
                return None
            audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return None

    env = np.abs(audio)
    win = max(1, int(target_sr * 0.05))
    kernel = np.ones(win, dtype=np.float32) / float(win)
    env_smooth = np.convolve(env, kernel, mode="same")

    p95 = float(np.percentile(env_smooth, 95)) if env_smooth.size else 0.0
    if p95 <= 1e-6:
        return None

    return np.clip(env_smooth / p95, 0.0, 1.0).astype(np.float32)

def audio_features(video_path: str, start_s: float, end_s: float) -> Dict[str, float]:
    if np is None:
        return {"audio_peak": 0.0, "audio_delta": 0.0, "audio_available": 0.0}

    tmp_wav = JOBS_DIR / f"tmp_audio_{uuid.uuid4().hex}.wav"
    try:
        env = extract_audio_envelope_ffmpeg(video_path, tmp_wav)
        if env is None:
            return {"audio_peak": 0.0, "audio_delta": 0.0, "audio_available": 0.0}

        with wave.open(str(tmp_wav), "rb") as wf:
            sr = wf.getframerate()

        s0 = int(max(0.0, start_s) * sr)
        s1 = int(max(0.0, end_s) * sr)
        s1 = min(s1, env.size)
        s0 = min(s0, s1)

        seg = env[s0:s1]
        if seg.size < max(10, int(sr * 0.1)):
            return {"audio_peak": 0.0, "audio_delta": 0.0, "audio_available": 0.0}

        peak = float(np.max(seg))
        med = float(np.median(seg))
        delta = float(max(0.0, peak - med))
        return {"audio_peak": clamp01(peak), "audio_delta": clamp01(delta), "audio_available": 1.0}
    finally:
        try:
            if tmp_wav.exists():
                tmp_wav.unlink()
        except Exception:
            pass

# -----------------------------
# Motion features
# -----------------------------
def motion_score(video_path: str, start_s: float, end_s: float) -> Dict[str, float]:
    if cv2 is None or np is None:
        return {"motion": 0.0, "motion_available": 0.0}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"motion": 0.0, "motion_available": 0.0}

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

        step = max(1, int(fps * 0.25))
        start_f = int(max(0.0, start_s) * fps)
        end_f = int(max(0.0, end_s) * fps)
        if total_frames > 0:
            end_f = min(end_f, int(total_frames) - 1)

        if end_f <= start_f + step:
            return {"motion": 0.0, "motion_available": 0.0}

        diffs: List[float] = []
        prev_gray = None

        f = start_f
        while f <= end_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diffs.append(float(np.mean(diff)) / 255.0)

            prev_gray = gray
            f += step

        if len(diffs) < 3:
            return {"motion": 0.0, "motion_available": 0.0}

        p95 = float(np.percentile(diffs, 95))
        if p95 <= 1e-6:
            return {"motion": 0.0, "motion_available": 1.0}

        motion = float(np.mean(diffs) / p95)
        return {"motion": clamp01(motion), "motion_available": 1.0}
    finally:
        cap.release()

# -----------------------------
# Captions decision
# -----------------------------
def decide_captions_mode(features: Dict[str, float]) -> Dict[str, Any]:
    motion = safe_float(features.get("motion", 0.0))
    audio_peak = safe_float(features.get("audio_peak", 0.0))
    audio_delta = safe_float(features.get("audio_delta", 0.0))

    combined = (0.45 * motion) + (0.35 * audio_peak) + (0.20 * audio_delta)

    rule1 = combined >= 0.68
    rule2 = (motion >= 0.55 and audio_peak >= 0.85)
    rule3 = (motion >= 0.70 and audio_peak >= 0.65 and audio_delta >= 0.25)

    if rule1 or rule2 or rule3:
        return {
            "captions_mode_used": "headline",
            "captions_reason": "headline (motion+audio peak/delta high)",
            "headline_text": "Here’s the fastest way:",
            "combined_score": round(float(combined), 4),
            "score_breakdown": {
                "motion": round(float(motion), 4),
                "audio_peak": round(float(audio_peak), 4),
                "audio_delta": round(float(audio_delta), 4),
            },
        }

    return {
        "captions_mode_used": "whisper",
        "captions_reason": "ok (no strong hook detected)",
        "headline_text": None,
        "combined_score": round(float(combined), 4),
        "score_breakdown": {
            "motion": round(float(motion), 4),
            "audio_peak": round(float(audio_peak), 4),
            "audio_delta": round(float(audio_delta), 4),
        },
    }

# -----------------------------
# Rendering
# -----------------------------
def aspect_dims(aspect: str) -> tuple[int, int]:
    if aspect == "vertical":
        return (1080, 1920)
    if aspect == "square":
        return (1080, 1080)
    return (1920, 1080)

def render_clip_ffmpeg(job_id: str, idx: int, video_path: str, start_s: float, end_s: float, aspect: str) -> str:
    if not ffmpeg_exists():
        raise RuntimeError("ffmpeg not found. Install ffmpeg to render MP4 clips.")

    w, h = aspect_dims(aspect)
    clip_id = f"{job_id}_clip_{idx + 1}"
    out_mp4 = OUTPUTS_DIR / f"{clip_id}_{aspect}.mp4"

    vf = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(max(0.0, start_s)),
        "-to", str(max(0.0, end_s)),
        "-i", video_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_mp4),
    ]
    run_cmd(cmd)
    return str(out_mp4)

def write_metadata_json(job_id: str, idx: int, payload: Dict[str, Any]) -> str:
    clip_id = f"{job_id}_clip_{idx + 1}"
    out_json = OUTPUTS_DIR / f"{clip_id}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    return str(out_json)

# -----------------------------
# Job runner (credits + upsell A)
# -----------------------------
def run_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if not job:
        return

    job["status"] = "running"
    job["updated_at"] = now_ts()

    email: str = job["email"]
    video_path: str = job["video_path"]
    segments: List[Dict[str, float]] = job["segments"]
    top_n: int = int(job.get("top_n", 3))
    export_top_only: bool = bool(job.get("export_top_only", True))
    aspect: str = job.get("aspect", "vertical")
    include_metadata_json: bool = bool(job.get("include_metadata_json", True))

    analysis_cost_per_segment: int = int(job.get("analysis_cost_per_segment", 1))
    export_cost_per_mp4: int = int(job.get("export_cost_per_mp4", 5))

    credits_before = get_credits(email)

    # Charge analysis
    credits_charged_analysis = len(segments) * analysis_cost_per_segment
    if credits_charged_analysis > 0:
        deduct_credits(email, credits_charged_analysis)

    credits_after_analysis = get_credits(email)

    # Score segments
    scored: List[Dict[str, Any]] = []
    for i, seg in enumerate(segments):
        start_s = safe_float(seg.get("start", 0.0))
        end_s = safe_float(seg.get("end", start_s + 6.0))

        m = motion_score(video_path, start_s, end_s)
        a = audio_features(video_path, start_s, end_s)
        feats = {**m, **a}
        decision = decide_captions_mode(feats)

        scored.append({
            "index": i + 1,
            "start": start_s,
            "end": end_s,
            "features": feats,
            "captions_mode_used": decision["captions_mode_used"],
            "captions_reason": decision["captions_reason"],
            "headline_text": decision["headline_text"],
            "combined_score": decision["combined_score"],
            "score_breakdown": decision["score_breakdown"],
        })

    ranked = sorted(scored, key=lambda x: float(x.get("combined_score", 0.0)), reverse=True)
    winners_by_score = ranked[: min(top_n, len(ranked))]
    selected_indices = set([x["index"] for x in winners_by_score])

    # Export budget
    if export_cost_per_mp4 <= 0:
        max_affordable_exports = len(winners_by_score)
    else:
        max_affordable_exports = credits_after_analysis // export_cost_per_mp4

    winners_limited = winners_by_score[: min(len(winners_by_score), int(max_affordable_exports))]
    exported_indices = set([x["index"] for x in winners_limited])

    # Upsell only if blocked
    blocked_exports_count = len(winners_by_score) - len(winners_limited)
    missing_credits = blocked_exports_count * export_cost_per_mp4 if blocked_exports_count > 0 else 0
    upsell = build_upsell_payload(missing_credits) if blocked_exports_count > 0 else None

    # Charge exports
    credits_charged_exports = len(exported_indices) * export_cost_per_mp4
    if credits_charged_exports > 0:
        deduct_credits(email, credits_charged_exports)

    credits_after = get_credits(email)
    credits_total_charged = credits_charged_analysis + credits_charged_exports

    # Render results
    results: List[Dict[str, Any]] = []
    for item in scored:
        idx = item["index"]
        selected = idx in selected_indices
        exported = idx in exported_indices
        output_mp4 = None
        meta_json = None

        if (not export_top_only) or exported:
            if exported:
                output_mp4 = render_clip_ffmpeg(job_id, idx - 1, video_path, item["start"], item["end"], aspect)

        if include_metadata_json:
            meta_payload = {
                **item,
                "selected": selected,
                "exported": exported,
                "output": output_mp4,
                "aspect": aspect,
                "credits": {
                    "credits_before": credits_before,
                    "credits_after": credits_after,
                    "credits_charged_analysis": credits_charged_analysis,
                    "credits_charged_exports": credits_charged_exports,
                    "credits_total_charged": credits_total_charged,
                    "analysis_cost_per_segment": analysis_cost_per_segment,
                    "export_cost_per_mp4": export_cost_per_mp4,
                },
                "upsell": upsell,
            }
            meta_json = write_metadata_json(job_id, idx - 1, meta_payload)

        results.append({**item, "selected": selected, "exported": exported, "output": output_mp4, "meta_json": meta_json})

        job["progress"] = round(len(results) / max(1, len(scored)), 4)
        job["updated_at"] = now_ts()
        job["results"] = results

    job["credits_before"] = credits_before
    job["credits_after"] = credits_after
    job["credits_charged_analysis"] = credits_charged_analysis
    job["credits_charged_exports"] = credits_charged_exports
    job["credits_total_charged"] = credits_total_charged
    job["analysis_cost_per_segment"] = analysis_cost_per_segment
    job["export_cost_per_mp4"] = export_cost_per_mp4
    job["selected_exports_requested"] = len(winners_by_score)
    job["exports_completed"] = len(exported_indices)
    job["exports_blocked"] = blocked_exports_count
    job["upsell"] = upsell

    job["status"] = "done"
    job["progress"] = 1.0
    job["updated_at"] = now_ts()

# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": APP_NAME, "stripe_mode": get_stripe_mode()}

# Credits endpoints (local testing)
@app.get("/credits")
def credits_get(email: str = Query(...)) -> Dict[str, Any]:
    ensure_user(email)
    return {"ok": True, "email": email, "credits": get_credits(email)}

@app.post("/credits/add")
def credits_add(email: str = Query(...), amount: int = Query(...)) -> Dict[str, Any]:
    add_credits(email, int(amount))
    return {"ok": True, "email": email, "credits": get_credits(email)}

@app.post("/credits/set")
def credits_set(email: str = Query(...), credits: int = Query(...)) -> Dict[str, Any]:
    set_credits(email, int(credits))
    return {"ok": True, "email": email, "credits": get_credits(email)}

# Stripe: create checkout session for a pack_id (starter/creator/pro)
@app.post("/stripe/create-checkout-session")
def stripe_create_checkout_session(
    email: str = Query(...),
    pack_id: str = Query(...),
) -> Dict[str, Any]:
    pack_id = pack_id.strip().lower()
    if pack_id not in {"starter", "creator", "pro"}:
        raise HTTPException(status_code=400, detail="pack_id must be starter|creator|pro")

    stripe_enabled_or_throw()
    mode = get_stripe_mode()

    pid = price_id_for_pack(pack_id)
    if not pid:
        raise HTTPException(status_code=500, detail=f"price id not configured for {mode}:{pack_id}")

    success_url, cancel_url = stripe_urls()

    session = stripe.checkout.Session.create(
        mode="payment",
        line_items=[{"price": pid, "quantity": 1}],
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "email": email.strip().lower(),
            "pack_id": pack_id,
            "credits": str(pack_credits(pack_id)),
            "mode": mode,
        },
    )
    return {"ok": True, "checkout_url": session.url, "mode": mode, "pack_id": pack_id, "price_id": pid}

# Stripe webhook (minimal): credits user on successful checkout
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> Dict[str, Any]:
    stripe_enabled_or_throw()
    whsec = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
    if not whsec:
        raise HTTPException(status_code=500, detail="STRIPE_WEBHOOK_SECRET not set")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig, secret=whsec)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid webhook: {e}")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        md = session.get("metadata", {}) or {}
        email = (md.get("email") or "").strip().lower()
        credits = int(md.get("credits") or "0")

        if email and credits > 0:
            add_credits(email, credits)

    return {"ok": True}

@app.post("/jobs", response_model=CreateJobResponse)
def create_job(payload: CreateJobRequest) -> CreateJobResponse:
    email = payload.email.strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="email is required")

    video_path = payload.video_path.strip()
    if not video_path:
        raise HTTPException(status_code=400, detail="video_path is required")

    p = Path(video_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"video not found: {video_path}")

    duration_s = get_video_duration_seconds(str(p))

    # Determine segments
    if payload.segments:
        segments_raw = payload.segments
        segments = normalize_and_validate_segments(segments_raw, duration_s=duration_s)
    else:
        if duration_s is None:
            segments = [{"start": 0.0, "end": float(payload.seg_len_s)}]
        else:
            segments_raw = generate_sliding_segments(duration_s, payload.seg_len_s, payload.seg_step_s, payload.max_candidates)
            segments = normalize_and_validate_segments(segments_raw, duration_s=duration_s)

    if not segments:
        raise HTTPException(status_code=400, detail="No valid segments after validation.")

    ensure_user(email)

    # Must afford analysis (exports can be partially blocked -> upsell)
    analysis_total = len(segments) * int(payload.analysis_cost_per_segment)
    if analysis_total > get_credits(email):
        raise HTTPException(
            status_code=402,
            detail=f"insufficient credits for analysis: have {get_credits(email)}, need {analysis_total}"
        )

    job_id = str(uuid.uuid4())
    JOBS[job_id] = {
        "job_id": job_id,
        "email": email,
        "status": "queued",
        "progress": 0.0,
        "video_path": str(p),
        "duration_s": duration_s,
        "segments": segments,
        "top_n": payload.top_n,
        "export_top_only": payload.export_top_only,
        "aspect": payload.aspect,
        "include_metadata_json": payload.include_metadata_json,
        "analysis_cost_per_segment": payload.analysis_cost_per_segment,
        "export_cost_per_mp4": payload.export_cost_per_mp4,
        "results": [],
        "created_at": now_ts(),
        "updated_at": now_ts(),
    }

    run_job(job_id)
    return CreateJobResponse(ok=True, job_id=job_id)

@app.get("/status/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {"ok": True, **job}
from fastapi.responses import FileResponse
from pathlib import Path

# -----------------------------
# Serve outputs (read-only)
# -----------------------------

@app.get("/outputs/{job_id}")
def list_outputs(job_id: str):
    files = []
    for p in OUTPUTS_DIR.glob(f"{job_id}_*"):
        files.append({
            "filename": p.name,
            "url": f"/outputs/file/{p.name}",
            "type": p.suffix.lstrip("."),
        })

    if not files:
        raise HTTPException(status_code=404, detail="no outputs for this job")

    return {
        "ok": True,
        "job_id": job_id,
        "files": files,
    }


@app.get("/outputs/file/{filename}")
def get_output_file(filename: str):
    p = OUTPUTS_DIR / filename

    if not p.exists():
        raise HTTPException(status_code=404, detail="file not found")

    return FileResponse(
        path=str(p),
        filename=p.name,
        media_type="application/octet-stream",
    )

@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "motionforge_saas_backend",
        "message": "MotionForge backend is running"
    }


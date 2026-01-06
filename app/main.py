from __future__ import annotations

import os
import time
import json
import hmac
import base64
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware


# =============================================================================
# Config
# =============================================================================
SERVICE_NAME = "motionforge_saas_backend"
ENV = os.getenv("ENV", "production")

ADMIN_KEY = os.getenv("ADMIN_KEY", "")
TOKEN_SIGNING_SECRET = os.getenv("TOKEN_SIGNING_SECRET", ADMIN_KEY or "dev-secret")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "86400"))

CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "https://www.getmotionforge.com")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]

BASE_DIR = Path(__file__).resolve().parent
COURSE_DIR = BASE_DIR / "static" / "course"

# This is ONLY a sales/marketing fallback URL (not the course itself).
# IMPORTANT: Do NOT point this to /course/ to avoid loops.
COURSE_SALES_URL = os.getenv("COURSE_SALES_URL", "https://www.getmotionforge.com")


# =============================================================================
# App
# =============================================================================
app = FastAPI(title=SERVICE_NAME, version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Minimal credits store (file-based, simple, robust)
# =============================================================================
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CREDITS_FILE = DATA_DIR / "credits.json"


def _load_credits() -> Dict[str, int]:
    if not CREDITS_FILE.exists():
        return {}
    try:
        return json.loads(CREDITS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_credits(d: Dict[str, int]) -> None:
    CREDITS_FILE.write_text(json.dumps(d, indent=2, sort_keys=True), encoding="utf-8")


def _require_admin(request: Request) -> None:
    # Accept ADMIN_KEY via header or query param
    key = request.headers.get("x-admin-key") or request.query_params.get("admin_key")
    if not ADMIN_KEY or key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="admin key required")


# =============================================================================
# Token helpers (simple signed token: base64(payload).base64(sig))
# =============================================================================
def _now() -> int:
    return int(time.time())


def _sign(msg: bytes) -> str:
    sig = hmac.new(TOKEN_SIGNING_SECRET.encode("utf-8"), msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode("utf-8").rstrip("=")


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def make_token(email: str) -> str:
    payload = {"email": email, "iat": _now(), "exp": _now() + TOKEN_TTL_SECONDS}
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    msg = _b64(raw).encode("utf-8")
    sig = _sign(msg)
    return f"mf_{msg.decode('utf-8')}.{sig}"


def verify_token(token: str) -> Dict[str, Any]:
    if not token or not token.startswith("mf_") or "." not in token:
        raise HTTPException(status_code=401, detail="invalid token")

    body = token[3:]
    msg_b64, sig = body.split(".", 1)
    expected = _sign(msg_b64.encode("utf-8"))
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=401, detail="invalid token signature")

    payload = json.loads(_b64d(msg_b64).decode("utf-8"))
    exp = int(payload.get("exp", 0))
    if _now() > exp:
        raise HTTPException(status_code=401, detail="token expired")

    email = payload.get("email")
    if not email or "@" not in email:
        raise HTTPException(status_code=401, detail="invalid token payload")

    return payload


def _get_token_from_request(request: Request) -> Optional[str]:
    # Prefer querystring token for course pages
    t = request.query_params.get("token")
    if t:
        return t
    # fallback: Authorization: Bearer ...
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


# =============================================================================
# Helpers: serving course files safely (NO redirects on missing/invalid token)
# =============================================================================
def _safe_course_file(name: str) -> FileResponse:
    path = (COURSE_DIR / name).resolve()
    if COURSE_DIR not in path.parents and path != COURSE_DIR:
        raise HTTPException(status_code=400, detail="bad path")
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"course file not found: {name}")
    return FileResponse(str(path), media_type="text/html; charset=utf-8")


def _course_index_response(request: Request) -> FileResponse:
    # Always serve index.html (even without token). Module pages can show their own note.
    return _safe_course_file("index.html")


def _course_missing_html(msg: str) -> HTMLResponse:
    html = f"""
    <!doctype html>
    <html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Course page not found</title>
    <style>
      body{{background:#070912;color:#e8ecff;font-family:Arial,Helvetica,sans-serif;margin:0;padding:24px}}
      .card{{max-width:760px;margin:0 auto;background:#0b1020;border:1px solid rgba(255,255,255,.10);
             border-radius:16px;padding:18px}}
      a{{color:#00eaff;text-decoration:none;font-weight:800}}
      p{{color:#a9b2d6;line-height:1.6}}
    </style></head>
    <body><div class="card">
      <h2>Course page not found</h2>
      <p>{msg}</p>
      <p><a href="/course/">Back to dashboard</a></p>
    </div></body></html>
    """
    return HTMLResponse(html, status_code=404)


# =============================================================================
# Core endpoints
# =============================================================================
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "env": ENV,
        "cors_origins": CORS_ORIGINS,
        "token_ttl_seconds": TOKEN_TTL_SECONDS,
        "course_dir": str(COURSE_DIR),
        "course_dir_exists": COURSE_DIR.exists(),
        "course_files": sorted([p.name for p in COURSE_DIR.glob("*.html")]) if COURSE_DIR.exists() else [],
    }


# =============================================================================
# Auth endpoints (simple)
# =============================================================================
@app.post("/auth/dev-token")
def auth_dev_token(email: str, request: Request) -> Dict[str, Any]:
    _require_admin(request)
    return {"email": email, "token": make_token(email)}


@app.get("/auth/session")
def auth_session(request: Request) -> Dict[str, Any]:
    token = _get_token_from_request(request)
    payload = verify_token(token or "")
    return {"ok": True, "email": payload["email"], "exp": payload["exp"]}


@app.get("/me")
def me(request: Request) -> Dict[str, Any]:
    # Optional: returns email if token valid, else ok false
    token = _get_token_from_request(request)
    if not token:
        return {"ok": False}
    try:
        payload = verify_token(token)
        return {"ok": True, "email": payload["email"]}
    except HTTPException:
        return {"ok": False}


# =============================================================================
# Credits endpoints (simple)
# =============================================================================
@app.get("/credits")
def credits_get(email: str) -> Dict[str, Any]:
    d = _load_credits()
    return {"email": email, "credits": int(d.get(email.lower(), 0))}


@app.post("/credits/admin/add")
def credits_admin_add(email: str, amount: int, request: Request) -> Dict[str, Any]:
    _require_admin(request)
    if amount == 0:
        return {"ok": True, "email": email, "credits": credits_get(email)["credits"]}
    d = _load_credits()
    key = email.lower()
    d[key] = int(d.get(key, 0)) + int(amount)
    if d[key] < 0:
        d[key] = 0
    _save_credits(d)
    return {"ok": True, "email": email, "credits": d[key]}


# =============================================================================
# Course endpoints (NO REDIRECT LOOPS, always returns HTML)
# =============================================================================
@app.get("/course/")
def course_index(request: Request):
    # Token is OPTIONAL for viewing dashboard; live gating happens by your purchase flow.
    return _course_index_response(request)


@app.get("/course/module-{n}")
def course_module(request: Request, n: int):
    # Friendly route without .html
    return _safe_course_file(f"module-{n}.html")


@app.get("/course/module-{n}.html")
def course_module_html(request: Request, n: int):
    return _safe_course_file(f"module-{n}.html")


# =============================================================================
# Stripe webhook placeholder (so the app doesn't crash if Railway hits it)
# =============================================================================
@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    # Keep this endpoint alive; you can wire real Stripe logic later.
    # If STRIPE_WEBHOOK_SECRET is missing, we still return 200 to avoid crash loops.
    _ = await request.body()
    return JSONResponse({"ok": True})


# =============================================================================
# Root
# =============================================================================
@app.get("/")
def root():
    return RedirectResponse(url="/status")

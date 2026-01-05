# app/main.py
from __future__ import annotations

import os
import json
import time
import hmac
import base64
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

import stripe

# =============================================================================
# Config
# =============================================================================

SERVICE_NAME = "motionforge_saas_backend"
DB_PATH = os.getenv("DB_PATH", "motionforge.db")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_MODE = os.getenv("STRIPE_MODE", "live").lower()  # informational only

# IMPORTANT: must be JSON like (single line recommended in Railway):
# {"price_...":10,"price_...":50,"price_...":250,"price_...":500}
STRIPE_PRICE_ID_TO_CREDITS_RAW = os.getenv("STRIPE_PRICE_ID_TO_CREDITS", "{}")

# Token signing secret (NOT the Stripe webhook secret)
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET", "")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "86400"))

# Optional admin key (for dev-only helpers)
ADMIN_KEY = os.getenv("ADMIN_KEY", "")

# CORS
CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "https://www.getmotionforge.com")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]

# Where to send people if they try to access the course without a valid token
COURSE_SALES_URL = os.getenv("COURSE_SALES_URL", "https://getmotionforge.com/course.html")

# Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY


def _clean_json_env(raw: str) -> str:
    """
    Railway / shells sometimes wrap JSON in quotes or keep newlines.
    We normalize to a parsable JSON string.
    """
    s = (raw or "").strip()

    # Strip surrounding quotes if someone pasted: " {...} "
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()

    # Remove accidental trailing semicolon
    if s.endswith(";"):
        s = s[:-1].strip()

    return s or "{}"


def _parse_price_map() -> Dict[str, int]:
    """
    Returns dict[price_id] = credits_per_unit
    """
    s = _clean_json_env(STRIPE_PRICE_ID_TO_CREDITS_RAW)
    try:
        d = json.loads(s)
        out: Dict[str, int] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if not isinstance(k, str):
                    continue
                try:
                    out[k.strip()] = int(v)
                except Exception:
                    continue
        return out
    except Exception:
        return {}


STRIPE_PRICE_ID_TO_CREDITS = _parse_price_map()

# =============================================================================
# DB helpers
# =============================================================================


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS credits (
            email TEXT PRIMARY KEY,
            credits INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tokens (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    )

    # Idempotency for Stripe webhook events
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS processed_events (
            event_id TEXT PRIMARY KEY,
            created_at INTEGER NOT NULL
        );
        """
    )

    conn.commit()
    conn.close()


def get_credits(email: str) -> int:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT credits FROM credits WHERE email = ?;", (email.lower(),))
    row = cur.fetchone()
    conn.close()
    return int(row["credits"]) if row else 0


def add_credits(email: str, amount: int) -> int:
    if amount <= 0:
        return get_credits(email)

    now = int(time.time())
    em = email.lower()

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT credits FROM credits WHERE email = ?;", (em,))
    row = cur.fetchone()

    if row:
        new_val = int(row["credits"]) + int(amount)
        cur.execute("UPDATE credits SET credits=?, updated_at=? WHERE email=?;", (new_val, now, em))
    else:
        new_val = int(amount)
        cur.execute("INSERT INTO credits(email, credits, updated_at) VALUES(?,?,?);", (em, new_val, now))

    conn.commit()
    conn.close()
    return new_val


def mark_event_processed(event_id: str) -> bool:
    """
    Returns True if we successfully marked it now.
    Returns False if it was already processed.
    """
    now = int(time.time())
    conn = db()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO processed_events(event_id, created_at) VALUES(?,?);", (event_id, now))
        conn.commit()
        return True
    except Exception:
        # already exists
        return False
    finally:
        conn.close()


# =============================================================================
# Tokens (simple signed + stored tokens)
# =============================================================================


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("utf-8").rstrip("=")


def _b64url_decode(s: str) -> bytes:
    padding = "=" * ((4 - len(s) % 4) % 4)
    return base64.urlsafe_b64decode((s + padding).encode("utf-8"))


def mint_token(email: str, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    if not ACCESS_TOKEN_SECRET:
        raise HTTPException(status_code=500, detail="ACCESS_TOKEN_SECRET not set")

    now = int(time.time())
    exp = now + int(ttl_seconds)

    payload = {"email": email.lower(), "exp": exp, "iat": now}
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = hmac.new(ACCESS_TOKEN_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256).digest()

    token = f"mf_{_b64url(payload_bytes)}.{_b64url(sig)}"

    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO tokens(token, email, expires_at, created_at) VALUES(?,?,?,?);",
        (token, email.lower(), exp, now),
    )
    conn.commit()
    conn.close()

    return token


def verify_token(token: str) -> str:
    if not token or not token.startswith("mf_") or "." not in token:
        raise HTTPException(status_code=401, detail="invalid token")

    if not ACCESS_TOKEN_SECRET:
        raise HTTPException(status_code=500, detail="ACCESS_TOKEN_SECRET not set")

    try:
        _, rest = token.split("mf_", 1)
        payload_b64, sig_b64 = rest.split(".", 1)
        payload_bytes = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)

        expected = hmac.new(ACCESS_TOKEN_SECRET.encode("utf-8"), payload_bytes, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            raise HTTPException(status_code=401, detail="invalid token signature")

        payload = json.loads(payload_bytes.decode("utf-8"))
        email = str(payload.get("email", "")).lower()
        exp = int(payload.get("exp", 0))

        if not email or exp <= int(time.time()):
            raise HTTPException(status_code=401, detail="token expired")

        # verify stored token exists
        conn = db()
        cur = conn.cursor()
        cur.execute("SELECT email, expires_at FROM tokens WHERE token=?;", (token,))
        row = cur.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=401, detail="token not recognized")
        if int(row["expires_at"]) <= int(time.time()):
            raise HTTPException(status_code=401, detail="token expired")

        return email
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="invalid token")


def bearer_email(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing Authorization: Bearer <token>")
    token = auth.split(" ", 1)[1].strip()
    return verify_token(token)


# =============================================================================
# Course (server-side gated HTML)
# =============================================================================

COURSE_DIR = Path(__file__).resolve().parent / "static" / "course"


def _course_redirect() -> RedirectResponse:
    return RedirectResponse(url=COURSE_SALES_URL, status_code=302)


def _require_course_token(request: Request) -> str:
    token = request.query_params.get("token") or ""
    # Use your real verification (signature + expiry + DB existence)
    verify_token(token)
    return token


def _serve_course_file(filename: str) -> FileResponse:
    path = COURSE_DIR / filename
    if not path.exists():
        return FileResponse(COURSE_DIR / "index.html")
    return FileResponse(path)


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


@app.on_event("startup")
def _startup() -> None:
    init_db()


# =============================================================================
# Public status endpoints
# =============================================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "stripe_mode": STRIPE_MODE,
        "refill_price_ids_loaded": sorted(list(STRIPE_PRICE_ID_TO_CREDITS.keys())),
        "cors_origins": CORS_ORIGINS,
        "token_ttl_seconds": TOKEN_TTL_SECONDS,
        "course_dir_exists": COURSE_DIR.exists(),
    }


# =============================================================================
# Course endpoints (TOKEN REQUIRED)
# =============================================================================

@app.get("/course/")
def course_index(request: Request):
    try:
        _require_course_token(request)
    except HTTPException:
        return _course_redirect()
    return _serve_course_file("index.html")


# Serve /course/module-1  and /course/module-1.html (both)
@app.get("/course/module-{n}")
def course_module(request: Request, n: int):
    try:
        _require_course_token(request)
    except HTTPException:
        return _course_redirect()
    return _serve_course_file(f"module-{n}.html")


@app.get("/course/module-{n}.html")
def course_module_html(request: Request, n: int):
    try:
        _require_course_token(request)
    except HTTPException:
        return _course_redirect()
    return _serve_course_file(f"module-{n}.html")


# =============================================================================
# Auth endpoints
# =============================================================================

@app.post("/auth/dev-token")
def auth_dev_token(email: str, admin_key: Optional[str] = None) -> Dict[str, Any]:
    if not ADMIN_KEY or (admin_key or "") != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="admin key required")
    token = mint_token(email=email)
    return {"token": token, "email": email.lower(), "expires_in": TOKEN_TTL_SECONDS}


@app.get("/auth/session")
def auth_from_checkout_session(session_id: str) -> Dict[str, Any]:
    """
    Used by success.html:
      https://www.getmotionforge.com/success.html?session_id=cs_...
    Frontend calls:
      GET /auth/session?session_id=cs_...
    Returns token + recommended access url.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=500, detail="STRIPE_SECRET_KEY not set")

    try:
        sess = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid session_id: {str(e)[:200]}")

    email = None
    if isinstance(sess, dict):
        cd = (sess.get("customer_details") or {}) if isinstance(sess.get("customer_details"), dict) else {}
        email = cd.get("email") or sess.get("customer_email")
    else:
        email = getattr(getattr(sess, "customer_details", None), "email", None) or getattr(sess, "customer_email", None)

    if not email:
        raise HTTPException(status_code=400, detail="no email on checkout session")

    token = mint_token(email=email)

    # Keep your existing access page
    access_url = f"https://www.getmotionforge.com/access.html?token={token}"

    # Also provide a direct, gated course link (best)
    course_url = f"https://motionforgesaasbackend-production.up.railway.app/course/?token={token}"

    return {"email": email.lower(), "token": token, "access_url": access_url, "course_url": course_url}


# =============================================================================
# Credits endpoints (require Bearer token)
# =============================================================================

@app.get("/me")
def me(request: Request) -> Dict[str, Any]:
    email = bearer_email(request)
    return {"email": email, "credits": get_credits(email)}


@app.get("/credits")
def credits(request: Request) -> Dict[str, Any]:
    email = bearer_email(request)
    return {"email": email, "credits": get_credits(email)}


@app.post("/credits/admin/add")
def credits_admin_add(email: str, amount: int, admin_key: Optional[str] = None) -> Dict[str, Any]:
    if not ADMIN_KEY or (admin_key or "") != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="admin key required")
    new_val = add_credits(email, int(amount))
    return {"email": email.lower(), "credits": new_val}


# =============================================================================
# Stripe webhook
# =============================================================================

@app.post("/stripe/webhook")
async def stripe_webhook(request: Request) -> JSONResponse:
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="STRIPE_WEBHOOK_SECRET not set")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    if not sig:
        raise HTTPException(status_code=400, detail="missing stripe-signature header")

    try:
        event = stripe.Webhook.construct_event(
            payload=payload,
            sig_header=sig,
            secret=STRIPE_WEBHOOK_SECRET,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid signature: {str(e)[:200]}")

    etype = event.get("type")
    event_id = event.get("id") or ""

    # ACK unknown events to avoid retries
    if etype != "checkout.session.completed":
        return JSONResponse({"status": "ignored", "type": etype})

    # Idempotency guard
    if event_id and not mark_event_processed(event_id):
        return JSONResponse({"status": "ok", "type": etype, "deduped": True})

    data_obj = (event.get("data") or {}).get("object") or {}
    session_id = data_obj.get("id")
    if not session_id:
        return JSONResponse(status_code=200, content={"status": "error", "reason": "missing checkout session id"})

    cd = data_obj.get("customer_details") or {}
    email = cd.get("email") if isinstance(cd, dict) else None
    email = email or data_obj.get("customer_email")

    try:
        sess = stripe.checkout.Session.retrieve(
            session_id,
            expand=["line_items.data.price"],
        )
    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"status": "error", "reason": "failed to retrieve checkout session", "detail": str(e)[:200]},
        )

    if not email:
        if isinstance(sess, dict):
            cd2 = (sess.get("customer_details") or {}) if isinstance(sess.get("customer_details"), dict) else {}
            email = cd2.get("email") or sess.get("customer_email")
        else:
            email = getattr(getattr(sess, "customer_details", None), "email", None) or getattr(sess, "customer_email", None)

    if not email:
        return JSONResponse(status_code=200, content={"status": "error", "reason": "no email on checkout session"})

    price_ids: List[str] = []
    qtys: List[int] = []

    if isinstance(sess, dict):
        li = sess.get("line_items") or {}
        items = li.get("data", []) if isinstance(li, dict) else []
        for it in items:
            if not isinstance(it, dict):
                continue
            q = int(it.get("quantity") or 1)
            p = it.get("price") or {}
            pid = p.get("id") if isinstance(p, dict) else None
            if pid:
                price_ids.append(str(pid))
                qtys.append(q)
    else:
        items = getattr(getattr(sess, "line_items", None), "data", []) or []
        for it in items:
            q = int(getattr(it, "quantity", 1) or 1)
            p = getattr(it, "price", None)
            pid = getattr(p, "id", None) if p is not None else None
            if pid:
                price_ids.append(str(pid))
                qtys.append(q)

    if not price_ids:
        return JSONResponse(
            status_code=200,
            content={"status": "error", "reason": "no price ids found", "session_id": session_id},
        )

    credits_to_add = 0
    matched: List[Dict[str, Any]] = []
    for pid, q in zip(price_ids, qtys):
        pid_clean = pid.strip()
        if pid_clean in STRIPE_PRICE_ID_TO_CREDITS:
            per_unit = int(STRIPE_PRICE_ID_TO_CREDITS[pid_clean])
            credits_to_add += per_unit * int(q)
            matched.append({"price_id": pid_clean, "qty": int(q), "credits_per_unit": per_unit})

    if credits_to_add <= 0:
        return JSONResponse(
            status_code=200,
            content={
                "status": "error",
                "reason": "no matching price_id in STRIPE_PRICE_ID_TO_CREDITS",
                "prices_seen": price_ids,
                "mapping_loaded": STRIPE_PRICE_ID_TO_CREDITS,
                "email": email.lower(),
                "session_id": session_id,
            },
        )

    new_total = add_credits(email, credits_to_add)

    return JSONResponse(
        {
            "status": "ok",
            "type": etype,
            "event_id": event_id,
            "email": email.lower(),
            "matched": matched,
            "credits_added": credits_to_add,
            "credits_total": new_total,
        }
    )

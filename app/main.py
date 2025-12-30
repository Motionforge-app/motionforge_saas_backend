# app/main.py
from __future__ import annotations

import os
import json
import time
import hmac
import base64
import hashlib
import sqlite3
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import JSONResponse, RedirectResponse
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

# IMPORTANT: must be JSON like:
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

# Stripe
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

def _parse_price_map() -> Dict[str, int]:
    try:
        d = json.loads(STRIPE_PRICE_ID_TO_CREDITS_RAW or "{}")
        out: Dict[str, int] = {}
        for k, v in d.items():
            if not isinstance(k, str):
                continue
            try:
                out[k] = int(v)
            except Exception:
                continue
        return out
    except Exception:
        return {}

STRIPE_PRICE_ID_TO_CREDITS = _parse_price_map()

# =============================================================================
# DB
# =============================================================================

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS credits (
            email TEXT PRIMARY KEY,
            credits INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            token TEXT PRIMARY KEY,
            email TEXT NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL
        );
    """)
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

        # also verify stored token exists (allows revocation later)
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
    }

# =============================================================================
# Auth endpoints
# =============================================================================

@app.post("/auth/dev-token")
def auth_dev_token(email: str, admin_key: Optional[str] = None) -> Dict[str, Any]:
    if not ADMIN_KEY:
        raise HTTPException(status_code=403, detail="admin key required")
    if (admin_key or "") != ADMIN_KEY:
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
        # stripe object
        email = getattr(getattr(sess, "customer_details", None), "email", None) or getattr(sess, "customer_email", None)

    if not email:
        raise HTTPException(status_code=400, detail="no email on checkout session")

    token = mint_token(email=email)
    # recommended access page: add token as query param
    access_url = f"https://www.getmotionforge.com/access.html?token={token}"
    return {"email": email.lower(), "token": token, "access_url": access_url}

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

# Optional admin-only helper to manually add credits (keep or delete)
@app.post("/credits/admin/add")
def credits_admin_add(email: str, amount: int, admin_key: Optional[str] = None) -> Dict[str, Any]:
    if not ADMIN_KEY:
        raise HTTPException(status_code=403, detail="admin key required")
    if (admin_key or "") != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="admin key required")
    new_val = add_credits(email, int(amount))
    return {"email": email.lower(), "credits": new_val}

# =============================================================================
# Stripe webhook (robust, no crashes)
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
    data_obj = (event.get("data") or {}).get("object") or {}

    # Always ACK unknown events to avoid retries
    if etype != "checkout.session.completed":
        return JSONResponse({"status": "ignored", "type": etype})

    session_id = data_obj.get("id")
    if not session_id:
        raise HTTPException(status_code=400, detail="missing checkout session id")

    # Email
    cd = data_obj.get("customer_details") or {}
    email = None
    if isinstance(cd, dict):
        email = cd.get("email")
    email = email or data_obj.get("customer_email")
    if not email:
        raise HTTPException(status_code=400, detail="no email in checkout session")

    # Fetch line items safely
    try:
        line_items = stripe.checkout.Session.list_line_items(session_id, limit=5)
        items = line_items.get("data", []) if isinstance(line_items, dict) else getattr(line_items, "data", [])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to fetch line items: {str(e)[:200]}")

    if not items:
        raise HTTPException(status_code=400, detail="no line items found for checkout session")

    # Price id (support both shapes)
    price_ids = []
    for it in items:
        price = it.get("price") if isinstance(it, dict) else getattr(it, "price", None)
        if isinstance(price, dict) and price.get("id"):
            price_ids.append(price["id"])
        elif price is not None and getattr(price, "id", None):
            price_ids.append(getattr(price, "id"))

    if not price_ids:
        raise HTTPException(status_code=400, detail="no price ids found in line items")

    # Find matching mapping (first match)
    credits_to_add = None
    matched_price = None
    for pid in price_ids:
        if pid in STRIPE_PRICE_ID_TO_CREDITS:
            credits_to_add = int(STRIPE_PRICE_ID_TO_CREDITS[pid])
            matched_price = pid
            break

    if not credits_to_add:
        raise HTTPException(
            status_code=400,
            detail=f"no matching price_id in STRIPE_PRICE_ID_TO_CREDITS for session line items (prices={price_ids})"
        )

    new_total = add_credits(email, credits_to_add)
    return JSONResponse(
        {
            "status": "ok",
            "type": etype,
            "email": email.lower(),
            "price_id": matched_price,
            "credits_added": credits_to_add,
            "credits_total": new_total,
        }
    )

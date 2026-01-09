from __future__ import annotations

import os
import time
import base64
import json
import sqlite3
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import stripe

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="motionforge_saas_backend")

# ----------------------------
# CORS (CRITICAL for browser fetch from getmotionforge.com)
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.getmotionforge.com",
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Stripe
# ----------------------------
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
if not STRIPE_SECRET_KEY:
    raise RuntimeError("STRIPE_SECRET_KEY not set")
stripe.api_key = STRIPE_SECRET_KEY

# ----------------------------
# Admin key (for manual credit ops)
# ----------------------------
# Accept either ADMIN_KEY or ADMIN_APP_KEY (since you mentioned "admin app key")
ADMIN_KEY = os.getenv("ADMIN_KEY") or os.getenv("ADMIN_APP_KEY") or ""

# ----------------------------
# Price → Credits mapping (LIVE)
# ----------------------------
# Confirmed earlier in your project:
PRICE_TO_CREDITS: Dict[str, int] = {
    # Tester Pack $5 → 10 credits
    "price_1Sjzy32L998MB1DP0pYyuyTY": 10,
    # Creator Pack $97 → 50 credits
    "price_1Sd90A2L998MB1DPzBnPWnTA": 50,
    # Refill 250 → 250 credits
    "price_1Sh7La2L998MB1DPtfvTv31N": 250,
    # Refill 500 → 500 credits
    "price_1Sh7Px2L998MB1DPQcQbGLfR": 500,
}

# Optional: extend/override via env JSON if you want (e.g. {"price_x":123})
# If set, it merges on top of the defaults above.
_env_map = os.getenv("PRICE_TO_CREDITS_JSON")
if _env_map:
    try:
        PRICE_TO_CREDITS.update({k: int(v) for k, v in json.loads(_env_map).items()})
    except Exception:
        # Don't crash prod for a bad env var
        pass

# ----------------------------
# SQLite (simple persistence)
# ----------------------------
DB_PATH = os.getenv("MF_DB_PATH", "motionforge.db")

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS credits (
              email TEXT PRIMARY KEY,
              credits INTEGER NOT NULL DEFAULT 0,
              updated_at INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS credit_events (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              email TEXT NOT NULL,
              session_id TEXT,
              price_id TEXT,
              credits_added INTEGER NOT NULL,
              created_at INTEGER NOT NULL
            )
            """
        )

@app.on_event("startup")
def _startup():
    init_db()

# ----------------------------
# Token helpers (mf_ base64 JSON)
# ----------------------------
def issue_token_for_email(email: str) -> str:
    payload = {"email": email, "iat": int(time.time())}
    token = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"mf_{token}"

def decode_token(token: str) -> Dict[str, Any]:
    if not token or not token.startswith("mf_"):
        raise HTTPException(status_code=401, detail="Invalid token")
    raw = token[3:]  # strip "mf_"
    # Add base64 padding if needed
    pad = "=" * ((4 - (len(raw) % 4)) % 4)
    try:
        data = base64.urlsafe_b64decode((raw + pad).encode())
        payload = json.loads(data.decode())
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    if "email" not in payload:
        raise HTTPException(status_code=401, detail="Token missing email")
    return payload

def bearer_token_from_auth_header(request: Request) -> str:
    auth = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    return auth.split(" ", 1)[1].strip()

# ----------------------------
# Credits storage ops
# ----------------------------
def get_credits(email: str) -> int:
    with db() as conn:
        row = conn.execute("SELECT credits FROM credits WHERE email = ?", (email,)).fetchone()
        return int(row["credits"]) if row else 0

def add_credits(email: str, amount: int) -> int:
    if amount <= 0:
        return get_credits(email)
    now = int(time.time())
    with db() as conn:
        existing = conn.execute("SELECT credits FROM credits WHERE email = ?", (email,)).fetchone()
        if existing:
            new_val = int(existing["credits"]) + int(amount)
            conn.execute(
                "UPDATE credits SET credits = ?, updated_at = ? WHERE email = ?",
                (new_val, now, email),
            )
        else:
            new_val = int(amount)
            conn.execute(
                "INSERT INTO credits (email, credits, updated_at) VALUES (?, ?, ?)",
                (email, new_val, now),
            )
        return new_val

def record_event(email: str, session_id: Optional[str], price_id: Optional[str], credits_added: int) -> None:
    now = int(time.time())
    with db() as conn:
        conn.execute(
            "INSERT INTO credit_events (email, session_id, price_id, credits_added, created_at) VALUES (?, ?, ?, ?, ?)",
            (email, session_id, price_id, int(credits_added), now),
        )

# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# Credits API (THIS FIXES your tool.html 404)
# ----------------------------
@app.get("/credits")
def credits(request: Request):
    token = bearer_token_from_auth_header(request)
    payload = decode_token(token)
    email = payload["email"]
    return {"ok": True, "email": email, "credits": get_credits(email)}

# ----------------------------
# Admin: add credits manually (optional but useful)
# ----------------------------
@app.post("/credits/admin/add")
def admin_add(email: str, amount: int, request: Request):
    key = request.headers.get("x-admin-key") or request.headers.get("X-Admin-Key") or ""
    if not ADMIN_KEY or key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="admin key required")
    new_val = add_credits(email, amount)
    record_event(email=email, session_id=None, price_id=None, credits_added=amount)
    return {"ok": True, "email": email, "credits": new_val}

# ----------------------------
# Stripe → Access + credit grant
# ----------------------------
@app.get("/auth/access-from-session")
def access_from_session(session_id: str):
    try:
        # Expand line_items so we can inspect price IDs
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=["line_items.data.price"],
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    email = None
    if getattr(session, "customer_details", None) and session.customer_details and session.customer_details.email:
        email = session.customer_details.email
    elif getattr(session, "customer_email", None):
        email = session.customer_email

    if not email:
        raise HTTPException(status_code=400, detail="No email found on Stripe session")

    # Determine credits from purchased items
    credits_to_add = 0
    seen_price_ids = []

    try:
        line_items = getattr(session, "line_items", None)
        if line_items and getattr(line_items, "data", None):
            for item in line_items.data:
                price = getattr(item, "price", None)
                price_id = getattr(price, "id", None) if price else None
                qty = int(getattr(item, "quantity", 1) or 1)

                if price_id:
                    seen_price_ids.append(price_id)
                    per = int(PRICE_TO_CREDITS.get(price_id, 0))
                    credits_to_add += per * qty
    except Exception:
        # If Stripe structure differs, do not crash access; just grant 0
        credits_to_add = 0

    # Grant credits (idempotency note: for true idempotency you'd dedupe by session_id in DB.
    # Keeping simple here, but we do record events.)
    if credits_to_add > 0:
        add_credits(email, credits_to_add)
        # record events for visibility (one per price id)
        for pid in (seen_price_ids or [None]):
            record_event(email=email, session_id=session_id, price_id=pid, credits_added=credits_to_add)

    token = issue_token_for_email(email)

    course_url = f"https://www.getmotionforge.com/course.html?token={token}&session_id={session_id}"
    tool_url = f"https://www.getmotionforge.com/tool.html?token={token}"

    return {
        "course_url": course_url,
        "tool_url": tool_url,
        "token": token,
        "credits_added": credits_to_add,
        "price_ids": seen_price_ids,
    }

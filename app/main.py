from __future__ import annotations

import os
import time
import base64
import json
import sqlite3
from typing import Dict, Any, Optional, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import stripe

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="motionforge_saas_backend")

# ----------------------------
# CORS
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
# Admin key (optional)
# ----------------------------
ADMIN_KEY = os.getenv("ADMIN_KEY") or os.getenv("ADMIN_APP_KEY") or ""

# ----------------------------
# Price → Credits mapping (LIVE)
# ----------------------------
PRICE_TO_CREDITS: Dict[str, int] = {
    # Tester Pack $5 → 10 credits
    # You have TWO similar IDs in the wild; support both (0 vs O).
    "price_1Sjzy32L998MB1DP0pYyuyTY": 10,  # ...DP0...
    "price_1Sjzy32L998MB1DPOpYyuyTY": 10,  # ...DPO... (THIS is the one Stripe returned)

    # Creator Pack $97 → 50 credits
    "price_1Sd90A2L998MB1DPzBnPWnTA": 50,

    # Refill 250 → 250 credits
    "price_1Sh7La2L998MB1DPtfvTv31N": 250,

    # Refill 500 → 500 credits
    "price_1Sh7Px2L998MB1DPQcQbGLfR": 500,
}

_env_map = os.getenv("PRICE_TO_CREDITS_JSON")
if _env_map:
    try:
        PRICE_TO_CREDITS.update({k: int(v) for k, v in json.loads(_env_map).items()})
    except Exception:
        pass

# ----------------------------
# SQLite
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
        conn.execute("CREATE INDEX IF NOT EXISTS idx_credit_events_session ON credit_events(session_id)")

@app.on_event("startup")
def _startup():
    init_db()

# ----------------------------
# Token helpers
# ----------------------------
def issue_token_for_email(email: str) -> str:
    payload = {"email": email, "iat": int(time.time())}
    token = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"mf_{token}"

def decode_token(token: str) -> Dict[str, Any]:
    if not token or not token.startswith("mf_"):
        raise HTTPException(status_code=401, detail="Invalid token")

    raw = token[3:]
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
# Credits ops
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

def session_has_positive_grant(session_id: str) -> bool:
    """Idempotency rule: only block re-processing if we already granted > 0 credits for this session."""
    with db() as conn:
        row = conn.execute(
            "SELECT 1 FROM credit_events WHERE session_id = ? AND credits_added > 0 LIMIT 1",
            (session_id,),
        ).fetchone()
        return row is not None

# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# Credits API (tool.html depends on this)
# ----------------------------
@app.get("/credits")
def credits(request: Request):
    token = bearer_token_from_auth_header(request)
    payload = decode_token(token)
    email = payload["email"]
    return {"ok": True, "email": email, "credits": get_credits(email)}

# ----------------------------
# Admin add (optional)
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
    # 1) Retrieve session (email)
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    email = None
    if getattr(session, "customer_details", None) and session.customer_details and session.customer_details.email:
        email = session.customer_details.email
    elif getattr(session, "customer_email", None):
        email = session.customer_email

    if not email:
        raise HTTPException(status_code=400, detail="No email found on Stripe session")

    credits_added_total = 0
    seen_price_ids: List[str] = []

    # 2) Only block if we already granted >0 for this session
    if not session_has_positive_grant(session_id):
        grants: List[Tuple[str, int]] = []

        # 3) Reliable line-items fetch
        try:
            items = stripe.checkout.Session.list_line_items(session_id, limit=100)
            for item in items.data:
                price = getattr(item, "price", None)
                price_id = getattr(price, "id", None) if price else None
                qty = int(getattr(item, "quantity", 1) or 1)

                if not price_id:
                    continue

                seen_price_ids.append(price_id)
                per = int(PRICE_TO_CREDITS.get(price_id, 0))
                if per <= 0:
                    continue

                add_amt = per * qty
                credits_added_total += add_amt
                grants.append((price_id, add_amt))

        except Exception:
            credits_added_total = 0
            seen_price_ids = []
            grants = []

        # 4) Apply + record
        if credits_added_total > 0:
            add_credits(email, credits_added_total)
            for pid, amt in grants:
                record_event(email=email, session_id=session_id, price_id=pid, credits_added=amt)
        else:
            # Keep a trace for debugging, but allow future re-tries (since it was 0)
            record_event(email=email, session_id=session_id, price_id="(no_match)", credits_added=0)

    # 5) Token + URLs
    token = issue_token_for_email(email)
    course_url = f"https://www.getmotionforge.com/course.html?token={token}&session_id={session_id}"
    tool_url = f"https://www.getmotionforge.com/tool.html?token={token}"

    return {
        "course_url": course_url,
        "tool_url": tool_url,
        "token": token,
        "credits_added": credits_added_total,
        "price_ids": seen_price_ids,
    }

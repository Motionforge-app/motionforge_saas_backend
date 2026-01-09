from __future__ import annotations

import os
import time
import base64
import json

from fastapi import FastAPI, HTTPException
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
# Helpers
# ----------------------------
def issue_token_for_email(email: str) -> str:
    payload = {
        "email": email,
        "iat": int(time.time())
    }
    token = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).decode().rstrip("=")
    return f"mf_{token}"

# ----------------------------
# Health
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------------
# Stripe → Course/Tool access
# ----------------------------
@app.get("/auth/access-from-session")
def access_from_session(session_id: str):
    try:
        session = stripe.checkout.Session.retrieve(session_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    email = None
    if session.customer_details and session.customer_details.email:
        email = session.customer_details.email
    elif session.customer_email:
        email = session.customer_email

    if not email:
        raise HTTPException(status_code=400, detail="No email found on Stripe session")

    token = issue_token_for_email(email)

    # Include session_id in course_url so course can always bounce back to access.html?session_id=...
    course_url = f"https://www.getmotionforge.com/course.html?token={token}&session_id={session_id}"

    # Direct tool URL (best for demo + avoiding loops)
    tool_url = f"https://www.getmotionforge.com/tool.html?token={token}"

    return {
        "course_url": course_url,
        "tool_url": tool_url,
        "token": token,
    }

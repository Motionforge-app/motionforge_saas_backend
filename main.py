import os
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import stripe

# -----------------------------
# CONFIG
# -----------------------------

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

if not STRIPE_SECRET_KEY:
    print("⚠️  WARNING: STRIPE_SECRET_KEY is empty. Set this in your environment on Railway.")

if not STRIPE_WEBHOOK_SECRET:
    print("⚠️  WARNING: STRIPE_WEBHOOK_SECRET is empty. Set this in your environment on Railway.")

stripe.api_key = STRIPE_SECRET_KEY


# -----------------------------
# FASTAPI APP
# -----------------------------

app = FastAPI(title="MotionForge SAAS Backend")

origins = [
    "https://getmotionforge.com",
    "https://www.getmotionforge.com",
    "http://localhost:5173",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# HELPERS
# -----------------------------

def _parse_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_credits_for_session(session_id: str):
    """
    Haal Stripe Checkout Session op, koppel aan Customer en zorg dat
    credits uit deze sessie maar één keer toegevoegd worden.
    """
    try:
        session = stripe.checkout.Session.retrieve(
            session_id,
            expand=["line_items.data.price.product"]
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid session_id: {str(e)}")

    customer_id = session.get("customer")
    if not customer_id:
        raise HTTPException(status_code=400, detail="No customer for this session")

    customer = stripe.Customer.retrieve(customer_id)
    metadata = customer.get("metadata", {}) or {}

    current_credits = _parse_int(metadata.get(CUSTOMER_CREDITS_FIELD, 0))
    last_session_id = metadata.get(CUSTOMER_LAST_SESSION_FIELD)

    # Nog niet verwerkt? Dan credits toevoegen
    if last_session_id != session_id:
        line_items = session.get("line_items", {}).get("data", [])
        added_credits = 0

        for item in line_items:
            price = item.get("price", {})
            product = price.get("product")
            quantity = item.get("quantity", 1)
            if product in PRODUCT_CREDITS:
                added_credits += PRODUCT_CREDITS[product] * (quantity or 1)

        if added_credits > 0:
            new_total = current_credits + added_credits
            new_metadata = {
                **metadata,
                CUSTOMER_CREDITS_FIELD: str(new_total),
                CUSTOMER_LAST_SESSION_FIELD: session_id,
            }
            customer = stripe.Customer.modify(customer_id, metadata=new_metadata)
            current_credits = new_total

    return customer, current_credits


def update_customer_credits(customer: stripe.Customer, new_credits: int) -> stripe.Customer:
    metadata = customer.get("metadata", {}) or {}
    new_metadata = {
        **metadata,
        CUSTOMER_CREDITS_FIELD: str(max(new_credits, 0)),
    }
    return stripe.Customer.modify(customer.id, metadata=new_metadata)


# -----------------------------
# MODELS
# -----------------------------

class CreditsResponse(BaseModel):
    credits: int


class StatusResponse(BaseModel):
    message: str
    credits: int


# -----------------------------
# ENDPOINTS
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "motionforge_saas_backend"}


@app.get("/check_credits", response_model=CreditsResponse)
def check_credits(session_id: str):
    """
    Wordt aangeroepen door tool.html om credit-saldo op te halen.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    customer, current_credits = ensure_credits_for_session(session_id)
    return CreditsResponse(credits=current_credits)


@app.post("/process_clip", response_model=StatusResponse)
async def process_clip(session_id: str, file: UploadFile = File(...)):
    """
    Verwerkt één clip en trekt 1 credit af.
    Hier komt later jouw echte MotionForge-verwerkingslogica.
    """
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")

    customer, current_credits = ensure_credits_for_session(session_id)

    if current_credits <= 0:
        raise HTTPException(status_code=402, detail="You have no credits left. Please buy more credits.")

    # TODO: hier je echte video-verwerking:
    # contents = await file.read()
    # ... verwerk contents met MotionForge ...
    # output_path = ...

    new_credits = current_credits - 1
    update_customer_credits(customer, new_credits)

    return StatusResponse(
        message="Clip processed successfully. 1 credit used.",
        credits=new_credits,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
import stripe
import json
import os
from fastapi import Request, HTTPException

# Zet je Stripe secret key
stripe.api_key = STRIPE_SECRET_KEY


# Path voor credits storage
CREDITS_FILE = "credits.json"

def load_credits():
    if not os.path.exists(CREDITS_FILE):
        return {}
    with open(CREDITS_FILE, "r") as f:
        return json.load(f)

def save_credits(data):
    with open(CREDITS_FILE, "w") as f:
        json.dump(data, f, indent=4)


@app.post("/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

       endpoint_secret = STRIPE_WEBHOOK_SECRET


    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Alleen reageren op betalingen die succesvol zijn
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        # Email van de koper
        customer_email = session.get("customer_details", {}).get("email")

        if not customer_email:
            return {"status": "no email found"}

        # Credits toevoegen
        credits = load_credits()
        current = credits.get(customer_email, 0)
        credits[customer_email] = current + 10  # aantal credits per aankoop

        save_credits(credits)

        print(f"Credits toegevoegd aan {customer_email}. Nieuw totaal: {credits[customer_email]}")

    return {"status": "success"}


from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SERVICE_NAME = "motionforge_saas_backend"

BASE_DIR = Path(__file__).resolve().parent
COURSE_DIR = BASE_DIR / "static" / "course"

CORS_ORIGINS_RAW = os.getenv("CORS_ORIGINS", "https://www.getmotionforge.com")
CORS_ORIGINS = [o.strip() for o in CORS_ORIGINS_RAW.split(",") if o.strip()]

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title=SERVICE_NAME, version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def serve_course_file(filename: str) -> FileResponse:
    path = (COURSE_DIR / filename).resolve()

    if COURSE_DIR not in path.parents:
        raise HTTPException(status_code=400, detail="Invalid path")

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Course file not found: {filename}")

    return FileResponse(path, media_type="text/html; charset=utf-8")

# -----------------------------------------------------------------------------
# Health / Status
# -----------------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "service": SERVICE_NAME}


@app.get("/status")
def status() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "course_dir_exists": COURSE_DIR.exists(),
        "course_files": sorted(p.name for p in COURSE_DIR.glob("*.html")),
    }

# -----------------------------------------------------------------------------
# Course routes (STABLE – NO REDIRECT LOOPS)
# -----------------------------------------------------------------------------
@app.get("/course/")
def course_index(request: Request):
    return serve_course_file("course.html")



# IMPORTANT:
# .html route MUST come FIRST to avoid FastAPI parsing errors
@app.get("/course/module-{n}.html")
def course_module_html(n: int):
    return serve_course_file(f"module-{n}.html")


@app.get("/course/module-{n}")
def course_module(n: int):
    return serve_course_file(f"module-{n}.html")

# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/course/")

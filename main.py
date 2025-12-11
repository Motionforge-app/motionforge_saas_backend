from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uuid
import os
import shutil
import logging
from moviepy.video.io.VideoFileClip import VideoFileClip
import imageio_ffmpeg  # NEW

# Register FFmpeg binary for MoviePy (required on Railway)
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging
logging.basicConfig(level=logging.INFO)

# File paths
BASE_DIR = "outputs"
os.makedirs(BASE_DIR, exist_ok=True)


# -----------------------
# Helper function: split
# -----------------------
def split_video(input_path: str, output_dir: str, clip_length: int = 15):
    try:
        video = VideoFileClip(input_path)
        duration = int(video.duration)

        clips = []

        start = 0
        index = 1

        while start < duration:
            end = min(start + clip_length, duration)

            output_path = os.path.join(output_dir, f"clip_{index}.mp4")

            # MoviePy write settings COMPACT & Railway-safe
            subclip = video.subclip(start, end)
            subclip.write_videofile(
                output_path,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=f"{output_dir}/temp_audio_{index}.m4a",
                remove_temp=True,
                threads=1
            )

            clips.append(output_path)

            start += clip_length
            index += 1

        video.close()
        return clips

    except Exception as e:
        logging.error(f"Error splitting video: {e}")
        raise


# -----------------------------------
# Upload endpoint
# -----------------------------------
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith((".mp4", ".mov", ".mkv")):
            raise HTTPException(status_code=400, detail="Invalid file format.")

        file_id = str(uuid.uuid4())

        # Gebruik altijd de echte extensie in lowercase (.mp4, .mov, .mkv)
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in [".mp4", ".mov", ".mkv"]:
            ext = ".mp4"

        input_path = os.path.join(BASE_DIR, f"{file_id}{ext}")

        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {"file_id": file_id, "message": "Upload successful"}

    except Exception as e:
        logging.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")


# -----------------------------------
# Split endpoint
# -----------------------------------
@app.post("/split/{file_id}")
async def split_endpoint(file_id: str):
    try:
        # Zoek het bestand ongeacht de extensie (.mp4, .MP4, .mov, etc.)
        possible_files = [f for f in os.listdir(BASE_DIR) if f.startswith(file_id)]
        if not possible_files:
            raise HTTPException(status_code=404, detail="File not found")

        input_path = os.path.join(BASE_DIR, possible_files[0])

        if not os.path.exists(input_path):
            raise HTTPException(status_code=404, detail="File not found")

        output_dir = os.path.join(BASE_DIR, f"{file_id}_clips")
        os.makedirs(output_dir, exist_ok=True)

        # Perform split
        clips = split_video(input_path, output_dir)

        # List downloadable files
        download_urls = [
            f"/download/{file_id}/{os.path.basename(clip)}" for clip in clips
        ]

        return {"clips": download_urls}

    except Exception as e:
        logging.error(f"Split failed: {e}")
        raise HTTPException(status_code=500, detail="Split error")


# -----------------------------------
# Download endpoint
# -----------------------------------
@app.get("/download/{file_id}/{filename}")
async def download_clip(file_id: str, filename: str):
    file_path = os.path.join(BASE_DIR, f"{file_id}_clips", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Clip not found")

    return FileResponse(path=file_path, filename=filename, media_type='video/mp4')


# -----------------------------------
# Placeholder endpoint for credits
# -----------------------------------
@app.get("/credits/{user_id}")
async def get_credits(user_id: str):
    return {"user_id": user_id, "credits": 97}  # later DB-powered


# -----------------------------------
# Placeholder endpoint for AI
# -----------------------------------
@app.post("/ai/captions")
async def ai_captions(data: dict):
    text = data.get("text", "")
    return {
        "hooks": [
            f"Hook idea based on: {text}",
            f"Alternative hook for: {text}"
        ],
        "captions": [
            f"Caption suggestion for: {text}"
        ],
        "hashtags": ["#motionforge", "#ai", "#shorts"]
    }


@app.get("/")
def root():
    return {"status": "ok", "service": "motionforge_backend"}

import os
import shutil
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import TypedDict

import pysubs2
import whisperx
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile

device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
storage_path = Path("storage")
model_path = storage_path / "whisperx_models"

app = FastAPI()


class Status(Enum):
    PROCESSING = "Processing"
    FINISHED = "Finished"


class SessionProcess(TypedDict):
    transcription: list[str]
    status: Status
    processed: int
    total: int


current_process: dict[str, SessionProcess] = {}


async def save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    try:
        with destination.open("wb") as buffer:
            while content := await upload_file.read(1024):
                buffer.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}",
        )


def transcribe_line(session_path: Path, sub_path: Path, audio_path: Path):
    sub = pysubs2.load(sub_path)

    segments_dir_path = session_path / "segments"
    segments_dir_path.mkdir(exist_ok=True)

    session_id = session_path.parts[-1]

    for i, line in enumerate(sub):
        segment_file_path = segments_dir_path / f"segment_{i}{Path(audio_path).suffix}"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                audio_path.as_posix(),
                "-ss",
                pysubs2.time.ms_to_str(line.start),
                "-to",
                pysubs2.time.ms_to_str(line.end),
                "-c",
                "copy",
                segment_file_path,
            ],
        )

        model = whisperx.load_model(
            "large-v2",
            device,
            compute_type=compute_type,
            download_root=model_path.as_posix(),
            language="ja",
        )

        whisper_audio = whisperx.load_audio(segment_file_path)
        result = model.transcribe(whisper_audio, batch_size=batch_size, language="ja")

        transcription = result["segments"][0]["text"] if result["segments"] else ""
        if session_id in current_process:
            current_process[session_id]["transcription"].append(transcription)
            current_process[session_id]["processed"] = i
        else:
            current_process[session_id] = {
                "status": Status.PROCESSING,
                "transcription": [transcription],
                "processed": i,
                "total": len(sub),
            }

        current_process[session_id]["status"] = Status.PROCESSING

    with open(session_path / "transcription.txt", "w") as f:
        f.write("\n".join(current_process[session_id]["transcription"]))

    current_process[session_id]["status"] = Status.FINISHED


@app.get("/session")
async def get_session():
    session_path = storage_path
    session_list = [f.path for f in os.scandir(session_path) if f.is_dir()]
    return {"session_list": ", ".join(session_list)}


@app.post("/session")
async def create_session():
    session_id = str(uuid.uuid4())
    session_path = storage_path / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    session_path = storage_path / session_id
    shutil.rmtree(session_path)
    return {"session_id": session_id}


@app.post("/upload/original/{session_id}")
async def upload_audio(
    session_id: str,
    audio: UploadFile,
    subtitle: UploadFile,
    background_tasks: BackgroundTasks,
):
    session_path = storage_path / session_id
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found")

    if (
        session_id in current_process
        and current_process[session_id]["status"] == Status.PROCESSING
    ):
        raise HTTPException(status_code=409, detail="File is currently being processed")

    # Check if audio.filename is None
    if audio.filename is None:
        raise HTTPException(status_code=400, detail="Invalid audio file")

    audio_path = session_path / audio.filename
    await save_upload_file(audio, audio_path)

    if subtitle.filename is None:
        raise HTTPException(status_code=400, detail="Invalid subtitle file")
    sub_path = session_path / subtitle.filename
    await save_upload_file(subtitle, sub_path)

    background_tasks.add_task(transcribe_line, session_path, sub_path, audio_path)

    return {
        "audio": audio.filename,
        "subtitle": subtitle.filename,
        "asd": current_process[session_id]["status"]
        if session_id in current_process
        else "",
    }


@app.post("/upload/reference/{session_id}")
async def upload_subtitle(session_id: str, subtitle: list[UploadFile]):
    session_path = storage_path / session_id
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found")

    ref_dir_path = session_path / "reference"
    ref_dir_path.mkdir(exist_ok=True)

    for sub in subtitle:
        if sub.filename is None:
            raise HTTPException(status_code=400, detail="Invalid subtitle file")

        ref_save_path = ref_dir_path / sub.filename
        await save_upload_file(sub, ref_save_path)

    return {
        "status": "success",
        "subtitle": ",".join(str(sub.filename) for sub in subtitle),
    }


@app.get("/transcription/{session_id}")
async def get_transcription(session_id: str):
    session_path = storage_path / session_id
    if session_id in current_process:
        return current_process[session_id]

    with open(session_path / "transcription.txt", "r") as f:
        transcription = f.read().split("\n")
        return {"transcription": transcription}

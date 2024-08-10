import json
import os
import shutil
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import TypedDict

import jaconv
import pysubs2
import whisperx
from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fugashi import Tagger
from rapidfuzz import fuzz

device = "cuda"
batch_size = 16
compute_type = "float16"
storage_path = Path("storage")
model_path = storage_path / "whisperx_models"

app = FastAPI()

tagger = Tagger()


class Status(str, Enum):
    PROCESSING = "Processing"
    FINISHED = "Finished"


class SubMatch(TypedDict):
    score: float
    matched_text: str


class SubMatches(TypedDict):
    start_time: str
    end_time: str
    text: str
    matches: list[SubMatch]


class SessionProcess(TypedDict):
    transcription: list[SubMatches]
    status: Status
    processed: int
    total: int


current_process: dict[str, SessionProcess] = {}


async def save_upload_file(
    upload_files: UploadFile | list[UploadFile], destination: Path
) -> Path | list[Path]:
    try:
        destination.mkdir(parents=True, exist_ok=True)
        if isinstance(upload_files, list):
            save_paths = []
            for file in upload_files:
                if file.filename is None:
                    raise HTTPException(status_code=400, detail="Invalid file")
                save_path = destination / file.filename
                with save_path.open("wb") as buffer:
                    while content := await file.read(1024):
                        buffer.write(content)
                save_paths.append(save_path)
            return save_paths
        else:
            if upload_files.filename is None:
                raise HTTPException(status_code=400, detail="Invalid file")

            save_path = destination / upload_files.filename
            with save_path.open("wb") as buffer:
                while content := await upload_files.read(1024):
                    buffer.write(content)
            return save_path

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}",
        )


def load_ja_sub(ja_sub_paths: Path | list[Path]) -> list[str]:
    all_subs = []
    if isinstance(ja_sub_paths, list):
        for file in ja_sub_paths:
            subs = pysubs2.load(file)
            all_subs.extend([sub.text for sub in subs])
    else:
        subs = pysubs2.load(ja_sub_paths)
        all_subs.extend([sub.text for sub in subs])
    return all_subs


def normalize_japanese(text):
    tokens = tagger(text)
    # Get readings, prioritizing 'pron', then 'kana', then surface form
    readings = [
        token.feature.pron or token.feature.kana or token.surface for token in tokens
    ]
    return "".join(readings)


def match_japanese_string(
    transcribed: str, official_list: list[str], threshold=70
) -> list[SubMatch]:
    matches: list[SubMatch] = []

    # Normalize transcribed text
    transcribed_normalized = normalize_japanese(transcribed)

    for official in official_list:
        # Normalize official text
        official_normalized = normalize_japanese(official)

        # Convert to hiragana for comparison
        t_hira = jaconv.kata2hira(jaconv.alphabet2kana(transcribed_normalized))
        o_hira = jaconv.kata2hira(jaconv.alphabet2kana(official_normalized))

        # Calculate similarity score
        score = fuzz.ratio(t_hira, o_hira)

        if score >= threshold:
            matches.append({"score": score, "matched_text": official})

    return sorted(matches, key=lambda d: d["score"], reverse=True)


def process_sub(
    session_path: Path,
    ja_sub_paths: Path | list[Path],
    eng_sub_path: Path | None = None,
    audio_path: Path | None = None,
    is_transcribed: bool = False,
):
    segments_dir_path = session_path / "segments"
    session_id = session_path.parts[-1]
    official_ja_subs = load_ja_sub(ja_sub_paths)

    sub: pysubs2.SSAFile | list[SubMatches] = {}
    if is_transcribed:
        with open("processed.json") as f:
            sub = json.load(f)["transcription"]
    else:
        segments_dir_path.mkdir(exist_ok=True)
        sub = pysubs2.load(eng_sub_path)

    for i, line in enumerate(sub):
        transcription = ""
        start_time = ""
        end_time = ""

        if is_transcribed:
            transcription = line.text
            start_time = line.start_time
            end_time = line.end_time
        else:
            segment_file_path = (
                segments_dir_path / f"segment_{i}{Path(audio_path).suffix}"
            )
            start_time = pysubs2.time.ms_to_str(line.start)
            end_time = pysubs2.time.ms_to_str(line.end)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path.as_posix(),
                    "-ss",
                    start_time,
                    "-to",
                    end_time,
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
            result = model.transcribe(
                whisper_audio, batch_size=batch_size, language="ja"
            )

            transcription = result["segments"][0]["text"] if result["segments"] else ""

        matches = (
            match_japanese_string(transcription, official_ja_subs, 30)
            if transcription
            else []
        )

        if session_id in current_process:
            current_process[session_id]["transcription"].append(
                {
                    "text": transcription,
                    "matches": matches,
                    "start_time": start_time,
                    "end_time": end_time,
                }
            )
            current_process[session_id]["processed"] = i + 1
        else:
            current_process[session_id] = {
                "status": Status.PROCESSING,
                "transcription": [
                    {
                        "text": transcription,
                        "matches": matches,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                ],
                "processed": i,
                "total": len(sub),
            }

        current_process[session_id]["status"] = Status.PROCESSING

    with open(session_path / "processed.json", "w", encoding="utf-8") as f:
        json.dump(current_process[session_id], f, ensure_ascii=False, indent=4)

    current_process[session_id]["status"] = Status.FINISHED

    print("😊😊😊😊😊😊")


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


@app.post("/match-sub/{session_id}")
async def upload_audio(
    session_id: str,
    audio: UploadFile,
    eng_subtitle: UploadFile,
    ja_subtitles: list[UploadFile],
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

    audio_path: Path = await save_upload_file(audio, session_path)
    eng_sub_path: Path = await save_upload_file(eng_subtitle, session_path)
    ja_sub_paths = await save_upload_file(ja_subtitles, session_path / "references")

    background_tasks.add_task(
        process_sub, session_path, ja_sub_paths, eng_sub_path, audio_path
    )

    return {
        "audio": audio,
        "eng_subtitle": eng_subtitle,
        "ja_subtitle": ja_subtitles,
    }


@app.post("/ref-sub/{session_id}")
async def upload_subtitle(
    session_id: str,
    subtitles: list[UploadFile],
    background_tasks: BackgroundTasks,
):
    session_path = storage_path / session_id
    if not session_path:
        raise HTTPException(status_code=404, detail="Session not found")

    await save_upload_file(subtitles, session_path / "references")

    background_tasks.add_task(process_sub, session_path, subtitles, is_transcribed=True)

    return {
        "status": "success",
        "subtitle": ",".join(str(sub.filename) for sub in subtitles),
    }


@app.get("/processed/{session_id}")
async def get_transcription(session_id: str):
    session_path = storage_path / session_id
    if session_id in current_process:
        return current_process[session_id]

    with open(session_path / "processed.json", "r") as f:
        processed = json.load(f)
        return processed

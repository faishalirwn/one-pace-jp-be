import json
import mimetypes
import os
import shutil
import subprocess
import uuid
from enum import Enum
from pathlib import Path
from typing import TypedDict

import jaconv
import librosa
import pysubs2
import whisperx
from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fugashi import Tagger
from pydantic import BaseModel
from pysubs2 import SSAEvent, SSAFile
from rapidfuzz import fuzz

device = "cuda"
batch_size = 16
compute_type = "float16"
storage_path = Path("storage")
storage_session_path = storage_path / "sessions"
ref_sub_dirname = "references"
ori_dirname = "original"
transcription_filename = "transcription.json"
final_sub_filename = "final.srt"
manual_ref_sub_filename = "ref_sub.txt"
model_path = storage_path / "whisperx_models"

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


tagger = Tagger()

model = whisperx.load_model(
    "large-v2",
    device,
    compute_type=compute_type,
    download_root=model_path.as_posix(),
    language="ja",
)


class Status(str, Enum):
    PROCESSING = "Processing"
    FINISHED = "Finished"


class SubMatch(TypedDict):
    score: float
    matched_text: str
    normalized: str


class SubMatches(TypedDict):
    start_time: str
    end_time: str
    text: str
    matches: list[SubMatch]
    match: str | None
    merge: bool


class SessionProcess(TypedDict):
    transcription: list[SubMatches]
    status: Status
    processed: int
    total: int


class SubMatchesPartial(BaseModel):
    match: str | None
    merge: bool


class SaveSubReq(BaseModel):
    transcription: list[SubMatchesPartial]


class FileTypes(str, Enum):
    AUDIO = "audio"
    ORIGINAL_SUB = "original_sub"
    REF_SUB = "ref_sub"
    REF_SUB_MANUAL = "ref_sub_manual"


class Response(BaseModel):
    message: str = "Success"


class UploadResponse(Response):
    filename: list[str]


class SessionListResponse(Response):
    session_list: list[str]


class SessionIdResponse(Response):
    session_id: str


class FilesResponse(Response):
    files: dict[FileTypes, str]


current_process: dict[str, SessionProcess] = {}
current_process_err = ""


def get_session_path(session_id: str) -> Path:
    session_path = storage_session_path / session_id
    if not session_path.is_dir():
        raise HTTPException(status_code=404, detail="Session not found")
    return storage_session_path / session_id


def get_session_files(session_id: str, filename: bool = False):
    session_path = get_session_path(session_id)

    ori_files = get_dir_files(session_path / ori_dirname)

    if not ori_files:
        return None

    audio_path = [
        file
        for file in ori_files
        if mimetypes.guess_type(file)[0].split("/")[0] == "audio"
    ]
    ori_sub_path = [
        file
        for file in ori_files
        if mimetypes.guess_type(file)[0].split("/")[0] == "text"
    ]

    ref_subs_path = get_dir_files(session_path / ref_sub_dirname)

    if not ref_subs_path or not audio_path or not ori_sub_path:
        return None
    else:
        if filename:
            ref_sub_files = []
            for path in ref_subs_path:
                ref_sub_files.append(path.name)
            return {
                FileTypes.AUDIO: audio_path[0].name,
                FileTypes.ORIGINAL_SUB: ori_sub_path[0].name,
                FileTypes.REF_SUB: ",".join(ref_sub_files),
            }
        else:
            return {
                "audio_path": audio_path[0],
                "ori_sub_path": ori_sub_path[0],
                "ref_subs_path": ref_subs_path,
            }


def get_dir_files(dir_path: Path):
    try:
        flist = [p for p in Path(dir_path).iterdir() if p.is_file()]
    except Exception:
        return None
    return flist


async def extract_audio(video: Path, output_path: Path) -> Path:
    output_path_full = output_path / f"{Path(video).stem}.opus"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video,
        "-vn",  # Disable video
        "-c:a",
        "libopus",  # Use Opus codec
        "-b:a",
        "128k",  # Set bitrate (adjust as needed)
        output_path_full,
    ]

    try:
        # Run the FFmpeg command
        subprocess.run(command, check=True, stderr=subprocess.PIPE)
        print(f"Audio extracted and converted to Opus: {output_path_full}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr.decode()}")

    return output_path_full


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

                print(f"Saving {file.filename}...")
                save_path = destination / file.filename
                with save_path.open("wb") as buffer:
                    while content := await file.read(1024):
                        buffer.write(content)
                save_paths.append(save_path)
            return save_paths
        else:
            if upload_files.filename is None:
                raise HTTPException(status_code=400, detail="Invalid file")

            print(f"Saving {upload_files.filename}...")
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
            matches.append(
                {
                    "score": score,
                    "matched_text": official,
                    "normalized": official_normalized,
                }
            )

    return sorted(matches, key=lambda d: d["score"], reverse=True)


def transcribe_and_match(
    session_path: Path,
    ja_sub_paths: Path | list[Path],
    eng_sub_path: Path,
    audio_path: Path,
    transcribe: bool = True,
):
    try:
        transcription_file = session_path / transcription_filename
        transcription_file_exists = transcription_file.is_file()

        if not transcribe and not transcription_file_exists:
            # how to convey/send this to frontend? this is background task
            # fe shouldn't allow this, consult to fe
            print("must transcribe, there is no previous transcription")
            return

        segments_dir_path = session_path / "segments"
        session_id = session_path.parts[-1]

        current_process[session_id] = {
            "status": Status.PROCESSING,
        }

        official_ja_subs = load_ja_sub(ja_sub_paths)

        sub: pysubs2.SSAFile | list[SubMatches] = {}

        use_prev_transcription = not transcribe

        if use_prev_transcription:
            with open(transcription_file) as f:
                transcription_content: SessionProcess = json.load(f)
                sub = transcription_content["transcription"]
        else:
            segments_dir_path.mkdir(exist_ok=True)
            sub = pysubs2.load(eng_sub_path)

        audio_duration = librosa.get_duration(path=audio_path)

        for i, line in enumerate(sub):
            transcription = ""
            start_time = ""
            end_time = ""
            merge = False

            if line.start / 1000 > audio_duration:
                break

            if use_prev_transcription:
                transcription = line.text
                start_time = line.start_time
                end_time = line.end_time
                merge = line.merge
            else:
                segment_file_path = (
                    segments_dir_path / f"segment_{i}{Path(audio_path).suffix}"
                )
                start_time = line.start
                end_time = line.end
                start_time_ffmpeg = pysubs2.time.ms_to_str(line.start)
                end_time_ffmpeg = pysubs2.time.ms_to_str(line.end)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        audio_path.as_posix(),
                        "-ss",
                        start_time_ffmpeg,
                        "-to",
                        end_time_ffmpeg,
                        "-c",
                        "copy",
                        segment_file_path,
                    ],
                )

                whisper_audio = whisperx.load_audio(segment_file_path)
                result = model.transcribe(
                    whisper_audio, batch_size=batch_size, language="ja"
                )

                transcription = (
                    result["segments"][0]["text"] if result["segments"] else ""
                )

            print(transcription)

            matches = (
                match_japanese_string(transcription, official_ja_subs, 30)
                if transcription
                else []
            )

            transcription_entry: SubMatches = {
                "text": transcription,
                "matches": matches,
                "start_time": start_time,
                "end_time": end_time,
                "merge": merge,
                "match": None,
            }

            current_process[session_id]["processed"] = i + 1

            if i == 0:
                current_process[session_id] = {
                    "total": len(sub),
                    "transcription": [transcription_entry],
                }
            else:
                current_process[session_id]["transcription"].append(transcription_entry)

        current_process[session_id]["status"] = Status.FINISHED

        with open(session_path / transcription_filename, "w", encoding="utf-8") as f:
            json.dump(current_process[session_id], f, ensure_ascii=False, indent=4)

        print("😊😊😊😊😊😊", line.start / 1000, audio_duration)
    except Exception as e:
        global current_process_err
        current_process_err = str(e)


def get_transcription(session_id: str) -> SessionProcess | None:
    session_path = get_session_path(session_id)

    transcription_file = session_path / transcription_filename
    transcription_file_exists = transcription_file.is_file()

    if transcription_file_exists:
        with open(transcription_file) as f:
            transcription = json.load(f)
            return transcription
    else:
        return None


@app.post("/test")
async def test(files: list[UploadFile]) -> UploadResponse:
    print(files)
    filenamesArr = []
    for file in files:
        filenamesArr.append(file.filename)
    return {"filename": filenamesArr}


@app.get("/sessions")
async def get_sessions() -> SessionListResponse:
    if not storage_session_path.is_dir():
        return {"session_list": ""}
    session_list = [f.name for f in os.scandir(storage_session_path) if f.is_dir()]
    # TODO: response model for all endpoint so gutaiteki response structure shows in swagger api docs
    return {"session_list": session_list}


@app.post("/session")
async def create_session() -> SessionIdResponse:
    session_id = str(uuid.uuid4())
    session_path = storage_session_path / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    return {"session_id": session_id}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    session_path = storage_session_path / session_id
    shutil.rmtree(session_path)
    return {"session_id": session_id}


@app.get("/process-sub/{session_id}")
async def get_processing_status(session_id: str):
    if session_id in current_process:
        if current_process_err:
            raise HTTPException(status_code=409, detail=current_process_err)
        if current_process[session_id]["status"] == Status.PROCESSING:
            return {"message": "sub currently being processed"}
        if current_process[session_id]["status"] == Status.FINISHED:
            return {"message": "sub processing is finished"}
    else:
        return {"message": "start process sub to get status"}


@app.post("/process-sub/{session_id}")
async def process_sub(
    session_id: str,
    background_tasks: BackgroundTasks,
):
    global current_process_err
    current_process_err = ""
    session_path = get_session_path(session_id)

    if (
        session_id in current_process
        and current_process[session_id]["status"] == Status.PROCESSING
    ):
        raise HTTPException(status_code=409, detail="File is currently being processed")

    session_files = get_session_files(session_id)

    if not session_files:
        return {"message": "submit required files"}

    audio_path = session_files["audio_path"]
    eng_sub_path = session_files["ori_sub_path"]
    ja_sub_paths = session_files["ref_subs_path"]

    background_tasks.add_task(
        transcribe_and_match, session_path, ja_sub_paths, eng_sub_path, audio_path
    )

    return {"message": "processing sub matching sub started"}


@app.put("/sub/{session_id}")
async def save_sub(session_id: str, sub_req: SaveSubReq):
    transcription = get_transcription(session_id)

    if not transcription:
        return {"message": "you shouldn't be able to request this brother"}

    transcription_list = transcription["transcription"]

    new_transcription = sub_req.transcription

    for i, sub in enumerate(transcription_list):
        sub = {
            **sub,
            "merge": new_transcription[i].merge,
            "match": new_transcription[i].match,
        }
        transcription_list[i] = sub

    transcription["transcription"] = transcription_list

    session_path = get_session_path(session_id)

    with open(session_path / transcription_filename, "w", encoding="utf-8") as f:
        json.dump(current_process[session_id], f, ensure_ascii=False, indent=4)

    return {"message": "sub transcription updated saved"}


@app.get("/sub/{session_id}")
async def get_sub(session_id: str):
    transcription = get_transcription(session_id)

    if not transcription:
        return {"message": "you shouldn't be able to request this brother"}

    return transcription


@app.post("/sub/{session_id}")
async def download_sub(session_id: str):
    session_path = get_session_path(session_id)
    transcription = get_transcription(session_id)

    if not transcription:
        return {"message": "you shouldn't be able to request this brother"}

    transcription_list = transcription["transcription"]

    subs = SSAFile()
    i = 0
    while i < len(transcription_list):
        sub = SSAEvent()
        sub.start = transcription_list[i]["start_time"]

        if transcription_list[i]["merge"]:
            while (
                i + 1 < len(transcription_list) and transcription_list[i + 1]["merge"]
            ):
                i += 1
            sub.end = transcription_list[i]["end_time"]
        else:
            sub.end = transcription_list[i]["end_time"]

        sub.text = (
            transcription_list[i]["match"] if transcription_list[i]["match"] else ""
        )
        subs.append(sub)

        i += 1

    final_sub_path = session_path / final_sub_filename
    subs.save(final_sub_path)

    return FileResponse(final_sub_path)


@app.get("/files/{session_id}")
async def get_files(session_id: str) -> FilesResponse:
    session_files = get_session_files(session_id)

    if session_files:
        return {"files": session_files}
    else:
        return {"message": "all required files must alredy been uploaded"}


@app.post("/files/{session_id}/{file_type}")
async def upload_files(
    session_id: str,
    file_type: FileTypes,
    files: list[UploadFile] = File(None),
    sub: str = Form(None),
) -> UploadResponse:
    session_path = get_session_path(session_id)

    uploaded_files = []

    ORIGINAL_SUB_MAX = 3 * 1024 * 1024

    if file_type == FileTypes.REF_SUB_MANUAL and not sub:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sub manual file must be str aight.",
        )

    if file_type == FileTypes.AUDIO and files[0].content_type.split("/")[0] != "audio":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"That ain't audio, but {files[0].content_type.split('/')[0]}.",
        )

    if file_type == FileTypes.ORIGINAL_SUB or file_type == FileTypes.REF_SUB:
        if files[0].filename.split(".")[-1] not in ["srt", "ass"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"That ain't text, but {files[0].filename.split('.')[-1]}.",
            )

        if files[0].size and files[0].size > 2 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Sub file size can't be more than {ORIGINAL_SUB_MAX} bytes.",
            )

    if file_type == FileTypes.AUDIO or file_type == FileTypes.ORIGINAL_SUB:
        ori_files = get_dir_files(session_path / ori_dirname)

        if ori_files:
            audio_path = [
                file
                for file in ori_files
                if mimetypes.guess_type(file)[0].split("/")[0] == "audio"
            ]
            if audio_path and file_type == FileTypes.AUDIO:
                audio_path[0].unlink()

            ori_sub_path = [
                file
                for file in ori_files
                if mimetypes.guess_type(file)[0].split("/")[0] == "text"
            ]
            if ori_sub_path and file_type == FileTypes.ORIGINAL_SUB:
                ori_sub_path[0].unlink()

        await save_upload_file(files[0], session_path / ori_dirname)
        uploaded_files.append(files[0].filename)

    if file_type == FileTypes.REF_SUB:
        for file in files:
            await save_upload_file(file, session_path / ref_sub_dirname)
            uploaded_files.append(file.filename)

    if file_type == FileTypes.REF_SUB_MANUAL:
        with open(session_path / ref_sub_dirname / manual_ref_sub_filename, "w") as f:
            f.write(sub)
            uploaded_files.append(manual_ref_sub_filename)

    return UploadResponse(message="upload success", filename=uploaded_files)


@app.delete("/files/{session_id}/{file_type}")
async def delete_file(session_id: str, file_type: FileTypes, filename: str):
    session_path = get_session_path(session_id)

    if file_type == FileTypes.REF_SUB:
        sub_path = session_path / ref_sub_dirname / filename
        try:
            sub_path.unlink()
        except Exception as e:
            raise HTTPException(status_code=404, detail=str(e))
        return {"message": f"{filename} deleted"}

    return {"message": "not ref sub file can't delete'"}

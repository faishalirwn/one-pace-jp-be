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

origins = ["http://localhost:3000", "https://one-pace-jp-fe.vercel.app"]

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
    PROCESSING = "processing"
    FINISHED = "finished"
    NOT_STARTED = "not_started"


class SubMatch(TypedDict):
    score: float
    matched_text: str
    normalized: str


class SubMatches(TypedDict):
    start_time: str
    end_time: str
    ori_text: str
    text: str
    matches: list[SubMatch]
    match: str | None
    merge: bool


class SessionProcess(BaseModel):
    transcription: list[SubMatches]
    status: Status
    processed: int
    total: int


class SubMatchesPartial(BaseModel):
    match: str | None
    merge: bool = False


class SaveSubReq(BaseModel):
    transcription: list[SubMatchesPartial]


class FileTypes(str, Enum):
    AUDIO = "audio"
    ORIGINAL_SUB = "original_sub"
    REF_SUB = "ref_sub"
    REF_SUB_MANUAL = "ref_sub_manual"


class Response(BaseModel):
    message: str = "Success"


class StatusResponse(Response):
    status: Status
    processed: int
    total: int


class UploadResponse(Response):
    filename: list[str]


class SessionListResponse(Response):
    session_list: list[str] | None


class SessionIdResponse(Response):
    session_id: str


class SessionFiles(BaseModel):
    files: dict[FileTypes, str | Path | list[Path] | None]


class FilesResponse(Response):
    files: dict[FileTypes, str] = {
        FileTypes.AUDIO: "",
        FileTypes.ORIGINAL_SUB: "",
        FileTypes.REF_SUB: "",
    }


class TranscriptionResponse(Response):
    transcription: SessionProcess | None


current_process: dict[str, SessionProcess] = {}
current_process_err = ""


def get_session_path(session_id: str) -> Path | None:
    session_path = storage_session_path / session_id
    if not session_path.is_dir():
        return None
    return session_path


def get_session_files(session_id: str, filename: bool = False) -> SessionFiles:
    session_path = get_session_path(session_id)

    ori_files = get_dir_files(session_path / ori_dirname)

    if not ori_files:
        return None

    def is_audio_file(file):
        if file.lower().endswith(".ass"):
            return False

        mime_type, _ = mimetypes.guess_type(file)
        return mime_type and mime_type.startswith("audio/")

    audio_path = next((file for file in ori_files if is_audio_file(file)), None)

    ori_sub_path = [file for file in ori_files if file.suffix in [".srt", ".ass"]][0]

    ref_subs_path = get_dir_files(session_path / ref_sub_dirname)

    ref_subs = [file for file in ref_subs_path if file.suffix in [".srt", ".ass"]]
    ref_subs_string = ",".join([ref_sub.stem for ref_sub in ref_subs])

    ref_sub_manual_path = session_path / ref_sub_dirname / manual_ref_sub_filename
    ref_sub_manual_content = ""

    if not ref_sub_manual_path.is_file():
        ref_sub_manual_path = None
    else:
        ref_sub_manual_content = []
        with open(ref_sub_manual_path, "r") as f:
            for line in f:
                ref_sub_manual_content.append(line.rstrip())
        ref_sub_manual_content = "\n".join(ref_sub_manual_content)

    if not ref_subs_path or not audio_path or not ori_sub_path:
        return None
    else:
        if filename:
            # TODO: manual sub, transcribt match n oman ula sub
            return SessionFiles(
                files={
                    FileTypes.AUDIO: audio_path.name,
                    FileTypes.ORIGINAL_SUB: ori_sub_path.name,
                    FileTypes.REF_SUB: ref_subs_string,
                    FileTypes.REF_SUB_MANUAL: ref_sub_manual_content,
                }
            )
        else:
            return SessionFiles(
                files={
                    FileTypes.AUDIO: audio_path,
                    FileTypes.ORIGINAL_SUB: ori_sub_path,
                    FileTypes.REF_SUB: ref_subs,
                    FileTypes.REF_SUB_MANUAL: ref_sub_manual_path,
                }
            )


def get_dir_files(dir_path: Path | None):
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
            if file.name == manual_ref_sub_filename:
                with open(file, "r") as f:
                    for line in f:
                        all_subs.append(line.rstrip())
            else:
                subs = pysubs2.load(file)
                all_subs.extend([sub.text for sub in subs])
    else:
        if ja_sub_paths.name == manual_ref_sub_filename:
            with open(file, "r") as f:
                for line in f:
                    all_subs.append(line.rstrip())
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
    def remove_duplicates(matches):
        seen = set()
        return [
            x
            for x in matches
            if not (x["matched_text"] in seen or seen.add(x["matched_text"]))
        ]

    try:
        transcription_file = session_path / transcription_filename
        transcription_file_exists = transcription_file.is_file()

        if not transcribe and not transcription_file_exists:
            print("must transcribe, there is no previous transcription")
            return

        segments_dir_path = session_path / "segments"
        session_id = session_path.parts[-1]

        print(session_id, "transcribe_and_match started....")

        current_process[session_id] = {
            "status": Status.PROCESSING,
            "processed": 0,
            "total": 0,
        }

        print("loading reference subs...")
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

        print(f"reading audio file duration... {audio_path}")

        audio_duration = librosa.get_duration(path=audio_path)

        print(session_id, "for i, line in enumerate(sub):....")
        for i, line in enumerate(sub):
            transcription = ""
            start_time = ""
            end_time = ""
            ori_text = ""
            merge = False

            if line.start / 1000 > audio_duration:
                break

            if use_prev_transcription:
                transcription = line.text
                start_time = line.start_time
                end_time = line.end_time
                merge = line.merge
                ori_text = line.ori_text
            else:
                segment_file_path = (
                    segments_dir_path / f"segment_{i}{Path(audio_path).suffix}"
                )
                start_time = line.start
                end_time = line.end
                ori_text = line.text
                start_time_ffmpeg = pysubs2.time.ms_to_str(line.start, fractions=True)
                end_time_ffmpeg = pysubs2.time.ms_to_str(line.end, fractions=True)
                print("- ", segment_file_path.name, start_time_ffmpeg, end_time_ffmpeg)
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

            # display transcription line by line to see if the process is running
            print("😊", transcription)

            matches = (
                match_japanese_string(transcription, official_ja_subs, 30)
                if transcription
                else []
            )

            transcription_entry: SubMatches = {
                "text": transcription,
                "matches": remove_duplicates(matches),
                "start_time": start_time,
                "end_time": end_time,
                "merge": merge,
                "match": matches[0]["matched_text"] if matches else None,
                "ori_text": ori_text,
            }

            current_process[session_id]["processed"] = i + 1

            if i == 0:
                current_process[session_id]["total"] = len(sub)
                current_process[session_id]["transcription"] = [transcription_entry]
            else:
                current_process[session_id]["transcription"].append(transcription_entry)

        current_process[session_id]["status"] = Status.FINISHED

        with open(session_path / transcription_filename, "w", encoding="utf-8") as f:
            json.dump(current_process[session_id], f, ensure_ascii=False, indent=4)

    except Exception as e:
        global current_process_err
        current_process_err = str(e)
        current_process[session_id]["status"] == Status.NOT_STARTED
        print("💀💀💀💀💀💀💀💀💀", e)


def get_transcription(session_id: str) -> SessionProcess | None:
    session_path = get_session_path(session_id)

    # do I have to do this everytime session_path?
    if not session_path:
        return None

    transcription_file = session_path / transcription_filename
    transcription_file_exists = transcription_file.is_file()

    if transcription_file_exists:
        with open(transcription_file) as f:
            transcription = json.load(f)
            return transcription
    else:
        return None


def is_session_processing(session_id: str) -> bool:
    if (
        session_id in current_process
        and current_process[session_id]["status"] == Status.PROCESSING
    ):
        return True
    else:
        return False


@app.get("/sessions")
async def get_sessions() -> SessionListResponse:
    if not storage_session_path.is_dir():
        return SessionListResponse(session_list=None)
    session_list = [f.name for f in os.scandir(storage_session_path) if f.is_dir()]
    return SessionListResponse(session_list=session_list)


@app.get("/sessions/{session_id}")
async def get_is_session_exist(session_id: str) -> SessionIdResponse:
    session_path = get_session_path(session_id)
    if session_path:
        return SessionIdResponse(session_id=session_path.name)
    else:
        return SessionIdResponse(session_id="", message="Session doesn't exist")


@app.post("/session")
async def create_session() -> SessionIdResponse:
    session_id = str(uuid.uuid4())
    session_path = storage_session_path / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    return SessionIdResponse(session_id=session_id)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> SessionIdResponse:
    session_path = storage_session_path / session_id
    shutil.rmtree(session_path)
    return SessionIdResponse(
        message="Session successfully deleted", session_id=session_id
    )


@app.get("/process-sub/{session_id}")
async def get_processing_status(session_id: str) -> StatusResponse:
    if session_id in current_process:
        if current_process_err:
            print("💀", current_process_err)
            raise HTTPException(status_code=409, detail=current_process_err)
        if current_process[session_id]["status"] == Status.PROCESSING:
            return StatusResponse(
                message="Sub is currently being processed. Please wait...",
                status=Status.PROCESSING,
                processed=current_process[session_id]["processed"],
                total=current_process[session_id]["total"],
            )
        if current_process[session_id]["status"] == Status.FINISHED:
            return StatusResponse(
                message="Sub processing is finished",
                status=Status.FINISHED,
                processed=current_process[session_id]["processed"],
                total=current_process[session_id]["total"],
            )
    else:
        transcription = get_transcription(session_id)

        if not transcription:
            return StatusResponse(
                message="Start process sub to get status",
                status=Status.NOT_STARTED,
                processed=0,
                total=0,
            )

        return StatusResponse(
            status=transcription["status"],
            processed=transcription["processed"],
            total=transcription["total"],
        )


@app.post("/process-sub/{session_id}")
async def process_sub(
    session_id: str,
    background_tasks: BackgroundTasks,
) -> Response:
    global current_process_err
    current_process_err = ""
    session_path = get_session_path(session_id)

    if is_session_processing(session_id):
        raise HTTPException(status_code=409, detail="File is currently being processed")

    session_files = get_session_files(session_id).files

    if not session_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="submit required files"
        )

    audio_path = session_files[FileTypes.AUDIO]
    eng_sub_path = session_files[FileTypes.ORIGINAL_SUB]
    ja_sub_paths = session_files[FileTypes.REF_SUB]

    if session_files[FileTypes.REF_SUB_MANUAL]:
        ja_sub_paths.append(session_files[FileTypes.REF_SUB_MANUAL])

    print("audio", audio_path)
    print("ori sub", eng_sub_path)
    print("ref sub", ja_sub_paths)

    background_tasks.add_task(
        transcribe_and_match, session_path, ja_sub_paths, eng_sub_path, audio_path
    )

    return Response(message="processing sub matching sub started")


@app.put("/sub/{session_id}")
async def save_sub(session_id: str, sub_req: SaveSubReq) -> Response:
    transcription = get_transcription(session_id)

    if not transcription:
        return Response(message="you shouldn't be able to request this brother")

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
        json.dump(transcription, f, ensure_ascii=False, indent=4)

    return Response(message="sub transcription updated saved")


@app.get("/sub/{session_id}")
async def get_sub(session_id: str) -> TranscriptionResponse:
    transcription = get_transcription(session_id)

    if not transcription:
        return TranscriptionResponse(
            message="you shouldn't be able to request this brother"
        )

    return TranscriptionResponse(transcription=transcription)


@app.get("/download-sub/{session_id}")
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

    return FileResponse(
        final_sub_path, filename=f"{session_id}.srt", media_type="text/plain"
    )


@app.get("/files/{session_id}")
async def get_files(session_id: str) -> FilesResponse:
    session_files = get_session_files(session_id, filename=True).files

    if session_files:
        return FilesResponse(files=session_files)
    else:
        return FilesResponse(message="all required files must alredy been uploaded")


@app.post("/files/{session_id}/{file_type}")
async def upload_files(
    session_id: str,
    file_type: FileTypes,
    files: list[UploadFile] = File(None),
    sub: str = Form(None),
) -> UploadResponse:
    ORIGINAL_SUB_MAX = 3 * 1024 * 1024

    if is_session_processing(session_id):
        raise HTTPException(status_code=409, detail="File is currently being processed")

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

    session_path = get_session_path(session_id)

    uploaded_files = []

    if file_type == FileTypes.AUDIO or file_type == FileTypes.ORIGINAL_SUB:
        ori_dir = session_path / ori_dirname
        ori_files = get_dir_files(ori_dir)

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

        await save_upload_file(files[0], ori_dir)
        uploaded_files.append(files[0].filename)

    if file_type == FileTypes.REF_SUB:
        ref_dir = session_path / ref_sub_dirname
        ref_files = get_dir_files(ref_dir)

        if ref_files:
            ref_subs_path = [
                file for file in ref_files if file.suffix in [".srt", ".ass"]
            ]

            for ref_subs in ref_subs_path:
                ref_subs.unlink()

        for file in files:
            await save_upload_file(file, ref_dir)
            uploaded_files.append(file.filename)

    if file_type == FileTypes.REF_SUB_MANUAL:
        with open(session_path / ref_sub_dirname / manual_ref_sub_filename, "w") as f:
            f.write(sub)
            uploaded_files.append(sub)

    return UploadResponse(message="upload success", filename=uploaded_files)

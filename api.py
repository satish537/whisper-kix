from services.audio_participants import process_audio_file
from services.audio_trans_nemo_test import process_audio_file1
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form, status, BackgroundTasks, Body
from typing import List, Dict
from services.utilservice import *
import requests, json
import whisperx
from services.audio_to_transcript_test import process_transcript_test
import torch
from services.audio_whisperx import process_audio_file_whisper
from services.audio_whisperx import cleanup
from whisperx import load_model
import time  

app = FastAPI()

DATA_PATH = "././data"

# Load the Whisper model at the module level
#device = "cuda" if torch.cuda.is_available() else "cpu"
#whisper_model = load_model("large", device=device, compute_type="float32")

@app.post("/audio-trans-video")
async def audio_to_transcript_endpoint_1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recording_id: str = Form(...)
):
    try:
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    # Add transcription task to the background (NON-BLOCKING)
    background_tasks.add_task(handle_transcription_background_task_whisper, audio_path, recording_id)

    # Return a success response immediately while the transcription is processed in the background
    return {"status": "success", "message": "Transcription request received"}

# Background task to handle transcription process
async def handle_transcription_background_task_whisper(audio_path, recording_id):
    try:
        timestamp = int(time.time())
        output_dir = f"/home/ubuntu/files/{recording_id}_{timestamp}/"
        os.makedirs(output_dir, exist_ok=True)

        response = process_audio_file_whisper(
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True,
        )

        # Callback URL to notify that transcription is complete
        url = 'https://app.hapie.ai/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
        }

        # Send the result to the callback URL
        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")

    finally:
        # Cleanup the entire output directory after processing is complete
        if output_dir:
            cleanup(output_dir)

@app.post("/audio-transcribe")
async def audio_transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    participants: str = Form(None),
    recording_id: str = Form(...)
):
    try:
        # Save the uploaded file
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")

        # Split the participants string into a list
        participant_list = [name.strip() for name in participants.split(',')]

        # Add transcription task to the background (NON-BLOCKING)
        background_tasks.add_task(
            handle_transcription_background_task,
            audio_path,
            recording_id,
            participant_list
        )

        # Return a success response immediately while the transcription is processed in the background
        return {"status": "success", "message": "Transcription request received"}

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error processing request: {str(e)}")

async def handle_transcription_background_task(audio_path: str, recording_id: str, participants: List[str]):
    try:
        output_dir = "/home/ubuntu/files/"
        response = process_audio_file(
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True,
            participants=participants
        )

        # Callback URL to notify that transcription is complete
        url = 'http://localhost:3000/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
        }

        # Send the result to the callback URL
        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")






@app.post("/audio-trans-video2")
async def audio_to_transcript_endpoint_1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recording_id: str = Form(...)
):
    try:
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    # Add transcription task to the background (NON-BLOCKING)
    background_tasks.add_task(handle_transcription_background_task1, audio_path, recording_id)

    # Return a success response immediately while the transcription is processed in the background
    return {"status": "success", "message": "Transcription request received"}

# Background task to handle transcription process
async def handle_transcription_background_task1(audio_path, recording_id):
    try:
        output_dir = "/home/ubuntu/files/"
        response = process_audio_file1(  # Remove whisper_model parameter
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True
        )

        # Callback URL to notify that transcription is complete
        url = 'https://app.hapie.ai/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
        }

        # Send the result to the callback URL
        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")




def get_device_for_model():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        return {"device": "cuda", "device_index": list(range(gpu_count))}
    elif gpu_count == 1:
        return {"device": "cuda", "device_index": 0}
    else:
        return {"device": "cpu", "device_index": 0}

#device_config = get_device_for_model()
#whisperx_model = whisperx.load_model("large", language="en", device=device_config["device"], device_index=device_config["device_index"], compute_type="float16")

@app.post("/audio-trans-video1", tags=["Testing"])
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    recording_id: str = Form(...)
):
    try:
        # Save the uploaded audio file using rename_and_save_file
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")

        print("Audio file saved, starting transcription task.")

        # Add transcription task to the background (NON-BLOCKING)
        background_tasks.add_task(handle_transcription_background_task_nemo, audio_path, recording_id)

        # Return a success response immediately while the transcription is processed in the background
        return JSONResponse(content={"status": "success", "message": "Transcription request received"})

    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

# Background task to handle transcription process
async def handle_transcription_background_task_nemo(audio_path, recording_id):
    try:
        result = await process_transcript_test(whisperx_model, DATA_PATH, audio_path, recording_id)

        # Handle callback or further processing here if needed
        callback_url = 'https://app.hapie.ai/api/recording/transcript-callback'
        data = {
            "recordingId": recording_id,
            "response": json.dumps(result),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="  # Replace with your actual API key
        }

        # Send the result to the callback URL
        requests.post(callback_url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")



@app.post("/audio-trans-video-whisper")
async def audio_to_transcript_endpoint_1(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    id: str = Form(...),
    questionId: str = Form(None),
    interpreterId: str = Form(None),
    participantId: str = Form(None),
    isOffline: bool = Form(...)
):
    try:
        audio_path, audiofile = await rename_and_save_file(file, document_name="document", version_id="0.0.1")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error saving file: {str(e)}")

    # Add transcription task to the background (NON-BLOCKING)
    background_tasks.add_task(handle_transcription_background_task_whisper, audio_path, id, questionId, interpreterId, participantId, isOffline)

    # Return a success response immediately while the transcription is processed in the background
    return {"status": "success", "message": "Transcription request received"}

# Background task to handle transcription process
async def handle_transcription_background_task_whisper(audio_path, id, questionId, interpreterId, participantId, isOffline):
    try:
        timestamp = int(time.time())
        output_dir = "/home/ubuntu/files/"
        os.makedirs(output_dir, exist_ok=True)

        response = process_audio_file_whisper(
            audio_path,
            output_dir,
            enable_stemming=True,
            batch_size=8,
            suppress_numerals=True,
        )

        # Callback URL to notify that transcription is complete
        url = 'https://new.kix.co/api/transcript/callback'
        data = {
            "id": id,
            "questionId": questionId,
            "interpreterId": interpreterId,
            "participantId": participantId,
            "isOffline": isOffline,
            "response": json.dumps(response),
            "apiKey": "SADIGIalsdfnIJJKBDSNFOBSasbdnbdigasnsaiubfjk=="
        }

        # Send the result to the callback URL
        requests.post(url, data=data)

    except Exception as e:
        print(f"Error processing transcription for recording id {recording_id}: {e}")

    finally:
        # Cleanup the entire output directory after processing is complete
        if output_dir:
            cleanup(output_dir)






import ffmpeg
import torch
import torchaudio
import whisperx
import numpy as np
import asyncio
import warnings
import os
import wget
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from omegaconf import OmegaConf
import json
import wget
import shutil
from datetime import datetime
import time

warnings.filterwarnings("ignore", message=".*torchaudio._backend.set_audio_backend.*")
DATA_PATH = "./data"

# Function to detect available GPUs
def get_device_for_model():
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        return {"device": "cuda", "device_index": list(range(gpu_count))}
    elif gpu_count == 1:
        return {"device": "cuda", "device_index": 0}
    else:
        return {"device": "cpu", "device_index": 0}


# Function to extract audio from video using ffmpeg
async def extract_audio_from_video(video_path, output_audio_path):
    try:
        file_extension = os.path.splitext(video_path)[1].lower()
        if file_extension in ['.mp4', '.mp3', '.webm']:  # Added .webm support
            output_audio_path = output_audio_path if output_audio_path.endswith('.wav') else f"{output_audio_path}.wav"
            ffmpeg.input(video_path).output(output_audio_path).run(quiet=True, overwrite_output=True)
            return output_audio_path
        elif file_extension == '.wav':
            return video_path
        else:
            print("Unsupported file format. Please use MP4, MP3, WAV, or WEBM.")
            return None
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

# Function to create NeMo config
def create_config(output_dir):
    DOMAIN_TYPE = "telephonic"
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
    MODEL_CONFIG = os.path.join(output_dir, CONFIG_FILE_NAME)
    if not os.path.exists(MODEL_CONFIG):
        MODEL_CONFIG = wget.download(CONFIG_URL, output_dir)

    config = OmegaConf.load(MODEL_CONFIG)

    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    meta = {
        "audio_filepath": os.path.join(output_dir, "mono_file.wav"),
        "offset": 0,
        "duration": None,
        "label": "infer",
        "text": "-",
        "rttm_filepath": None,
        "uem_filepath": None,
    }
    with open(os.path.join(data_dir, "input_manifest.json"), "w") as fp:
        json.dump(meta, fp)
        fp.write("\n")

    config.num_workers = 0
    config.diarizer.manifest_filepath = os.path.join(data_dir, "input_manifest.json")
    config.diarizer.out_dir = output_dir
    config.diarizer.oracle_vad = False
    config.diarizer.clustering.parameters.oracle_num_speakers = False
    config.diarizer.vad.model_path = "vad_multilingual_marblenet"
    config.diarizer.vad.parameters.onset = 0.8
    config.diarizer.vad.parameters.offset = 0.6
    config.diarizer.vad.parameters.pad_offset = -0.05
    config.diarizer.msdd_model.model_path = "diar_msdd_telephonic"

    return config

# Function to perform speaker diarization using NeMo
async def perform_speaker_diarization(audio_path, output_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Save audio as mono for NeMo compatibility
    wav, _ = torchaudio.load(audio_path)
    wav = wav.mean(dim=0).numpy()  # Convert to mono
    torchaudio.save(os.path.join(output_dir, "mono_file.wav"), torch.tensor(wav).unsqueeze(0), 16000)

    # Speaker Diarization using NeMo
    msdd_model = NeuralDiarizer(cfg=create_config(output_dir)).to(device)
    msdd_model.diarize()
    del msdd_model
    torch.cuda.empty_cache()

    # Map speakers to segments according to RTTM file
    speaker_segments = []
    with open(os.path.join(output_dir, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            start = float(line_list[5])
            duration = float(line_list[8])
            speaker_id = int(line_list[11].split("_")[-1])
            speaker_segments.append({"start": start, "end": start + duration, "speaker": speaker_id})

    return speaker_segments

# Function to clean up temporary files and directories
def cleanup_temp_files(output_dir):
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)  # Remove the entire output directory
            print(f"Cleaned up temporary files in {output_dir}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


# Function to transcribe audio using WhisperX with VAD
async def transcribe_audio_with_whisperx(audio_path, whisperx_model):
    transcription = whisperx_model.transcribe(audio_path)
    if not transcription.get("segments"):
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_a, metadata = whisperx.load_align_model(language_code=transcription["language"], device=device)
    aligned_transcription = whisperx.align(transcription["segments"], model_a, metadata, audio_path, device=device)
    return aligned_transcription

# Function to extract word-level segments from WhisperX transcription
async def extract_word_level_segments(aligned_transcription):
    word_segments = []
    for segment in aligned_transcription["segments"]:
        word_segments.extend(segment["words"])
    return word_segments


# Function to refine speaker segmentation based on WhisperX transcription and Resemblyzer outputs
async def refine_speaker_segmentation(transcription_segments, speaker_segments):
    refined_segments = []
    min_len = min(len(transcription_segments), len(speaker_segments))

    for i in range(min_len):
        try:
            text = transcription_segments[i]['text']
            speaker = speaker_segments[i]['speaker']
            refined_segments.append({
                "start": float(transcription_segments[i]['start']),
                "end": float(transcription_segments[i]['end']),
                "text": text,
                "speaker": int(speaker)
            })
        except KeyError as e:
            print(f"KeyError while refining speaker segmentation: {e}")
            continue

    return refined_segments


# Function to format transcription with speaker labels as plain text
async def format_plain_text_transcription(refined_segments):
    formatted_transcription = []
    current_speaker = None

    for segment in refined_segments:
        if segment["speaker"] != current_speaker:
            current_speaker = segment["speaker"]
            formatted_transcription.append(f"Speaker {current_speaker}: {segment['text']}\n")
        else:
            formatted_transcription.append(f"Speaker {current_speaker}: {segment['text']}\n")

    return "\n".join(formatted_transcription)


# Function to format transcription with word-level details as JSON
async def format_json_word_transcription(refined_segments, aligned_transcription, delay=0.01):
    word_segments = await extract_word_level_segments(aligned_transcription)
    word_diarized_segments = []

    previous_end_time = 0.0
    for word in word_segments:
        word_start = word.get("start", previous_end_time + delay)
        word_end = word.get("end", word_start + delay)

        previous_end_time = word_end

        for segment in refined_segments:
            if segment["start"] <= word_start <= segment["end"]:
                word_diarized_segments.append({
                    "word": word["word"],
                    "start": float(word_start),
                    "end": float(word_end),
                    "confidence": float(word.get("confidence", 1.0)),
                    "speaker": int(segment["speaker"]),
                    "speaker_confidence": 1.0
                })

    return word_diarized_segments


async def process_transcript_test(whisperx_model, data_path, audio_path, recording_id):
    output_dir = os.path.join(data_path, f"output_{recording_id}_{int(time.time())}")
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Step 1: Extract audio from video (if necessary)
        audio_output_path = os.path.join(output_dir, f"extracted_audio_{recording_id}.wav")
        audio_path = await extract_audio_from_video(audio_path, output_audio_path=audio_output_path)

        if audio_path is None:
            print("Skipping transcription as no audio was extracted.")
            return None

        # Step 2: Perform speaker diarization using NeMo
        diarization_output = await perform_speaker_diarization(audio_path, output_dir)

        # Step 3: Transcribe audio using WhisperX
        transcription = await transcribe_audio_with_whisperx(audio_path, whisperx_model)

        # Step 4: Refine the segmentation
        refined_segments = await refine_speaker_segmentation(transcription["segments"], diarization_output)

        # Step 5: Format the transcription outputs
        plain_text = await format_plain_text_transcription(refined_segments)
        json_transcription = await format_json_word_transcription(refined_segments, transcription)

        # Step 6: Merge the outputs
        merged_output = {
            "transcript": plain_text,
            "words": json_transcription
        }

        return merged_output

    except Exception as e:
        print(f"Error processing transcript: {e}")
        return None

    finally:
        # Cleanup temporary files and directories
        cleanup_temp_files(output_dir)

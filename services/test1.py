import os
import wget
from omegaconf import OmegaConf
import json
import shutil
import torch
import torchaudio
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from deepmultilingualpunctuation import PunctuationModel
import re
import logging
import nltk
import whisperx
from whisperx.utils import LANGUAGES, TO_LANGUAGE_CODE
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
from typing import List, Dict


print("hello")

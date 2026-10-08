# SpeakNext

Bilingual Speech-to-Suggestion Assistant for English and Bangla Presentations

SpeakNext is an end-to-end presentation assistant that listens to spoken speech, transcribes the utterances in real time, and generates contextually relevant next-sentence suggestions. Built with parameter-efficient fine-tuning (LoRA), modern sequence-to-sequence and causal language models, and text-to-speech synthesis, SpeakNext supports both English and Bangla speaker workflows.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Model Directory Layout](#model-directory-layout)
- [Running the Application](#running-the-application)
- [Training and Evaluation Workflows](#training-and-evaluation-workflows)
- [Hardware and Precision Guidelines](#hardware-and-precision-guidelines)
- [Contributing and License](#contributing-and-license)

---

## Overview

Public speakers and presenters often face pauses, loss of momentum, or difficulty transitioning between presentation points. SpeakNext acts as a real-time copilot:

1. Captures live audio or uploaded recordings.
2. Performs speech recognition using language-adapted Whisper models.
3. Retrieves relevant background dialogue context when applicable.
4. Generates multiple plausible continuations using fine-tuned language models.
5. Optionally speaks out the suggested sentence using text-to-speech engines.

The system is designed with a modular structure, enabling researchers and developers to run the full pipeline locally via Streamlit or retrain individual stages using provided Jupyter notebooks.

---

## Key Features

- Dual Language Support: Full pipeline support for English and Bangla speech recognition and text generation.
- Parameter-Efficient ASR: Fine-tuned `openai/whisper-small` using Low-Rank Adaptation (LoRA) for high accuracy and minimal weight footprints.
- Flexible Generation Models:
  - English: GPT-2, mT5, and BART completion models.
  - Bangla: Fine-tuned GPT-2 causal language model and mT5 sequence-to-sequence model.
- Semantic Context Retrieval: Optional sentence retrieval using Sentence-BERT (`all-MiniLM-L6-v2`) and dialogue corpora to enhance prompt context.
- Integrated Text-to-Speech (TTS):
  - English: Tacotron2 synthesis via Coqui TTS.
  - Bangla: Facebook MMS-TTS (`facebook/mms-tts-ben` via VITS).
- Streamlit Web Dashboard: User interface featuring in-browser audio recording, drag-and-drop audio file upload, manual transcript editing, customizable suggestion counts, and audio playback.

---

## System Architecture

```
[ Audio Source ]
  (Microphone Record or Audio File Upload)
         │
         ▼
[ ASR Module: Whisper-small + LoRA Adapter ]
         │
         ▼
[ Transcript Buffer ] ◄── (Manual User Edit / Example Input)
         │
         ├─── [ Context Retrieval (S-BERT + DailyDialog) ] (Optional, English)
         │
         ▼
[ Suggestion Engine ]
  ├── Causal LM (GPT-2, GPT-Neo)
  └── Seq2Seq (mT5, BART, FLAN-T5)
         │
         ▼
[ Ranked Next-Sentence Suggestions ]
         │
         ▼
[ Speech Synthesis (TTS) ] (Optional)
  ├── English: Coqui TTS (Tacotron2-DDC)
  └── Bangla: Facebook MMS-TTS (VITS)
```

---

## Repository Structure

```
SpeakNext/
├── app.py                         # Streamlit user interface and inference pipeline
├── requirements.txt               # Python package dependencies
├── english-model-training.ipynb   # English ASR and text generation training notebook
├── ml-bn-dataset.ipynb            # Bangla ASR, LM training, and benchmark notebook
├── Bangla_ML_Output_Report.pdf    # Experimental findings and performance metrics for Bangla
├── .gitignore                     # Git ignore rules for virtualenvs and models
└── README.md                      # Project documentation
```

---

## Installation and Setup

### 1. Prerequisites

- Python 3.9, 3.10, or 3.11 recommended
- CUDA-compatible GPU (recommended for real-time inference) or CPU
- FFmpeg installed and available in the system `PATH` (required for audio reading via `librosa` and `torchaudio`)
- Optional (for English Coqui TTS): `espeak-ng` installed on your operating system

### 2. Clone the Repository

```bash
git clone https://github.com/Junaed93/SpeakNext.git
cd SpeakNext
```

### 3. Create a Virtual Environment

On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux or macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install PyTorch according to your CUDA version from [pytorch.org](https://pytorch.org/), then install the remaining requirements:

```bash
pip install -r requirements.txt
```

---

## Model Directory Layout

The Streamlit application expects trained models to reside under a root folder (default: `./models`). You can adjust this path in the sidebar of the web application.

Organize the directory as follows:

```
models/
├── english/
│   ├── whisper-en-lora/          # PEFT LoRA adapter (adapter_config.json, adapter_model.bin/safetensors)
│   ├── gpt2-en-continuation/     # Hugging Face GPT-2 model and tokenizer files
│   ├── mt5-en-completion/        # Hugging Face mT5 model and tokenizer files
│   └── bart-en-completion/       # Hugging Face BART model and tokenizer files
└── bangla/
    ├── whisper-bn-lora/          # PEFT LoRA adapter for Bangla speech
    ├── gpt2-bn-continuation/     # Fine-tuned Bangla GPT-2 model and tokenizer
    └── mt5-bn-continuation/      # Fine-tuned Bangla mT5 model and tokenizer
```

Note: If a model folder is missing when selected in the interface, the app will show a notification indicating the expected folder path.

---

## Running the Application

Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

After starting the server, open the local URL printed in your console (usually `http://localhost:8501`).

### User Workflow

1. Configure Settings (Sidebar):
   - Set the models root path (defaults to `./models`).
   - Select Language: `English` or `Bangla`.
   - Select the suggestion model: `GPT-2`, `mT5`, or `BART` (for English).
   - Adjust the number of suggestions (1 to 5).
   - Toggle context retrieval (English only).
   - Enable or disable Text-to-Speech (TTS) playback.
2. Provide Speech or Text:
   - Audio tab: Click Record to speak through your microphone, or upload an audio file (`.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`).
   - Click "Transcribe" to run the fine-tuned Whisper model.
   - Text tab: Alternatively, type directly or click "Insert example sentence".
3. Review Transcript: Edit any transcription inaccuracies in the text area.
4. Generate Suggestions: Click "Suggest what comes next" to produce ranked completions.
5. Listen (Optional): If TTS is activated, the top suggestion will be synthesized and playable directly in the browser.

---

## Training and Evaluation Workflows

The repository includes complete self-contained Jupyter notebooks covering data preparation, training, evaluation, and export:

### English Pipeline (`english-model-training.ipynb`)

- Data Preparation: Loads paired audio datasets and cleans text transcriptions.
- Whisper LoRA Fine-Tuning: Trains low-rank adapters on `openai/whisper-small` with evaluation metrics (WER and CER via `evaluate` and `jiwer`).
- Text Language Models:
  - Fine-tunes GPT-2 with next-token causal loss, early stopping, and repetition penalties.
  - Trains GPT-Neo 125M as an alternative causal baseline.
  - Fine-tunes FLAN-T5 for conditional sentence completion.
- Evaluation and Export: Produces training curves and exports checkpoint weights in FP16 format for deployment.

### Bangla Pipeline (`ml-bn-dataset.ipynb`)

- Dataset Audit: Standardizes Unicode representations and speech splits across Bengali audio corpora.
- Whisper Bangla Adaptation: Evaluates zero-shot vs. fine-tuned Whisper performance for Bengali phonetics and vocabulary.
- Bengali Language Modeling:
  - Trains Bangla GPT-2 on clean Bengali text.
  - Implements an mT5-small baseline for sequence-to-sequence continuation.
- Benchmark and Reporting: Includes comparative evaluations across generation models. Experimental metrics and findings are documented in [Bangla_ML_Output_Report.pdf](Bangla_ML_Output_Report.pdf).

---

## Hardware and Precision Guidelines

- GPU Acceleration: Both training and inference automatically leverage CUDA if available (`torch.cuda.is_available()`).
- Precision Settings:
  - Causal LM (GPT-2) and Seq2Seq (mT5) are loaded in float32 in `app.py` to prevent known numerical underflow issues in fp16 inference with certain T5/mT5 tokenizers.
  - LoRA adapters for Whisper run on top of standard Whisper feature extractors.
- Audio Sample Rate: Speech models expect single-channel mono audio sampled at 16,000 Hz (`sr=16000`). Resampling is handled automatically by the application.

---

## Troubleshooting

- Missing Folder Error: Ensure the path provided in the sidebar "Models folder" matches the physical location where model checkpoints were unpacked.
- Audio Codec Issues: If audio decoding fails during file upload, verify that FFmpeg is installed and accessible from your system terminal.
- Coqui TTS on Windows: Coqui TTS requires Microsoft C++ Build Tools and `espeak-ng`. If you only require text suggestions without English voice playback, keep the TTS checkbox disabled.

---

## Contributing and License

Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request on the GitHub repository.

This project is licensed under the MIT License.

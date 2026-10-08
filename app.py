"""
Presentation-Assist: Streamlit front-end for the English + Bangla pipelines.

Pipeline: audio -> Whisper (LoRA) -> [context retrieval, English only] -> GPT-2 / mT5 / BART suggestions -> optional TTS

Expected folder layout (set the root in the sidebar, default ./models):

models/
  english/
    whisper-en-lora/          (adapter_config.json + adapter weights)
    gpt2-en-continuation/
    mt5-en-completion/
    bart-en-completion/
  bangla/
    whisper-bn-lora/
    gpt2-bn-continuation/
    mt5-bn-continuation/

Run:  streamlit run app.py
"""
import html
import os
import re
import tempfile
import time

import librosa
import numpy as np
import streamlit as st
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

APP_NAME = "SpeakNext"  # change this to rename the app
APP_TAGLINE = "Speech to suggestion assistant"

st.set_page_config(page_title=APP_NAME, layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=Noto+Sans+Bengali:wght@400;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', 'Noto Sans Bengali', sans-serif;
}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2rem; max-width: 1100px;}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 60%, #312e81 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {color: #e2e8f0 !important;}
[data-testid="stSidebar"] label p {font-weight: 700; letter-spacing: .02em;}
[data-testid="stSidebar"] div[data-baseweb="select"] > div,
[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    border-radius: 10px !important;
}
.brand {padding: .4rem 0 1.2rem 0; margin-bottom: 1rem; border-bottom: 1px solid rgba(255,255,255,0.15);}
.brand-name {
    font-size: 2rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1;
    background: linear-gradient(90deg, #a5b4fc, #f0abfc);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.brand-tag {font-size: .78rem; font-weight: 600; opacity: .75; margin-top: .35rem; text-transform: uppercase; letter-spacing: .12em;}
.side-label {font-size: .75rem; font-weight: 800; letter-spacing: .18em; opacity: .6; margin-bottom: .5rem;}

/* ---------- Hero ---------- */
.hero {
    padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
    background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #c026d3 100%);
    box-shadow: 0 18px 40px -18px rgba(79,70,229,.7);
}
.hero h1 {color: #fff !important; font-weight: 800; font-size: 2.3rem; margin: 0; letter-spacing: -.02em;}
.hero p {color: rgba(255,255,255,.88); font-weight: 600; margin: .4rem 0 0 0; font-size: 1.02rem;}
.pill {
    display: inline-block; margin-top: .9rem; padding: .25rem .8rem; border-radius: 999px;
    background: rgba(255,255,255,.18); color: #fff; font-weight: 700; font-size: .8rem; letter-spacing: .06em;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {gap: .5rem;}
.stTabs [data-baseweb="tab"] {font-weight: 700; padding: .6rem 1.1rem; border-radius: 10px 10px 0 0;}

/* ---------- Buttons ---------- */
.stButton > button {border-radius: 12px; font-weight: 700; padding: .55rem 1.3rem; transition: all .15s ease;}
.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #6366f1, #8b5cf6); border: none; color: #fff;
    box-shadow: 0 8px 20px -8px rgba(99,102,241,.8);
}
.stButton > button:hover {transform: translateY(-1px);}

/* ---------- Inputs ---------- */
.stTextArea textarea {border-radius: 14px; font-size: 1.05rem; font-weight: 500;}
[data-testid="stFileUploader"] section {border-radius: 14px;}

/* ---------- Suggestions ---------- */
.section-title {font-weight: 800; font-size: 1.35rem; margin: 1.4rem 0 .2rem 0; letter-spacing: -.01em;}
.section-sub {font-weight: 600; opacity: .6; font-size: .85rem; margin-bottom: .9rem;}
.sugg {
    display: flex; gap: 1rem; align-items: flex-start; padding: 1rem 1.2rem; margin-bottom: .75rem;
    border-radius: 16px; background: rgba(99,102,241,.08); border: 1px solid rgba(99,102,241,.25);
    border-left: 6px solid #7c3aed; box-shadow: 0 10px 24px -16px rgba(79,70,229,.6);
}
.sugg-num {
    min-width: 2rem; height: 2rem; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #6366f1, #c026d3); color: #fff; font-weight: 800; font-size: .9rem;
}
.sugg-text {font-weight: 700; font-size: 1.08rem; line-height: 1.55;}
</style>
""",
    unsafe_allow_html=True,
)

LANGS = {
    "English": {
        "code": "en",
        "dir": "english",
        "whisper": "whisper-en-lora",
        "gpt2": "gpt2-en-continuation",
        "seq2seq": {"mT5": "mt5-en-completion", "BART": "bart-en-completion"},
        "example": "The main challenge in this project is",
    },
    "Bangla": {
        "code": "bn",
        "dir": "bangla",
        "whisper": "whisper-bn-lora",
        "gpt2": "gpt2-bn-continuation",
        "seq2seq": {"mT5": "mt5-bn-continuation"},
        "example": "আজকের উপস্থাপনার মূল বিষয় হলো বাংলা ভাষা।",
    },
}

# ----------------------------------------------------------------- sidebar
st.sidebar.markdown(
    f'<div class="brand"><div class="brand-name">{APP_NAME}</div>'
    f'<div class="brand-tag">{APP_TAGLINE}</div></div>'
    '<div class="side-label">SETTINGS</div>',
    unsafe_allow_html=True,
)
MODELS_ROOT = st.sidebar.text_input("Models folder", "./models")
lang_name = st.sidebar.selectbox("Language", list(LANGS))
cfg = LANGS[lang_name]
lang_dir = os.path.join(MODELS_ROOT, cfg["dir"])

generator_choice = st.sidebar.selectbox("Suggestion model", ["GPT-2"] + list(cfg["seq2seq"]))
n_sugg = st.sidebar.slider("Number of suggestions", 1, 5, 3)
use_context = False
if lang_name == "English":
    use_context = st.sidebar.checkbox("Use DailyDialog context retrieval (slower first load)", value=False)
use_tts = st.sidebar.checkbox("Speak the top suggestion (TTS)", value=False)
st.sidebar.caption(f"Device: **{device}**")


def need(path):
    if not os.path.isdir(path):
        st.error(f"Folder not found: `{path}`. Check the models folder in the sidebar.")
        st.stop()
    return path


# ----------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Loading Whisper + LoRA adapter...")
def load_whisper(adapter_path, code):
    proc = WhisperProcessor.from_pretrained("openai/whisper-small", language=code, task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    base.generation_config.language = code
    base.generation_config.task = "transcribe"
    base.generation_config.forced_decoder_ids = None
    model = PeftModel.from_pretrained(base, adapter_path).to(device).eval()
    return proc, model


@st.cache_resource(show_spinner="Loading GPT-2...")
def load_gpt2(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path).float().to(device).eval()  # .float(): weights may be saved in fp16
    return tok, model


@st.cache_resource(show_spinner="Loading seq2seq model...")
def load_seq2seq(path):
    tok = AutoTokenizer.from_pretrained(path)
    tok.truncation_side = "left"
    model = AutoModelForSeq2SeqLM.from_pretrained(path).float().to(device).eval()  # mT5 breaks in fp16
    return tok, model


@st.cache_resource(show_spinner="Loading Sentence-BERT + DailyDialog (one-off)...")
def load_context_pool():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    attempts = [
        ("li2017dailydialog/daily_dialog", dict(split="train"), "dialog"),
        ("roskoN/dailydialog", dict(split="train"), "utterances"),
    ]
    pool = []
    for name, kw, field in attempts:
        try:
            ds = load_dataset(name, **kw)
            for ex in ds:
                for t in ex[field]:
                    t = t.strip()
                    if len(t) > 10:
                        pool.append(t)
            break
        except Exception:
            continue
    if not pool:
        return sbert, None, []
    pool = pool[:5000]
    emb = sbert.encode(pool, convert_to_tensor=True, batch_size=128)
    return sbert, emb, pool


@st.cache_resource(show_spinner="Loading TTS...")
def load_tts(code):
    if code == "en":
        from TTS.api import TTS

        return "coqui", TTS("tts_models/en/ljspeech/tacotron2-DDC").to(device)
    from transformers import VitsModel

    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
    model = VitsModel.from_pretrained("facebook/mms-tts-ben").to(device).eval()
    return "mms", (tok, model)


# ----------------------------------------------------------------- inference helpers
def load_audio_bytes(data: bytes, suffix=".wav"):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        y, _ = librosa.load(path, sr=16000, mono=True)
    finally:
        os.remove(path)
    return y


def transcribe(proc, model, audio):
    feats = proc.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        ids = model.generate(input_features=feats, max_new_tokens=128)
    return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()


def trim_to_sentence(text):
    text = text.strip()
    m = re.search(r"[।?!.]", text)
    return text[: m.end()].strip() if m else text


def last_context(text, max_words=40):
    sents = [s for s in re.split(r"(?<=[।?!.])\s+", text.strip()) if s]
    words = " ".join(sents[-2:]).split()
    return " ".join(words[-max_words:])


def gpt2_suggest(tok, model, prompt, n, max_new_tokens=35):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][:, -200:].to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.92,
            temperature=0.8,
            num_return_sequences=n,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=tok.eos_token_id,
        )
    return [trim_to_sentence(tok.decode(o[ids.shape[1]:], skip_special_tokens=True)) for o in out]


def seq2seq_suggest(tok, model, prompt, n, beam):
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    kw = (
        dict(num_beams=max(5, n), num_return_sequences=n)
        if beam
        else dict(do_sample=True, top_p=0.92, temperature=0.9, num_return_sequences=n)
    )
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=50, no_repeat_ngram_size=3, **kw)
    return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]


def speak(text, code):
    kind, obj = load_tts(code)
    path = os.path.join(tempfile.gettempdir(), "assist_tts.wav")
    if kind == "coqui":
        obj.tts_to_file(text=text, file_path=path)
    else:
        from scipy.io import wavfile

        tok, model = obj
        enc = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            wav = model(**enc).waveform[0].cpu().numpy()
        wavfile.write(path, model.config.sampling_rate, wav)
    return path


def get_suggestions(text):
    """Returns (suggestions, context_used, seconds)."""
    t0 = time.perf_counter()
    context = []
    if generator_choice == "GPT-2":
        tok, model = load_gpt2(need(os.path.join(lang_dir, cfg["gpt2"])))
        prompt = last_context(text) if lang_name == "Bangla" else text
        sugg = gpt2_suggest(tok, model, prompt, n_sugg)
    else:
        tok, model = load_seq2seq(need(os.path.join(lang_dir, cfg["seq2seq"][generator_choice])))
        if lang_name == "English":
            if use_context:
                from sentence_transformers import util

                sbert, emb, pool = load_context_pool()
                if emb is not None:
                    q = sbert.encode(text, convert_to_tensor=True)
                    idx = util.cos_sim(q, emb)[0].topk(3).indices.tolist()
                    context = [pool[i] for i in idx]
            prompt = f"context: {' | '.join(context)} incomplete: {text}"
            sugg = seq2seq_suggest(tok, model, prompt, n_sugg, beam=True)
        else:
            sugg = seq2seq_suggest(tok, model, last_context(text), n_sugg, beam=False)
    return sugg, context, time.perf_counter() - t0


# ----------------------------------------------------------------- UI
st.markdown(
    f'<div class="hero"><h1>{APP_NAME}</h1>'
    f"<p>Speak, and get the next sentence suggested instantly.</p>"
    f'<span class="pill">{lang_name.upper()} MODE</span></div>',
    unsafe_allow_html=True,
)

if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "last_lang" not in st.session_state or st.session_state.last_lang != lang_name:
    st.session_state.transcript = ""
    st.session_state.last_lang = lang_name

tab_audio, tab_text = st.tabs(["Audio", "Type text (skip ASR)"])

with tab_audio:
    c1, c2 = st.columns(2)
    with c1:
        rec = st.audio_input("Record")
    with c2:
        up = st.file_uploader("...or upload", type=["wav", "mp3", "flac", "m4a", "ogg"])

    src = rec or up
    if src is not None:
        st.audio(src)
        if st.button("Transcribe", type="primary"):
            proc, asr = load_whisper(need(os.path.join(lang_dir, cfg["whisper"])), cfg["code"])
            t0 = time.perf_counter()
            suffix = os.path.splitext(getattr(src, "name", "rec.wav"))[1] or ".wav"
            audio = load_audio_bytes(src.getvalue(), suffix)
            st.session_state.transcript = transcribe(proc, asr, audio)
            st.session_state.asr_time = time.perf_counter() - t0
        if "asr_time" in st.session_state and st.session_state.transcript:
            st.caption(f"ASR time: {st.session_state.asr_time:.2f}s")

with tab_text:
    if st.button("Insert example sentence"):
        st.session_state.transcript = cfg["example"]

text = st.text_area(
    "Transcript (edit it if the ASR made mistakes)",
    key="transcript",
    height=100,
)

if st.button("Suggest what comes next", type="primary", disabled=not text.strip()):
    with st.spinner("Generating..."):
        sugg, ctx, secs = get_suggestions(text.strip())
    st.markdown(
        f'<div class="section-title">Suggestions</div>'
        f'<div class="section-sub">{generator_choice} &nbsp;|&nbsp; {secs:.2f}s</div>',
        unsafe_allow_html=True,
    )
    for i, s in enumerate(sugg, 1):
        st.markdown(
            f'<div class="sugg"><div class="sugg-num">{i}</div>'
            f'<div class="sugg-text">{html.escape(s)}</div></div>',
            unsafe_allow_html=True,
        )
    if ctx:
        with st.expander("Retrieved context"):
            for c in ctx:
                st.write("•", c)
    if use_tts and sugg and sugg[0]:
        try:
            st.audio(speak(sugg[0], cfg["code"]))
        except Exception as e:
            st.warning(f"TTS failed: {e}")

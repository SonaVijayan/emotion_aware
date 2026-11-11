
"""
Multilingual Multimodal Emotion-Aware Conversational AI
Complete Streamlit Research Prototype – FINAL FIXED VERSION

All known errors fixed:
- AudioClassificationPipeline requires key "raw"
- Whisper ASR uses key "array"
- asyncio.run() crash in Streamlit fixed with ThreadPoolExecutor
- Device handling modernized
"""

import os
import re
import time
import asyncio
import tempfile
import concurrent.futures
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Multilingual Multimodal Emotion AI",
    page_icon="🎭",
    layout="wide"
)

TEXT_EMOTION_MODEL = os.getenv(
    "TEXT_EMOTION_MODEL",
    "j-hartmann/emotion-english-distilroberta-base"
)

WHISPER_MODEL = os.getenv(
    "WHISPER_MODEL",
    "openai/whisper-small"
)

WAV2VEC2_SER_MODEL = os.getenv(
    "WAV2VEC2_SER_MODEL",
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
)

MLLM_MODEL = os.getenv(
    "MLLM_MODEL",
    "Qwen/Qwen2.5-0.5B-Instruct"
)

EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]

LABEL_ALIASES = {
    "happy": "joy",
    "happiness": "joy",
    "excited": "joy",
    "angry": "anger",
    "mad": "anger",
    
    "sad": "sadness",
    "depressed": "sadness",
    "scared": "fear",
    "fearful": "fear",
    "anxious": "fear",
    "astonishment": "surprise",
    "surprised": "surprise",
    "disappointed": "sadness",
    "calm": "neutral",
    "other": "neutral"
}

VOICE_BY_LANGUAGE = {
    "en": "en-US-AriaNeural",
    "hi": "hi-IN-SwaraNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "it": "it-IT-ElsaNeural",
    "pt": "pt-BR-FranciscaNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural"
}

EMOTION_TTS_STYLE = {
    "joy": ("+8%", "+8Hz"),
    "sadness": ("-12%", "-8Hz"),
    "anger": ("+5%", "+3Hz"),
    "fear": ("-5%", "+2Hz"),
    "surprise": ("+10%", "+10Hz"),
    "disgust": ("-3%", "-2Hz"),
    "neutral": ("+0%", "+0Hz")
}

# ============================================================
# USER INTERFACE STYLE
# ============================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    color: white;
}
h1 {
    color: #feca57;
    font-weight: 800;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,.07);
    padding: 12px;
    border-radius: 12px;
}
section[data-testid="stSidebar"] {
    background: rgba(10,10,30,.92);
}
.stButton>button {
    border-radius: 10px;
    font-weight: 650;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DEVICE CONFIGURATION
# ============================================================

def get_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_text_emotion_model(model_name):
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=model_name,
        top_k=None,
        device=get_device()
    )


@st.cache_resource(show_spinner=False)
def load_asr_model(model_name):
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=get_device(),
        chunk_length_s=30,
        stride_length_s=(5, 5)
    )


@st.cache_resource(show_spinner=False)
def load_speech_emotion_model(model_name):
    from transformers import pipeline
    return pipeline(
        "audio-classification",
        model=model_name,
        top_k=None,
        device=get_device()
    )


@st.cache_resource(show_spinner=False)
def load_mllm(model_name):
    from transformers import pipeline
    return pipeline(
        "text-generation",
        model=model_name,
        device=get_device()
    )


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(label):
    key = str(label).strip().lower()
    key = re.sub(r"^[a-z0-9]+_", "", key)
    key = key.replace(" ", "_")
    key = LABEL_ALIASES.get(key, key)
    if key in EMOTION_LABELS:
        return key
    return "neutral"


def normalize_distribution(raw):
    result = {label: 0.0 for label in EMOTION_LABELS}
    for item in raw:
        label = normalize_label(item.get("label", "neutral"))
        score = float(item.get("score", 0.0))
        result[label] += max(0.0, score)
    total = sum(result.values())
    if total <= 0:
        return {label: 1.0 / len(EMOTION_LABELS) for label in EMOTION_LABELS}
    return {key: value / total for key, value in result.items()}


# ============================================================
# CONFIDENCE CALCULATION
# ============================================================

def entropy_confidence(probabilities):
    p = np.asarray(list(probabilities.values()), dtype=float)
    p = p[p > 0]
    if len(p) <= 1:
        return 1.0
    entropy = -np.sum(p * np.log(p))
    confidence = 1.0 - (entropy / np.log(len(EMOTION_LABELS)))
    return float(np.clip(confidence, 0.0, 1.0))


def top_emotion(probabilities):
    label = max(probabilities, key=probabilities.get)
    return label, float(probabilities[label])


# ============================================================
# CONFIDENCE-AWARE MULTIMODAL FUSION
# ============================================================

def confidence_aware_fusion(
    text_probs=None,
    audio_probs=None,
    text_quality=1.0,
    audio_quality=1.0
):
    gates = {}
    if text_probs is not None:
        gates["text"] = entropy_confidence(text_probs) * np.clip(text_quality, 0, 1)
    if audio_probs is not None:
        gates["audio"] = entropy_confidence(audio_probs) * np.clip(audio_quality, 0, 1)
    if not gates:
        raise ValueError("No valid modality available for fusion.")
    total_gate = sum(gates.values())
    if total_gate <= 1e-8:
        weights = {key: 1.0 / len(gates) for key in gates}
    else:
        weights = {key: value / total_gate for key, value in gates.items()}
    fused = {label: 0.0 for label in EMOTION_LABELS}
    if text_probs is not None:
        for label in EMOTION_LABELS:
            fused[label] += weights.get("text", 0.0) * text_probs[label]
    if audio_probs is not None:
        for label in EMOTION_LABELS:
            fused[label] += weights.get("audio", 0.0) * audio_probs[label]
    total = sum(fused.values())
    if total > 0:
        fused = {key: value / total for key, value in fused.items()}
    return fused, weights


# ============================================================
# LANGUAGE DETECTION
# ============================================================

def detect_language(text):
    if not text or len(text.strip()) < 3:
        return "unknown"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        return detect(text)
    except Exception:
        return "unknown"


# ============================================================
# AUDIO PREPROCESSING
# ============================================================

def load_audio_16k(audio_path):
    """
    Decode audio and resample to mono 16 kHz.
    Returns (audio_array, sampling_rate)
    """
    import librosa

    audio, sample_rate = librosa.load(
        audio_path,
        sr=16000,
        mono=True
    )

    if audio is None or len(audio) == 0:
        raise ValueError("Audio file is empty.")

    audio = np.asarray(audio, dtype=np.float32)

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numerical values.")

    peak = np.max(np.abs(audio))
    if peak < 1e-5:
        raise ValueError("Audio is silent or its volume is too low.")

    return audio, 16000


# ============================================================
# MULTILINGUAL WHISPER TRANSCRIPTION
# ============================================================

def transcribe_audio(asr, audio_data):
    """
    Automatic multilingual transcription.
    audio_data must be a dict with key "array"
    """
    result = asr(
        audio_data,
        generate_kwargs={"task": "transcribe"}
    )

    transcript = ""
    if isinstance(result, dict):
        transcript = result.get("text", "").strip()
    else:
        transcript = str(result).strip()

    transcript = re.sub(r"\s+", " ", transcript).strip()

    if not transcript:
        return "", "unknown"

    # Reject obvious repeated-character hallucinations
    characters = [c for c in transcript if not c.isspace()]
    if len(characters) >= 30:
        counts = Counter(characters)
        dominant_ratio = max(counts.values()) / len(characters)
        if dominant_ratio > 0.70:
            return "", "unknown"

    # Reject highly repetitive token sequences
    words = transcript.split()
    if len(words) >= 8 and len(set(words)) <= 2:
        return "", "unknown"

    language = detect_language(transcript)
    return transcript, language


# ============================================================
# SPEECH EMOTION RECOGNITION
# ============================================================

def classify_audio_emotion(ser, audio_data):
    """
    audio_data must be a dict with key "raw"
    """
    raw = ser(audio_data)
    if isinstance(raw, dict):
        raw = [raw]
    probabilities = normalize_distribution(raw)
    confidence = entropy_confidence(probabilities)
    return probabilities, confidence


# ============================================================
# DIALOGUE EMOTION MEMORY
# ============================================================

def init_memory():
    if "emotion_memory" not in st.session_state:
        st.session_state.emotion_memory = []
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
    if "generated_tts" not in st.session_state:
        st.session_state.generated_tts = None


def update_emotion_memory(text, language, probabilities, source, gates):
    label, confidence = top_emotion(probabilities)
    st.session_state.emotion_memory.append({
        "turn": len(st.session_state.emotion_memory) + 1,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "language": language,
        "input_source": source,
        "text": text[:300],
        "emotion": label,
        "confidence": confidence,
        "text_gate": gates.get("text", 0.0),
        "audio_gate": gates.get("audio", 0.0)
    })


def memory_dataframe():
    return pd.DataFrame(st.session_state.emotion_memory)


def memory_context(max_turns=6):
    rows = st.session_state.emotion_memory[-max_turns:]
    return "\n".join(
        f"Turn {r['turn']}: "
        f"emotion={r['emotion']}, "
        f"confidence={r['confidence']:.3f}, "
        f"language={r['language']}"
        for r in rows
    )


# ============================================================
# OPTIONAL MULTILINGUAL MLLM
# ============================================================

def contextual_interpretation(generator, user_text):
    history = memory_context()
    prompt = (
        "You are a careful multilingual conversational assistant. "
        "Understand the user's message and emotional context. "
        "Respond in the same language as the latest user message. "
        "Do not diagnose the user or claim certainty about emotions.\n\n"
        f"Recent dialogue emotion memory:\n"
        f"{history or 'No previous turns.'}\n\n"
        f"Latest message: {user_text}\n\n"
        "Give a short, natural and empathetic response."
    )
    result = generator(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        return_full_text=False
    )
    return result[0]["generated_text"].strip()


# ============================================================
# MULTILINGUAL TEXT-TO-SPEECH (robust against event-loop errors)
# ============================================================

async def _edge_tts_save(text, voice, rate, pitch, output_path):
    import edge_tts
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch
    )
    await communicate.save(output_path)


def create_tts_audio(text, language, emotion):
    if not text or not text.strip():
        return None

    language = language if language in VOICE_BY_LANGUAGE else "en"
    voice = VOICE_BY_LANGUAGE[language]
    rate, pitch = EMOTION_TTS_STYLE.get(emotion, ("+0%", "+0Hz"))

    output_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            output_path = f.name

        def _run_async():
            asyncio.run(_edge_tts_save(text, voice, rate, pitch, output_path))

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Running inside Streamlit / Jupyter → use a thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_async)
                future.result(timeout=60)
        else:
            _run_async()

        with open(output_path, "rb") as f:
            audio_bytes = f.read()
        return audio_bytes

    except Exception as exc:
        st.error(f"TTS generation failed: {exc}")
        return None
    finally:
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except Exception:
                pass


# ============================================================
# RESEARCH METRICS
# ============================================================

def expected_calibration_error(y_true, probabilities, labels):
    probs = np.asarray(probabilities, dtype=float)
    if len(probs) == 0:
        return float("nan")
    pred_idx = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    predictions = np.asarray(labels)[pred_idx]
    correct = (np.asarray(y_true) == predictions)
    ece = 0.0
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (confidence > edges[i]) & (confidence <= edges[i + 1])
        if np.any(mask):
            ece += (
                np.mean(mask)
                * abs(np.mean(correct[mask]) - np.mean(confidence[mask]))
            )
    return float(ece)


def classification_metrics(y_true, y_pred, labels, probabilities):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return pd.DataFrame([
        ["Accuracy", accuracy_score(y_true, y_pred)],
        ["Macro Precision", p],
        ["Macro Recall", r],
        ["Macro F1-score", f],
        ["Weighted Precision", pw],
        ["Weighted Recall", rw],
        ["Weighted F1-score", fw],
        ["Expected Calibration Error", expected_calibration_error(y_true, probabilities, labels)]
    ], columns=["Metric", "Value"])


# ============================================================
# APPLICATION INITIALIZATION
# ============================================================

init_memory()

st.title("🎭 Multilingual Multimodal Emotion-Aware AI")

st.caption(
    "Multilingual Speech Recognition | Emotion Recognition | "
    "Confidence-Aware Fusion | Dialogue Memory | Multilingual TTS"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("Model Configuration")
    st.caption("Models are downloaded from Hugging Face on first use.")
    text_model_name = st.text_input("Text emotion model", TEXT_EMOTION_MODEL)
    whisper_model_name = st.text_input("Multilingual Whisper model", WHISPER_MODEL)
    ser_model_name = st.text_input("Speech emotion model", WAV2VEC2_SER_MODEL)
    mllm_model_name = st.text_input("mLLM model", MLLM_MODEL)
    use_mllm = st.checkbox("Enable optional mLLM response", value=False)
    st.markdown("---")
    st.info(
        "Whisper performs automatic multilingual transcription. "
        "Emotion classification depends on the language coverage "
        "of the selected emotion models."
    )


# ============================================================
# TABS
# ============================================================

tab_live, tab_memory, tab_eval, tab_method = st.tabs([
    "Live Multimodal Analysis",
    "Dialogue Emotion Memory",
    "Research Evaluation",
    "Methodology and Limitations"
])


# ============================================================
# LIVE MULTIMODAL ANALYSIS
# ============================================================

with tab_live:
    st.subheader("Text and Audio Input")

    text_input = st.text_area(
        "Enter text or a conversation turn",
        placeholder="Enter English, Hindi, Hinglish, or another language...",
        height=120
    )

    audio_file = st.file_uploader(
        "Upload speech audio",
        type=["wav", "mp3", "m4a", "flac", "ogg"]
    )

    analyze = st.button("Analyze Emotion", type="primary", use_container_width=True)

    if analyze:
        if not text_input.strip() and audio_file is None:
            st.warning("Enter text or upload an audio file.")
        else:
            try:
                with st.spinner("Loading models and analyzing input..."):
                    text_probs = None
                    audio_probs = None
                    transcript = ""
                    language = "unknown"
                    source = "text"

                    # TEXT EMOTION
                    if text_input.strip():
                        language = detect_language(text_input)
                        text_model = load_text_emotion_model(text_model_name)
                        text_raw = text_model(text_input)[0]
                        text_probs = normalize_distribution(text_raw)

                    # AUDIO PROCESSING
                    if audio_file is not None:
                        suffix = os.path.splitext(audio_file.name)[1] or ".wav"

                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                            f.write(audio_file.getbuffer())
                            audio_path = f.name

                        try:
                            # Decode once
                            audio_array, sampling_rate = load_audio_16k(audio_path)

                            # Correct dictionaries for each pipeline
                            asr_input = {
                                "array": audio_array,
                                "sampling_rate": sampling_rate
                            }
                            ser_input = {
                                "raw": audio_array,          # REQUIRED by AudioClassificationPipeline
                                "sampling_rate": sampling_rate
                            }

                            asr = load_asr_model(whisper_model_name)
                            ser = load_speech_emotion_model(ser_model_name)

                            transcript, asr_language = transcribe_audio(asr, asr_input)

                            # SER runs independently of ASR success
                            audio_probs, _ = classify_audio_emotion(ser, ser_input)

                            source = "audio" if not text_input.strip() else "text + audio"

                            if not text_input.strip():
                                language = asr_language if asr_language != "unknown" else "unknown"

                        finally:
                            if os.path.exists(audio_path):
                                os.unlink(audio_path)

                    # FUSION
                    if text_probs is None and audio_probs is None:
                        st.error("No valid emotion prediction was generated.")
                        st.stop()

                    fused_probs, gates = confidence_aware_fusion(text_probs, audio_probs)
                    dominant, confidence = top_emotion(fused_probs)

                    content = text_input.strip() or transcript or "Audio input"

                    # MEMORY UPDATE
                    update_emotion_memory(content, language, fused_probs, source, gates)

                    # Persist analysis
                    st.session_state.last_analysis = {
                        "transcript": transcript,
                        "language": language,
                        "source": source,
                        "content": content,
                        "probabilities": fused_probs,
                        "gates": gates,
                        "dominant": dominant,
                        "confidence": confidence
                    }

                    # Response text
                    if use_mllm:
                        generator = load_mllm(mllm_model_name)
                        tts_text = contextual_interpretation(generator, content)
                    else:
                        if language == "hi":
                            tts_text = (
                                f"मैं आपकी बात समझ रहा हूँ। "
                                f"आपकी भावना {dominant} हो सकती है। "
                                f"मैं आपकी बात सुनने के लिए यहाँ हूँ।"
                            )
                        else:
                            tts_text = (
                                "I understand what you are expressing. "
                                f"The detected emotional tone may be {dominant}. "
                                "I am here to listen."
                            )

                    st.session_state.tts_text = tts_text
                    st.session_state.tts_language = (
                        language if language in VOICE_BY_LANGUAGE else "en"
                    )
                    st.session_state.tts_emotion = dominant
                    st.session_state.generated_tts = None

                st.success("Analysis completed.")

            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                st.exception(exc)

    # DISPLAY PERSISTED ANALYSIS
    analysis = st.session_state.last_analysis

    if analysis is not None:
        transcript = analysis["transcript"]

        if transcript:
            st.markdown("#### Whisper Transcription")
            st.info(transcript)
            st.caption(
                "Language detected from the recognized transcript: "
                + analysis["language"]
            )
        elif analysis["source"] in ["audio", "text + audio"]:
            st.warning(
                "Whisper did not produce a reliable transcript. "
                "The audio emotion model was still evaluated independently."
            )

        st.markdown("#### Fused Emotion Distribution")
        result_df = pd.DataFrame({
            "Emotion": EMOTION_LABELS,
            "Probability": [analysis["probabilities"][x] for x in EMOTION_LABELS]
        }).sort_values("Probability", ascending=False)

        st.dataframe(
            result_df.style.format({"Probability": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant Emotion", analysis["dominant"].capitalize())
        c2.metric("Confidence", f"{analysis['confidence']:.2%}")
        c3.metric("Detected Language", analysis["language"])

        st.markdown("#### Confidence-Aware Modality Gates")
        gate_df = pd.DataFrame([
            {"Modality": key, "Fusion Weight": value}
            for key, value in analysis["gates"].items()
        ])
        st.dataframe(
            gate_df.style.format({"Fusion Weight": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        if use_mllm and st.session_state.get("tts_text"):
            st.markdown("#### Contextual mLLM Response")
            st.write(st.session_state.tts_text)

        # TTS
        st.markdown("#### Emotion-Conditioned Multilingual TTS")
        if st.button("Generate Spoken Response"):
            with st.spinner("Generating speech..."):
                audio_bytes = create_tts_audio(
                    st.session_state.tts_text,
                    st.session_state.tts_language,
                    st.session_state.tts_emotion
                )
                if audio_bytes:
                    st.session_state.generated_tts = audio_bytes

        if st.session_state.generated_tts:
            st.audio(st.session_state.generated_tts, format="audio/mp3")
            st.download_button(
                "Download Generated Speech",
                data=st.session_state.generated_tts,
                file_name="emotion_conditioned_response.mp3",
                mime="audio/mpeg"
            )


# ============================================================
# DIALOGUE EMOTION MEMORY
# ============================================================

with tab_memory:
    st.subheader("Dialogue Emotion Memory")

    memory_df = memory_dataframe()

    if memory_df.empty:
        st.info("No conversation turns have been analyzed in this session.")
    else:
        st.dataframe(memory_df, use_container_width=True, hide_index=True)

        counts = (
            memory_df["emotion"]
            .value_counts()
            .rename_axis("Emotion")
            .reset_index(name="Turns")
        )

        st.markdown("#### Emotion Distribution Across Dialogue")
        st.bar_chart(counts.set_index("Emotion"))

        st.download_button(
            "Download Dialogue Memory CSV",
            memory_df.to_csv(index=False).encode("utf-8"),
            "dialogue_emotion_memory.csv",
            "text/csv"
        )

        if st.button("Clear Dialogue Memory"):
            st.session_state.emotion_memory = []
            st.session_state.last_analysis = None
            st.session_state.generated_tts = None
            st.rerun()


# ============================================================
# RESEARCH EVALUATION
# ============================================================

with tab_eval:
    st.subheader("Quantitative Research Evaluation")

    st.write("Upload a labelled CSV dataset to calculate classification metrics.")

    st.code(
        "text,label\n"
        "I am very happy today,joy\n"
        "I feel disappointed,sadness\n"
        "I am angry,anger\n"
        "I am afraid,fear",
        language="csv"
    )

    eval_file = st.file_uploader(
        "Upload Evaluation Dataset (CSV)",
        type=["csv"],
        key="evaluation_csv"
    )

    if st.button("Run Text Classification Evaluation", type="primary"):
        if eval_file is None:
            st.warning("Upload a labelled CSV first.")
        else:
            try:
                df = pd.read_csv(eval_file)

                if not {"text", "label"}.issubset(df.columns):
                    raise ValueError("CSV must contain text and label columns.")

                model = load_text_emotion_model(text_model_name)

                y_true = []
                y_pred = []
                probability_rows = []
                latencies = []
                predictions = []

                with st.spinner("Running evaluation..."):
                    for _, row in df.iterrows():
                        text = str(row["text"])
                        truth = normalize_label(row["label"])

                        start = time.perf_counter()
                        raw = model(text)[0]
                        probs = normalize_distribution(raw)
                        predicted, confidence = top_emotion(probs)
                        elapsed = (time.perf_counter() - start) * 1000

                        y_true.append(truth)
                        y_pred.append(predicted)
                        probability_rows.append([probs[label] for label in EMOTION_LABELS])
                        latencies.append(elapsed)

                        predictions.append({
                            "Text": text,
                            "Ground Truth": truth,
                            "Prediction": predicted,
                            "Confidence": confidence,
                            "Latency (ms)": elapsed
                        })

                labels = sorted(set(y_true))
                metrics = classification_metrics(y_true, y_pred, labels, probability_rows)

                metrics.loc[len(metrics)] = ["Mean Inference Latency (ms)", np.mean(latencies)]
                metrics.loc[len(metrics)] = ["Median Inference Latency (ms)", np.median(latencies)]
                metrics.loc[len(metrics)] = ["Samples Evaluated", len(y_true)]

                st.success(f"Evaluation completed: {len(y_true)} samples.")

                st.markdown("#### Research Metrics")
                st.dataframe(
                    metrics.style.format({"Value": "{:.4f}"}),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "Download Research Metrics",
                    metrics.to_csv(index=False).encode("utf-8"),
                    "research_metrics.csv",
                    "text/csv"
                )

                predictions_df = pd.DataFrame(predictions)
                st.markdown("#### Sample Predictions")
                st.dataframe(predictions_df, use_container_width=True, hide_index=True)

                st.download_button(
                    "Download Predictions",
                    predictions_df.to_csv(index=False).encode("utf-8"),
                    "sample_predictions.csv",
                    "text/csv"
                )

                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                st.dataframe(
                    pd.DataFrame(cm, index=labels, columns=labels),
                    use_container_width=True
                )

            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
                st.exception(exc)


# ============================================================
# METHODOLOGY AND LIMITATIONS
# ============================================================

with tab_method:
    st.subheader("Implemented Research Modules")

    method_table = pd.DataFrame([
        ["Speech Recognition", "OpenAI Whisper Small", "Automatic multilingual speech transcription"],
        ["Speech Emotion Recognition", "Wav2Vec2", "Audio emotion probability distribution"],
        ["Text Emotion Recognition", "Transformers classifier", "Text emotion probability distribution"],
        ["Multimodal Fusion", "Entropy-confidence weighted late fusion", "Reliability-weighted probability combination"],
        ["Dialogue Memory", "Streamlit session state", "Conversation emotion history"],
        ["Contextual Response", "Optional Qwen mLLM", "Context-aware text response"],
        ["Text-to-Speech", "Microsoft Edge TTS", "Multilingual speech generation"],
        ["Research Evaluation", "Scikit-learn", "Accuracy, precision, recall, F1 and ECE"]
    ], columns=["Module", "Technology", "Function"])

    st.dataframe(method_table, use_container_width=True, hide_index=True)

    st.markdown("### Important Research Limitations")
    st.warning(
        "Whisper supports multilingual speech recognition. "
        "However, the default text emotion model and Wav2Vec2 "
        "emotion checkpoint are English-oriented. Their predictions "
        "must not be presented as validated multilingual emotion "
        "recognition without language-specific evaluation."
    )

    st.markdown("""
**Research methodology considerations**

1. Whisper automatically detects speech language during transcription.
2. Audio is converted to mono 16 kHz before model inference.
3. Speech emotion recognition is evaluated independently of ASR output.
4. Text and audio probabilities are combined using entropy-confidence weights.
5. Dialogue emotion memory stores analyzed turns during the active session.
6. TTS uses language-specific voices and emotion-dependent rate/pitch settings.
7. The current fusion method is a transparent late-fusion baseline, not a trained neural gating network.
8. TTS prosody adjustments are approximations and should not be described as validated expressive-speech synthesis.

**Recommended research metrics**

- Accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Weighted F1-score
- Expected Calibration Error (ECE)
- Mean and median inference latency
- Word Error Rate (WER) and Character Error Rate (CER) for ASR evaluation

For a multilingual research paper, report emotion classification performance separately for each language and use language-appropriate labelled test datasets.
""")

st.markdown("---")
st.caption("Multilingual Multimodal Emotion-Aware AI | Research Prototype")


import streamlit as st
from transformers import pipeline
import librosa
import tempfile
import os
from streamlit_mic_recorder import mic_recorder
from datetime import datetime

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Emotion-Aware AI",
    page_icon="🎭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
    h1 {
        background: linear-gradient(90deg, #ff6b6b, #feca57, #48dbfb, #ff9ff3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        text-align: center;
    }
    .subtitle {
        text-align: center;
        color: #dfe6e9;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
    }
    .section-box {
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.8rem;
    }
    .stTextArea textarea {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1e1e2f !important;
        border-radius: 12px !important;
    }
    .stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #feca57) !important;
        color: #1e1e2f !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        width: 100%;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("<h1>🎭 Emotion-Aware AI</h1>", unsafe_allow_html=True)
st.markdown('<p class="subtitle">Text + Speech Emotion Recognition</p>', unsafe_allow_html=True)

# ====================== LOAD MODELS ======================
@st.cache_resource
def load_text_model():
    return pipeline(
        "text-classification",
        model="tabularisai/multilingual-emotion-classification",
        top_k=None
    )

@st.cache_resource
def load_speech_model():
    return pipeline(
        "audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )

with st.spinner("Loading models..."):
    text_classifier = load_text_model()
    speech_classifier = load_speech_model()

# ====================== TEXT SECTION ======================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("### 📝 Text Emotion Detection")
user_text = st.text_area("Enter your text:", height=110, label_visibility="collapsed",
                         placeholder="Type in any language...")

if st.button("Detect Text Emotion", key="text_btn"):
    if user_text.strip():
        with st.spinner("Analyzing..."):
            results = text_classifier(user_text)[0]
            results = sorted(results, key=lambda x: x['score'], reverse=True)
        st.success(f"**Dominant Emotion:** {results[0]['label']} ({results[0]['score']:.1%})")
        for r in results[:5]:
            st.progress(r['score'], text=f"{r['label']}: {r['score']:.1%}")
    else:
        st.warning("Please enter some text.")
st.markdown('</div>', unsafe_allow_html=True)

# ====================== SPEECH SECTION ======================
st.markdown('<div class="section-box">', unsafe_allow_html=True)
st.markdown("### 🎤 Speech Emotion Detection")

# --- Option 1: Record Audio ---
st.markdown("#### 1. Record Audio")
audio = mic_recorder(
    start_prompt="Start Recording",
    stop_prompt="Stop Recording",
    just_once=False,
    use_container_width=True,
    format="wav",
    key="recorder"
)

if audio:
    st.audio(audio['bytes'])

    # Let user choose filename
    default_name = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    file_name = st.text_input("Save as (filename):", value=default_name)

    if st.button("Save Recording as WAV"):
        save_path = file_name if file_name.endswith(".wav") else file_name + ".wav"
        with open(save_path, "wb") as f:
            f.write(audio['bytes'])
        st.success(f"Saved successfully as: `{save_path}`")
        st.session_state['last_saved_audio'] = save_path

# --- Option 2: Upload existing WAV ---
st.markdown("#### 2. Or Upload existing WAV file")
audio_file = st.file_uploader("Upload WAV file", type=["wav"], label_visibility="collapsed")

# --- Detect Emotion ---
st.markdown("#### 3. Detect Emotion")
if st.button("Detect Speech Emotion", key="speech_btn"):
    audio_path = None

    # Priority: newly recorded & saved > uploaded file
    if 'last_saved_audio' in st.session_state and os.path.exists(st.session_state['last_saved_audio']):
        audio_path = st.session_state['last_saved_audio']
    elif audio_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.getvalue())
            audio_path = tmp.name

    if audio_path:
        with st.spinner("Analyzing speech emotion..."):
            try:
                speech, sr = librosa.load(audio_path, sr=16000, mono=True)
                results = speech_classifier(speech, sampling_rate=16000)
                results = sorted(results, key=lambda x: x['score'], reverse=True)

                st.success(f"**Dominant Emotion:** {results[0]['label']} ({results[0]['score']:.1%})")
                for r in results:
                    st.progress(r['score'], text=f"{r['label']}: {r['score']:.1%}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please record & save audio, or upload a WAV file first.")

st.markdown('</div>', unsafe_allow_html=True)

st.caption("Emotion-Aware Multilingual Conversational AI")
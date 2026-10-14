from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError


class_names = np.load(Path(__file__).with_name("classes.npy"), allow_pickle=False)
display_names = {
    "kylian_mbappe": "Kylian Mbappé",
}
player_names = [
    display_names.get(str(name), str(name).replace("_", " ").title())
    for name in class_names
]

st.set_page_config(
    page_title="ScoutIQ | Player Recognition Lab",
    page_icon=":material/face_retouching_natural:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #07100b; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { max-width: 1420px; padding: 24px 42px 48px; }
    [data-testid="stAppDeployButton"], #MainMenu, footer { visibility: hidden; }
    .brand-lockup { display: flex; align-items: center; gap: 12px; min-height: 48px; }
    .brand-mark {
        display: grid; place-items: center; width: 44px; height: 44px; flex: 0 0 44px;
        color: #0a130b; background: #c1ff35; border-radius: 13px;
        box-shadow: 0 0 30px rgba(193, 255, 53, 0.12);
    }
    .brand-mark svg { width: 25px; height: 25px; }
    .brand-name { color: #f4f6f1; font-family: 'DM Sans', sans-serif; font-size: 19px; font-weight: 800; line-height: 1; }
    .brand-name span { color: #c1ff35; }
    .brand-descriptor { margin-top: 5px; color: #737d75; font-size: 10px; font-weight: 700; line-height: 1; }
    .model-status {
        display: inline-flex; align-items: center; justify-content: center; gap: 9px; float: right;
        padding: 10px 14px; color: #aab3ac; background: #101913; border: 1px solid #202c24;
        border-radius: 999px; font-size: 11px; font-weight: 800;
    }
    .status-light { width: 9px; height: 9px; border-radius: 50%; background: #c1ff35; box-shadow: 0 0 12px rgba(193, 255, 53, 0.65); }
    .eyebrow {
        display: inline-flex; align-items: center; gap: 8px; margin: 2px 0 26px; padding: 8px 13px;
        color: #c1ff35; background: rgba(193, 255, 53, 0.08); border: 1px solid rgba(193, 255, 53, 0.2);
        border-radius: 999px; font-size: 11px; font-weight: 800;
    }
    .eyebrow-dot { width: 7px; height: 7px; border-radius: 50%; background: #c1ff35; }
    .stMarkdown h1.hero-title {
        margin: 0; color: #f7f8f5; font-family: 'Archivo Black', sans-serif;
        font-size: 90px; font-weight: 900; line-height: 0.94;
    }
    .stMarkdown h1.hero-title > span > span { color: #c1ff35; }
    .hero-copy { max-width: 520px; margin: 28px 0 0; color: #929b94; font-size: 18px; font-weight: 500; line-height: 1.65; }
    .proof-row { display: flex; flex-wrap: wrap; gap: 24px; margin-top: 30px; color: #858f87; font-size: 11px; font-weight: 800; }
    .proof-row span { display: inline-flex; align-items: center; gap: 8px; }
    .proof-mark { color: #c1ff35; font-size: 15px; }
    .roster-label { margin-top: 30px; color: #c1ff35; font-size: 10px; font-weight: 800; }
    .roster-names { max-width: 510px; margin: 7px 0 0; color: #aab3ac; font-size: 13px; line-height: 1.8; }
    .st-key-match_panel { padding: 24px; background: #0e1913; border: 1px solid #202c24; border-radius: 24px; }
    .panel-heading { margin: 0 0 3px; color: #f0f3ee; font-size: 16px; font-weight: 800; }
    .panel-help { margin: 0 0 16px; color: #768178; font-size: 12px; }
    .st-key-match_panel [data-testid="stFileUploader"] section {
        min-height: 330px; align-content: center; background: #0a140e;
        border: 1px dashed #2a3930; border-radius: 20px;
    }
    .st-key-match_panel [data-testid="stFileUploader"] section:hover { border-color: #89b62d; background: #0d1911; }
    .st-key-match_panel [data-testid="stFileUploaderDropzoneInstructions"] { color: #dce2dc; }
    .st-key-match_panel [data-testid="stFileUploader"] button {
        color: #c1ff35; background: #162319; border: 1px solid #2a3b2b; border-radius: 9px;
    }
    .st-key-match_panel [data-testid="stImage"] img { max-height: 310px; object-fit: contain; border-radius: 12px; }
    .st-key-match_panel [data-testid="stButton"] button {
        min-height: 50px; color: #0a130b; background: #9bc52f; border: 0; border-radius: 13px;
        font-size: 13px; font-weight: 900; text-transform: uppercase;
    }
    .st-key-match_panel [data-testid="stButton"] button:hover { background: #c1ff35; border: 0; }
    .result-heading { margin: 21px 0 4px; color: #f4f6f1; font-family: 'Barlow Condensed', Impact, sans-serif; font-size: 36px; font-weight: 800; }
    .result-note { color: #929b94; font-size: 13px; }
    .result-kicker { margin-top: 20px; color: #c1ff35; font-size: 10px; font-weight: 800; }
    .footer-note { margin-top: 26px; color: #667168; font-size: 11px; line-height: 1.5; }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] { padding: 18px 20px 36px; }
        .stMarkdown h1.hero-title { font-size: 48px; }
        .hero-copy { margin-top: 18px; font-size: 15px; }
        .eyebrow { margin-bottom: 20px; }
        .proof-row { margin-top: 20px; gap: 15px; }
        .roster-label { margin-top: 22px; }
        .st-key-match_panel { padding: 18px; border-radius: 20px; }
        .st-key-match_panel [data-testid="stFileUploader"] section { min-height: 220px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

header_brand, header_status = st.columns([3, 1], vertical_alignment="center")
with header_brand:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
              <path d="M8 4H6a2 2 0 0 0-2 2v2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M4 16v2a2 2 0 0 0 2 2h2"/>
              <path d="M8.5 10v.2M15.5 10v.2M8 15c1.1 1.2 2.4 1.8 4 1.8s2.9-.6 4-1.8"/>
            </svg>
          </div>
          <div><div class="brand-name">SCOUT<span>IQ</span></div><div class="brand-descriptor">PLAYER RECOGNITION LAB</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_status:
    st.markdown('<div class="model-status"><span class="status-light"></span> MODEL ONLINE</div>', unsafe_allow_html=True)

st.space(72)
intro, matcher = st.columns([1, 1], gap="large", vertical_alignment="top")

with intro:
    st.markdown('<div class="eyebrow"><span class="eyebrow-dot"></span> AI-POWERED SCOUTING</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">WHO IS<br><span>THAT<br>PLAYER?</span></h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-copy">Upload a footballer\'s photo and let our vision model read the game. Fast, focused, and built for the beautiful game.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="proof-row"><span><b class="proof-mark">✦</b> {len(player_names)} PLAYER CLASSES</span><span><b class="proof-mark">◈</b> UPLOADS NOT SAVED</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="roster-label">MODEL ROSTER</div>', unsafe_allow_html=True)
    roster = ", ".join(escape(name) for name in player_names)
    st.markdown(f'<p class="roster-names">{roster}</p>', unsafe_allow_html=True)

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


def clear_prediction():
    st.session_state.prediction_result = None


with matcher:
    with st.container(key="match_panel"):
        st.markdown('<div class="panel-heading">Match a player</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-help">JPEG, PNG or WEBP up to 10 MB</div>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader(
            "Choose a player image",
            type=("jpg", "jpeg", "png", "webp"),
            max_upload_size=10,
            key="player_image",
            on_change=clear_prediction,
            label_visibility="collapsed",
        )

        if uploaded_image is not None:
            try:
                with Image.open(uploaded_image) as source_image:
                    image = source_image.convert("RGB")
                st.image(image, caption=uploaded_image.name, alt="Uploaded football player photo", width="stretch")
            except (UnidentifiedImageError, OSError):
                image = None
                st.error("This image could not be opened. Choose a valid JPEG, PNG, or WEBP file.")
        else:
            image = None

        identify_clicked = st.button(
            "Identify player",
            type="primary",
            icon=":material/arrow_forward:",
            width="stretch",
        )
        if identify_clicked:
            if image is None:
                st.warning("Upload a clear face photo first.")
            else:
                try:
                    with st.spinner("Reading the player photo..."):
                        from predict import predict_image

                        st.session_state.prediction_result = predict_image(image)
                except Exception:
                    st.session_state.prediction_result = None
                    st.error("The recognition model could not run. Check the project setup and try again.")

        result = st.session_state.prediction_result
        if result is not None:
            if result["status"] == "recognized":
                player_name = result["player"].replace("_", " ").title()
                st.markdown('<div class="result-kicker">PLAYER IDENTIFIED</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="result-heading">{player_name}</div>', unsafe_allow_html=True)
                st.metric("Classifier confidence", f"{result['confidence']:.1%}")
                with st.expander("Top predictions"):
                    for player, probability in result["top_predictions"]:
                        st.write(f"{player.replace('_', ' ').title()}: {probability:.1%}")
            elif result["status"] == "unknown":
                st.markdown('<div class="result-kicker">NO ROSTER MATCH</div>', unsafe_allow_html=True)
                st.markdown('<div class="result-heading">Unknown player</div>', unsafe_allow_html=True)
                st.markdown('<div class="result-note">This face is outside the model\'s known player classes.</div>', unsafe_allow_html=True)
            else:
                st.warning("No face detected. Try a clear, front-facing photo.")

st.markdown(
    '<div class="footer-note">The model recognizes seven trained player classes and can reject faces outside that roster. Uploaded images are processed in memory and are not saved by this app.</div>',
    unsafe_allow_html=True,
)
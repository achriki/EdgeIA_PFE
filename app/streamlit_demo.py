"""
streamlit_demo.py — Live demo app for EdgeIA PFE presentation
"""

import io
import time
 
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

PI_URL = "http://100.77.212.107:30080"
PREDICT_ENDPOINT = f"{PI_URL}/predict"
HEALTH_ENDPOINT = f"{PI_URL}/health"
 
st.set_page_config(page_title="EdgeIA - Détection de personnes", page_icon="🎯", layout="wide")

# Helpers
def check_pi_health():
    """Quick health check so the UI can show a clear status instead of a silent hang."""
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False
 
 
def call_predict(image_bytes: bytes):
    """
    Send image bytes to the Pi's /predict endpoint.
    Returns (response_json, latency_ms) or raises an exception the caller handles.
    """
    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
    start = time.time()
    response = requests.post(PREDICT_ENDPOINT, files=files, timeout=10)
    latency_ms = (time.time() - start) * 1000
    response.raise_for_status()
    return response.json(), latency_ms

def draw_detections(image: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes + confidence labels on a copy of the input image."""
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
 
    box_color = (0, 200, 83)       # green
    text_bg_color = (0, 200, 83)
    text_color = (255, 255, 255)
 
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
 
    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        conf = det["conf"]
 
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
 
        label = f"person {conf:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
 
        draw.rectangle([x1, max(0, y1 - text_h - 6), x1 + text_w + 8, y1], fill=text_bg_color)
        draw.text((x1 + 4, max(0, y1 - text_h - 4)), label, fill=text_color, font=font)
 
    return annotated
 
 
def run_inference_and_display(image_bytes: bytes, original_image: Image.Image):
    """Shared logic for both upload and webcam paths: call API, show results."""
    with st.spinner("Inférence en cours sur le Raspberry Pi..."):
        try:
            result, latency_ms = call_predict(image_bytes)
        except requests.exceptions.ConnectionError:
            st.error(
                f"Impossible de joindre le Raspberry Pi à {PI_URL}. "
                "Vérifiez que le pod est actif et que l'IP est correcte."
            )
            return
        except requests.exceptions.Timeout:
            st.error("Le Raspberry Pi n'a pas répondu à temps (timeout).")
            return
        except requests.exceptions.HTTPError as e:
            st.error(f"Erreur du serveur d'inférence : {e}")
            return
 
    detections = result.get("detections", [])
    n_persons = result.get("persons", 0)
 
    annotated = draw_detections(original_image, detections)
 
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image(annotated, caption="Détections", use_container_width=True)
    with col2:
        st.metric("Personnes détectées", n_persons)
        st.metric("Latence d'inférence", f"{latency_ms:.0f} ms")
        st.metric("Modèle", "YOLOv5n FP16")
        if detections:
            with st.expander("Détails des détections"):
                for i, det in enumerate(detections, 1):
                    st.write(f"**Personne {i}** — confiance {det['conf']:.2f}")

# UI

st.title("🎯 EdgeIA - Détection de personnes en temps réel")
st.caption("Démonstration live - inférence exécutée sur Raspberry Pi 3B (edge AI)")
 
# Health status banner
pi_alive = check_pi_health()
status_col1, status_col2 = st.columns([1, 4])
with status_col1:
    if pi_alive:
        st.success("🟢 Pi connecté")
    else:
        st.error("🔴 Pi inaccessible")
with status_col2:
    st.caption(f"Endpoint : `{PREDICT_ENDPOINT}`")
 
st.divider()
 
tab_upload, tab_webcam = st.tabs(["📁 Charger une image", "📷 Webcam"])
 
with tab_upload:
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes))
        run_inference_and_display(image_bytes, original_image)
 
with tab_webcam:
    st.caption("Prenez une photo directement depuis la webcam de cet ordinateur.")
    camera_image = st.camera_input("Capture")
    if camera_image is not None:
        image_bytes = camera_image.getvalue()
        original_image = Image.open(io.BytesIO(image_bytes))
        run_inference_and_display(image_bytes, original_image)
 
st.divider()
st.caption(
    "Pipeline : Webcam/Upload -> PC (Streamlit) -> réseau local -> "
    "Raspberry Pi 3B (Flask + TFLite YOLOv5n) -> résultat JSON -> affichage"
)
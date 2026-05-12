import os
import base64
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Fuerza Total — Asistente Virtual",
    page_icon="💪",
    layout="centered"
)

# ── Imagen de fondo ────────────────────────────────────────────────────────────
@st.cache_data
def load_bg_image():
    for fname in ["gym_bg.jpg", "gym_bg.jpeg", "gym_bg.png"]:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = "jpeg" if fname.endswith((".jpg", ".jpeg")) else "png"
            return f"data:image/{ext};base64,{data}"
    return None

bg_src = load_bg_image()

if bg_src:
    bg_css = f"""
    .stApp {{
        background-image:
            linear-gradient(rgba(0,0,0,0.78), rgba(0,0,0,0.78)),
            url('{bg_src}');
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
    }}
    """
else:
    bg_css = """
    .stApp {
        background-color: #0f0f0f;
        background-image:
            radial-gradient(circle at 50% 0%, rgba(255,107,0,0.06) 0%, transparent 55%),
            radial-gradient(circle, rgba(255,107,0,0.055) 1px, transparent 1px);
        background-size: cover, 28px 28px;
        background-attachment: fixed;
    }
    """

# Inyectar el fondo primero (separado porque tiene f-string)
st.markdown(f"<style>{bg_css}</style>", unsafe_allow_html=True)

# ── CSS estático ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.stStatusWidget { display: none !important; }
.block-container {
    padding-top: 2rem !important;
    max-width: 760px !important;
}

/* ── Header ── */
.gym-header {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #ff6b00 0%, #d94e00 100%);
    border-radius: 20px;
    padding: 36px 40px 28px;
    margin-bottom: 14px;
    box-shadow: 0 16px 48px rgba(255,107,0,0.35), 0 1px 0 rgba(255,255,255,0.12) inset;
}
.gym-header::before {
    content: '';
    position: absolute;
    top: -55px; right: -35px;
    width: 230px; height: 230px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
    pointer-events: none;
}
.gym-header::after {
    content: '';
    position: absolute;
    bottom: -75px; left: -15px;
    width: 190px; height: 190px;
    background: rgba(0,0,0,0.1);
    border-radius: 50%;
    pointer-events: none;
}
.gym-name {
    font-family: 'Bebas Neue', cursive;
    font-size: 3.5rem;
    letter-spacing: 5px;
    color: #fff;
    margin: 0;
    line-height: 1;
    position: relative;
    z-index: 1;
}
.gym-tagline {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 3.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.65);
    margin: 7px 0 18px;
    position: relative;
    z-index: 1;
}
.open-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,0.16);
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #fff;
    position: relative;
    z-index: 1;
}
.pulse {
    width: 7px;
    height: 7px;
    background: #00ff88;
    border-radius: 50%;
    flex-shrink: 0;
    animation: pulse-anim 1.8s infinite;
}
@keyframes pulse-anim {
    0%   { box-shadow: 0 0 0 0 rgba(0,255,136,0.55); }
    70%  { box-shadow: 0 0 0 7px rgba(0,255,136,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,255,136,0); }
}

/* ── Info bar ── */
.info-bar {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.info-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(0,0,0,0.5);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 10px;
    padding: 8px 16px;
    color: rgba(255,255,255,0.7);
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    transition: border-color 0.2s, color 0.2s;
}
.info-chip:hover {
    border-color: rgba(255,107,0,0.35);
    color: rgba(255,255,255,0.9);
}
.chip-icon { color: #ff6b00; flex-shrink: 0; }

/* ── Mensajes — glass morphism ── */
[data-testid="stChatMessage"] {
    background: rgba(0,0,0,0.52) !important;
    backdrop-filter: blur(14px) !important;
    -webkit-backdrop-filter: blur(14px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    margin-bottom: 8px !important;
    transition: border-color 0.25s;
}
[data-testid="stChatMessage"]:hover {
    border-color: rgba(255,107,0,0.22) !important;
}

/* Avatar más pequeño */
[data-testid="stChatMessageAvatar"] {
    width: 30px !important;
    height: 30px !important;
    min-width: 30px !important;
    align-self: flex-start !important;
    margin-top: 2px !important;
}
[data-testid="stChatMessageAvatar"] > div,
[data-testid="stChatMessageAvatar"] img {
    width: 30px !important;
    height: 30px !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    background: rgba(255,107,0,0.18) !important;
}

/* ── Botones de consulta rápida ── */
.quick-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(255,255,255,0.28);
    margin: 4px 0 10px;
}
.stButton > button {
    background: rgba(0,0,0,0.48) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,107,0,0.3) !important;
    border-radius: 20px !important;
    color: #ff8c00 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    height: auto !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: rgba(255,107,0,0.15) !important;
    border-color: rgba(255,107,0,0.55) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(255,107,0,0.22) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Chat input */
[data-testid="stChatInputContainer"] {
    background: rgba(0,0,0,0.52) !important;
    backdrop-filter: blur(14px) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* ── Footer ── */
.gym-footer {
    text-align: center;
    color: rgba(255,255,255,0.2);
    font-family: 'Inter', sans-serif;
    font-size: 0.67rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 32px;
    padding-top: 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gym-header">
    <div class="gym-name">Gimnasio Fuerza Total</div>
    <div class="gym-tagline">Asistente virtual</div>
    <span class="open-badge">
        <span class="pulse"></span>
        En línea ahora
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-bar">
    <span class="info-chip">
        <svg class="chip-icon" width="13" height="13" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
        </svg>
        Lun–Vie &nbsp;6:00 – 22:00
    </span>
    <span class="info-chip">
        <svg class="chip-icon" width="13" height="13" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/>
            <circle cx="12" cy="10" r="3"/>
        </svg>
        Av. 18 de Julio 1234
    </span>
    <span class="info-chip">
        <svg class="chip-icon" width="13" height="13" viewBox="0 0 24 24" fill="none"
             stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07
                     A19.5 19.5 0 0 1 4.68 13 19.79 19.79 0 0 1 1.65 4.45
                     A2 2 0 0 1 3.62 2h3a2 2 0 0 1 2 1.72
                     c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L7.91 9.91
                     a16 16 0 0 0 6.29 6.29l.91-.91a2 2 0 0 1 2.11-.45
                     c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/>
        </svg>
        099 123 456
    </span>
</div>
""", unsafe_allow_html=True)


# ── Backend ───────────────────────────────────────────────────────────────────
@st.cache_resource
def cargar_sistema():
    with open("datos_gimnasio.txt", "r", encoding="utf-8") as f:
        texto = f.read()
    fragmentos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = modelo.encode(fragmentos)
    return modelo, fragmentos, embeddings


def buscar_contexto(pregunta, modelo, fragmentos, embeddings):
    emb_pregunta = modelo.encode([pregunta])
    similitudes = np.dot(embeddings, emb_pregunta.T).flatten()
    top3 = np.argsort(similitudes)[-3:][::-1]
    return "\n".join([fragmentos[i] for i in top3])


def responder(pregunta, contexto):
    cliente = Groq(api_key=GROQ_API_KEY)
    respuesta = cliente.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": f"""Sos el asistente virtual del Gimnasio Fuerza Total.
Respondé en español, de forma amable, breve y directa. Usá emojis ocasionalmente.
Usá solo esta información para responder:

{contexto}

Si no sabés la respuesta, invitá a contactar por WhatsApp al 099 123 456."""
            },
            {"role": "user", "content": pregunta}
        ],
        temperature=0.2
    )
    return respuesta.choices[0].message.content


# ── Estado ────────────────────────────────────────────────────────────────────
if "mensajes" not in st.session_state:
    st.session_state.mensajes = [
        {
            "rol": "assistant",
            "texto": "Hola, soy el asistente de **Fuerza Total**. Puedo ayudarte con horarios, precios, clases y más. ¿En qué te puedo ayudar?"
        }
    ]

# Flag para mostrar botones de acceso rápido solo antes de la primera consulta
if "show_quick" not in st.session_state:
    st.session_state.show_quick = True

# ── Chat ──────────────────────────────────────────────────────────────────────
for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["texto"])

# Botones de consulta rápida
pregunta_rapida = None
if st.session_state.show_quick:
    st.markdown('<div class="quick-label">Consultas frecuentes</div>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Horarios"):
            st.session_state.show_quick = False
            pregunta_rapida = "¿Cuáles son los horarios?"
    with col2:
        if st.button("Precios"):
            st.session_state.show_quick = False
            pregunta_rapida = "¿Cuáles son los precios?"
    with col3:
        if st.button("Clases"):
            st.session_state.show_quick = False
            pregunta_rapida = "¿Qué clases tienen?"
    with col4:
        if st.button("Ubicación"):
            st.session_state.show_quick = False
            pregunta_rapida = "¿Dónde están ubicados?"

pregunta_texto = st.chat_input("Escribí tu consulta...")
pregunta = pregunta_rapida or pregunta_texto

if pregunta:
    st.session_state.show_quick = False
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})

    with st.spinner(""):
        modelo, fragmentos, embeddings = cargar_sistema()
        contexto = buscar_contexto(pregunta, modelo, fragmentos, embeddings)
        respuesta = responder(pregunta, contexto)

    st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta})
    st.rerun()

st.markdown("""
<div class="gym-footer">
    Gimnasio Fuerza Total &nbsp;·&nbsp; Asistente IA &nbsp;·&nbsp; 2025
</div>
""", unsafe_allow_html=True)

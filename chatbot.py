import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np

# Cargar API key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Fuerza Total - Asistente Virtual",
    page_icon="💪",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a2e 100%);
    }
    .header-container {
        background: linear-gradient(90deg, #ff6b00, #ff8c00);
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(255, 107, 0, 0.3);
    }
    .header-container h1 {
        color: white;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }
    .header-container p {
        color: rgba(255,255,255,0.85);
        margin: 6px 0 0 0;
        font-size: 0.95rem;
    }
    .badges {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255,107,0,0.12);
        border: 1px solid rgba(255,107,0,0.3);
        border-radius: 20px;
        padding: 6px 14px;
        color: #ff8c00;
        font-size: 0.82rem;
        font-weight: 500;
    }
    .stChatMessage {
        background: rgba(255,255,255,0.04) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        margin-bottom: 8px !important;
    }
    .footer {
        text-align: center;
        color: rgba(255,255,255,0.3);
        font-size: 0.75rem;
        margin-top: 24px;
        padding-top: 16px;
        border-top: 1px solid rgba(255,255,255,0.06);
    }
    #MainMenu, footer, header {visibility: hidden;}
    .stStatusWidget {display: none !important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-container">
    <h1>💪 Gimnasio Fuerza Total</h1>
    <p>Asistente virtual — Respondemos al instante</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="badges">
    <span class="badge">🕐 Lun-Vie 6:00 - 22:00</span>
    <span class="badge">📍 Av. 18 de Julio 1234</span>
    <span class="badge">📱 099 123 456</span>
</div>
""", unsafe_allow_html=True)

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
    # Similitud coseno manual
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

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []
    st.session_state.mensajes.append({
        "rol": "assistant",
        "texto": "¡Hola! 👋 Soy el asistente virtual de **Fuerza Total**. Puedo ayudarte con información sobre horarios, precios, clases y más. ¿En qué te puedo ayudar?"
    })

if len(st.session_state.mensajes) == 1:
    st.markdown("""
    <div class="badges">
        <span class="badge">🕐 Horarios</span>
        <span class="badge">💰 Precios</span>
        <span class="badge">🏋️ Clases</span>
        <span class="badge">📍 Ubicación</span>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.mensajes:
    with st.chat_message(msg["rol"]):
        st.markdown(msg["texto"])

pregunta = st.chat_input("Escribí tu consulta...")

if pregunta:
    with st.chat_message("user"):
        st.markdown(pregunta)
    st.session_state.mensajes.append({"rol": "user", "texto": pregunta})

    with st.spinner("Pensando..."):
        modelo, fragmentos, embeddings = cargar_sistema()
        contexto = buscar_contexto(pregunta, modelo, fragmentos, embeddings)
        respuesta = responder(pregunta, contexto)

    with st.chat_message("assistant"):
        st.markdown(respuesta)
    st.session_state.mensajes.append({"rol": "assistant", "texto": respuesta})

st.markdown("""
<div class="footer">
    Powered by IA · Gimnasio Fuerza Total © 2025
</div>
""", unsafe_allow_html=True)
import streamlit as st
import openai
import chromadb
import os

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS para Interface Customizada GoEvo ---
st.markdown("""
<style>
    /* 1. Esconde Header, Footer e Menus nativos */
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    
    /* 2. Remove badges e botão fullscreen (incluindo o Built with Streamlit do iframe) */
    div[class*="viewerBadge"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
    .st-emotion-cache-1cvow4s {display: none !important;} /* Oculta o rodapé teimoso */

    /* 3. Força fundo branco e ADICIONA A BORDA AZUL no chat inteiro */
    .stApp {
        background-color: #FFFFFF !important;
        border: 2px solid #0986D5 !important; /* Borda cor GoEvo */
        border-radius: 12px !important; /* Arredonda os cantos da tela */
    }
    [data-testid="stAppViewContainer"] {
        background-color: transparent !important;
    }

    /* 4. ZERA o preenchimento e ajusta container para a rolagem perfeita */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 140px !important; /* AUMENTADO: Garante que a última mensagem pare acima da caixa de input */
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* 5. Ajuste global de fontes para PRETO */
    html, body {
        font-size: 14px;
        color: #000000 !important;
    }

    /* 6. Balões de chat: Fundo claro e texto preto */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        background-color: #F0F2F6 !important; /* Cinza leve para as mensagens */
        color: #000000 !important;
        border-radius: 8px !important;
    }
    
    /* Garante que o texto dentro do chat seja preto */
    [data-testid="stChatMessageContent"] p, 
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        color: #000000 !important;
    }

    /* Ajusta a cor dos ícones de avatar */
    [data-testid="stChatMessage"] .st-emotion-cache-1p7n9v6 {
        background-color: #E0E0E0 !important;
    }

    /* 7. Remove padding extra do topo do chat */
    [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }

    /* --- 8. AJUSTES DA CAIXA DE TEXTO (Fundo Sólido, Placeholder Cinza) --- */
    
    /* RODAPÉ SÓLIDO: Fundo branco impede que o chat apareça rolando por trás da caixa */
    [data-testid="stBottom"] {
        background-color: #FFFFFF !important;
        z-index: 9999 !important;
    }
    [data-testid="stBottomBlock"] > div {
        background-color: #FFFFFF !important;
    }

    /* Pinta a moldura externa do input com a cor AZUL GOEVO */
    [data-testid="stChatInput"] > div {
        background-color: #0986D5 !important;
        border: 1px solid #0986D5 !important;
        border-radius: 10px !important; 
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }

    /* Fundo branco e texto preto na área onde o usuário digita */
    [data-testid="stChatInput"] textarea {
        background-color: #FFFFFF !important; 
        color: #000000 !important; 
        -webkit-text-fill-color: #000000 !important; /* Força o preto no Chrome */
        padding-left: 12px !important; 
        border-radius: 6px !important;
    }

    /* COR FORÇADA PARA O PLACEHOLDER ("Como posso te ajudar?") = CINZA CLARO */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #888888 !important; 
        -webkit-text-fill-color: #888888 !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] textarea::-webkit-input-placeholder {
        color: #888888 !important; 
        -webkit-text-fill-color: #888888 !important;
        opacity: 1 !important;
    }

    /* Cor da setinha de enviar (Branca para contrastar com a moldura azul) */
    [data-testid="stChatInput"] button {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }

   /* --- 9. COMPENSAÇÃO DO CORTE DO IFRAME E ALINHAMENTO --- */
    
    /* Levanta o rodapé branco para ele escapar do corte invisível do iframe */
    [data-testid="stBottom"] {
        bottom: 45px !important;
        padding-top: 10px !important; /* Dá um respiro visual acima da caixa azul */
        padding-bottom: 10px !important;
    }
    
</style>
""", unsafe_allow_html=True)

# --- 3. Configuração de APIs ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHRO

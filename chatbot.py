import streamlit as st
import openai
import chromadb

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo Assist", page_icon="🤖", layout="wide")

# --- 2. CSS Minimalista ---
st.markdown("""
<style>
    /* Importação de fonte moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    /* Reset Geral para Minimalismo */
    html, body, [data-testid="stAppViewContainer"], .main {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        background-color: #0E1117 !important; /* Fundo escuro limpo */
        color: #E0E0E0 !important;
    }

    /* Esconde elementos nativos do Streamlit */
    header, footer, [data-testid="stHeader"], [data-testid="stFooter"] {visibility: hidden; display: none !important;}
    div[class*="container_1upux"], div[class*="viewerBadge"], button[title="View fullscreen"] {display: none !important;}

    /* Ajuste do container para ocupar a tela toda sem margens */
    .block-container {
        padding: 1rem !important;
        max-width: 100% !important;
    }

    /* Balões de Chat Minimalistas */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        margin-bottom: 0.8rem !important;
    }

    /* Estilo das mensagens do Assistente */
    [data-testid="stChatMessageContent"] {
        font-size: 13px !important;
        line-height: 1.5 !important;
        font-weight: 400 !important;
        color: #D1D1D1 !important;
    }

    /* Estilo das mensagens do Usuário (Destaque sutil) */
    [data-testid="stChatMessage"][data-testid="user"] {
        background-color: #1A1C24 !important;
        padding: 10px !important;
        border-radius: 12px !important;
    }

    /* Caixa de Input Minimalista */
    [data-testid="stChatInput"] {
        border-radius: 10px !important;
        border: 1px solid #30363D !important;
        background-color: #0D1117 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Configuração de API ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("Erro de configuração nas Secrets.")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- 4. Lógica do Chat ---
@st.cache_resource
def carregar_colecoes():
    try:
        _client = chromadb.CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
        return _client.get_collection("colecao_funcionalidades"), _client.get_collection("colecao_parametros")
    except: return None, None

def rotear_pergunta(pergunta):
    res = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Categorize: SAUDACAO, FUNCIONALIDADE ou PARAMETRO. Pergunta: '{pergunta}'"}],
        max_tokens=5
    )
    return res.choices[0].message.content.strip().upper()

# (As funções de busca e síntese permanecem iguais à versão anterior para manter a funcionalidade)
# ... [Código de busca omitido para brevidade, mas deve ser mantido conforme sua versão anterior]

# --- 5. Interface ---
RES_SAUDACAO = "Olá! Eu sou o Evo, suporte inteligente da GoEvo. Como posso ajudar?"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pergunta := st.chat_input("Escreva sua dúvida aqui..."):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"): st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner(""):
            # Lógica de resposta EVO aqui...
            res_final = "Resposta processada pelo Evo." # Substituir pela sua lógica de síntese
            st.markdown(res_final)
    st.session_state.messages.append({"role": "assistant", "content": res_final})

import streamlit as st
import openai
import chromadb
import os

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS para Ajustar Tamanho da Logo e Cores ---
st.markdown("""
<style>
    /* Esconde Header e Footer */
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}

    /* ZERA o preenchimento para o chat ocupar a tela */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* --- AJUSTE DOS ÍCONES (AVATARES) --- */

    /* 1. Ícone do Usuário: Cinza Claro */
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #D3D3D3 !important;
    }

    /* 2. Ícone da IA (Evo): Remove fundo e ajuste de tamanho */
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: transparent !important;
        width: 40px !important; /* Aumenta levemente o container */
        height: 40px !important;
        border-radius: 0% !important; /* Remove o corte circular se preferir a logo quadrada */
    }

    /* Força a logo a ocupar 100% do espaço sem margens internas */
    div[data-testid="stChatMessageAvatarAssistant"] img {
        width: 32px !important; 
        height: 32px !important;
        object-fit: contain !important;
    }

    /* Remove o padding padrão do Streamlit que esmaga a imagem */
    div[data-testid="stChatMessageAvatarAssistant"] > div {
        padding: 0px !important;
    }

    /* Estilo do texto */
    [data-testid="stChatMessageContent"] p {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Configuração de APIs (Secrets) ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: Configure as chaves de API no Streamlit Secrets.")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- 4. Funções do Core ---
@st.cache_resource
def carregar_colecao():
    try:
        _client = chromadb.CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
        return _client.get_collection("colecao_funcionalidades")
    except: return None

def rotear_pergunta(pergunta):
    try:
        resp = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"Classifique: SAUDACAO, AGRADECIMENTO ou FUNCIONALIDADE: '{pergunta}'"}],
            temperature=0, max_tokens=10
        )
        return resp.choices[0].message.content.strip().upper()
    except: return "SAUDACAO"

def buscar_contexto_seguro(pergunta, colecao):
    if colecao is None: return "", None, ""
    try:
        emb = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small").data[0].embedding
        res = colecao.query(query_embeddings=[emb], n_results=1)
        if not res['metadatas'][0]: return "", None, ""
        meta = res['metadatas'][0][0]
        fragmentos = colecao.query(query_embeddings=[emb], where={"fonte": meta.get('fonte')}, n_results=15)
        contexto = "\n\n".join([f.get('texto_original', '') for f in fragmentos['metadatas'][0]])
        return contexto, meta.get('video_url'), meta.get('fonte')
    except: return "", None, ""

def gerar_resposta(pergunta, contexto, nome_feature):
    prompt = f"Você é o Evo... Responda sobre {nome_feature}. Contexto: {contexto}"
    try:
        resp = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": pergunta}], temperature=0)
        return resp.choices[0].message.content
    except: return "Erro ao processar."

# --- 5. Execução do Chat ---

LOGO_IA = "logo-goevo.png" # Certifique-se que o nome está exato como no GitHub
RES_SAUDACAO = "Olá! Eu sou o Evo, suporte da GoEvo. Como posso te ajudar hoje?"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

# Renderiza histórico
for msg in st.session_state.messages:
    avatar = LOGO_IA if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Entrada do usuário
if pergunta := st.chat_input("Como posso te ajudar?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar=LOGO_IA):
        with st.spinner("Escrevendo..."):
            intencao = rotear_pergunta(pergunta)
            if "AGRADECIMENTO" in intencao:
                res_final = "De nada! 😊"
            elif "SAUDACAO" in intencao:
                res_final = RES_SAUDACAO
            else:
                ctx, video, nome_f = buscar_contexto_seguro(pergunta, colecao_func := carregar_colecao())
                if ctx:
                    res_final = gerar_resposta(pergunta, ctx, nome_f)
                    if video: res_final += f"\n\n---\n**🎥 Vídeo:** [Abrir]({video})"
                else:
                    res_final = "Ainda não tenho essa informação na base."

            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

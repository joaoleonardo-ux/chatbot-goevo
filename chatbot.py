import streamlit as st
import openai
import chromadb
import os

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS ---
st.markdown("""
<style>
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-size: 14px;
        background-color: transparent !important;
    }

    /* Balões de chat compactos */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Ícone do Usuário: Cinza Claro */
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #D3D3D3 !important;
    }

    /* AJUSTE PARA O LOGO DA IA: Removemos o fundo azul para a logo aparecer limpa */
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: transparent !important;
    }
    
    /* Garante que a imagem do logo preencha o espaço corretamente */
    div[data-testid="stChatMessageAvatarAssistant"] img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }

    [data-testid="stChatMessageContent"] p {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Configuração de APIs ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: Configure as chaves de API no Streamlit Secrets.")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- 4. Funções do Core (Resumidas para o exemplo) ---
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

# ... (Funções buscar_contexto_seguro e gerar_resposta permanecem iguais) ...
def buscar_contexto_seguro(pergunta, colecao):
    if colecao is None: return "", None, ""
    try:
        emb = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small").data[0].embedding
        res = colecao.query(query_embeddings=[emb], n_results=1)
        if not res['metadatas'][0]: return "", None, ""
        meta = res['metadatas'][0][0]
        contexto = "\n\n".join([f.get('texto_original', '') for f in colecao.query(query_embeddings=[emb], where={"fonte": meta.get('fonte')}, n_results=15)['metadatas'][0]])
        return contexto, meta.get('video_url'), meta.get('fonte')
    except: return "", None, ""

def gerar_resposta(pergunta, contexto, nome_feature):
    prompt = f"Você é o Evo... Responda: {nome_feature} Contexto: {contexto}"
    try:
        resp = client_openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": prompt}, {"role": "user", "content": pergunta}], temperature=0)
        return resp.choices[0].message.content
    except: return "Erro ao processar."

# --- 5. Execução do Chat ---

# DEFINIMOS O CAMINHO DA LOGO AQUI
LOGO_IA = "logo-goevo.png"
RES_SAUDACAO = "Olá! Eu sou o Evo, suporte da GoEvo. Como posso te ajudar hoje?"
RES_AGRADECIMENTO = "De nada! 😊"
colecao_func = carregar_colecao()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

# Renderiza histórico (Usando o parâmetro avatar)
for msg in st.session_state.messages:
    # Se for assistente, usa a logo; se for user, usa o ícone padrão
    avatar = LOGO_IA if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Entrada do usuário
if pergunta := st.chat_input("Como posso te ajudar?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    # Resposta da IA
    with st.chat_message("assistant", avatar=LOGO_IA): # Aqui aplicamos a logo
        with st.spinner("Escrevendo..."):
            intencao = rotear_pergunta(pergunta)
            if "AGRADECIMENTO" in intencao: res_final = RES_AGRADECIMENTO
            elif "SAUDACAO" in intencao: res_final = RES_SAUDACAO
            else:
                ctx, video, nome_f = buscar_contexto_seguro(pergunta, colecao_func)
                if ctx:
                    res_final = gerar_resposta(pergunta, ctx, nome_f)
                    if video: res_final += f"\n\n---\n**🎥 Vídeo:** [Abrir]({video})"
                else: res_final = "Não encontrei essa funcionalidade."

            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

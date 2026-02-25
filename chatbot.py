import streamlit as st
import openai
import chromadb

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS (Ajustes de Cores e Visibilidade) ---
st.markdown("""
<style>
    /* Esconde elementos nativos */
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    
    /* ZERA o preenchimento superior */
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* FUNDO BRANCO EM TUDO */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stBottom"] {
        background-color: #FFFFFF !important;
    }

    /* AJUSTE DA CAIXA DE DIGITAÇÃO (Fundo Claro) */
    [data-testid="stBottom"] > div {
        background-color: #FFFFFF !important;
    }

    [data-testid="stChatInput"] {
        background-color: #F7F9FB !important; /* Cinza bem clarinho */
        border-radius: 10px !important;
        border: 1px solid #E0E0E0 !important;
    }

    /* FORÇAR COR DO TEXTO (Preto para ser visível no branco) */
    /* Isso resolve o problema do passo a passo sumindo */
    [data-testid="stChatMessageContent"] p, 
    [data-testid="stChatMessageContent"] li, 
    [data-testid="stChatMessageContent"] ol {
        color: #31333F !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    /* BALÕES DE CHAT */
    [data-testid="stChatMessage"] {
        padding: 0.8rem !important;
        margin-bottom: 0.5rem !important;
        border-radius: 12px;
        background-color: #F8F9FB !important;
        border: 1px solid #F0F2F6;
    }

    /* CORES DOS ÍCONES (AVATARES) */
    /* Usuário: Cinza | IA Evo: Azul GoEvo */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #808080 !important;
    }
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #004aad !important;
    }

    /* Centralização da Logo no Topo */
    .stImage {
        display: flex;
        justify-content: center;
        margin-top: -10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Logo da GoEvo (Nome ajustado para o seu arquivo no Git) ---
CAMINHO_LOGO = "logo-goevo.png" 

# Criamos colunas apenas para centralizar a imagem pequena no meio
col1, col2, col3 = st.columns([1, 0.2, 1])
with col2:
    try:
        st.image(CAMINHO_LOGO, width=80) # Logo bem pequena como solicitado
    except:
        st.write("") # Se der erro, não exibe nada

# --- 4. Configuração de APIs ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except:
    st.error("Erro nas chaves de API.")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- 5. Funções do Chatbot ---

@st.cache_resource
def carregar_colecao():
    try:
        _client = chromadb.CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)
        return _client.get_collection("colecao_funcionalidades")
    except: return None

def rotear_pergunta(pergunta):
    try:
        prompt = f"Classifique: SAUDACAO, AGRADECIMENTO ou FUNCIONALIDADE. Responda apenas uma palavra: '{pergunta}'"
        res = client_openai.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=15
        )
        intencao = res.choices[0].message.content.strip().upper()
        if "FUNCIONALIDADE" in intencao: return "FUNCIONALIDADE"
        if "AGRADECIMENTO" in intencao: return "AGRADECIMENTO"
        return "SAUDACAO"
    except: return "SAUDACAO"

def buscar_contexto(pergunta, colecao):
    if colecao is None: return "", None, ""
    try:
        emb = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small").data[0].embedding
        res = colecao.query(query_embeddings=[emb], n_results=1)
        if not res['metadatas'][0]: return "", None, ""
        meta = res['metadatas'][0][0]
        f_alvo, video = meta.get('fonte'), meta.get('video_url')
        res_comp = colecao.query(query_embeddings=[emb], where={"fonte": f_alvo}, n_results=15)
        ctx = "\n\n".join([f.get('texto_original', '') for f in res_comp['metadatas'][0] if f.get('texto_original')])
        return ctx, video, f_alvo
    except: return "", None, ""

def gerar_resposta(pergunta, contexto, nome_feature):
    prompt = f"Você é o Evo. Responda: 'Para realizar {nome_feature}, siga estes passos:' seguido de uma lista numerada técnica."
    try:
        res = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"CONTEXTO: {contexto}\nPERGUNTA: {pergunta}"}],
            temperature=0
        )
        return res.choices[0].message.content
    except: return "Erro ao processar."

# --- 6. Fluxo do Chat ---

RES_SAUDACAO = "Olá! Eu sou o Evo, suporte inteligente da GoEvo. Como posso ajudar você hoje?"
RES_AGRADECIMENTO = "De nada! Fico feliz em ajudar. Se precisar de algo mais, é só chamar! 😊"
colecao_func = carregar_colecao()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if pergunta := st.chat_input("Como posso te ajudar?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"): st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            intencao = rotear_pergunta(pergunta)
            if intencao == "AGRADECIMENTO":
                res_final = RES_AGRADECIMENTO
            elif intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                ctx, video, nome_f = buscar_contexto(pergunta, colecao_func)
                if ctx:
                    res_final = gerar_resposta(pergunta, ctx, nome_f)
                    if video: res_final += f"\n\n---\n**🎥 Tutorial:** [Clique aqui para assistir]({video})"
                else:
                    res_final = "Ainda não tenho o passo a passo para essa funcionalidade."
            
            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

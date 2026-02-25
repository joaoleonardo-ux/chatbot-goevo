import streamlit as st
import openai
import chromadb
import os

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS (Reseta Rodapé, Centraliza Logo e Ajusta Cores) ---
st.markdown("""
<style>
    /* Esconde elementos nativos */
    header, footer, [data-testid="stHeader"] {display: none !important;}
    
    /* 1. RESET DE FUNDO (Resolve o problema do escuro) */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stBottom"] {
        background-color: #FFFFFF !important;
    }

    /* 2. AJUSTE DO RODAPÉ (Diminui a sessão e muda a cor) */
    [data-testid="stBottom"] > div {
        background-color: #FFFFFF !important;
        padding-bottom: 15px !important;
        padding-top: 0px !important;
    }

    [data-testid="stChatInput"] {
        background-color: #F7F9FB !important;
        border-radius: 10px !important;
        border: 1px solid #E0E0E0 !important;
    }

    /* Placeholder ("Como posso te ajudar?") cinza claro */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #D1D5DB !important;
    }

    /* 3. VISIBILIDADE DO TEXTO (Resolve o passo a passo invisível) */
    [data-testid="stChatMessageContent"] p, 
    [data-testid="stChatMessageContent"] li, 
    [data-testid="stChatMessageContent"] ol {
        color: #31333F !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    /* 4. BALÕES DE CHAT COMPACTOS */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        border-radius: 12px;
        background-color: #F8F9FB !important;
        border: 1px solid #F0F2F6;
    }

    /* 5. CORES DOS ÍCONES (AVATARES) */
    [data-testid="stChatMessageAvatarUser"] { background-color: #808080 !important; }
    [data-testid="stChatMessageAvatarAssistant"] { background-color: #0986D5 !important; }

    /* 6. FORÇAR CENTRALIZAÇÃO DA LOGO NO CONTEINER */
    [data-testid="stImage"] img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    
    .block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Logo da GoEvo (Centralização via Colunas) ---
CAMINHO_LOGO = "logo-goevo.png"

# Criamos 3 colunas: a do meio é onde a logo fica "presa" centralizada
c1, c2, c3 = st.columns([1, 0.3, 1])
with c2:
    if os.path.exists(CAMINHO_LOGO):
        st.image(CAMINHO_LOGO, width=65) # Tamanho pequeno e centralizado
    else:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)

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

# --- 5. Funções do Chatbot (Roteamento e Busca) ---

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
    prompt = f"Você é o Evo. Responda: 'Para realizar {nome_feature}, siga estes passos:' seguido de lista numerada."
    try:
        res = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"CONTEXTO: {contexto}\nPERGUNTA: {pergunta}"}],
            temperature=0
        )
        return res.choices[0].message.content
    except: return "Erro ao processar."

# --- 6. Fluxo do Chat ---

RES_SAUDACAO = "Olá! Eu sou o Evo, suporte inteligente da GoEvo. Como posso ajudar?"
RES_AGRADECIMENTO = "De nada! Se precisar de algo mais, é só chamar! 😊"
colecao_func = carregar_colecao()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

if pergunta := st.chat_input("Como posso te ajudar?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"): st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("..."):
            intencao = rotear_pergunta(pergunta)
            if intencao == "AGRADECIMENTO":
                res_final = RES_AGRADECIMENTO
            elif intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                ctx, video, nome_f = buscar_contexto(pergunta, colecao_func)
                if ctx:
                    res_final = gerar_resposta(pergunta, ctx, nome_f)
                    if video: res_final += f"\n\n---\n**🎥 Tutorial:** [Clique aqui]({video})"
                else:
                    res_final = "Ainda não tenho esse passo a passo."
            
            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

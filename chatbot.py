import streamlit as st
import openai
import chromadb

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS Refinado ---
st.markdown("""
<style>
    /* Esconde Header e Footer nativos */
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

    /* FUNDO BRANCO DA PÁGINA */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FFFFFF !important;
    }

    /* CORREÇÃO DO RODAPÉ (Caixa de digitação) */
    /* Remove o fundo preto do container do rodapé */
    [data-testid="stBottom"], [data-testid="stBottom"] > div {
        background-color: #FFFFFF !important;
    }

    /* Estilização da caixa de input para cinza claro */
    [data-testid="stChatInput"] {
        background-color: #F7F9FB !important;
        border-radius: 10px !important;
        border: 1px solid #E0E0E0 !important;
    }

    /* Garante que o texto digitado pelo usuário seja visível (preto) */
    [data-testid="stChatInput"] textarea {
        color: #31333F !important;
    }

    /* FORÇAR COR DO TEXTO (Resolve o problema do texto invisível) */
    [data-testid="stChatMessageContent"] p, 
    [data-testid="stChatMessageContent"] li, 
    [data-testid="stChatMessageContent"] ol,
    [data-testid="stChatMessageContent"] ul {
        color: #31333F !important; /* Cinza escuro/preto */
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
    /* Usuário (Cinza) */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #808080 !important;
    }

    /* IA Evo (Azul GoEvo) */
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #004aad !important;
    }

    /* Centralização da Logo Pequena */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 5px 0 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Logo da GoEvo (Pequena) ---
URL_LOGO = "https://s3.amazonaws.com//beta-img.b2bstack.net/uploads/production/product/product_image/396/Marca-GOEVO.jpg"

st.markdown(f"""
    <div class="logo-container">
        <img src="{URL_LOGO}" width="80">
    </div>
""", unsafe_allow_html=True)

# --- 4. Configuração de APIs ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: Configure as chaves de API no Streamlit Secrets.")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- 5. Funções de IA ---

@st.cache_resource
def carregar_colecao():
    try:
        _client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY, 
            tenant=CHROMA_TENANT, 
            database=CHROMA_DATABASE
        )
        return _client.get_collection("colecao_funcionalidades")
    except Exception as e:
        st.error(f"Erro de conexão: {e}")
        return None

def rotear_pergunta(pergunta):
    try:
        prompt_roteador = f"Classifique: SAUDACAO, AGRADECIMENTO ou FUNCIONALIDADE. Responda apenas uma palavra. Pergunta: '{pergunta}'"
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_roteador}],
            temperature=0, 
            max_tokens=15
        )
        intencao = resposta.choices[0].message.content.strip().upper()
        if "FUNCIONALIDADE" in intencao: return "FUNCIONALIDADE"
        if "AGRADECIMENTO" in intencao: return "AGRADECIMENTO"
        return "SAUDACAO"
    except:
        return "SAUDACAO"

def buscar_contexto_seguro(pergunta, colecao):
    if colecao is None: return "", None, ""
    try:
        emb_response = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small")
        emb = emb_response.data[0].embedding
        res_topo = colecao.query(query_embeddings=[emb], n_results=1)
        if not res_topo['metadatas'][0]: return "", None, ""
        meta = res_topo['metadatas'][0][0]
        fonte_alvo = meta.get('fonte')
        video_url = meta.get('video_url')
        res_completos = colecao.query(query_embeddings=[emb], where={"fonte": fonte_alvo}, n_results=15)
        fragmentos = res_completos.get('metadatas', [[]])[0]
        contexto = "\n\n".join([f.get('texto_original', '') for f in fragmentos if f.get('texto_original')])
        return contexto, video_url, fonte_alvo
    except:
        return "", None, ""

def gerar_resposta_padronizada(pergunta, contexto, nome_feature):
    prompt_sistema = f"""Você é o Evo, assistente da GoEvo. 
    REGRAS RÍGIDAS:
    1. Responda: "Para realizar {nome_feature}, siga estes passos:"
    2. Liste os passos numerados conforme o contexto.
    3. Use tom profissional e direto.
    4. Se não souber, diga: "Não encontrei o procedimento exato na base." """
    try:
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema}, 
                {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPERGUNTA:\n{pergunta}"}
            ],
            temperature=0
        )
        return resposta.choices[0].message.content
    except:
        return "Erro ao processar."

# --- 6. Fluxo do Chat ---

RES_SAUDACAO = "Olá! Eu sou o Evo, suporte inteligente da GoEvo. Como posso ajudar você hoje?"
RES_AGRADECIMENTO = "De nada! Fico feliz em poder ajudar. Se precisar de algo mais, estou à disposição! 😊"
colecao_func = carregar_colecao()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if pergunta := st.chat_input("Como posso te ajudar?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Consultando base..."):
            intencao = rotear_pergunta(pergunta)
            if intencao == "AGRADECIMENTO":
                res_final = RES_AGRADECIMENTO
            elif intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                ctx, video, nome_f = buscar_contexto_seguro(pergunta, colecao_func)
                if ctx:
                    res_final = gerar_resposta_padronizada(pergunta, ctx, nome_f)
                    if video:
                        res_final += f"\n\n---\n**🎥 Tutorial em Vídeo:** [Clique aqui para assistir]({video})"
                else:
                    res_final = "Desculpe, ainda não tenho o passo a passo para essa funcionalidade."

            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

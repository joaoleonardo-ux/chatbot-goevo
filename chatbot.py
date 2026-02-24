import streamlit as st
import openai
import chromadb

# --- 1. Configuração da Página (Deve ser o primeiro comando) ---
st.set_page_config(page_title="GoEvo Assist", page_icon="🤖", layout="wide")

# --- 2. Injeção de CSS para Limpeza Total da Interface ---
st.markdown("""
<style>
    /* Esconde o Header, Footer e Menus nativos */
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    
    /* Remove a barra específica identificada (div._container_1upux_1) */
    div[class*="container_1upux"] {display: none !important;}
    div[class*="viewerBadge"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}

    /* Ajusta o container principal para não ter margens sobrando */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Ajuste global de fontes para visualização em janelas pequenas */
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 14px;
    }

    /* Estilização compacta dos balões de chat */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stChatMessageContent"] p {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        overflow-wrap: break-word;
    }

    /* Ajuste da altura do título para caber no popup */
    h1 {
        font-size: 1.3rem !important;
        margin-bottom: 0.2rem !important;
        padding-top: 0px !important;
    }
    
    .stCaption {
        font-size: 0.85rem !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Leo - Assistente Virtual")
st.caption("Suporte inteligente GoEvo")

# --- 3. Configuração das Chaves de API ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: Configure as chaves de API no arquivo .streamlit/secrets.toml")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- 4. Funções do Agente de IA ---
@st.cache_resource
def carregar_colecoes_chroma():
    try:
        _client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY, 
            tenant=CHROMA_TENANT, 
            database=CHROMA_DATABASE
        )
        colecao_funcionalidades = _client.get_collection("colecao_funcionalidades")
        colecao_parametros = _client.get_collection("colecao_parametros")
        return colecao_funcionalidades, colecao_parametros
    except Exception as e:
        st.error(f"Erro ao conectar com a base de conhecimento: {e}")
        return None, None

def rotear_pergunta(pergunta):
    prompt_roteador = f"""
    Classifique a pergunta do usuário em: SAUDACAO, FUNCIONALIDADE ou PARAMETRO.
    Responda APENAS com a palavra correspondente.
    Pergunta: "{pergunta}"
    """
    resposta = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_roteador}],
        temperature=0, max_tokens=10
    )
    intencao = resposta.choices[0].message.content.strip().upper()
    if "FUNCIONALIDADE" in intencao: return "FUNCIONALIDADE"
    if "PARAMETRO" in intencao: return "PARAMETRO"
    return "SAUDACAO"

def buscar_e_sintetizar_contexto(pergunta, colecao, n_results_inicial=10):
    if colecao is None: return "", None
    embedding_pergunta = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small").data[0].embedding
    resultados_iniciais = colecao.query(query_embeddings=[embedding_pergunta], n_results=n_results_inicial)
    metadados_iniciais = resultados_iniciais.get('metadatas', [[]])[0]
    
    if not metadados_iniciais: return "", None

    fontes_relevantes = list(set([doc['fonte'] for doc in metadados_iniciais]))
    resultados_filtrados = colecao.query(
        query_embeddings=[embedding_pergunta],
        where={"fonte": {"$in": fontes_relevantes}},
        n_results=50 
    )

    metadados_completos = resultados_filtrados.get('metadatas', [[]])[0]
    if not metadados_completos: return "", None

    contexto_texto = "\n\n---\n\n".join([doc['texto_original'] for doc in metadados_completos])
    video_url = metadados_iniciais[0].get('video_url')
    return contexto_texto, video_url

def gerar_resposta_sintetizada(pergunta, contexto, prompt_especialista):
    resposta = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_especialista},
            {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPERGUNTA:\n{pergunta}"}
        ],
        temperature=0.5
    )
    return resposta.choices[0].message.content

# --- 5. Prompts e Lógica do Chat ---
prompt_funcionalidades = "Você é o GoEvo Assist. Responda de forma direta e numerada usando o contexto. Não cite suporte externo."
prompt_parametros = "Você é o especialista técnico GoEvo. Explique o parâmetro de forma curta e objetiva."
RESPOSTA_SAUDACAO = "Olá! Eu sou o Leo, Assistente Virtual do GoEvo. Como posso ajudar?"

colecao_func, colecao_param = carregar_colecoes_chroma()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibe histórico
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "video" in message and message["video"]:
            st.video(message["video"])

# Entrada do Usuário
if pergunta_usuario := st.chat_input("Qual a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            intencao = rotear_pergunta(pergunta_usuario)
            video_para_mostrar = None
            
            if intencao == "SAUDACAO":
                resposta_final = RESPOSTA_SAUDACAO
            else:
                colecao = colecao_func if intencao == "FUNCIONALIDADE" else colecao_param
                prompt = prompt_funcionalidades if intencao == "FUNCIONALIDADE" else prompt_parametros
                contexto, video_para_mostrar = buscar_e_sintetizar_contexto(pergunta_usuario, colecao)
                
                if contexto:
                    resposta_final = gerar_resposta_sintetizada(pergunta_usuario, contexto, prompt)
                else:
                    resposta_final = "Não encontrei informações sobre isso na base. Pode reformular sua pergunta?"

            st.markdown(resposta_final)
            if video_para_mostrar:
                st.video(video_para_mostrar)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta_final, "video": video_para_mostrar})

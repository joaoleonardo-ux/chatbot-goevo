import streamlit as st
import openai
import chromadb

# --- Configuração da Página (Deve ser o primeiro comando Streamlit) ---
st.set_page_config(page_title="GoEvo Assist", page_icon="🤖", layout="wide")

# --- CÓDIGO PARA AJUSTE VISUAL (CSS) ---
st.markdown("""
<style>
    /* 1. Ajustes Gerais de Fonte e Redução de Espaços */
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 14px;
    }
    
    /* Remove a barra de menu superior e o preenchimento extra */
    header {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    
    /* Reduz drasticamente o padding do container principal */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* 2. Ajuste do Título e Legenda */
    h1 {
        font-size: 1.2rem !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    .stCaption {
        font-size: 0.8rem !important;
        margin-bottom: 10px !important;
    }

    /* 3. Estilização das Mensagens de Chat (Compactas) */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stChatMessageContent"] p {
        font-size: 0.9rem !important;
        line-height: 1.4 !important;
        overflow-wrap: break-word;
    }

    /* Ajuste da área de input (deixa mais discreta) */
    [data-testid="stChatInput"] {
        padding-bottom: 20px !important;
    }

    /* 4. Esconder o rodapé "Built with Streamlit" e o botão de Fullscreen */
    [data-testid="stFooter"] {display: none !important;}
    .viewerBadge_container__1QS1n {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Leo - Assistente Virtual")
st.caption("Suporte inteligente GoEvo")

# --- Configuração das Chaves de API ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: Configure seu arquivo .streamlit/secrets.toml")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Funções do Agente de IA ---
@st.cache_resource
def carregar_colecoes_chroma():
    try:
        _client = chromadb.CloudClient(
            api_key=st.secrets["CHROMA_API_KEY"], 
            tenant=st.secrets["CHROMA_TENANT"], 
            database=st.secrets["CHROMA_DATABASE"]
        )
        colecao_funcionalidades = _client.get_collection("colecao_funcionalidades")
        colecao_parametros = _client.get_collection("colecao_parametros")
        return colecao_funcionalidades, colecao_parametros
    except Exception as e:
        st.error(f"Erro na base: {e}")
        return None, None

def rotear_pergunta(pergunta):
    prompt_roteador = f"""
    Classifique a pergunta: SAUDACAO, FUNCIONALIDADE ou PARAMETRO.
    Pergunta: "{pergunta}"
    Responda apenas a palavra.
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

# --- Prompts ---
prompt_assistente_funcionalidades = "Você é o GoEvo Assist. Responda de forma direta e numerada com base no contexto. Não use saudações longas."
prompt_especialista_parametros = "Você é o especialista técnico GoEvo. Explique o parâmetro de forma curta e objetiva."

# --- Interface do Chat ---
RESPOSTA_SAUDACAO = "Olá! Sou o Leo. Como posso ajudar com o sistema hoje?"
colecao_func, colecao_param = carregar_colecoes_chroma()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "video" in message and message["video"]:
            st.video(message["video"])

if pergunta_usuario := st.chat_input("Qual a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Buscando..."):
            intencao = rotear_pergunta(pergunta_usuario)
            if intencao == "SAUDACAO":
                resposta_final = RESPOSTA_SAUDACAO
                video_para_mostrar = None
            else:
                colecao = colecao_func if intencao == "FUNCIONALIDADE" else colecao_param
                prompt = prompt_assistente_funcionalidades if intencao == "FUNCIONALIDADE" else prompt_especialista_parametros
                contexto, video_para_mostrar = buscar_e_sintetizar_contexto(pergunta_usuario, colecao)
                
                if contexto:
                    resposta_final = gerar_resposta_sintetizada(pergunta_usuario, contexto, prompt)
                else:
                    resposta_final = "Não encontrei essa informação. Pode reformular?"

            st.markdown(resposta_final)
            if video_para_mostrar:
                st.video(video_para_mostrar)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta_final, "video": video_para_mostrar})

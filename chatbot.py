import streamlit as st
import openai
import chromadb

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS para Interface Profissional ---
st.markdown("""
<style>
    /* Esconde Header, Footer e Menus nativos */
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    
    /* ZERA o preenchimento superior */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* FUNDO BRANCO DA PÁGINA */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stBottom"] {
        background-color: #FFFFFF !important;
    }

    /* AJUSTE DA CAIXA DE DIGITAÇÃO (Removendo o fundo preto) */
    [data-testid="stBottom"] > div {
        background-color: #FFFFFF !important;
    }

    /* COR DO TEXTO (Garante visibilidade no fundo branco) */
    [data-testid="stChatMessageContent"] p, 
    [data-testid="stChatMessageContent"] li, 
    [data-testid="stChatMessageContent"] ol {
        color: #31333F !important;
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    /* ESTILO DOS BALÕES DE CHAT */
    [data-testid="stChatMessage"] {
        padding: 0.8rem !important;
        margin-bottom: 0.5rem !important;
        border-radius: 12px;
        background-color: #F8F9FB !important; /* Fundo suave para o Evo */
        border: 1px solid #F0F2F6;
    }

    /* AJUSTE DOS ÍCONES (AVATARES) */
    /* Ícone do Usuário (Cinza) */
    [data-testid="stChatMessageAvatarUser"] {
        background-color: #808080 !important;
    }

    /* Ícone da IA (Azul GoEvo) */
    [data-testid="stChatMessageAvatarAssistant"] {
        background-color: #004aad !important;
    }

    /* Caixa de entrada estilizada */
    [data-testid="stChatInput"] {
        border-radius: 10px !important;
        border: 1px solid #E0E0E0 !important;
    }

    /* Centralização da Logo */
    .logo-container {
        display: flex;
        justify-content: center;
        padding-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Logo da GoEvo no Topo ---
# Substitua a URL abaixo pelo link da sua imagem (.png ou .jpg)
URL_LOGO = "https://sua-url-da-logo-aqui.png" 

with st.container():
    st.markdown(f'<div class="logo-container">', unsafe_allow_html=True)
    # Se você preferir usar um arquivo local, use st.image("caminho/para/logo.png")
    # st.image(URL_LOGO, width=150) 
    st.markdown('</div>', unsafe_allow_html=True)

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

# --- 5. Funções do Core ---

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
        st.error(f"Erro ao conectar com a base: {e}")
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
        meta_principal = res_topo['metadatas'][0][0]
        fonte_alvo = meta_principal.get('fonte')
        video_url = meta_principal.get('video_url')
        res_completos = colecao.query(query_embeddings=[emb], where={"fonte": fonte_alvo}, n_results=15)
        fragmentos = res_completos.get('metadatas', [[]])[0]
        contexto = "\n\n".join([f.get('texto_original', '') for f in fragmentos if f.get('texto_original')])
        return contexto, video_url, fonte_alvo
    except Exception as e:
        return "", None, ""

def gerar_resposta(pergunta, contexto, nome_feature):
    prompt_sistema = f"""Você é o Evo, o assistente técnico da GoEvo. 
    Sua missão é fornecer instruções idênticas e padronizadas.
    REGRAS:
    1. Comece com: "Para realizar {nome_feature}, siga estes passos:"
    2. Use listas numeradas.
    3. Seja direto e técnico. 
    4. Responda apenas com base no contexto. Se não souber, diga que não encontrou."""

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

# --- 6. Execução do Chat ---

RES_SAUDACAO = "Olá! Eu sou o Evo, suporte da GoEvo. Como posso te ajudar hoje?"
RES_AGRADECIMENTO = "De nada! Fico feliz em ajudar. Se precisar de algo mais, é só chamar! 😊"
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
        with st.spinner("Escrevendo..."):
            intencao = rotear_pergunta(pergunta)
            if intencao == "AGRADECIMENTO":
                res_final = RES_AGRADECIMENTO
            elif intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                ctx, video, nome_f = buscar_contexto_seguro(pergunta, colecao_func)
                if ctx:
                    res_final = gerar_resposta(pergunta, ctx, nome_f)
                    if video:
                        res_final += f"\n\n---\n\n**🎥 Vídeo explicativo:**\n[Clique aqui para abrir o vídeo]({video})"
                else:
                    res_final = "Ainda não encontrei esse procedimento. Pode detalhar melhor?"

            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

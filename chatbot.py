import streamlit as st
import openai
import chromadb
import os

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS para Interface Totalmente Limpa, Cores de Ícones e Barra de Digitação ---
st.markdown("""
<style>
    /* Esconde Header, Footer e Menus nativos */
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    
    /* Remove a barra de rodapé e o badge "Built with Streamlit" */
    div[class*="container_1upux"] {display: none !important;}
    div[class*="viewerBadge"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}

    /* ZERA o preenchimento superior para o chat começar do topo */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Ajuste global de fontes */
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 14px;
        background-color: transparent !important;
    }

    /* Balões de chat compactos */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stChatMessageContent"] p {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        overflow-wrap: break-word;
    }

    /* --- AJUSTE DE CORES DOS ÍCONES (AVATARES) --- */
    
    /* Ícone do Usuário: Cinza Claro */
    div[data-testid="stChatMessageAvatarUser"] {
        background-color: #D3D3D3 !important;
    }

    /* Ícone da IA (Assistant): Azul GoEvo */
    div[data-testid="stChatMessageAvatarAssistant"] {
        background-color: #0882c8 !important;
    }

    /* --- AJUSTE DE COR DA BARRA DE DIGITAÇÃO E BOTÃO DE ENVIO --- */
    
    /* Cor da borda da barra de chat ao focar e em hover */
    div[data-testid="stChatInput"] input:focus {
        border-color: #0882c8 !important;
        box-shadow: 0 0 0 0.2rem rgba(8, 130, 200, 0.25) !important; /* Adiciona um brilho suave azul ao redor */
    }

    div[data-testid="stChatInput"]:has(input:hover) {
        border-color: #0882c8 !important;
    }

    /* Cor da seta do botão de envio */
    button[data-testid="stChatInputButton"] {
        color: #0882c8 !important;
    }

    /* Cor de fundo do botão de envio em hover (opcional, para feedback tátil) */
    button[data-testid="stChatInputButton"]:hover {
        background-color: rgba(8, 130, 200, 0.1) !important;
    }

    /* Remove padding extra do topo do chat */
    [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0px !important;
        padding-top: 0px !important;
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

# --- 4. Funções do Core do Chatbot ---
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
        st.error(f"Erro ao conectar com a base ChromaDB: {e}")
        return None

def rotear_pergunta(pergunta):
    """Classifica com temperatura 0 para identificar SAUDACAO, AGRADECIMENTO ou FUNCIONALIDADE."""
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
        if not res_topo['metadatas'][0]:
            return "", None, ""

        meta_principal = res_topo['metadatas'][0][0]
        fonte_alvo = meta_principal.get('fonte')
        video_url = meta_principal.get('video_url')

        res_completos = colecao.query(
            query_embeddings=[emb], 
            where={"fonte": fonte_alvo}, 
            n_results=15
        )
        
        fragmentos = res_completos.get('metadatas', [[]])[0]
        contexto = "\n\n".join([f.get('texto_original', '') for f in fragmentos if f.get('texto_original')])
        
        return contexto, video_url, fonte_alvo
    except Exception as e:
        st.error(f"Erro na recuperação: {e}")
        return "", None, ""

def gerar_resposta(pergunta, contexto, nome_feature):
    """Sintetiza a resposta com temperatura 0 e regras rígidas de formatação."""
    prompt_sistema = f"""Você é o Evo, o assistente técnico da GoEvo. 
    Sua missão é fornecer instruções idênticas e padronizadas.
    
    REGRAS DE OURO:
    1. Comece sempre com: "Para realizar {nome_feature}, siga estes passos:"
    2. Use estritamente listas numeradas para as ações.
    3. Seja direto. Não peça informações adicionais se o contexto já permitir responder.
    4. Não faça sugestões ou comentários fora do contexto fornecido.
    5. Mantenha tom profissional e técnico.
    6. Se o contexto não permitir responder, diga: "Não encontrei o procedimento exato na base de conhecimento." """

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
        return "Desculpe, tive um problema ao processar sua resposta. Pode tentar novamente?"

# --- 5. Execução do Chat ---

RES_SAUDACAO = "Olá! Eu sou o Evo, suporte da GoEvo. Como posso te ajudar com as funcionalidades do sistema hoje?"
RES_AGRADECIMENTO = "De nada! Fico feliz em ajudar. Se tiver mais alguma dúvida sobre as funcionalidades, é só chamar! 😊"
colecao_func = carregar_colecao()

# Inicializa histórico
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

# Renderiza histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada do usuário
if pergunta := st.chat_input("Como posso te ajudar?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Escrevendo..."):
            intencao = rotear_pergunta(pergunta)
            
            # Lógica de resposta baseada na intenção
            if intencao == "AGRADECIMENTO":
                res_final = RES_AGRADECIMENTO
            elif intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                ctx, video, nome_f = buscar_contexto_seguro(pergunta, colecao_func)
                if ctx:
                    res_final = gerar_resposta(pergunta, ctx, nome_f)
                    if video:
                        res_final += f"\n\n---\n\n**🎥 Vídeo explicativo:**\nAssista ao passo a passo detalhado: [Clique aqui para abrir o vídeo]({video})"
                else:
                    res_final = "Ainda não encontrei um passo a passo para essa funcionalidade. Pode detalhar melhor sua dúvida?"

            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

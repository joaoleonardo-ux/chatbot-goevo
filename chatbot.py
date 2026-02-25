import streamlit as st
import openai
import chromadb

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo Assist", page_icon="🤖", layout="wide")

# --- 2. Injeção de CSS para Interface Totalmente Limpa ---
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

    /* Remove padding extra do topo do chat */
    [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# REMOVIDOS: st.title e st.caption para limpar o topo conforme solicitado

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
        st.error(f"Erro ao conectar com a base: {e}")
        return None, None

def rotear_pergunta(pergunta):
    prompt_roteador = f"Classifique a pergunta: SAUDACAO, FUNCIONALIDADE ou PARAMETRO. Responda apenas a palavra. Pergunta: '{pergunta}'"
    try:
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_roteador}],
            temperature=0, max_tokens=10
        )
        intencao = resposta.choices[0].message.content.strip().upper()
        if "FUNCIONALIDADE" in intencao: return "FUNCIONALIDADE"
        if "PARAMETRO" in intencao: return "PARAMETRO"
        return "SAUDACAO"
    except Exception as e:
        return "SAUDACAO" # Fallback em caso de erro na API

def buscar_e_sintetizar_contexto(pergunta, colecao, n_results_inicial=10):
    if colecao is None: return "", None
    try:
        emb_response = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small")
        emb = emb_response.data[0].embedding
        
        res_iniciais = colecao.query(query_embeddings=[emb], n_results=n_results_inicial)
        meta_iniciais = res_iniciais.get('metadatas', [[]])[0]
        
        if not meta_iniciais: return "", None

        # Garante que a chave 'fonte' existe antes de acessar
        fontes = list(set([doc.get('fonte') for doc in meta_iniciais if doc.get('fonte')]))
        
        res_filtrados = colecao.query(query_embeddings=[emb], where={"fonte": {"$in": fontes}}, n_results=50)
        meta_completos = res_filtrados.get('metadatas', [[]])[0]
        
        # Garante que a chave 'texto_original' existe antes de acessar
        contexto = "\n\n---\n\n".join([doc.get('texto_original', '') for doc in meta_completos if doc.get('texto_original')])
        
        # Tenta pegar o vídeo do primeiro resultado, se existir
        video = meta_iniciais[0].get('video_url') if meta_iniciais else None
        return contexto, video
    except Exception as e:
        st.error(f"Erro na busca: {e}")
        return "", None

def gerar_resposta_sintetizada(pergunta, contexto, prompt):
    try:
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPERGUNTA:\n{pergunta}"}],
            temperature=0.5
        )
        return resposta.choices[0].message.content
    except Exception as e:
        return "Erro ao gerar resposta."

# --- 5. Lógica do Chat ---
p_func = "Você é o Evo. Responda de forma direta e numerada usando o contexto."
p_param = "Você é o especialista técnico Evo. Explique o parâmetro de forma curta."
RES_SAUDACAO = "Olá! Eu sou o Evo, suporte inteligente da GoEvo. Como posso ajudar?"

colecao_func, colecao_param = carregar_colecoes_chroma()

# Inicializa o chat já com a mensagem de saudação
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": RES_SAUDACAO}
    ]

# Exibe histórico de mensagens (sem player de vídeo)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # A linha 'if msg["role"] == "assistant" and "video" in msg and msg["video"]: st.video(msg["video"])' foi removida

# Processa a entrada do usuário
if pergunta := st.chat_input("Qual a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            intencao = rotear_pergunta(pergunta)
            video_mostrar = None
            res_final = ""
            
            if intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                col = colecao_func if intencao == "FUNCIONALIDADE" else colecao_param
                p = p_func if intencao == "FUNCIONALIDADE" else p_param
                
                if col:
                    ctx, video_mostrar = buscar_e_sintetizar_contexto(pergunta, col)
                    if ctx:
                        res_final = gerar_resposta_sintetizada(pergunta, ctx, p)
                    else:
                        res_final = "Não encontrei essa informação na minha base de conhecimento."
                else:
                     res_final = "Desculpe, a base de conhecimento não está disponível."

            # --- AJUSTE: Adiciona o link do vídeo ao final da resposta, se houver ---
            if video_mostrar:
                res_final += f"\n\n---\n\n**🎥 Vídeo Explicativo:**\n\nEssa funcionalidade possui um vídeo explicativo de como funciona. [Clique aqui para visualizar]({video_mostrar})."
            # ---------------------------------------------------------------------

            st.markdown(res_final)
            # A linha 'if video_mostrar: st.video(video_mostrar)' foi removida
    
    # Salva a resposta completa (com o link) no histórico.
    st.session_state.messages.append({"role": "assistant", "content": res_final})

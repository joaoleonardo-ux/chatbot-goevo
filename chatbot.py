import streamlit as st
import openai
import chromadb
import os
from collections import Counter # <-- IMPORTAÇÃO NOVA NECESSÁRIA PARA O AJUSTE

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
# Tenta obter dos segredos do Streamlit, se não, tenta das variáveis de ambiente
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    # Fallback para variáveis de ambiente
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
    CHROMA_TENANT = os.environ.get("CHROMA_TENANT")
    CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE")

# Verifica se as chaves foram carregadas
if not OPENAI_API_KEY:
    st.error("ERRO: Chave de API da OpenAI não configurada.")
    st.stop()
if not CHROMA_API_KEY or not CHROMA_TENANT or not CHROMA_DATABASE:
    st.error("ERRO: Chaves de API do ChromaDB não configuradas.")
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
        # Verifica se as coleções existem antes de tentar obter
        colecoes_existentes = [col.name for col in _client.list_collections()]
        
        colecao_funcionalidades = None
        if "colecao_funcionalidades" in colecoes_existentes:
            colecao_funcionalidades = _client.get_collection("colecao_funcionalidades")
        else:
            st.warning("Aviso: Coleção 'colecao_funcionalidades' não encontrada no banco de dados.")

        colecao_parametros = None
        if "colecao_parametros" in colecoes_existentes:
            colecao_parametros = _client.get_collection("colecao_parametros")
        else:
            st.warning("Aviso: Coleção 'colecao_parametros' não encontrada no banco de dados.")
            
        return colecao_funcionalidades, colecao_parametros
    except Exception as e:
        st.error(f"Erro ao conectar com a base de dados Chroma: {e}")
        return None, None

def rotear_pergunta(pergunta):
    prompt_roteador = f"""Classifique a pergunta do usuário em uma das seguintes categorias:
- FUNCIONALIDADE: Se a pergunta for sobre como usar, configurar ou entender um recurso ou processo do sistema.
- PARAMETRO: Se a pergunta for sobre o significado ou propósito de um campo, opção ou configuração específica.
- SAUDACAO: Se a pergunta for uma saudação, despedida ou conversa fiada.

Responda APENAS com uma das palavras: FUNCIONALIDADE, PARAMETRO ou SAUDACAO.

Pergunta: '{pergunta}'"""
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
        st.error(f"Erro ao rotear pergunta com OpenAI: {e}")
        return "SAUDACAO" # Fallback seguro

def buscar_e_sintetizar_contexto(pergunta, colecao, n_results_inicial=10):
    if colecao is None:
        st.warning("Tentativa de busca em uma coleção inexistente.")
        return "", None
    try:
        emb_response = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small")
        emb = emb_response.data[0].embedding
        
        # Busca os N resultados mais similares (ex: top 10)
        res_iniciais = colecao.query(query_embeddings=[emb], n_results=n_results_inicial)
        meta_iniciais = res_iniciais.get('metadatas', [[]])[0]
        
        if not meta_iniciais: return "", None

        # --- INÍCIO DO AJUSTE: Lógica de Seleção do Vídeo mais Frequente ---
        video = None
        # Extrai URLs de vídeo válidas (não nulas e não vazias) dos resultados iniciais
        videos_encontrados = [m.get('video_url') for m in meta_iniciais if m.get('video_url')]

        if videos_encontrados:
            # Usa Counter para encontrar o vídeo mais comum na lista
            # most_common(1) retorna uma lista com uma tupla: [(video_url, contagem)]
            video_mais_comum = Counter(videos_encontrados).most_common(1)
            if video_mais_comum:
                # Pega a URL do primeiro elemento da tupla
                video = video_mais_comum[0][0]
        # --- FIM DO AJUSTE ---

        # Garante que a chave 'fonte' existe para evitar KeyErrors na filtragem
        fontes = list(set([doc.get('fonte') for doc in meta_iniciais if doc.get('fonte')]))
        
        # Expande a busca para pegar todo o contexto das fontes identificadas
        res_filtrados = colecao.query(query_embeddings=[emb], where={"fonte": {"$in": fontes}}, n_results=50)
        meta_completos = res_filtrados.get('metadatas', [[]])[0]
        
        # Monta o contexto final
        contexto = "\n\n---\n\n".join([doc.get('texto_original', '') for doc in meta_completos if doc.get('texto_original')])
        
        return contexto, video
    except Exception as e:
        st.error(f"Erro durante busca e síntese de contexto: {e}")
        return "", None

def gerar_resposta_sintetizada(pergunta, contexto, prompt_sistema):
    prompt_usuario = f"""Use o seguinte contexto para responder à pergunta do usuário.
Se a resposta não puder ser encontrada no contexto, diga que você não tem essa informação.
Seja claro, conciso e direto.

CONTEXTO:
{contexto}

PERGUNTA:
{pergunta}

RESPOSTA:"""
    try:
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": prompt_usuario}
            ],
            temperature=0.3
        )
        return resposta.choices[0].message.content
    except Exception as e:
        st.error(f"Erro ao gerar resposta sintetizada com OpenAI: {e}")
        return "Desculpe, ocorreu um erro ao gerar a resposta."

# --- 5. Lógica do Chat ---
# Prompts de Sistema Mais Robustos
P_FUNC_SYSTEM = """Você é o Evo, um assistente virtual especializado no sistema GoEvo.
Sua função é ajudar usuários com dúvidas sobre funcionalidades e processos do sistema.
- Suas respostas devem ser baseadas **exclusivamente** no contexto fornecido.
- Seja direto, claro e objetivo.
- Use listas numeradas ou bullet points para instruções passo a passo.
- Se o contexto não contiver a informação, admita que não sabe. Não invente."""

P_PARAM_SYSTEM = """Você é o Evo, um especialista técnico do sistema GoEvo.
Sua função é explicar o significado e o propósito de parâmetros, campos e configurações específicas do sistema.
- Suas explicações devem ser curtas, precisas e fáceis de entender.
- Baseie-se **exclusivamente** no contexto técnico fornecido.
- Se o contexto não tiver a definição, diga que não encontrou a informação."""

RES_SAUDACAO = "Olá! Eu sou o Evo, seu assistente virtual para o sistema GoEvo. Estou aqui para ajudar com dúvidas sobre funcionalidades e parâmetros. Como posso ser útil hoje?"

colecao_func, colecao_param = carregar_colecoes_chroma()

# Inicializa o chat já com a mensagem de saudação
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": RES_SAUDACAO}
    ]

# Exibe histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "video" in msg and msg["video"]:
            st.video(msg["video"])

# Processa a entrada do usuário
if pergunta := st.chat_input("Qual a sua dúvida sobre o GoEvo?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        with st.spinner("Processando sua pergunta..."):
            intencao = rotear_pergunta(pergunta)
            video_mostrar = None
            res_final = ""
            
            if intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                # Seleciona a coleção e o prompt com base na intenção
                if intencao == "FUNCIONALIDADE":
                    col = colecao_func
                    prompt_sis = P_FUNC_SYSTEM
                else: # PARAMETRO
                    col = colecao_param
                    prompt_sis = P_PARAM_SYSTEM
                
                if col:
                    ctx, video_mostrar = buscar_e_sintetizar_contexto(pergunta, col)
                    if ctx:
                        res_final = gerar_resposta_sintetizada(pergunta, ctx, prompt_sis)
                    else:
                        res_final = "Desculpe, não encontrei informações relevantes sobre isso na minha base de conhecimento."
                else:
                     res_final = "Desculpe, a base de conhecimento necessária não está disponível no momento."

            st.markdown(res_final)
            if video_mostrar:
                st.video(video_mostrar)
    
    st.session_state.messages.append({"role": "assistant", "content": res_final, "video": video_mostrar})

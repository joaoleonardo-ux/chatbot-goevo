import streamlit as st
import openai
import chromadb
import os
# A importação de Counter não é mais necessária nesta abordagem, mas pode deixar
from collections import Counter 

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo Assist", page_icon="🤖", layout="wide")

# --- 2. Injeção de CSS (Mantido igual) ---
st.markdown("""
<style>
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    div[class*="container_1upux"] {display: none !important;}
    div[class*="viewerBadge"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
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
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    [data-testid="stChatMessageContent"] p {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        overflow-wrap: break-word;
    }
    [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Configuração das Chaves de API (Mantido igual) ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    CHROMA_API_KEY = os.environ.get("CHROMA_API_KEY")
    CHROMA_TENANT = os.environ.get("CHROMA_TENANT")
    CHROMA_DATABASE = os.environ.get("CHROMA_DATABASE")

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
    prompt_roteador = f"""Classifique a pergunta do usuário. Responda APENAS: FUNCIONALIDADE, PARAMETRO ou SAUDACAO. Pergunta: '{pergunta}'"""
    try:
        resposta = client_openai.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt_roteador}], temperature=0, max_tokens=10
        )
        intencao = resposta.choices[0].message.content.strip().upper()
        if "FUNCIONALIDADE" in intencao: return "FUNCIONALIDADE"
        if "PARAMETRO" in intencao: return "PARAMETRO"
        return "SAUDACAO"
    except Exception as e:
        return "SAUDACAO"

# --- FUNÇÃO DE BUSCA REFORMULADA (A SOLUÇÃO) ---
def buscar_e_sintetizar_contexto(pergunta, colecao, n_results_inicial=10):
    if colecao is None:
        st.warning("Tentativa de busca em uma coleção inexistente.")
        return "", None
    try:
        # 1. Gera o embedding da pergunta
        emb_response = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small")
        emb = emb_response.data[0].embedding
        
        # --- NOVO: Pré-filtragem baseada em Palavras-Chave Críticas ---
        pergunta_lower = pergunta.lower()
        filtro_hard = None # Padrão: sem filtro

        # Mapa de palavras-chave para substrings dos nomes das features no JSON
        # Se o usuário disser a chave, forçamos o filtro pelo valor.
        keyword_map = {
            "pedido": "Pedido",        # Vai casar com "Acompanhamento de Pedidos..."
            "solicitação": "Solicitação", # Vai casar com "Solicitação de Compra..."
            "solicitacao": "Solicitação"  # Variação sem acento
            # Adicione outros pares críticos aqui se necessário
        }

        conditions = []
        for keyword, feature_substring in keyword_map.items():
            # Se a palavra-chave crítica está na pergunta...
            if keyword in pergunta_lower:
                # ...adiciona uma condição de filtro para o banco de dados.
                # Usamos "$contains" para buscar a substring no nome completo da feature.
                conditions.append({"feature_name": {"$contains": feature_substring}})
        
        # Monta o filtro final para o ChromaDB
        if conditions:
            # Usamos um 'set' de tuplas para remover duplicatas (ex: se achar "solicitação" e "solicitacao")
            unique_conditions = [dict(t) for t in {tuple(d.items()) for d in conditions}]
            
            if len(unique_conditions) == 1:
                filtro_hard = unique_conditions[0]
            elif len(unique_conditions) > 1:
                # Se o usuário mencionou ambos (raro, mas possível), permite ambos.
                filtro_hard = {"$or": unique_conditions}
            
            # print(f"DEBUG: Aplicando filtro HARD por palavra-chave: {filtro_hard}")
        # -------------------------------------------------------------

        # 2. Realiza a busca semântica, APLICANDO O FILTRO SE HOUVER
        res = colecao.query(
            query_embeddings=[emb], 
            n_results=n_results_inicial,
            where=filtro_hard # <-- Aqui está a mágica. Se tiver filtro, ele usa.
        )
        meta = res.get('metadatas', [[]])[0]
        
        if not meta:
            return "", None

        # 3. Seleção de Vídeo Simplificada e Confiável
        # Como já filtramos os resultados para conter APENAS a funcionalidade correta
        # (se a palavra-chave foi usada), podemos pegar o primeiro vídeo que aparecer com segurança.
        video = None
        for m in meta:
            v_url = m.get('video_url')
            if v_url:
                video = v_url
                break
        
        # Monta o contexto
        contexto = "\n\n---\n\n".join([doc.get('texto_original', '') for doc in meta if doc.get('texto_original')])
        
        return contexto, video
    except Exception as e:
        st.error(f"Erro durante busca e síntese de contexto: {e}")
        return "", None
# --- FIM DA FUNÇÃO REFORMULADA ---

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
        return "Desculpe, ocorreu um erro ao gerar a resposta."

# --- 5. Lógica do Chat (Mantido igual) ---
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

RES_SAUDACAO = "Olá! Eu sou o Evo, seu assistente virtual para o sistema GoEvo. Como posso ser útil hoje?"

colecao_func, colecao_param = carregar_colecoes_chroma()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "video" in msg and msg["video"]:
            st.video(msg["video"])

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
                col = colecao_func if intencao == "FUNCIONALIDADE" else colecao_param
                prompt_sis = P_FUNC_SYSTEM if intencao == "FUNCIONALIDADE" else P_PARAM_SYSTEM
                
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

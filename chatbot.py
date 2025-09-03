import streamlit as st
import google.generativeai as genai
import chromadb
import unicodedata

# --- Configuração da Página ---
st.set_page_config(page_title="GoEvo Assist", page_icon="🤖")
st.title("🤖 Agente de Suporte GoEvo Compras")
st.caption("Faça uma pergunta.")

# --- Configuração Segura das Chaves de API ---
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: As chaves de API não foram encontradas. Configure seu arquivo .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- Funções do Agente de IA ---

@st.cache_resource
def carregar_colecao_chroma():
    try:
        client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE
        )
        collection = client.get_collection("manual_do_sistema_v2")
        st.success("Conectado à base de conhecimento (ChromaDB)!")
        return collection
    except Exception as e:
        st.error(f"Erro ao conectar com o ChromaDB: {e}")
        return None

# --- NOVA FUNÇÃO DE CLASSIFICAÇÃO DE INTENÇÃO COM GEMINI ---
def identificar_intencao_gemini(pergunta):
    try:
        prompt_classificador = f"""
        Classifique a frase do usuário em uma de duas categorias: SAUDACAO ou PERGUNTA_TECNICA.
        - SAUDACAO: Cumprimentos, conversas casuais (oi, olá, tudo bem?), agradecimentos, despedidas.
        - PERGUNTA_TECNICA: Dúvidas sobre o sistema, pedidos de ajuda, perguntas sobre funcionalidades.
        Responda APENAS com a palavra SAUDACAO ou a palavra PERGUNTA_TECNICA.

        Frase do usuário: "{pergunta}"
        """
        model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
        # Configuração para uma resposta curta e rápida
        config = genai.types.GenerationConfig(max_output_tokens=5, temperature=0)
        
        resposta = model.generate_content(prompt_classificador, generation_config=config)
        intencao = resposta.text.strip().upper()
        
        if "SAUDACAO" in intencao:
            return "SAUDACAO"
        else:
            return "PERGUNTA_TECNICA"
            
    except Exception as e:
        print(f"Erro na classificação de intenção: {e}")
        return "PERGUNTA_TECNICA" # Em caso de erro, assume que é pergunta técnica

def buscar_contexto(pergunta, colecao, n_results=5):
    if colecao is None: return "", None
    
    embedding_pergunta = genai.embed_content(
        model="models/text-embedding-004", content=pergunta, task_type="RETRIEVAL_QUERY"
    )["embedding"]

    resultados = colecao.query(query_embeddings=[embedding_pergunta], n_results=n_results)
    contexto_completo = resultados['metadatas'][0]
    chunks_relevantes = [doc['texto_original'] for doc in contexto_completo]
    contexto_texto = "\n\n---\n\n".join(chunks_relevantes)
    
    video_url = None
    for doc in contexto_completo:
        if doc.get('video_url'): 
            video_url = doc['video_url']
            break
            
    return contexto_texto, video_url

def gerar_resposta_com_gemini(pergunta, contexto):
    # Coloque seu prompt profissional aqui
    prompt_sistema = """
    ## Persona:
    Você é o GoEvo Assist, o especialista virtual... 
    (etc...)
    """
    model = genai.GenerativeModel('models/gemini-1.5-pro-latest')
    resposta = model.generate_content(f"{prompt_sistema}\n\n**CONTEXTO:**\n{contexto}\n\n**PERGUNTA DO USUÁRIO:**\n{pergunta}")
    return resposta.text

# --- Lógica da Interface do Chat (ATUALIZADA) ---
RESPOSTA_SAUDACAO = "Olá! Eu sou o Léo, assistente virtual da GoEvo. Como posso te ajudar com o sistema hoje?"
colecao = carregar_colecao_chroma()

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
        with st.spinner("Pensando..."):
            
            resposta_final = ""
            video_para_mostrar = None # <-- Garante que o vídeo começa como nulo
            
            # --- NOVA LÓGICA DE ROTEAMENTO COM IA ---
            intencao = identificar_intencao_gemini(pergunta_usuario)

            # Rota 1: Saudação
            if intencao == "SAUDACAO":
                resposta_final = RESPOSTA_SAUDACAO
                # video_para_mostrar continua nulo, como esperado.
            
            # Rota 2: Pergunta Técnica
            else: # PERGUNTA_TECNICA
                if colecao is not None:
                    contexto_relevante, video_encontrado = buscar_contexto(pergunta_usuario, colecao)
                    resposta_final = gerar_resposta_com_gemini(pergunta_usuario, contexto_relevante)
                    video_para_mostrar = video_encontrado
                else:
                    resposta_final = "Desculpe, estou com problemas para acessar a base de conhecimento."
            
            st.markdown(resposta_final)
            if video_para_mostrar:
                st.video(video_para_mostrar)
    
    mensagem_assistente = {"role": "assistant", "content": resposta_final}
    if video_para_mostrar:
        mensagem_assistente["video"] = video_para_mostrar
    st.session_state.messages.append(mensagem_assistente)

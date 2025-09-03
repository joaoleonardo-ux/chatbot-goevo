import streamlit as st
import openai
import chromadb

# --- Configuração da Página ---
st.set_page_config(page_title="GoEvo Assist", page_icon="🤖")
st.title("🤖 Agente de Suporte GoEvo Compras")
st.caption("Faça uma pergunta.")

# --- Configuração Segura das Chaves de API ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: As chaves de API não foram encontradas. Configure seu arquivo .streamlit/secrets.toml")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Funções do Agente de IA ---
@st.cache_resource
def carregar_colecao_chroma():
    try:
        client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE
        )
        # ATUALIZADO: Nome da nova coleção
        collection = client.get_collection("manual_do_sistema_v2") 
        st.success("Conectado à base de conhecimento (ChromaDB)!")
        return collection
    except Exception as e:
        st.error(f"Erro ao conectar com o ChromaDB: {e}")
        return None

def buscar_contexto(pergunta, colecao, n_results=5):
    if colecao is None: return "", None
    
    embedding_pergunta = client_openai.embeddings.create(
        input=[pergunta], model="text-embedding-3-small"
    ).data[0].embedding

    resultados = colecao.query(
        query_embeddings=[embedding_pergunta],
        n_results=n_results
    )
    
    contexto_completo = resultados['metadatas'][0]
    chunks_relevantes = [doc['texto_original'] for doc in contexto_completo]
    contexto_texto = "\n\n---\n\n".join(chunks_relevantes)
    
    video_url = None
    for doc in contexto_completo:
        # ATUALIZADO: A chave no seu novo JSON é 'videoTutorial'
        if doc.get('video_url'): 
            video_url = doc['video_url']
            break
            
    return contexto_texto, video_url

def gerar_resposta_com_gpt(pergunta, contexto):
    prompt_sistema = """
    ## Persona:
    Você é o GoEvo Assist, o especialista virtual e assistente de treinamento do sistema de compras GoEvo...
    # (Seu prompt profissional completo aqui)
    """
    
    resposta = client_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt_sistema},
            {"role": "user", "content": f"**CONTEXTO:**\n{contexto}\n\n**PERGUNTA:**\n{pergunta}"}
        ],
        temperature=0.7
    )
    return resposta.choices[0].message.content

# --- Lógica da Interface do Chat ---
colecao = carregar_colecao_chroma()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "video" in message:
            st.video(message["video"])

if pergunta_usuario := st.chat_input("Qual a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Consultando a base de conhecimento..."):
            
            contexto_relevante, video_encontrado = buscar_contexto(pergunta_usuario, colecao)
            resposta_final = gerar_resposta_com_gpt(pergunta_usuario, contexto_relevante)
            
            st.markdown(resposta_final)
            if video_encontrado:
                st.video(video_encontrado)
    
    mensagem_assistente = {"role": "assistant", "content": resposta_final}
    if video_encontrado:
        mensagem_assistente["video"] = video_encontrado
    st.session_state.messages.append(mensagem_assistente)

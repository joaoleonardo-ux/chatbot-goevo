import streamlit as st
import openai
import chromadb

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo Assist", page_icon="🤖", layout="wide")

# --- 2. Interface (CSS Personalizado) ---
st.markdown("""
<style>
    /* Esconde Header, Footer e Menus nativos */
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    
    /* ZERA o preenchimento para tela cheia */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }

    /* Estilo de balões de chat */
    [data-testid="stChatMessage"] {
        padding: 0.7rem !important;
        margin-bottom: 0.5rem !important;
        border-radius: 10px;
    }
    
    [data-testid="stChatMessageContent"] p {
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
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
        # Foco exclusivo na coleção de funcionalidades
        return _client.get_collection("colecao_funcionalidades")
    except Exception as e:
        st.error(f"Erro ao conectar com a base ChromaDB: {e}")
        return None

def rotear_pergunta(pergunta):
    """Classifica se o usuário quer uma saudação ou suporte funcional."""
    try:
        prompt_roteador = f"Classifique: SAUDACAO ou FUNCIONALIDADE. Responda apenas uma palavra. Pergunta: '{pergunta}'"
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_roteador}],
            temperature=0, max_tokens=10
        )
        intencao = resposta.choices[0].message.content.strip().upper()
        return "FUNCIONALIDADE" if "FUNCIONALIDADE" in intencao else "SAUDACAO"
    except:
        return "SAUDACAO"

def buscar_contexto_seguro(pergunta, colecao):
    """Busca contexto garantindo que o vídeo pertença à mesma fonte do texto."""
    if colecao is None: return "", None
    try:
        # 1. Gera Embedding da pergunta
        emb_response = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small")
        emb = emb_response.data[0].embedding
        
        # 2. Busca o fragmento mais próximo para definir o TÓPICO (Fonte)
        res_topo = colecao.query(query_embeddings=[emb], n_results=1)
        if not res_topo['metadatas'][0]:
            return "", None

        meta_principal = res_topo['metadatas'][0][0]
        fonte_alvo = meta_principal.get('fonte')
        video_url = meta_principal.get('video_url')

        # 3. Busca todos os fragmentos QUE PERTENCEM a essa fonte específica
        # Isso evita misturar passos de uma função com vídeo de outra
        res_completos = colecao.query(
            query_embeddings=[emb], 
            where={"fonte": fonte_alvo}, 
            n_results=15
        )
        
        fragmentos = res_completos.get('metadatas', [[]])[0]
        contexto = "\n\n".join([f.get('texto_original', '') for f in fragmentos if f.get('texto_original')])
        
        return contexto, video_url
    except Exception as e:
        st.error(f"Erro na recuperação: {e}")
        return "", None

def gerar_resposta(pergunta, contexto):
    """Sintetiza a resposta final em formato de guia."""
    prompt_sistema = """Você é o Evo, o assistente inteligente da GoEvo.
    Sua missão é explicar o funcionamento do sistema usando o contexto fornecido.
    REGRAS:
    1. Use listas numeradas para o passo a passo.
    2. Seja cordial, direto e objetivo.
    3. Se o contexto for insuficiente, peça para o usuário ser mais específico sobre qual tela ou menu ele se refere.
    4. Não invente passos que não estão no texto."""

    try:
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_sistema}, 
                {"role": "user", "content": f"CONTEXTO:\n{contexto}\n\nPERGUNTA:\n{pergunta}"}
            ],
            temperature=0.3
        )
        return resposta.choices[0].message.content
    except:
        return "Desculpe, tive um problema ao processar sua resposta. Pode tentar novamente?"

# --- 5. Execução do Chat ---

RES_SAUDACAO = "Olá! Eu sou o Evo, suporte da GoEvo. Como posso te ajudar com as funcionalidades do sistema hoje?"
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
            
            if intencao == "SAUDACAO":
                res_final = RES_SAUDACAO
            else:
                ctx, video = buscar_contexto_seguro(pergunta, colecao_func)
                if ctx:
                    res_final = gerar_resposta(pergunta, ctx)
                    # Adiciona o link do vídeo se existir na fonte encontrada
                    if video:
                        res_final += f"\n\n---\n\n**🎥 Vídeo explicativo:**\nAssista ao passo a passo detalhado: [Clique aqui para abrir o vídeo]({video})"
                else:
                    res_final = "Ainda não encontrei um passo a passo para essa funcionalidade. Pode detalhar melhor?"

            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})

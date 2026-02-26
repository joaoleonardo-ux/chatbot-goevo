import streamlit as st
import openai
import chromadb
import os

# --- 1. Configuração da Página ---
st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

st.markdown("""
<style>
    header {visibility: hidden; height: 0px !important;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    .block-container { padding-top: 0rem !important; padding-bottom: 0rem !important; padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important; }
    html, body, [data-testid="stAppViewContainer"] { font-size: 14px; background-color: transparent !important; }
    [data-testid="stChatInput"] div:focus-within { border-color: #0882c8 !important; }
    [data-testid="stChatInput"] button { background-color: #0882c8 !important; border: none !important; }
    [data-testid="stChatInput"] button svg { color: white !important; }
    [data-testid="stChatInput"] textarea { caret-color: #0882c8 !important; }
    [data-testid="stChatMessage"] { padding: 0.5rem !important; margin-bottom: 0.5rem !important; }
    div[data-testid="stChatMessageAvatarUser"] { background-color: #D3D3D3 !important; }
    div[data-testid="stChatMessageAvatarAssistant"] { background-color: transparent !important; }
    [data-testid="stChatMessageContent"] p { font-size: 0.95rem !important; line-height: 1.4 !important; }
</style>
""", unsafe_allow_html=True)

# --- 2. Configuração de APIs ---
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]
    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]
    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]
except (FileNotFoundError, KeyError):
    st.error("ERRO: Configure as chaves de API no Streamlit Secrets.")
    st.stop()

client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- 3. Funções do Core ---

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
    """Refinado para diferenciar Solicitação de Compra (Solicitante) de Compra (Comprador)."""
    try:
        prompt_roteador = f"""
        Classifique a intenção do usuário no sistema GoEvo:
        1. SAUDACAO
        2. AGRADECIMENTO
        3. SOLICITACAO_AMBIGUA: O usuário quer fazer uma "Solicitação de Compra" (perfil Solicitante), mas não disse se quer incluir "um item por vez" ou "por lista".
        4. FUNCIONALIDADE: Outras dúvidas (incluindo se ele já for específico sobre Solicitação Spot/Lista ou se for sobre o módulo de Compras do Comprador).

        Pergunta: '{pergunta}'
        Responda apenas a categoria.
        """
        resposta = client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_roteador}],
            temperature=0, 
            max_tokens=20
        )
        return resposta.choices[0].message.content.strip().upper()
    except:
        return "FUNCIONALIDADE"

def buscar_contexto_seguro(pergunta, colecao):
    if colecao is None: return "", None, ""
    try:
        emb_response = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small")
        emb = emb_response.data[0].embedding
        
        res_topo = colecao.query(query_embeddings=[emb], n_results=1)
        if not res_topo['metadatas'] or not res_topo['metadatas'][0]:
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
        return "", None, ""

def gerar_resposta(pergunta, contexto, nome_feature):
    prompt_sistema = f"""Você é o Evo, o assistente técnico da GoEvo.
    REGRAS DE OURO:
    1. Comece com: "Para realizar {nome_feature}, siga estes passos:"
    2. Use listas numeradas para as ações.
    3. Seja direto e técnico. Diferencie claramente 'Solicitação de Compra' de 'Compra'.
    4. Se não encontrar o procedimento, informe que não localizou na base."""

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
        return "Desculpe, tive um problema ao processar. Pode repetir?"

# --- 4. Execução do Chat ---

LOGO_IA = "logo-goevo.png" 
RES_SAUDACAO = "Olá! Eu sou o Evo. Como posso te ajudar hoje?"
RES_AGRADECIMENTO = "De nada! Fico feliz em ajudar. 😊"
colecao_func = carregar_colecao()

# Inicializa estados de memória
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]
if "aguardando_tipo_solicitacao" not in st.session_state:
    st.session_state.aguardando_tipo_solicitacao = False

# Renderiza histórico
for msg in st.session_state.messages:
    avatar = LOGO_IA if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Entrada do usuário
if pergunta := st.chat_input("Como posso te ajudar?"):
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant", avatar=LOGO_IA):
        with st.spinner("Analisando sua solicitação..."):
            res_final = ""
            
            # CENÁRIO: Respondendo à pergunta de escolha do tipo de solicitação
            if st.session_state.aguardando_tipo_solicitacao:
                if any(x in pergunta.lower() for x in ["um", "1", "spot", "item por item"]):
                    termo_busca = "Solicitação de Compra Spot"
                else:
                    termo_busca = "Solicitação de Compra por Lista"
                
                ctx, video, nome_f = buscar_contexto_seguro(termo_busca, colecao_func)
                res_final = gerar_resposta(termo_busca, ctx, nome_f)
                if video: res_final += f"\n\n---\n**🎥 Vídeo:** [Assista aqui]({video})"
                st.session_state.aguardando_tipo_solicitacao = False
            
            else:
                intencao = rotear_pergunta(pergunta)
                
                if intencao == "SAUDACAO":
                    res_final = RES_SAUDACAO
                elif intencao == "AGRADECIMENTO":
                    res_final = RES_AGRADECIMENTO
                elif intencao == "SOLICITACAO_AMBIGUA":
                    res_final = ("Entendi que você deseja criar uma **Solicitação de Compra**. "
                                 "Para eu te passar o passo a passo correto, como você prefere incluir os itens?\n\n"
                                 "1. **Um item por vez** (Solicitação de Compra Spot)\n"
                                 "2. **Vários itens de uma única vez** (Solicitação de Compra por Lista)")
                    st.session_state.aguardando_tipo_solicitacao = True
                else:
                    # Fluxo normal: se for dúvida de Comprador ou se já foi específico na Solicitação
                    ctx, video, nome_f = buscar_contexto_seguro(pergunta, colecao_func)
                    if ctx:
                        res_final = gerar_resposta(pergunta, ctx, nome_f)
                        if video: res_final += f"\n\n---\n**🎥 Vídeo:** [Assista aqui]({video})"
                    else:
                        res_final = "Não encontrei o procedimento exato para essa funcionalidade. Pode detalhar melhor sua dúvida?"

            st.markdown(res_final)
            st.session_state.messages.append({"role": "assistant", "content": res_final})


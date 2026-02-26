import streamlit as st

import openai

import chromadb

import os



st.set_page_config(page_title="Evo IA", page_icon="✨", layout="wide")

# --- 2. Injeção de CSS para Interface Totalmente Limpa e Clara ---
st.markdown("""
<style>
    /* 1. Esconde Header, Footer e Menus nativos */
    header {visibility: hidden; height: 0px !important;}
    #MainMenu {visibility: hidden;}
    footer {display: none !important;}
    [data-testid="stHeader"] {display: none !important;}
    [data-testid="stFooter"] {display: none !important;}
    
    /* 2. Remove badges, botão fullscreen e "Built with Streamlit" agressivamente */
    div[class*="viewerBadge"] {display: none !important;}
    button[title="View fullscreen"] {display: none !important;}
    .st-emotion-cache-1cvow4s {display: none !important;} /* Classe comum do rodapé do iframe */

    /* 3. Força o fundo branco em toda a aplicação */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
    }

    /* 4. ZERA o preenchimento e ajusta container */
    .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
        background-color: #FFFFFF !important;
    }

    /* 5. Ajuste global de fontes para PRETO */
    html, body, [data-testid="stAppViewContainer"] {
        font-size: 14px;
        color: #000000 !important;
    }

    /* 6. Balões de chat: Fundo claro e texto preto */
    [data-testid="stChatMessage"] {
        padding: 0.5rem !important;
        margin-bottom: 0.5rem !important;
        background-color: #F0F2F6 !important; /* Cinza bem leve para distinguir as bolhas */
        color: #000000 !important;
    }
    
    /* Garante que o texto dentro do chat seja preto */
    [data-testid="stChatMessageContent"] p, 
    [data-testid="stChatMessageContent"] li,
    [data-testid="stChatMessageContent"] span {
        font-size: 0.95rem !important;
        line-height: 1.4 !important;
        color: #000000 !important;
    }

    /* Ajusta a cor dos ícones de avatar */
    [data-testid="stChatMessage"] .st-emotion-cache-1p7n9v6 {
        background-color: #E0E0E0 !important;
    }

    /* 7. Remove padding extra do topo do chat */
    [data-testid="stVerticalBlock"] > div:first-child {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }

    /* 8. AJUSTES DA CAIXA DE TEXTO E FUNDO INFERIOR (Onde o usuário digita) */
    
    /* Fundo branco para a área fixa inferior inteira */
    [data-testid="stBottom"], 
    [data-testid="stBottomBlock"] > div {
        background-color: #FFFFFF !important;
    }

    /* Fundo branco para o container do input */
    [data-testid="stChatInput"] {
        background-color: #FFFFFF !important;
    }

    /* Texto preto e fundo branco dentro da área de digitação */
    [data-testid="stChatInput"] textarea {
        color: #000000 !important;
        background-color: #FFFFFF !important;
    }

    /* Escurece o botão de enviar (setinha) para ele aparecer no fundo branco */
    [data-testid="stChatInput"] button {
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Logo da GoEvo (Centralização via Colunas) ---
CAMINHO_LOGO = "logo-goevo.png"

c1, c2, c3 = st.columns([1, 0.3, 1])
with c2:
    if os.path.exists(CAMINHO_LOGO):
        st.image(CAMINHO_LOGO, width=65)
    else:
        st.markdown('<div style="height: 10px;"></div>', unsafe_allow_html=True)



# --- 4. Configuração de APIs ---

try:

    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    CHROMA_API_KEY = st.secrets["CHROMA_API_KEY"]

    CHROMA_TENANT = st.secrets["CHROMA_TENANT"]

    CHROMA_DATABASE = st.secrets["CHROMA_DATABASE"]

except:

    st.error("Erro nas chaves de API.")

    st.stop()



client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)



# --- 5. Funções do Chatbot (Roteamento e Busca) ---



@st.cache_resource

def carregar_colecao():

    try:

        _client = chromadb.CloudClient(api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE)

        return _client.get_collection("colecao_funcionalidades")

    except: return None



def rotear_pergunta(pergunta):

    try:

        prompt = f"Classifique: SAUDACAO, AGRADECIMENTO ou FUNCIONALIDADE. Responda apenas uma palavra: '{pergunta}'"

        res = client_openai.chat.completions.create(

            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0, max_tokens=15

        )

        intencao = res.choices[0].message.content.strip().upper()

        if "FUNCIONALIDADE" in intencao: return "FUNCIONALIDADE"

        if "AGRADECIMENTO" in intencao: return "AGRADECIMENTO"

        return "SAUDACAO"

    except: return "SAUDACAO"



def buscar_contexto(pergunta, colecao):

    if colecao is None: return "", None, ""

    try:

        emb = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small").data[0].embedding

        res = colecao.query(query_embeddings=[emb], n_results=1)

        if not res['metadatas'][0]: return "", None, ""

        meta = res['metadatas'][0][0]

        f_alvo, video = meta.get('fonte'), meta.get('video_url')

        res_comp = colecao.query(query_embeddings=[emb], where={"fonte": f_alvo}, n_results=15)

        ctx = "\n\n".join([f.get('texto_original', '') for f in res_comp['metadatas'][0] if f.get('texto_original')])

        return ctx, video, f_alvo

    except: return "", None, ""



def gerar_resposta(pergunta, contexto, nome_feature):

    prompt = f"Você é o Evo. Responda: 'Para realizar {nome_feature}, siga estes passos:' seguido de lista numerada."

    try:

        res = client_openai.chat.completions.create(

            model="gpt-4o",

            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"CONTEXTO: {contexto}\nPERGUNTA: {pergunta}"}],

            temperature=0

        )

        return res.choices[0].message.content

    except: return "Erro ao processar."



# --- 6. Fluxo do Chat ---



RES_SAUDACAO = "Olá! Eu sou o Evo, suporte inteligente da GoEvo. Como posso ajudar?"

RES_AGRADECIMENTO = "De nada! Se precisar de algo mais, é só chamar! 😊"

colecao_func = carregar_colecao()



if "messages" not in st.session_state:

    st.session_state.messages = [{"role": "assistant", "content": RES_SAUDACAO}]



for msg in st.session_state.messages:

    with st.chat_message(msg["role"]): st.markdown(msg["content"])



if pergunta := st.chat_input(" Como posso te ajudar?"):

    st.session_state.messages.append({"role": "user", "content": pergunta})

    with st.chat_message("user"): st.markdown(pergunta)



    with st.chat_message("assistant"):

        with st.spinner("..."):

            intencao = rotear_pergunta(pergunta)

            if intencao == "AGRADECIMENTO":

                res_final = RES_AGRADECIMENTO

            elif intencao == "SAUDACAO":

                res_final = RES_SAUDACAO

            else:

                ctx, video, nome_f = buscar_contexto(pergunta, colecao_func)

                if ctx:

                    res_final = gerar_resposta(pergunta, ctx, nome_f)

                    if video: res_final += f"\n\n---\n**🎥 Tutorial:** [Clique aqui]({video})"

                else:

                    res_final = "Ainda não tenho esse passo a passo."

            

            st.markdown(res_final)

            st.session_state.messages.append({"role": "assistant", "content": res_final})




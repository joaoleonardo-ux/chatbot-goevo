import streamlit as st
import openai
import chromadb

# Adicione este bloco no início do seu chatbot.py
st.markdown("""
    <style>
        /* Esconde o header e o footer do Streamlit */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Configuração da Página ---
st.set_page_config(page_title="Agente de Suporte", page_icon="🤖")
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

# Cliente da OpenAI
client_openai = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Funções do Agente de IA ---

@st.cache_resource
def carregar_colecao_chroma():
    try:
        client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE
        )
        collection = client.get_collection("manual_do_sistema")
        st.success("Conectado à base de conhecimento (ChromaDB)!")
        return collection
    except Exception as e:
        st.error(f"Erro ao conectar com o ChromaDB: {e}")
        return None

# --- NOVA FUNÇÃO DE CLASSIFICAÇÃO DE INTENÇÃO ---
def identificar_intencao(pergunta):
    """
    Usa a IA para classificar a intenção do usuário.
    """
    try:
        prompt_classificador = f"""
        Você é um classificador de intenção de usuário para um chatbot de suporte técnico.
        Analise a frase do usuário e classifique-a em uma das seguintes categorias:

        - SAUDACAO: Cumprimentos, agradecimentos, despedidas, gentilezas.
        - PERGUNTA_TECNICA: Dúvidas sobre funcionalidades do sistema, como fazer algo no sistema.
        - ERRO: Relato de falha, erro técnico, problema de funcionamento.
        - TREINAMENTO: Solicitação de passo a passo, guia de uso, tutoriais.
        - FAQ: Perguntas comuns de negócio ou suporte que não envolvem o sistema em si.

        Responda APENAS com uma destas palavras: SAUDACAO, PERGUNTA_TECNICA, ERRO, TREINAMENTO ou FAQ.

        Frase do usuário: "{pergunta}"
        """

        resposta = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_classificador}],
            temperature=0,
            max_tokens=5
        )

        return resposta.choices[0].message.content.strip().upper()
            
    except Exception as e:
        print(f"Erro na classificação de intenção: {e}")
        return "PERGUNTA_TECNICA"



def buscar_contexto(pergunta, colecao, n_results=3):
    if colecao is None: return ""
    embedding_pergunta = client_openai.embeddings.create(
        input=[pergunta], model="text-embedding-3-small"
    ).data[0].embedding
    resultados = colecao.query(query_embeddings=[embedding_pergunta], n_results=n_results)
    chunks_relevantes = [doc['texto_original'] for doc in resultados['metadatas'][0]]
    contexto = "\n\n---\n\n".join(chunks_relevantes)
    return contexto

def gerar_resposta_com_gpt(pergunta, contexto):
    prompt_sistema = """
    ## Persona:
    Você é o GoEvo Assist, o especialista virtual e assistente de treinamento do sistema de compras GoEvo. Sua personalidade é profissional, prestativa e didática.

    ## Objetivo Principal:
    Seu objetivo é guiar os usuários de forma proativa para que eles consigam realizar o processo de compras com sucesso e autonomia, utilizando as melhores práticas do sistema.

    ## Regras de Comportamento e Tom de Voz:
    1.  **Seja Proativo:** Após responder a pergunta principal do usuário, se o contexto permitir, sugira o próximo passo lógico ou uma funcionalidade relacionada que possa ser útil.
        * Exemplo: Se o usuário perguntar como criar um fornecedor, você pode responder e, ao final, sugerir: "Agora que o fornecedor foi criado, gostaria de saber como cadastrar um produto para ele?".
    2.  **Seja Claro e Estruturado:** Sempre que a resposta envolver um processo, formate-a em passos numerados (1., 2., 3.) ou em uma lista com marcadores (•) para facilitar a leitura e o acompanhamento.
    3.  **Seja Interativo:** Termine suas respostas com uma pergunta aberta para incentivar a continuação da conversa e garantir que a dúvida do usuário foi completamente resolvida. Use frases como "Isso te ajudou?", "Ficou claro?" ou "Posso te ajudar com mais algum detalhe sobre este processo?".

    ## Regras Rígidas de Uso do Contexto (Regras Anti--Alucinação):
    1.  **REGRA MAIS IMPORTANTE:** Sua resposta deve ser baseada **única e exclusivamente** nas informações contidas no **CONTEXTO** fornecido abaixo. Não utilize nenhum conhecimento externo que você possua.
    2.  **SE A RESPOSTA NÃO ESTIVER NO CONTEXTO:** Se a pergunta do usuário não puder ser respondida com as informações do contexto, não invente uma solução. Responda de forma clara e honesta: "Não encontrei informações específicas sobre isso em nossa base de conhecimento. Você poderia tentar perguntar de uma forma diferente?"
    3.  **NÃO SE DESCULPE:** Não peça desculpas por usar apenas o contexto. Aja como um especialista consultando seu manual para fornecer a informação mais precisa.
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

# --- Interface do Chat ---
# REMOVEMOS A LISTA DE SAUDAÇÕES DAQUI
RESPOSTA_SAUDACAO = "Olá! Eu sou Léo, assistente virtual da GoEvo. Como posso te ajudar?"

colecao = carregar_colecao_chroma()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pergunta_usuario := st.chat_input("Qual a sua dúvida?"):
    st.session_state.messages.append({"role": "user", "content": pergunta_usuario})
    with st.chat_message("user"):
        st.markdown(pergunta_usuario)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            
            # --- LÓGICA DE ROTEAMENTO ATUALIZADA COM IA ---
            intencao_detectada = identificar_intencao(pergunta_usuario)
            
            # ROTA 1: A IA classificou como Saudação
            if intencao_detectada == "SAUDACAO":
                resposta_final = RESPOSTA_SAUDACAO
            
            # ROTA 2: A IA classificou como Pergunta Técnica
            else: # intencao_detectada == "PERGUNTA_TECNICA"
                if colecao is not None:
                    contexto_relevante = buscar_contexto(pergunta_usuario, colecao)
                    resposta_final = gerar_resposta_com_gpt(pergunta_usuario, contexto_relevante)
                else:
                    resposta_final = "Desculpe, estou com problemas para acessar a base de conhecimento."
            
            st.markdown(resposta_final)
    

    st.session_state.messages.append({"role": "assistant", "content": resposta_final})



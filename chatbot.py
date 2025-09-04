import streamlit as st
import openai
import chromadb

# --- Configuração da Página ---
st.set_page_config(page_title="GoEvo Assist", page_icon="🤖")
st.title("🤖 Agente de Suporte GoEvo Compras")
st.caption("Faça uma pergunta.")

# --- Configuração das Chaves de API ---
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
def carregar_colecoes_chroma():
    try:
        client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY, tenant=CHROMA_TENANT, database=CHROMA_DATABASE
        )
        colecao_funcionalidades = client.get_collection("colecao_funcionalidades")
        colecao_parametros = client.get_collection("colecao_parametros")
        st.success("Conectado aos especialistas de Funcionalidades e Parâmetros!")
        return colecao_funcionalidades, colecao_parametros
    except Exception as e:
        st.error(f"Erro ao conectar com a base de conhecimento: {e}")
        return None, None

def rotear_pergunta(pergunta):
    prompt_roteador = f"""
    Você é um roteador de perguntas para um assistente de suporte de sistema.
    Classifique a pergunta do usuário em uma de três categorias: SAUDACAO, FUNCIONALIDADE ou PARAMETRO.

    - SAUDACAO: Cumprimentos, conversas casuais (oi, olá, tudo bem?).
    - FUNCIONALIDADE: Perguntas sobre COMO FAZER algo no sistema (ex: "como eu crio uma cotação?", "onde eu aprovo a SC?", "tem vídeo de como cadastrar produto?").
    - PARAMETRO: Perguntas sobre O QUE É ou COMO CONFIGURAR um parâmetro (ex: "o que faz o parâmetro Liberação de SC?", "quais os parâmetros da omie?", "onde configuro o centro de custo?").

    Responda APENAS com a palavra SAUDACAO, FUNCIONALIDADE ou PARAMETRO.

    Pergunta do usuário: "{pergunta}"
    """
    resposta = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_roteador}],
        temperature=0, max_tokens=10
    )
    intencao = resposta.choices[0].message.content.strip().upper()
    if "FUNCIONALIDADE" in intencao: return "FUNCIONALIDADE"
    if "PARAMETRO" in intencao: return "PARAMETRO"
    return "SAUDACAO"

# --- FUNÇÃO DE BUSCA ATUALIZADA ---
def buscar_contexto(pergunta, colecao, n_results=15): # <-- AJUSTE 1: Aumentado para 15
    """
    Busca os chunks, usa todos para o contexto de TEXTO,
    mas usa APENAS O MAIS RELEVANTE para o VÍDEO.
    """
    if colecao is None: return "", None
    
    embedding_pergunta = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small").data[0].embedding
    
    resultados = colecao.query(
        query_embeddings=[embedding_pergunta],
        n_results=n_results
    )
    
    metadados_completos = resultados.get('metadatas', [[]])[0]
    
    contexto_texto = ""
    if metadados_completos:
        contexto_texto = "\n\n---\n\n".join([doc['texto_original'] for doc in metadados_completos])
    
    video_url = None
    if metadados_completos:
        primeiro_resultado_meta = metadados_completos[0]
        video_url = primeiro_resultado_meta.get('video_url')
            
    return contexto_texto, video_url

def gerar_resposta_com_gpt(pergunta, contexto, prompt_especialista):
    resposta = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_especialista},
            {"role": "user", "content": f"**CONTEXTO:**\n{contexto}\n\n**PERGUNTA:**\n{pergunta}"}
        ],
        temperature=0.5
    )
    return resposta.choices[0].message.content

# --- Definição dos Prompts dos Especialistas ---
prompt_assistente_funcionalidades = """
## Persona:
Você é o GoEvo Assist, o especialista virtual e assistente de treinamento do sistema de compras GoEvo. Sua personalidade é profissional, prestativa e didática.
## Objetivo Principal:
Seu objetivo é guiar os usuários de forma proativa para que eles consigam realizar o processo de compras com sucesso e autonomia, utilizando as melhores práticas do sistema.
## Regras de Comportamento e Tom de Voz:
1. Seja Proativo: Após responder a pergunta principal, se o contexto permitir, sugira o próximo passo lógico ou uma funcionalidade relacionada.
2. Seja Claro e Estruturado: Sempre que a resposta envolver um processo, formate-a em passos numerados (1., 2., 3.).
3. Seja Interativo: Termine suas respostas com uma pergunta aberta como "Isso te ajudou?" ou "Posso te ajudar com mais algum detalhe?".
## Regras Rígidas de Uso do Contexto:
1. Sua resposta deve ser baseada única e exclusivamente nas informações contidas no CONTEXTO.
2. Se a resposta não estiver no CONTEXTO, responda: "Não encontrei informações sobre isso em nossa base de conhecimento. Você poderia tentar perguntar de uma forma diferente?".
"""

# <-- AJUSTE 2: Prompt do Especialista em Parâmetros Aprimorado
prompt_especialista_parametros = """
## Persona:
Você é o GoEvo Assist, um especialista técnico nos parâmetros de configuração do sistema de compras GoEvo. Sua personalidade é precisa, completa e informativa.

## Objetivo Principal:
Seu objetivo é listar e explicar de forma clara todos os parâmetros relevantes encontrados no contexto que respondam à pergunta do usuário.

## Regras de Comportamento e Tom de Voz:
1.  **Seja Completo:** Se o contexto contiver múltiplos parâmetros que se encaixam na pergunta do usuário (ex: "parâmetros do omie"), **liste TODOS eles**. Não resuma ou omita informações.
2.  **Formate com Clareza:** Para cada parâmetro, estruture a resposta da seguinte forma, usando os dados do contexto:
    * **Parâmetro:** (Use o "Titulo do Parametro")
    * **Finalidade:** (Use a "Sugestão de Descrição")
    * **Quando é Utilizado:** (Use o "Quando é utilizado")
    * **Dependências:** (Use o "Necessário ativação de outro parametro")
3.  **Use Negrito:** Destaque os títulos de cada seção (como **Finalidade:**) para facilitar a leitura.

## Regras Rígidas de Uso do Contexto:
1.  **REGRA MAIS IMPORTANTE:** Sua resposta deve ser baseada **única e exclusivamente** nas informações contidas no **CONTEXTO**.
2.  **SE A RESPOSTA NÃO ESTIVER NO CONTEXTO:** Se o contexto não contiver informações para responder à pergunta, diga: "Não encontrei detalhes sobre este(s) parâmetro(s) em nossa base de conhecimento. Poderia ser mais específico?".
"""

# --- Lógica da Interface do Chat com Roteamento ---
RESPOSTA_SAUDACAO = "Olá! Eu sou o GoEvo Assist. Posso te ajudar com dúvidas sobre funcionalidades ou parâmetros do sistema. O que você gostaria de saber?"
colecao_func, colecao_param = carregar_colecoes_chroma()

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
        with st.spinner("Analisando sua pergunta..."):
            
            resposta_final = ""
            video_para_mostrar = None
            
            intencao = rotear_pergunta(pergunta_usuario)
            
            if intencao == "SAUDACAO":
                resposta_final = RESPOSTA_SAUDACAO
            
            else: # Se for FUNCIONALIDADE ou PARAMETRO
                colecao_para_buscar = None
                prompt_para_usar = None
                
                if intencao == "FUNCIONALIDADE":
                    st.spinner("Consultando o especialista em funcionalidades...")
                    colecao_para_buscar = colecao_func
                    prompt_para_usar = prompt_assistente_funcionalidades
                
                elif intencao == "PARAMETRO":
                    st.spinner("Consultando o especialista em parâmetros...")
                    colecao_para_buscar = colecao_param
                    prompt_para_usar = prompt_especialista_parametros

                if colecao_para_buscar:
                    contexto, video_encontrado = buscar_contexto(pergunta_usuario, colecao_para_buscar)
                    resposta_final = gerar_resposta_com_gpt(pergunta_usuario, contexto, prompt_para_usar)
                    
                    if "Não encontrei" in resposta_final:
                        video_para_mostrar = None
                    else:
                        video_para_mostrar = video_encontrado
                else:
                    resposta_final = "Desculpe, a base de conhecimento necessária não está disponível."
            
            st.markdown(resposta_final)
            if video_para_mostrar:
                st.video(video_para_mostrar)
    
    mensagem_assistente = {"role": "assistant", "content": resposta_final}
    if video_para_mostrar:
        mensagem_assistente["video"] = video_para_mostrar
    st.session_state.messages.append(mensagem_assistente)

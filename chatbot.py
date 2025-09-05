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
        # A anotação _client é para indicar que esta variável é interna da função cacheada
        _client = chromadb.CloudClient(
            api_key=st.secrets["CHROMA_API_KEY"], 
            tenant=st.secrets["CHROMA_TENANT"], 
            database=st.secrets["CHROMA_DATABASE"]
        )
        colecao_funcionalidades = _client.get_collection("colecao_funcionalidades")
        colecao_parametros = _client.get_collection("colecao_parametros")
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

def buscar_e_sintetizar_contexto(pergunta, colecao, n_results_inicial=10):
    if colecao is None: return "", None

    # Etapa A: Recuperação Ampla
    embedding_pergunta = client_openai.embeddings.create(input=[pergunta], model="text-embedding-3-small").data[0].embedding
    resultados_iniciais = colecao.query(
        query_embeddings=[embedding_pergunta],
        n_results=n_results_inicial
    )
    
    metadados_iniciais = resultados_iniciais.get('metadatas', [[]])[0]
    
    if not metadados_iniciais:
        return "", None

    # Identifica as fontes únicas dos documentos relevantes
    fontes_relevantes = list(set([doc['fonte'] for doc in metadados_iniciais]))
    # Removida (ou comentada) a linha abaixo para não exibir a mensagem no chat
    # st.info(f"Tópicos relevantes identificados: {', '.join(fontes_relevantes)}") 

    # Etapa B: Busca Filtrada e Completa
    resultados_filtrados = colecao.query(
        query_embeddings=[embedding_pergunta],
        where={"fonte": {"$in": fontes_relevantes}},
        n_results=50 
    )

    metadados_completos = resultados_filtrados.get('metadatas', [[]])[0]

    if not metadados_completos:
        return "", None

    # Etapa C: Monta o "Super-Documento"
    contexto_texto = "\n\n---\n\n".join([doc['texto_original'] for doc in metadados_completos])
    
    video_url = metadados_iniciais[0].get('video_url')
            
    return contexto_texto, video_url

def gerar_resposta_sintetizada(pergunta, contexto, prompt_especialista):
    resposta = client_openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_especialista},
            {"role": "user", "content": f"**CONTEXTO COMPLETO (DE VÁRIAS FONTES):**\n{contexto}\n\n**PERGUNTA DO USUÁRIO:**\n{pergunta}"}
        ],
        temperature=0.5
    )
    return resposta.choices[0].message.content

# --- Definição dos Prompts dos Especialistas ---
prompt_assistente_funcionalidades = """
## Persona:
Você é o GoEvo Assist, o especialista virtual e assistente de treinamento do sistema de compras GoEvo.
Sua missão é fornecer respostas claras, precisas e didáticas, utilizando EXCLUSIVAMENTE o contexto fornecido.
Você é o **único e completo ponto de suporte** para o usuário neste momento. NÃO se refira a "consultar o suporte do sistema GoEvo", "nossa equipe", ou qualquer outro tipo de "suporte externo".

## Formato de Resposta e Abordagem:
* Compreenda a pergunta do usuário e o contexto fornecido.
* Formule uma resposta **direta, natural e conversacional**, como se estivesse explicando a funcionalidade a um colega.
* Priorize a utilidade: se o contexto descreve um procedimento, apresente os **passos de forma clara e numerada**.
* Se houver dúvidas e respostas relevantes no contexto para a pergunta, integre as respostas de forma fluida.
* Vá direto ao ponto. NÃO inclua seções fixas como "Resumo", "Dicas Adicionais", "Observações Finais" ou qualquer texto genérico que não seja uma resposta direta e personalizada à pergunta do usuário, baseada no contexto.
* Mantenha um tom profissional, amigável e extremamente prestativo.
* NÃO adicione frases como "Se precisar de mais informações ou assistência, consulte o suporte do sistema GoEvo." ou similares. Você É o suporte.

Se o contexto não fornecer a informação para a pergunta do usuário, diga educadamente que, com base nas informações disponíveis, não foi possível encontrar a resposta específica e sugira que o usuário tente reformular a pergunta ou explore outro tópico.
"""
prompt_especialista_parametros = """
## Persona:
Você é o GoEvo Assist, um especialista técnico nos parâmetros de configuração do sistema de compras GoEvo.
Sua missão é explicar os parâmetros de forma clara, objetiva e útil, utilizando EXCLUSIVAMENTE o contexto fornecido.
Você é o **único e completo ponto de suporte** para o usuário. NÃO se refira a "consultar o suporte do sistema GoEvo" ou qualquer outro tipo de "suporte externo".

## Formato de Resposta e Abordagem:
* Compreenda a pergunta do usuário e o contexto fornecido.
* Formule uma resposta **direta, natural e conversacional**, explicando o parâmetro de forma compreensível.
* Priorize a clareza sobre o que o parâmetro faz, seu impacto e, se disponível no contexto, como configurá-lo.
* Vá direto ao ponto. NÃO inclua seções fixas como "Definição do Parâmetro", "Impacto", "Considerações Importantes" ou qualquer texto genérico.
* Mantenha um tom profissional e técnico, mas acessível.
* NÃO adicione frases como "Se precisar de mais informações ou assistência, consulte o suporte do sistema GoEvo." ou similares. Você É o suporte.

Se o contexto não fornecer a informação para a pergunta do usuário, diga educadamente que, com base nas informações disponíveis, não foi possível encontrar a resposta específica e sugira que o usuário tente reformular a pergunta ou explore outro tópico.
"""


# --- Lógica da Interface do Chat com Roteamento ---
RESPOSTA_SAUDACAO = "Olá! Eu sou o Leo, Assistente Virtual do GoEvo. em que posso ajudar?"
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
            
            else:
                colecao_para_buscar = colecao_func if intencao == "FUNCIONALIDADE" else colecao_param
                prompt_para_usar = prompt_assistente_funcionalidades if intencao == "FUNCIONALIDADE" else prompt_especialista_parametros
                
                if colecao_para_buscar:
                    st.spinner(f"Consultando especialista em {intencao.lower()}...")
                    
                    contexto_sintetizado, video_encontrado = buscar_e_sintetizar_contexto(pergunta_usuario, colecao_para_buscar)
                    
                    if contexto_sintetizado:
                        st.spinner("Elaborando a melhor resposta...")
                        resposta_final = gerar_resposta_sintetizada(pergunta_usuario, contexto_sintetizado, prompt_para_usar)
                        video_para_mostrar = video_encontrado
                    else:
                        resposta_final = "Não encontrei informações sobre isso em nossa base de conhecimento. Você poderia tentar perguntar de uma forma diferente?"
                else:
                    resposta_final = "Desculpe, a base de conhecimento necessária não está disponível."
            
            st.markdown(resposta_final)
            if video_para_mostrar:
                st.video(video_para_mostrar)
    
    mensagem_assistente = {"role": "assistant", "content": resposta_final}
    if video_para_mostrar:
        mensagem_assistente["video"] = video_para_mostrar
    st.session_state.messages.append(mensagem_assistente)

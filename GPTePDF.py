import streamlit as st
import os
import pytube
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import RagGeneratorChatModel
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import UnstructuredFileLoader, ImageCaptionLoader

# Configurações da interface do usuário
st.header("Upload your own file and ask questions like ChatGPT")
st.subheader('File types supported: PDF/DOCX/TXT/JPG/PNG/YouTube :city_sunrise:')

# Barra lateral para entrada de chave de API do OpenAI
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Carregamento do modelo LLM (GPT-3.5)
llm = RagGeneratorChatModel(
    model_name="gpt-3.5-turbo-16k",
    max_tokens=16000,
    temperature=0,
    streaming=True
)

# Carregar modelo RAG pré-treinado
rag_model = RagGeneratorChatModel.from_pretrained("facebook/rag-token-base")

# Função para carregar histórico de versões
def load_version_history():
    with open("version_history.txt", "r") as file:
        return file.read()

# Barra lateral para upload de arquivos e fornecimento de URL do YouTube
with st.sidebar:
    uploaded_files = st.file_uploader("Please upload your files", accept_multiple_files=True, type=None)
    youtube_url = st.text_input("YouTube URL")

    with st.sidebar.expander("**Version History**", expanded=False):
        st.write(load_version_history())

    st.info("Please refresh the browser if you decide to upload more files to reset the session", icon="🚨")

# Processamento de dados
if uploaded_files or youtube_url:
    if "processed_data" not in st.session_state:
        documents = []

        if uploaded_files:
            for uploaded_file in uploaded_files:
                file_path = os.path.join(os.getcwd(), uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                if file_path.endswith((".png", ".jpg")):
                    image_loader = ImageCaptionLoader(path_images=[file_path])
                    image_documents = image_loader.load()
                    documents.extend(image_documents)
                elif file_path.endswith((".pdf", ".docx", ".txt")):
                    loader = UnstructuredFileLoader(file_path)
                    loaded_documents = loader.load()
                    documents.extend(loaded_documents)

        if youtube_url:
            youtube_video = pytube.YouTube(youtube_url)
            streams = youtube_video.streams.filter(only_audio=True)
            stream = streams.first()
            stream.download(filename="youtube_audio.mp4")
            openai.api_key = openai_api_key
            with open("youtube_audio.mp4", "rb") as audio_file:
                transcript = openai.Audio.transcribe("whisper-1", audio_file)
            youtube_text = transcript['text']
            youtube_document = Document(page_content=youtube_text, metadata={})
            documents.append(youtube_document)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        document_chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_documents(document_chunks, embeddings)

        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

    else:
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    # Inicialização da cadeia de perguntas e respostas (Q&A) com RAG
    qa = ConversationalRetrievalChain.from_chat_model(rag_model, vectorstore.as_retriever())

    # Histórico do chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Entrada do usuário
    if prompt := st.chat_input("Ask your questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        history = [
            f"{message['role']}: {message['content']}" 
            for message in st.session_state.messages
        ]

        # Consulta o modelo RAG para obter resposta
        result = qa({
            "question": prompt, 
            "chat_history": history
        })

        # Exibe a resposta na interface do usuário
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = result["answer"]
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.write("Please upload your files and provide a YouTube URL for transcription.")

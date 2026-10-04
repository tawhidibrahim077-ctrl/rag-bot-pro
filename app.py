import os
import sqlite3
import streamlit as st
from datetime import datetime
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import base64

vectorstore = None
llm = None

st.set_page_config(page_title="Client RAG Bot Pro", layout="wide")
DB_NAME = "chat_history.db"

# 1. DATABASE: Chat History Save er jonno
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chats
                 (id INTEGER PRIMARY KEY, timestamp TEXT, role TEXT, message TEXT)''')
    conn.commit()
    conn.close()

def save_chat(role, message):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO chats (timestamp, role, message) VALUES (?, ?, ?)", (timestamp, role, message))
    conn.commit()
    conn.close()

def load_chat():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT role, message FROM chats ORDER BY id ASC")
    rows = c.fetchall()
    conn.close()
    return rows

# 2. MODEL LOAD
@st.cache_resource
def load_models():
    global vectorstore, llm
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash", 
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        streaming=True
    )    
    return vectorstore, llm
    
vectorstore, llm = load_models()
init_db()

# 3. PDF UPLOAD + DUPLICATE CHECK
def process_and_save_pdf(uploaded_file):
    global vectorstore
    with open(f"temp_{uploaded_file.name}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(f"temp_{uploaded_file.name}")
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)
    
    # Prottek chunk e source name add kore dicchi
    for chunk in chunks:
        chunk.metadata["source"] = uploaded_file.name
    
    vectorstore.add_documents(chunks)
    os.remove(f"temp_{uploaded_file.name}")
    return len(chunks), uploaded_file.name

# 4. DATABASE THEKE SOB PDF ER NAM BER KORA
def get_saved_sources():
    try:
        collections = vectorstore._collection.get()
        if collections['metadatas']:
            return set([os.path.basename(m['source']) for m in collections['metadatas']])
    except:
        pass
    return set()

# 5. CHAT EXPORT TO TXT
def export_chat():
    chat_data = load_chat()
    text = ""
    for role, msg in chat_data:
        text += f"{role.upper()}: {msg}\n\n"
    return text

# UI
st.title("📄 Client RAG Chatbot Pro")
tab1, tab2, tab3 = st.tabs(["📤 PDF Upload", "💬 Chat with AI", "⚙️ Settings"])

with tab1:
    st.header("Notun PDF Upload Koro")
    uploaded_files = st.file_uploader("Multiple PDF select koro", type="pdf", accept_multiple_files=True)
    
    saved_sources = get_saved_sources()

    # AUTO SAVE - KINTU DUPLICATE HOBE NA
    if uploaded_files:
        new_files_found = False
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in saved_sources:
                new_files_found = True
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    num_chunks, name = process_and_save_pdf(uploaded_file)
                st.success(f"✅ {name} - {num_chunks} ta chunk save hoise!")
            else:
                st.info(f"ℹ️ {uploaded_file.name} agei save kora ase")
        
        if new_files_found:
            st.rerun() # Notun file thaklei shudhu refresh

    st.subheader("Database e ki ase:")
    if saved_sources:
        for s in saved_sources:
            st.write(f"✅ {s}")
    else:
        st.write("Ekhono kono PDF nai")

with tab2:
    st.header("AI er sathe Kotha Bolo")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat()

    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    if prompt := st.chat_input("Prosno likho..."):
        st.chat_message("user").write(prompt)
        st.session_state.chat_history.append(("user", prompt))
        save_chat("user", prompt)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        template = """Tumi ekjon sahajjokari AI. Nicher context use kore uttor dao.
        Jodi uttor na thake bolo 'ami document e pai nai'. Bangla te uttor diba.
        Chat History: {chat_history}
        Context: {context}
        Prosno: {question}
        Uttor:"""
        prompt_temp = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([f"Source: {os.path.basename(doc.metadata['source'])}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}" for doc in docs])

        history_str = "\n".join([f"Human: {h}\nAI: {a}" for h, a in st.session_state.chat_history[-6:]])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough(), "chat_history": lambda x: history_str}
            | prompt_temp
            | llm
            | StrOutputParser()
        )
        
        with st.chat_message("assistant"):
            with st.spinner("Vabtese..."):
                full_response = st.write_stream(rag_chain.stream(prompt))
        
        st.session_state.chat_history.append(("assistant", full_response))
        save_chat("assistant", full_response)

        docs = retriever.invoke(prompt)
        st.write("--- **Source** ---")
        for i, doc in enumerate(docs):
            source_name = os.path.basename(doc.metadata['source'])
            page_num = doc.metadata.get('page', 'N/A')
            st.write(f"{i+1}. {source_name} - Page: {page_num}")
        st.write("----------------")

with tab3:
    st.header("Settings")
    if st.button("🗑️ Chat History Clear Koro"):
        conn = sqlite3.connect(DB_NAME)
        c = conn.cursor()
        c.execute("DELETE FROM chats")
        conn.commit()
        conn.close()
        st.session_state.chat_history = []
        st.success("Chat history clear hoye gese")

    if st.button("🗑️ Vector Database Clear Koro"):
        vectorstore.delete_collection()
        st.success("Database er sob PDF delete hoye gese")
        st.rerun()

    chat_export = export_chat()
    st.download_button(
        label="📥 Chat Export as TXT",
        data=chat_export,
        file_name=f"chat_export_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
    )

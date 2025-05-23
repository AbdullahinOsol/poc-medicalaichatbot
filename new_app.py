import streamlit as st
import ollama
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import fitz
from new_helper import vector_and_upsert, get_file_hash, check_domain
import time
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

model = OllamaLLM(model="llama3.2:3b", streaming=True)


new_template = """
You are a helpful and knowledgeable medical assistant. Use the context if relevant to answer the user's question accurately. Otherwise answer the question according to your knowledge. Do NOT answer any questions unrelated to medicine or healthcare. If the user asks something outside the medical domain, respond with:
"I'm a medical assistant and can help with health-related questions."

Here is the relevant context: {context_text}


Here is the conversation so far: {chat_history_prompt}


Here is the question to answer: {user_input}
"""

new_prompt = ChatPromptTemplate.from_template(new_template)
chain = new_prompt | model

# --- Configuration ---
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("INDEX_NAME")
namespace = os.getenv("NAMESPACE")
embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME")

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if 'indexed_file_hashes' not in st.session_state:
    st.session_state.indexed_file_hashes = set()

# --- Streamlit UI ---
st.set_page_config(page_title="🩺 Medical Assistant", page_icon="💬")
st.title("🩺 Medical Assistant Chatbot")
st.markdown("Ask anything medical-related. Uses RAG for context-aware answers.")

# --- File Upload Section ---
uploaded_file = st.file_uploader("📄 Upload a medical-related document (PDF or TXT)", type=["pdf", "txt"])

# --- Load Embedding Model ---
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    multi_process=False,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- Load Pinecone Index ---
try:
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)
except Exception as e:
    st.error(f"❌ Error connecting to Pinecone: {e}")
    st.stop()

# --- Process Uploaded File ---
if uploaded_file:
    file_bytes = uploaded_file.read()
    file_hash = get_file_hash(file_bytes)

    if file_hash not in st.session_state.indexed_file_hashes:
        with st.spinner('Processing uploaded file...'):
            if uploaded_file.type == 'application/pdf':
                doc = fitz.open(stream=uploaded_file.read(), filetype='pdf')
                full_text = '\n'.join([page.get_text() for page in doc])

            elif uploaded_file.type == 'text/plain':
                full_text = file_bytes.decode('utf-8')

            else:
                full_text = ''
                print('Unsupported file type.')

            if full_text.strip():
                data = vector_and_upsert(full_text)
                try:
                    index.upsert(data, namespace=namespace)
                    st.success('✅ File content indexed successfully.')
                    st.session_state.indexed_file_hashes.add(file_hash)
                
                except Exception as e:
                    st.error(f"❌ Failed to index document: {e}")

            else:
                st.warning('⚠️ No extractable text found in the file.')

# --- Initialize Ollama Client ---
client = ollama.Client()

# --- Chat Input ---
user_input = st.chat_input("Enter your medical question...")

# --- Display Chat History ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Process New Input ---
if user_input:
    # Add user message to history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
            response_container = st.empty()
            response_text = ""

    try:
        # Step 1: Embed user input
        vectorized_query = embedding_model.embed_query(user_input)

        # Step 2: Retrieve top document from Pinecone
        results = index.query(
            namespace=namespace,
            vector=vectorized_query,
            top_k=1,
            include_metadata=True,
        )

        # Step 3: Prepare context
        if not results["matches"]:
            context_text = ""
        else:
            context_text = results["matches"][0]["metadata"]["text"]

        print(f"the id of the context received is: {results['matches'][0]['id']}")
        print(f"The score for received context is: {results['matches'][0]['score']}")
        print(f"Context retrieved: {context_text}")

        # Step 4: Build conversation prompt
        chat_history_prompt = "\n".join([
            f"User: {msg['content']}" if msg['role'] == 'user' else f"Assistant: {msg['content']}"
            for msg in st.session_state.chat_history if msg['role'] != 'system'
        ])

        stream = chain.stream(
            {"context_text": context_text,
             "chat_history_prompt": chat_history_prompt,
             "user_input": user_input}
        )

        for chunk in stream:
            response_text += chunk
            response_container.markdown(response_text + "▌")

        response_container.markdown(response_text)

        # Save assistant message to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # Keep only the last 3 user-assistant turns(6 messages)
        st.session_state.chat_history = st.session_state.chat_history[-4:]

        # Display updated chat history
        # print('Chat history updated:')
        # print(st.session_state.chat_history)

    except Exception as e:
        response_container.error(f"❌ Error: {e}")
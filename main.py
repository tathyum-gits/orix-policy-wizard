import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Functions
def get_embedding(text):
    response = openai_client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

def search_pinecone(query, top_k=5):
    query_vector = get_embedding(query)
    result = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )
    return result.matches

def ask_gpt4(context, question):
    system_prompt = ("""
    You are a helpful AI assistant. 
    Answer the user's question based only on the provided context. Write your answer in bullet points. Do not write in paragraphs.
    If the answer is not contained in the context, say 'The answer is not available in the available documents.'.
    """)

    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def build_context(matches):
    context = ""
    for match in matches:
        chunk_text = match.metadata.get('text', '')
        context += chunk_text + "\n"
    return context

# Streamlit App
st.set_page_config(page_title="Orix Policy Wizard", page_icon="🔎")
st.title("🔎 Orix Policy Wizard")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar settings
st.sidebar.title("Settings")
st.sidebar.info("Ensure your environment variables (API Keys) are set correctly!")

# Optional: Display API status
try:
    openai_client.models.list()
    st.sidebar.success("OpenAI API connected ✅")
except Exception as e:
    st.sidebar.error(f"OpenAI Error: {e}")

try:
    pc.describe_index(os.getenv("PINECONE_INDEX_NAME"))
    st.sidebar.success("Pinecone API connected ✅")
except Exception as e:
    st.sidebar.error(f"Pinecone Error: {e}")

# Chat display
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

# Chat input
query = st.chat_input("Type your message...")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    with st.spinner("Thinking..."):
        matches = search_pinecone(query)

        if not matches:
            answer = "No relevant documents found."
        else:
            context = build_context(matches)
            answer = ask_gpt4(context, query)

    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").markdown(answer)

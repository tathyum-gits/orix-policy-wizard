import os
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv


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
st.set_page_config(page_title="Mini RAG App", page_icon="🔎")
st.title("🔎 Mini RAG App")
st.write("Ask questions and get AI answers based on your document collection!")

# User input
query = st.text_input("Enter your question:", placeholder="Type your question here...")

if query:
    with st.spinner("Searching for relevant documents..."):
        matches = search_pinecone(query)

    if not matches:
        st.warning("No relevant documents found.")
    else:
        context = build_context(matches)

        with st.spinner("Thinking with GPT-4..."):
            answer = ask_gpt4(context, query)

        st.subheader("🧠 GPT-4 Answer:")
        st.markdown(answer)

        with st.expander("Show Retrieved Context"):
            st.text(context)

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

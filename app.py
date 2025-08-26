import streamlit as st
import pinecone
from pinecone import ServerlessSpec, Pinecone
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import uuid
import os
import re

# Load API keys from Streamlit secrets
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)

# Embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Gemini setup
genai.configure(api_key=gemini_api_key)
chat_model = genai.GenerativeModel('gemini-1.5-flash')

# ---------------- UTILS ----------------
def sanitize_index_name(user_id: str) -> str:
    """Ensure Pinecone index name is valid"""
    name = f"neurallens-{user_id.lower()}"
    name = re.sub(r'[^a-z0-9-]', '-', name)   
    name = name.strip('-')                   
    return name

def get_user_index(user_id):
    index_name = sanitize_index_name(user_id)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=384,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
    return pc.Index(index_name)

def store_text(user_id, text, doc_id):
    try:
        index = get_user_index(user_id)
        chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
        for i, chunk in enumerate(chunks):
            embedding = model.encode(chunk).tolist()
            index.upsert(vectors=[(f"{doc_id}-{i}", embedding, {"text": chunk, "doc_id": doc_id})])
    except Exception as e:
        st.error(f"Error storing document: {e}")

def generate_summary(user_id, keyword):
    try:
        index = get_user_index(user_id)
        query_embedding = model.encode(keyword).tolist()
        results = index.query(vector=query_embedding, top_k=10, include_metadata=True)
        doc_chunks = {}
        for match in results['matches']:
            doc_id = match['metadata'].get('doc_id')
            text = match['metadata']['text']
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(text)
        if not doc_chunks:
            return "No relevant documents found for the keyword."
        complete_docs = ["\n".join(chunks) for chunks in doc_chunks.values()]
        combined_text = "\n".join(complete_docs)
        summary_prompt = f"Summarize the following content related to '{keyword}':\n\n{combined_text}"
        summary_response = chat_model.generate_content(summary_prompt)
        return summary_response.text.strip()
    except Exception as e:
        return f"Error generating summary: {e}"

def search_documents(user_id, query):
    try:
        index = get_user_index(user_id)
        if query.lower().startswith("summary of"):
            keyword = query[len("summary of"):].strip()
            return generate_summary(user_id, keyword)
        else:
            query_embedding = model.encode(query).tolist()
            results = index.query(vector=query_embedding, top_k=1, include_metadata=True)
            if results and results['matches']:
                document_text = results['matches'][0]['metadata']['text']
                gemini_response = chat_model.generate_content([
                    f"Context: {document_text}\n\nQuestion: {query}\n\nPlease provide a detailed and complete response. If not available, reply with 'Information not available in the given files'.\n\nAnswer:"
                ])
                return gemini_response.text.strip()
            return "No matching document found"
    except Exception as e:
        return f"Error searching document: {e}"

def extract_title(text):
    try:
        title_prompt = f"Extract the single most suitable title for the following text:\n\n{text}\n\nTitle:"
        title_response = chat_model.generate_content(title_prompt)
        return title_response.text.strip()
    except Exception as e:
        return f"Error extracting title: {e}"

# ---------------- STREAMLIT UI ----------------
st.title("NeuroDoc AI")

menu = st.sidebar.radio("Navigation", ["Home", "Store Document", "Search", "Summary", "Extract Title"])

if menu == "Home":
    st.write("Welcome to **NeuroDoc AI**")
    st.write("Store documents, search through them, generate summaries, and extract titles intelligently!")

elif menu == "Store Document":
    user_id = st.text_input("Enter your User ID")
    text = st.text_area("Paste your document text here")
    if st.button("Store Document"):
        if user_id and text:
            doc_id = str(uuid.uuid4())
            store_text(user_id, text, doc_id)
            st.success(f"Document stored successfully! (Doc ID: {doc_id})")
        else:
            st.error("Please enter both User ID and text")

elif menu == "Search":
    user_id = st.text_input("Enter your User ID")
    query = st.text_input("Enter your search query")
    if st.button("Search"):
        if user_id and query:
            answer = search_documents(user_id, query)
            st.write("### Answer:")
            st.write(answer)
        else:
            st.error("Please enter both User ID and query")

elif menu == "Summary":
    user_id = st.text_input("Enter your User ID")
    keyword = st.text_input("Enter keyword for summary")
    if st.button("Generate Summary"):
        if user_id and keyword:
            summary_text = generate_summary(user_id, keyword)
            st.write("### Summary:")
            st.write(summary_text)
        else:
            st.error("Please enter both User ID and keyword")

elif menu == "Extract Title":
    text = st.text_area("Paste text to extract title")
    if st.button("Extract Title"):
        if text:
            title = extract_title(text)
            st.write("### Suggested Title:")
            st.write(title)
        else:
            st.error("Please paste some text")



from flask import Flask, request, jsonify
import pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import uuid
import os
app = Flask(__name__)

pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

pc = pinecone.Pinecone(api_key=pinecone_api_key)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
genai.configure(api_key=gemini_api_key)
chat_model = genai.GenerativeModel('gemini-1.5-pro')

def get_user_index(user_id):
    index_name = f"neurallens-{user_id.lower()}"
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
        print(f"Error storing document: {e}")

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
                    f"Context: {document_text}\n\nQuestion: {query}\n\nPlease provide a detailed and complete response to the question in full sentences. If the information is not available, respond with 'Information not available in the given files':\n\nAnswer:"
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

@app.route('/')
def home():
    return "NeuroDoc AI is running!"

@app.route('/store', methods=['POST'])
def store():
    data = request.get_json()
    user_id = data.get('user_id')
    text = data.get('text')
    if not user_id or not text:
        return jsonify({"error": "User ID and text are required"}), 400
    doc_id = str(uuid.uuid4())
    store_text(user_id, text, doc_id)
    return jsonify({"message": "Document stored successfully!", "doc_id": doc_id})

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    user_id = data.get('user_id')
    query = data.get('query')
    if not user_id or not query:
        return jsonify({"error": "User ID and query are required"}), 400
    answer = search_documents(user_id, query)
    return jsonify({"answer": answer})

@app.route('/summary', methods=['POST'])
def summary():
    data = request.get_json()
    user_id = data.get('user_id')
    keyword = data.get('keyword')
    if not user_id or not keyword:
        return jsonify({"error": "User ID and keyword are required"}), 400
    summary_text = generate_summary(user_id, keyword)
    return jsonify({"summary": summary_text})

@app.route('/title', methods=['POST'])
def title():
    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "Text is required"}), 400
    title = extract_title(text)
    return jsonify({"title": title})

@app.route('/get/<user_id>/<doc_id>', methods=['GET'])
def get_document(user_id, doc_id):
    try:
        index = get_user_index(user_id)
        result = index.query(id=doc_id, top_k=1, include_values=True)
        if result and result['matches']:
            return jsonify({"doc_id": doc_id, "embedding": result['matches'][0]['values']})
        return jsonify({"error": "Document not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


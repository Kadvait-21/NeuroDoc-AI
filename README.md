# NeuroDoc AI

NeuroDoc AI is an intelligent document storage and retrieval system powered by **Pinecone**, **Sentence Transformers**, and **Google Gemini**.
It allows you to store documents, search contextually, generate summaries, and extract titles — all from a simple **Streamlit** interface.

---

## Features

* **Store Documents**: Save your text documents into a personal Pinecone index.
* **Semantic Search**: Ask questions or find contextually relevant information.
* **Summarization**: Generate concise summaries for topics or keywords.
* **Title Extraction**: Automatically suggest the best title for a document.

---

## Tech Stack

* [Streamlit](https://streamlit.io/) – Interactive UI
* [Pinecone](https://www.pinecone.io/) – Vector database
* [SentenceTransformers](https://www.sbert.net/) – Embeddings
* [Google Gemini](https://ai.google.dev/) – Generative AI for summaries & answers
* Python 3.9+

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/neurondoc-ai.git
cd neurondoc-ai
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

You’ll need API keys from **Pinecone** and **Google AI (Gemini)**.

Create a `.env` file in the project root:

```bash
PINECONE_API_KEY=your-pinecone-api-key
GEMINI_API_KEY=your-gemini-api-key
```

Or export them directly:

```bash
export PINECONE_API_KEY=your-pinecone-api-key
export GEMINI_API_KEY=your-gemini-api-key
```

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
neurondoc-ai/
│── app.py              # Main Streamlit app
│── requirements.txt    # Dependencies
│── README.md           # Documentation
│── .env                # API keys (not committed to git)
```

---

## 🎯 Usage

1. Launch the app using `streamlit run app.py`.
2. Use the sidebar to choose an action:

   * **Store Document**: Enter User ID + text → store in Pinecone.
   * **Search**: Ask a question and get AI-assisted answers.
   * **Summary**: Summarize documents around a keyword.
   * **Extract Title**: Get the best title for your text.
3. All data is stored in **per-user Pinecone indexes**.

---



##  Requirements

* Python 3.9+
* Pinecone API key
* Google Gemini API key

---

## Contributing

Pull requests are welcome! If you’d like to suggest major changes, please open an issue first to discuss.

---

## License

MIT License – feel free to use, modify, and share.

---

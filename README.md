# Multilingual_PDF_RAG_System


## 📘 Overview
This project implements a **Retrieval-Augmented Generation (RAG)** system designed to process **multilingual PDFs** — including **English, Hindi, Bengali, and Chinese** — and provide **summaries** and **answers to user queries** based on the document content.

The system can handle both **scanned** and **digital PDFs**, using a hybrid approach of **OCR-based extraction** and **semantic retrieval**. It demonstrates end-to-end integration of **text extraction**, **vector database search**, and **LLM-powered answer generation**.

---

## 🏗️ System Architecture

### **1. PDF Ingestion & Text Extraction**
- Detects document type (scanned vs. digital)
- Extracts text using:
  - **PyMuPDF / pdfplumber** for digital PDFs
  - **Tesseract OCR** (via `pytesseract`) for scanned PDFs
- Handles **multilingual content** using appropriate OCR language models (e.g., `hin`, `ben`, `chi_sim`).

### **2. Text Preprocessing & Chunking**
- Text is cleaned, normalized, and chunked using:
  - Semantic-aware segmentation
  - Overlap-based chunking to maintain context
- Metadata (document name, page, language, etc.) is preserved for filtering.

### **3. Embedding & Vector Store**
- Converts text chunks into embeddings using models such as:
  - `sentence-transformers/all-MiniLM-L6-v2` (small, efficient)
  - Multilingual alternatives like `paraphrase-multilingual-MiniLM-L12-v2`
- Stores embeddings in a **vector database**:
  - `FAISS` (for local experiments)
  - Optional support for scalable DBs like `Milvus`, `Pinecone`, or `Chroma`

### **4. Hybrid Search & Reranking**
- Combines **semantic search** (via embeddings) and **keyword search** (via BM25).
- Optionally reranks results using cross-encoder models (e.g., `ms-marco-MiniLM-L-6-v2`).

### **5. LLM Integration**
- Integrates with a local or API-based **LLM** for answer generation.
  - Example: `mistral-7b-instruct`, `Llama-3-8B`, or `gpt-4-mini`
- Supports **query decomposition** and **context-aware responses** with **chat memory**.

### **6. Metadata Filtering**
- Enables filtering results based on:
  - Language
  - Document type
  - Date or domain metadata

---

## ⚙️ Features

| Feature | Description |
|----------|-------------|
| 🈳 **Multilingual OCR** | Handles Hindi, Bengali, Chinese, and English documents |
| 🧩 **Hybrid Retrieval** | Combines semantic + keyword search |
| 💬 **Chat Memory** | Maintains conversation context |
| 🔍 **Query Decomposition** | Splits complex queries into manageable sub-queries |
| 🧠 **Reranking** | Improves result relevance |
| ⚡ **Scalability** | Designed to handle up to **1TB** of document data |
| 🧱 **Small Models** | Optimized for lightweight deployment without large GPUs |

---

## 🧪 Evaluation Metrics

| Metric | Description |
|---------|-------------|
| **Query Relevance** | Alignment of results with user intent |
| **Retrieval Test** | Whether the retrieved chunks match contextually |
| **Latency** | Time taken per query-response cycle |
| **Fluency** | Clarity and coherence of generated text |
| **Model Efficiency** | Performance with small models (<2B parameters) |

---

## 🚀 How to Run

### **1. Clone the repository**
```bash
git clone https://github.com/<your-username>/multilingual-rag-system.git
cd multilingual-rag-system
```

### **2. Install dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Jupyter notebook**
```bash
jupyter notebook RAG_system.ipynb
```

Or execute directly:
```bash
python run_rag_pipeline.py
```

### **4. Input PDFs**
Place all input PDFs (digital and scanned) in:
```
/data/input_pdfs/
```

### **5. Query the System**
You can interact with the system via:
- Notebook cell prompts  
- Web UI (if implemented)  
- CLI (`python query.py --query "Summarize document X"`)  

---

## 🧰 Tech Stack

| Category | Tools/Frameworks |
|-----------|------------------|
| PDF Extraction | `PyMuPDF`, `pdfplumber`, `pytesseract` |
| NLP Models | `HuggingFace Transformers`, `SentenceTransformers` |
| Vector DB | `FAISS`, `Chroma`, `Milvus` |
| Backend Logic | `LangChain`, `LlamaIndex`, `Python` |
| LLMs | `Mistral`, `Llama`, `GPT-4-mini` |
| Visualization & UI | `Streamlit` (optional) |

---

## 📊 Example Workflow

```text
PDF → OCR/Text Extraction → Chunking → Embedding → Vector Storage →
Query → Retrieval → Context Reranking → LLM Answer Generation
```

---

## 📈 Future Improvements
- Integration with **document summarization dashboards**
- Deployment as an **API or web service**
- **Language detection** automation and mixed-language query handling
- Distributed vector database support (e.g., `Weaviate`, `Milvus`)
- Enhanced **model fine-tuning** for domain-specific documents

---

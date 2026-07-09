# GST-RAG-MODEL
Team project: A Retrieval-Augmented Generation (RAG) system for GST document intelligence.

**Features**
- Retrieval-augmented QA over uploaded documents (PDF, DOCX, TXT, CSV, XLSX, PPTX, HTML, JSON, and more).
- Chroma-based persistent vector store located at `chroma/`.
- Configurable embedding and LLM models via the Streamlit sidebar.

**Available Models**
- **LLM models:**
	- meta-llama/llama-3.3-70b-instruct
	- openai/gpt-oss-120b
	- google/gemma-4-31b-it
	- nvidia/nemotron-3-nano-30b-a3b (default)
- **Embedding models:**
	- sentence-transformers/all-MiniLM-L6-v2
	- sentence-transformers/all-MiniLM-L12-v2 (default)
	- intfloat/e5-base-v2

**Quick Start**
1. Create and activate a Python virtual environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Provide an API key for your OpenRouter-compatible backend (or other OpenAI-compatible client) as the `API_KEY` environment variable or in a `.env` file:

```
export API_KEY="your_api_key"
# or create a .env file with API_KEY=your_api_key
```

4. Run the app:

```
streamlit run app.py
```

5. In the sidebar select the desired embedding and LLM models, then ask questions in the main chat area. Conversation turns are shown as repeated "User question" / "Model answer" pairs.

**Data**: The app stores embeddings and DB state in the `chroma/` directory by default.

**Notes**
- If you change the embedding model selection, the app will reload the selected embedding model (this may take time).
- The default LLM is `nvidia/nemotron-3-nano-30b-a3b` and the default embedding model is `sentence-transformers/all-MiniLM-L12-v2`.

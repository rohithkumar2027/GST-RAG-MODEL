import os
from typing import List, Any

import streamlit as st

from dotenv import load_dotenv

load_dotenv()

LLM_MODELS = [
	"meta-llama/llama-3.3-70b-instruct",
	"openai/gpt-oss-120b",
	"google/gemma-4-31b-it",
	"nvidia/nemotron-3-nano-30b-a3b",
]
DEFAULT_LLM_MODEL = "nvidia/nemotron-3-nano-30b-a3b"

EMBEDDING_MODELS = [
	"sentence-transformers/all-MiniLM-L6-v2",
	"sentence-transformers/all-MiniLM-L12-v2",
	"intfloat/e5-base-v2",
]
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"


@st.cache_resource
def load_sentence_model(name: str):
	from sentence_transformers import SentenceTransformer

	return SentenceTransformer(name)

@st.cache_resource
def load_cross_encoder(name: str):
	from sentence_transformers import CrossEncoder

	return CrossEncoder(name)


class EmbeddingManger:
	def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
		self.model_name = model_name
		self.model = None
		self.load_model()

	def load_model(self):
		try:
			self.model = load_sentence_model(self.model_name)
		except Exception as e:
			raise RuntimeError(f"Error loading the model {self.model_name}:{e}")

	def generate_embeddings(self, texts: List[str]):
		if not self.model:
			raise ValueError("Model not loaded")
		return self.model.encode(texts)


class VectorStore:
	def __init__(self, collection_name: str = "Pdf_documents", persist_directory: str = "chroma"):
		self.collection_name = collection_name
		self.persist_directory = persist_directory
		self.client = None
		self.collection = None
		self.initialize_store()

	def initialize_store(self):
		import chromadb

		os.makedirs(self.persist_directory, exist_ok=True)
		self.client = chromadb.PersistentClient(path=self.persist_directory)
		try:
			# try to get existing collection
			self.collection = self.client.get_collection(name=self.collection_name)
		except Exception:
			# create if not exists
			try:
				self.collection = self.client.create_collection(
					name=self.collection_name,
					metadata={"description": "PDF document embedding for RAG"},
				)
			except Exception as e:
				st.error(f"Failed to create/get collection: {e}")

	def query(self, query_embedding: List[float], top_k: int = 10):
		if self.collection is None:
			return None
		results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
		return results


class RAGRetriever:
	def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManger):
		self.vector_store = vector_store
		self.embedding_manager = embedding_manager

	def retrieve(self, query: str, top_k: int = 10, score_threshold: float = 0.0):
		query_embedding = self.embedding_manager.generate_embeddings([query])[0]

		if self.vector_store.collection is None:
			raise ValueError("Collection not initialized")

		try:
			results = self.vector_store.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

			retrieved_docs = []
			if results.get("documents") and results["documents"][0]:
				documents = results["documents"][0]
				metadatas = results.get("metadatas", [])[0] if results.get("metadatas") else [None] * len(documents)
				distances = results.get("distances", [])[0] if results.get("distances") else [0.0] * len(documents)
				ids = results.get("ids", [])[0] if results.get("ids") else [None] * len(documents)

				for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
					similarity_score = 1 - distance
					if similarity_score >= score_threshold:
						retrieved_docs.append(
							{
								"id": doc_id,
								"content": document,
								"metadata": metadata,
								"similarity_score": similarity_score,
								"distance": distance,
								"rank": i + 1,
							}
						)
			return retrieved_docs
		except Exception as e:
			st.error(f"Error during retrieval: {e}")
			return []


class ReRanker:
	def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
		self.model_name = model_name
		self.model = load_cross_encoder(self.model_name)

	def rerank(self, query: str, retriever: RAGRetriever, top_k: int = 3):
		retrieved_docs = retriever.retrieve(query, top_k=top_k)
		if not retrieved_docs:
			return []
		pairs = [(query, doc["content"]) for doc in retrieved_docs]
		scores = self.model.predict(pairs)
		for doc, score in zip(retrieved_docs, scores):
			doc["rerank_score"] = float(score)
		retrieved_docs.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
		return retrieved_docs[:top_k]


def sample_rag(query: str, retriever: RAGRetriever, llm, model_name: str, rerank_k: int = 3):
	from openai import OpenAI

	re_ranker = ReRanker()
	results = re_ranker.rerank(query, retriever, top_k=rerank_k)
	context = "\n\n".join([doc.get("content", "") for doc in results]) if results else ""

	if not results:
		return "No relevant documents found to answer the question.", results

	prompt = f"""Use the following context to answer the question:\nContext:\n{context}\nQuestion: {query}\nAnswer:"""

	try:
		response = llm.chat.completions.create(
			model=model_name,
			messages=[{"role": "user", "content": prompt}],
		)
		return response.choices[0].message.content, results
	except Exception as e:
		return f"LLM call failed: {e}", results


def main():
	st.set_page_config( page_title="GST AI Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded")
	st.title("📚 GST AI Assistant")
	top_k = st.sidebar.slider("Retriever top-k", 1, 20, 8)
	rerank_k = st.sidebar.slider("Rerank top-k", 1, 5, 3)
	selected_llm_model = st.sidebar.selectbox(
		"LLM model",
		options=LLM_MODELS,
		index=LLM_MODELS.index(DEFAULT_LLM_MODEL),
	)
	selected_embedding_model = st.sidebar.selectbox(
		"Embedding model",
		options=EMBEDDING_MODELS,
		index=EMBEDDING_MODELS.index(DEFAULT_EMBEDDING_MODEL),
	)

	if "embedding_manager" not in st.session_state or st.session_state.get("selected_embedding_model") != selected_embedding_model:
		try:
			st.session_state.embedding_manager = EmbeddingManger(model_name=selected_embedding_model)
			st.session_state.selected_embedding_model = selected_embedding_model
		except Exception as e:
			st.error(f"Failed to load embedding model: {e}")

	if "vector_store" not in st.session_state:
		try:
			st.session_state.vector_store = VectorStore(persist_directory="chroma")
		except Exception as e:
			st.error(f"Failed to initialize vector store: {e}")

	if "conversation" not in st.session_state:
		st.session_state.conversation = []

	user_question = st.chat_input("Ask a question about the uploaded PDFs:")

	if user_question:
		with st.spinner("Retrieving and generating answer..."):
			try:
				from openai import OpenAI

				llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.getenv("API_KEY"))

				retriever = RAGRetriever(st.session_state.vector_store, st.session_state.embedding_manager)
				answer, docs = sample_rag(user_question, retriever, llm, model_name=selected_llm_model, rerank_k=rerank_k)

				st.session_state.conversation.append(
					{
						"question": user_question,
						"answer": answer,
						"docs": [
							{
								"content": doc.get("content", ""),
								"score": doc.get("rerank_score", 0.0),
								"similarity": doc.get("similarity_score", 0.0),
								"metadata": doc.get("metadata"),
							}
							for doc in docs
						],
					}
				)
			except Exception as e:
				st.error(f"Error answering query: {e}")

	for turn in st.session_state.conversation:
		st.markdown("---")
		st.markdown("**User query**")
		st.write(turn["question"])
		st.markdown("**Model answer**")
		st.write(turn["answer"])
		if turn["docs"]:
			st.markdown("**Retrieved passages**")
			for i, doc in enumerate(turn["docs"], start=1):
				with st.expander(f"Passage {i} — score {doc['score']:.4f}"):
					st.write(doc["content"])
					st.markdown(f"**Similarity:** {doc['similarity']:.4f}")
					if doc["metadata"]:
						st.write(doc["metadata"])


if __name__ == "__main__":
	main()
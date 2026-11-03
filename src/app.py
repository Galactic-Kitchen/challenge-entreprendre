import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.mistralai import MistralAI
from llama_index.vector_stores.chroma import ChromaVectorStore

from src.config import Config

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = Flask(__name__)
app.config.from_object(Config)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/rag/query")
def rag_query():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")

    if not question:
        return jsonify({"error": "A 'question' field is required."}), 400

    if not app.config.get("MISTRAL_API_KEY"):
        return jsonify({"error": "MISTRAL_API_KEY is missing from the environment config."}), 500

    try:
        Settings.llm = MistralAI(
            model=app.config.get("MISTRAL_MODEL", "mistral-small-latest"),
            api_key=app.config["MISTRAL_API_KEY"],
        )

        client = chromadb.HttpClient(
            host=app.config["CHROMA_HOST"],
            port=app.config["CHROMA_PORT"],
        )
        collection = client.get_or_create_collection(app.config["CHROMA_COLLECTION"])
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        retriever = VectorIndexRetriever(index=index, similarity_top_k=app.config["TOP_K"])
        nodes = retriever.retrieve(question)

        if not nodes:
            return jsonify({"answer": "No relevant documents found in the vector store.", "sources": []})

        context = "\n\n".join(
            node.node.get_content() if hasattr(node.node, "get_content") else str(node.node)
            for node in nodes
        )

        response = Settings.llm.complete(
            "Use the following context to answer the user's question. If the answer cannot be derived from the context, say so clearly.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )

        sources = [
            {
                "score": getattr(node, "score", None),
                "id": getattr(node.node, "node_id", None),
                "text": node.node.get_content() if hasattr(node.node, "get_content") else str(node.node),
            }
            for node in nodes
        ]

        return jsonify({"answer": str(response), "sources": sources})
    except Exception as exc:  # pragma: no cover
        return jsonify({"error": str(exc)}), 500

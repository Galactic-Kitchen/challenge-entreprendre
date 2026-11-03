import logging
from typing import Any, Dict, List

import chromadb
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import Config

logger = logging.getLogger(__name__)


def initialiser_et_rempli_chromadb(donnees_brutes_git: List[Dict[str, Any]]) -> VectorStoreIndex:
    """Découpe, vectorise et stocke les fichiers de configuration dans ChromaDB.

    Cette fonction prend les données brutes extraites des dépôts Git, applique 
    un découpage (chunking), calcule les embeddings locaux via Hugging Face, 
    et enregistre le tout dans la base vectorielle ChromaDB.

    Args:
        donnees_brutes_git (List[Dict[str, Any]]): Liste de dictionnaires contenant 
            les fichiers extraits (ex: [{'nom_projet': 'X', 'contenu': '...'}])

    Returns:
        VectorStoreIndex: L'index LlamaIndex connecté à ChromaDB prêt à être interrogé.
    """ 
    logger.info("1. configuration du modele d'embeding local ...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    logger.info("2. Transformation des données brutes en objets Document LlamaIndex...")
    documents = []
    for item in donnees_brutes_git:
        # On crée un objet Document avec le contenu et les métadonnées indispensables
        doc = Document(
            text=item["contenu"],
            metadata={
                "nom_projet": item["nom_projet"],
                "nom_fichier": item["nom_fichier"],
                "type_fichier": item.get("type_fichier", "inconnu")
            }
        )
        documents.append(doc)
    logger.info("3. Chunking (Découpage des documents en morceaux)...")
    # Découpage par tranche de 500 caractères avec 50 caractères de recouvrement
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info("   -> %s documents découpés en %s chunks (nodes).", len(documents), len(nodes))

    logger.info("4. Connexion et stockage dans ChromaDB...")
    client = chromadb.HttpClient(host=Config.CHROMA_HOST, port=Config.CHROMA_PORT)
    chroma_collection = client.get_or_create_collection(Config.CHROMA_COLLECTION)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("5. Calcul des embeddings et enregistrement des vecteurs...")
    # Cette étape calcule les vecteurs pour chaque chunk et les insère dans ChromaDB
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )

    logger.info("✅ Ingestion terminée avec succès ! La base ChromaDB est remplie.")
    return index

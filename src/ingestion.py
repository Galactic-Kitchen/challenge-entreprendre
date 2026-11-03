import os
import chromadb
from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from typing import List, Dict, Any


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
    print("1. configuration du modele d'embeding local ...")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )
    print("2. Transformation des données brutes en objets Document LlamaIndex...")
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
    print("3. Chunking (Découpage des documents en morceaux)...")
    # Découpage par tranche de 500 caractères avec 50 caractères de recouvrement
    splitter = SentenceSplitter(chunk_size=500, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"   -> {len(documents)} documents découpés en {len(nodes)} chunks (nodes).")

    print("4. Connexion et stockage dans ChromaDB...")
    # Connexion au serveur ChromaDB (ou stockage local persistant)
    
    client = chromadb.HttpClient(host="localhost", port=8000)
    
    # Création ou récupération de la collection
    chroma_collection = client.get_or_create_collection("projets_entreprise")
    
    # Association de ChromaDB à LlamaIndex
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("5. Calcul des embeddings et enregistrement des vecteurs...")
    # Cette étape calcule les vecteurs pour chaque chunk et les insère dans ChromaDB
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True
    )

    print(" ✅ Ingestion terminée avec succès ! La base ChromaDB est remplie.")
    return index

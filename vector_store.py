"""
vector_store.py — ChromaDB wrapper for the CAM document processing pipeline.
Uses google-genai SDK directly for embeddings since the old google.generativeai
package is deprecated and no longer compatible with chromadb.
"""

import chromadb
import google.genai as genai

from config import CHROMA_PATH, COLLECTION_NAME


def init_vector_store(api_key: str):
    """Initialise ChromaDB with a persistent client and Gemini embedding model.
    
    Creates a custom embedding function using the new google-genai SDK
    since ChromaDB's built-in Google embedding function is deprecated.
    Returns (collection, embedding_fn) tuple.
    """
    # Connect to on-disk ChromaDB — data persists across runs
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Create a google-genai client for direct API access
    genai_client = genai.Client(api_key=api_key)

    # Custom embedding function wrapping the new google-genai SDK.
    # ChromaDB calls this automatically on add() and query().
    class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
        def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
            embeddings = []
            for text in input:
                # Call Gemini embedding API for each text chunk
                result = genai_client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=text,
                )
                # Extract the embedding vector from the response
                embeddings.append(result.embeddings[0].values)
            return embeddings

    embedding_fn = GeminiEmbeddingFunction()

    # Get existing collection or create new one with our embedding function
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    return (collection, embedding_fn)


def add_chunks(collection, chunks: list[dict]) -> None:
    """Insert a list of chunk dicts into the ChromaDB collection."""
    if not chunks:
        return

    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        documents.append(chunk["text"])

        metadatas.append({
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk["page_number"],
            "page_type": chunk["page_type"],
            "statement_type": chunk["statement_type"],
            "company_name": chunk["company_name"],
            "fiscal_year": chunk["fiscal_year"],
            "document_type": chunk["document_type"],
        })

        sanitized_name = chunk["company_name"].replace(" ", "_").lower()
        chunk_id = (
            f"{sanitized_name}"
            f"_{chunk['document_type']}"
            f"_{chunk['fiscal_year']}"
            f"_p{chunk['page_number']}"
            f"_c{chunk['chunk_index']}"
        )
        ids.append(chunk_id)

    collection.add(documents=documents, metadatas=metadatas, ids=ids)


def query_chunks(
    collection,
    query_text: str,
    filters: dict = None,
    n_results: int = 5,
) -> list[dict]:
    """Semantic similarity search against the ChromaDB collection."""
    kwargs = {
        "query_texts": [query_text],
        "n_results": n_results,
    }

    if filters:
        kwargs["where"] = filters

    results = collection.query(**kwargs)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    output = []
    for doc, meta, dist in zip(docs, metas, dists):
        output.append({
            "text": doc,
            "metadata": meta,
            "distance": dist,
        })

    return output


def delete_company_docs(
    collection,
    company_name: str,
    document_type: str,
) -> None:
    """Delete all chunks for a given company and document type."""
    collection.delete(
        where={
            "company_name": company_name,
            "document_type": document_type,
        }
    )
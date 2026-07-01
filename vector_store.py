"""
vector_store.py — ChromaDB wrapper for the CAM document processing pipeline.

Handles all vector store operations: initialisation, chunk insertion,
semantic querying, and document deletion. No LLM calls, no PDF processing,
no database calls — pure ChromaDB interaction.
"""

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PATH, COLLECTION_NAME


def init_vector_store(api_key: str):
    """Initialise ChromaDB with a persistent client and Google embedding model.

    Creates or connects to an on-disk ChromaDB database at CHROMA_PATH,
    sets up Google's text-embedding-004 as the embedding function, and
    returns the collection plus the embedding function as a tuple.
    """
    # PersistentClient stores all data on disk at the given path so
    # embeddings survive across process restarts without re-ingestion.
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Google's text-embedding-004 converts text into 768-dimensional
    # vectors optimised for semantic similarity search. The api_key
    # authenticates against the Gemini API for embedding generation.
    embedding_fn = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
        api_key=api_key,
        model_name="models/text-embedding-004",
    )

    # get_or_create_collection either opens an existing collection or
    # creates a new one. The embedding function is attached so that
    # ChromaDB automatically embeds documents on add() and queries on query().
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    return (collection, embedding_fn)


def add_chunks(collection, chunks: list[dict]) -> None:
    """Insert a list of chunk dicts into the ChromaDB collection.

    Builds the three parallel lists ChromaDB requires (documents, metadatas,
    ids), generates a deterministic unique ID per chunk from its metadata,
    and calls collection.add() in a single batch operation.
    """
    # Nothing to do if the ingestion pipeline produced no chunks
    # (e.g. a blank PDF or all pages were classified as "skip").
    if not chunks:
        return

    # ChromaDB.add() expects three parallel lists of equal length:
    # - documents: the raw text strings to embed and store
    # - metadatas: dicts of filterable fields attached to each document
    # - ids: unique string identifiers for deduplication and retrieval
    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        # The text field is stored as a ChromaDB "document" (the content
        # that gets embedded). Everything else becomes searchable metadata.
        documents.append(chunk["text"])

        # Build the metadata dict with every field except "text", since
        # text is already stored as the document itself and duplicating
        # it in metadata would waste storage.
        metadatas.append({
            "chunk_index": chunk["chunk_index"],
            "page_number": chunk["page_number"],
            "page_type": chunk["page_type"],
            "statement_type": chunk["statement_type"],
            "company_name": chunk["company_name"],
            "fiscal_year": chunk["fiscal_year"],
            "document_type": chunk["document_type"],
        })

        # Deterministic ID format ensures the same chunk always gets the
        # same ID, preventing duplicates on re-ingestion. Spaces in the
        # company name are replaced with underscores and lowercased so
        # the ID is filesystem-safe and consistent.
        sanitized_name = chunk["company_name"].replace(" ", "_").lower()
        chunk_id = (
            f"{sanitized_name}"
            f"_{chunk['document_type']}"
            f"_{chunk['fiscal_year']}"
            f"_p{chunk['page_number']}"
            f"_c{chunk['chunk_index']}"
        )
        ids.append(chunk_id)

    # Single batch add — ChromaDB embeds all documents in one API call
    # to the Google embedding model and stores them alongside metadata.
    collection.add(documents=documents, metadatas=metadatas, ids=ids)


def query_chunks(
    collection,
    query_text: str,
    filters: dict = None,
    n_results: int = 5,
) -> list[dict]:
    """Perform a semantic similarity search against the ChromaDB collection.

    Embeds the query_text using the same Google model, finds the closest
    n_results chunks by vector distance, and optionally filters by metadata
    fields (e.g. company_name, statement_type). Returns a list of result
    dicts each containing the matched text, its metadata, and distance score.
    """
    # Build the query keyword arguments. ChromaDB expects query_texts
    # as a list because it supports batched queries, but we always
    # send a single query at a time in this pipeline.
    kwargs = {
        "query_texts": [query_text],
        "n_results": n_results,
    }

    # The "where" clause lets us pre-filter by metadata before the
    # similarity search runs. For example, restricting results to
    # standalone statements or a specific company avoids irrelevant
    # matches from other documents in the same collection.
    if filters:
        kwargs["where"] = filters

    # Execute the semantic search — ChromaDB embeds the query text,
    # computes cosine distances against all stored embeddings (post-filter),
    # and returns the top n_results closest matches.
    results = collection.query(**kwargs)

    # ChromaDB returns nested lists because it supports batched queries.
    # Since we sent exactly one query, our results are at index [0].
    # Each of these three lists has n_results elements, aligned by position.
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    # Zip the three parallel lists into a single list of result dicts
    # so the caller gets one clean object per match with text, metadata,
    # and distance bundled together.
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
    """Delete all chunks for a given company and document type from ChromaDB.

    Used before re-ingestion to prevent duplicate chunks. The where clause
    matches on both company_name and document_type so that deleting an
    annual report does not accidentally remove audit report chunks.
    """
    # ChromaDB's delete with a "where" clause removes every document
    # whose metadata matches the filter. This is an atomic operation
    # that clears all chunks for the specified company + document type
    # pair, making the collection ready for a fresh re-ingestion.
    collection.delete(
        where={
            "company_name": company_name,
            "document_type": document_type,
        }
    )

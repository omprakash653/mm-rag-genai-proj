from __future__ import annotations

import os
import sys

from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_qdrant import (
    QdrantVectorStore,
    RetrievalMode,
)

from qdrant_client import (
    QdrantClient,
    models,
)


# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(
    PROJECT_ROOT / ".env"
)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from exception.custom_exception import (
    DocumentPortalException,
)

from logger.custom_logger import (
    CustomLogger,
)


logger = CustomLogger().get_logger(
    __name__
)


# ============================================================
# MULTIMODAL QDRANT RETRIEVER
# ============================================================

class MultimodalQdrantRetriever:
    """
    Retrieve documents from an existing Qdrant collection.

    Embedding model:
        all-MiniLM-L6-v2

    Embedding dimension:
        384

    Vector DB:
        Qdrant

    Supported content types:
        - page_text_plus_ocr
        - table
        - image
    """

    VALID_CONTENT_TYPES = {
        "page_text_plus_ocr",
        "text",
        "table",
        "image",
    }

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        collection_name: str | None = None,
    ) -> None:

        try:

            # ==================================================
            # COLLECTION
            # ==================================================

            self.collection_name = (

                collection_name

                or os.getenv(
                    "QDRANT_COLLECTION_NAME"
                )

                or "mm-rag-documents"
            )

            # ==================================================
            # EMBEDDING MODEL
            # ==================================================

            self.embedding_model = (

                os.getenv(
                    "EMBEDDING_MODEL"
                )

                or "all-MiniLM-L6-v2"
            )

            # ==================================================
            # EMBEDDING DIMENSION
            # ==================================================

            self.embedding_dimension = int(

                os.getenv(
                    "EMBEDDING_DIMENSION"
                )

                or 384
            )

            # ==================================================
            # QDRANT URL
            # ==================================================

            self.qdrant_url = (

                os.getenv(
                    "QDRANT_Cluster_Endpoint"
                )

                or os.getenv(
                    "QDRANT_URL"
                )
            )

            # ==================================================
            # QDRANT API KEY
            # ==================================================

            self.qdrant_api_key = os.getenv(
                "QDRANT_API_KEY"
            )

            # ==================================================
            # VALIDATE
            # ==================================================

            self._validate_config()

            # ==================================================
            # QDRANT CLIENT
            # ==================================================

            self.client = QdrantClient(

                url=self.qdrant_url,

                api_key=self.qdrant_api_key,
            )

            # ==================================================
            # CHECK COLLECTION
            # ==================================================

            self._validate_collection()

            # ==================================================
            # HUGGING FACE EMBEDDINGS
            # ==================================================

            logger.info(
                "loading_embedding_model",
                model=self.embedding_model,
                dimension=self.embedding_dimension,
            )

            self.embeddings = HuggingFaceEmbeddings(

                model_name=self.embedding_model,

                model_kwargs={
                    "device": "cpu"
                },

                encode_kwargs={
                    "normalize_embeddings": True
                },
            )

            # ==================================================
            # QDRANT VECTOR STORE
            # ==================================================

            self.vector_store = QdrantVectorStore(

                client=self.client,

                collection_name=self.collection_name,

                embedding=self.embeddings,

                retrieval_mode=RetrievalMode.DENSE,
            )

            logger.info(

                "retriever_initialized",

                collection_name=self.collection_name,

                embedding_model=self.embedding_model,

                embedding_dimension=self.embedding_dimension,
            )

        except Exception as exc:

            logger.exception(

                "retriever_initialization_failed",

                error=str(exc),
            )

            raise DocumentPortalException(

                "Failed to initialize Qdrant retriever",

                exc,

            ) from exc

    # ============================================================
    # CONFIGURATION VALIDATION
    # ============================================================

    def _validate_config(self) -> None:

        if not self.qdrant_url:

            raise ValueError(
                "QDRANT_Cluster_Endpoint is missing"
            )

        if not self.qdrant_api_key:

            raise ValueError(
                "QDRANT_API_KEY is missing"
            )

        if self.embedding_dimension <= 0:

            raise ValueError(
                "EMBEDDING_DIMENSION must be positive"
            )

    # ============================================================
    # COLLECTION VALIDATION
    # ============================================================

    def _validate_collection(self) -> None:

        """
        Retrieval uses an existing collection.

        It does not create a collection.
        """

        if not self.client.collection_exists(
            self.collection_name
        ):

            raise ValueError(

                "Qdrant collection does not exist: "

                f"{self.collection_name}"
            )

        # ==================================================
        # CHECK VECTOR DIMENSION
        # ==================================================

        collection_info = (
            self.client.get_collection(
                self.collection_name
            )
        )

        vectors_config = (
            collection_info.config.params.vectors
        )

        # Handle normal single-vector collection
        if hasattr(vectors_config, "size"):

            qdrant_dimension = (
                vectors_config.size
            )

            if (
                qdrant_dimension
                != self.embedding_dimension
            ):

                raise ValueError(

                    "Embedding dimension mismatch. "

                    f"Qdrant collection has "
                    f"{qdrant_dimension} dimensions, "

                    f"but your embedding model "
                    f"uses {self.embedding_dimension}."
                )

    # ============================================================
    # NORMALIZE CONTENT TYPES
    # ============================================================

    @classmethod
    def _normalize_content_types(

        cls,

        content_types: Iterable[str] | None,

    ) -> list[str]:

        if content_types is None:

            return []

        normalized = [

            str(content_type).strip()

            for content_type in content_types

            if str(content_type).strip()
        ]

        invalid = sorted(

            set(normalized)
            - cls.VALID_CONTENT_TYPES
        )

        if invalid:

            raise ValueError(

                "Unsupported content_types: "

                f"{invalid}. "

                "Valid values are: "

                f"{sorted(cls.VALID_CONTENT_TYPES)}"
            )

        return normalized

    # ============================================================
    # BUILD QDRANT FILTER
    # ============================================================

    def build_filter(

        self,

        *,

        filename: str | None = None,

        document_id: str | None = None,

        content_types: Iterable[str] | None = None,

        page_number: int | None = None,

        page_from: int | None = None,

        page_to: int | None = None,

    ) -> models.Filter | None:

        must_conditions: list[Any] = []

        # ==================================================
        # FILENAME
        # ==================================================

        if filename:

            must_conditions.append(

                models.FieldCondition(

                    key="metadata.filename",

                    match=models.MatchValue(

                        value=Path(
                            filename
                        ).name
                    ),
                )
            )

        # ==================================================
        # DOCUMENT ID
        # ==================================================

        if document_id:

            must_conditions.append(

                models.FieldCondition(

                    key="metadata.document_id",

                    match=models.MatchValue(

                        value=document_id
                    ),
                )
            )

        # ==================================================
        # CONTENT TYPE
        # ==================================================

        normalized_types = (
            self._normalize_content_types(
                content_types
            )
        )

        if normalized_types:

            must_conditions.append(

                models.FieldCondition(

                    key="metadata.content_type",

                    match=models.MatchAny(

                        any=normalized_types
                    ),
                )
            )

        # ==================================================
        # EXACT PAGE
        # ==================================================

        if page_number is not None:

            if page_number <= 0:

                raise ValueError(
                    "page_number must be greater than 0"
                )

            must_conditions.append(

                models.FieldCondition(

                    key="metadata.page_number",

                    match=models.MatchValue(

                        value=page_number
                    ),
                )
            )

        # ==================================================
        # PAGE RANGE
        # ==================================================

        elif (
            page_from is not None
            or page_to is not None
        ):

            if (
                page_from is not None
                and page_from <= 0
            ):

                raise ValueError(
                    "page_from must be greater than 0"
                )

            if (
                page_to is not None
                and page_to <= 0
            ):

                raise ValueError(
                    "page_to must be greater than 0"
                )

            if (
                page_from is not None
                and page_to is not None
                and page_from > page_to
            ):

                raise ValueError(
                    "page_from cannot be greater than page_to"
                )

            must_conditions.append(

                models.FieldCondition(

                    key="metadata.page_number",

                    range=models.Range(

                        gte=page_from,

                        lte=page_to,
                    ),
                )
            )

        # ==================================================
        # RETURN FILTER
        # ==================================================

        if not must_conditions:

            return None

        return models.Filter(
            must=must_conditions
        )

    # ============================================================
    # DENSE RETRIEVAL
    # ============================================================

    def retrieve(

        self,

        query: str,

        *,

        k: int = 5,

        filename: str | None = None,

        document_id: str | None = None,

        content_types: Iterable[str] | None = None,

        page_number: int | None = None,

        page_from: int | None = None,

        page_to: int | None = None,

    ) -> list[Document]:

        try:

            query = query.strip()

            if not query:

                raise ValueError(
                    "Query cannot be empty"
                )

            if k <= 0:

                raise ValueError(
                    "k must be greater than 0"
                )

            query_filter = self.build_filter(

                filename=filename,

                document_id=document_id,

                content_types=content_types,

                page_number=page_number,

                page_from=page_from,

                page_to=page_to,
            )

            # ==================================================
            # SIMILARITY SEARCH
            # ==================================================

            documents = (

                self.vector_store
                .similarity_search(

                    query=query,

                    k=k,

                    filter=query_filter,
                )
            )

            logger.info(

                "documents_retrieved",

                query=query,

                k=k,

                result_count=len(
                    documents
                ),
            )

            return documents

        except Exception as exc:

            logger.exception(

                "document_retrieval_failed",

                query=query,

                error=str(exc),
            )

            raise DocumentPortalException(

                "Failed to retrieve documents from Qdrant",

                exc,

            ) from exc

    # ============================================================
    # RETRIEVE WITH SCORES
    # ============================================================

    def retrieve_with_scores(

        self,

        query: str,

        *,

        k: int = 5,

        filename: str | None = None,

        document_id: str | None = None,

        content_types: Iterable[str] | None = None,

        page_number: int | None = None,

        page_from: int | None = None,

        page_to: int | None = None,

    ) -> list[
        tuple[Document, float]
    ]:

        try:

            query = query.strip()

            if not query:

                raise ValueError(
                    "Query cannot be empty"
                )

            if k <= 0:

                raise ValueError(
                    "k must be greater than 0"
                )

            query_filter = self.build_filter(

                filename=filename,

                document_id=document_id,

                content_types=content_types,

                page_number=page_number,

                page_from=page_from,

                page_to=page_to,
            )

            results = (

                self.vector_store
                .similarity_search_with_score(

                    query=query,

                    k=k,

                    filter=query_filter,
                )
            )

            logger.info(

                "documents_retrieved_with_scores",

                query=query,

                k=k,

                result_count=len(
                    results
                ),
            )

            return results

        except Exception as exc:

            logger.exception(

                "scored_document_retrieval_failed",

                query=query,

                error=str(exc),
            )

            raise DocumentPortalException(

                "Failed to retrieve scored documents from Qdrant",

                exc,

            ) from exc

    # ============================================================
    # LANGCHAIN RETRIEVER
    # ============================================================

    def as_langchain_retriever(

        self,

        *,

        search_type: str = "similarity",

        k: int = 5,

        fetch_k: int = 20,

        lambda_mult: float = 0.5,

        filename: str | None = None,

        document_id: str | None = None,

        content_types: Iterable[str] | None = None,

        page_number: int | None = None,

        page_from: int | None = None,

        page_to: int | None = None,

    ) -> VectorStoreRetriever:

        if search_type not in {
            "similarity",
            "mmr",
        }:

            raise ValueError(

                "search_type must be "
                "'similarity' or 'mmr'"
            )

        if k <= 0:

            raise ValueError(
                "k must be greater than 0"
            )

        query_filter = self.build_filter(

            filename=filename,

            document_id=document_id,

            content_types=content_types,

            page_number=page_number,

            page_from=page_from,

            page_to=page_to,
        )

        search_kwargs: dict[
            str, Any
        ] = {

            "k": k
        }

        if query_filter is not None:

            search_kwargs[
                "filter"
            ] = query_filter

        # ==================================================
        # MMR
        # ==================================================

        if search_type == "mmr":

            if fetch_k < k:

                raise ValueError(

                    "fetch_k must be "
                    "greater than or equal to k"
                )

            if not (
                0.0
                <= lambda_mult
                <= 1.0
            ):

                raise ValueError(

                    "lambda_mult must "
                    "be between 0 and 1"
                )

            search_kwargs.update(

                {

                    "fetch_k":
                        fetch_k,

                    "lambda_mult":
                        lambda_mult,
                }
            )

        # ==================================================
        # RETURN LANGCHAIN RETRIEVER
        # ==================================================

        return self.vector_store.as_retriever(

            search_type=search_type,

            search_kwargs=search_kwargs,
        )

    # ============================================================
    # FORMAT RESULTS
    # ============================================================

    @staticmethod
    def format_results(

        results: list[
            tuple[Document, float]
        ],

        *,

        content_preview_chars: int = 700,

    ) -> str:

        lines: list[str] = []

        for rank, (
            document,
            score
        ) in enumerate(
            results,
            start=1
        ):

            metadata = (
                document.metadata
            )

            content = (
                document.page_content
                .strip()
            )

            if len(content) > (
                content_preview_chars
            ):

                content = (

                    content[
                        :content_preview_chars
                    ]

                    + "..."
                )

            lines.extend(

                [

                    f"\n{'=' * 80}",

                    f"Rank: {rank}",

                    f"Score: {score:.4f}",

                    (
                        "Type: "
                        f"{metadata.get(
                            'content_type',
                            'unknown'
                        )}"
                    ),

                    (
                        "Page: "
                        f"{metadata.get(
                            'page_number',
                            'unknown'
                        )}"
                    ),

                    (
                        "File: "
                        f"{metadata.get(
                            'filename',
                            'unknown'
                        )}"
                    ),

                    f"Content:\n{content}",
                ]
            )

            image_path = (
                metadata.get(
                    "image_path"
                )
            )

            if image_path:

                lines.append(
                    f"Image path: {image_path}"
                )

        return "\n".join(lines)

    # ============================================================
    # COLLECTION STATUS
    # ============================================================

    def collection_status(
        self
    ) -> dict[str, Any]:

        try:

            info = (
                self.client.get_collection(
                    self.collection_name
                )
            )

            return {

                "collection_name":
                    self.collection_name,

                "points_count":
                    getattr(
                        info,
                        "points_count",
                        None
                    ),

                "vectors_count":
                    getattr(
                        info,
                        "vectors_count",
                        None
                    ),

                "status":
                    str(
                        getattr(
                            info,
                            "status",
                            "unknown"
                        )
                    ),

                "embedding_model":
                    self.embedding_model,

                "embedding_dimension":
                    self.embedding_dimension,
            }

        except Exception as exc:

            raise DocumentPortalException(

                "Failed to read Qdrant collection status",

                exc,

            ) from exc


# ============================================================
# MANUAL RETRIEVAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = (
        MultimodalQdrantRetriever()
    )

    print(
        "\nCollection status:"
    )

    print(
        retriever.collection_status()
    )

    print(
        "\nEmbedding model:",
        retriever.embedding_model
    )

    print(
        "Embedding dimension:",
        retriever.embedding_dimension
    )

    # ========================================================
    # TEST QUERY
    # ========================================================

    query = (
        "What are the important "
        "policies and incident details?"
    )

    results = (
        retriever.retrieve_with_scores(

            query=query,

            k=5,
        )
    )

    print(
        retriever.format_results(
            results
        )
    )
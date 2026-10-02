from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger

from prompt_library.prompt import (
    GENERATION_SMOKE_TEST_QUERY,
    MULTIMODAL_RAG_SYSTEM_PROMPT,
    format_multimodal_rag_user_prompt,
)

from src.retriever import MultimodalQdrantRetriever


logger = CustomLogger().get_logger(__name__)


# =====================================================================
# Multimodal RAG Generator
# =====================================================================

class MultimodalRAGGenerator:
    """
    Retrieve relevant content from Qdrant and generate
    a grounded answer using Groq.

    Architecture:

        Query
          ↓
        Qdrant Retriever
          ↓
        Relevant Documents
          ↓
        Context
          ↓
        Groq GPT-OSS-120B
          ↓
        Grounded Answer
    """

    def __init__(
        self,
        collection_name: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        max_context_chars: int = 30_000,
        max_images: int = 0,
    ) -> None:

        try:

            # ---------------------------------------------------------
            # Groq configuration
            # ---------------------------------------------------------

            self.model_name = (
                model_name
                or os.getenv("GROQ_MODEL")
                or "openai/gpt-oss-120b"
            )

            self.temperature = temperature
            self.max_context_chars = max_context_chars
            self.max_images = max(0, int(max_images))

            # ---------------------------------------------------------
            # Validate configuration
            # ---------------------------------------------------------

            self._validate_config()

            # ---------------------------------------------------------
            # Retriever
            # ---------------------------------------------------------

            self.retriever = MultimodalQdrantRetriever(
                collection_name=collection_name
            )

            # ---------------------------------------------------------
            # Groq LLM
            # ---------------------------------------------------------

            self.llm = ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"),
                model=self.model_name,
                temperature=self.temperature,
            )

            logger.info(
                "generation_initialized",
                model_name=self.model_name,
                collection_name=self.retriever.collection_name,
            )

        except Exception as exc:

            logger.exception(
                "generation_initialization_failed",
                error=str(exc),
            )

            raise DocumentPortalException(
                "Failed to initialize multimodal generation pipeline",
                exc,
            ) from exc

    # =================================================================
    # Configuration
    # =================================================================

    def _validate_config(self) -> None:

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is missing from .env"
            )

        if self.temperature < 0:
            raise ValueError(
                "temperature cannot be negative"
            )

        if self.max_context_chars <= 0:
            raise ValueError(
                "max_context_chars must be greater than 0"
            )

    # =================================================================
    # Source Label
    # =================================================================

    @staticmethod
    def _source_label(
        *,
        filename: str,
        page_number: Any,
    ) -> str:

        if page_number in {None, "", -1, "-1", "unknown"}:
            return f"[{filename}]"

        return f"[{filename} p.{page_number}]"

    # =================================================================
    # Prepare Retrieval Context
    # =================================================================

    def _prepare_context(
        self,
        results: list[tuple[Any, float]],
    ) -> tuple[
        str,
        list[dict[str, Any]],
    ]:

        context_sections: list[str] = []

        sources: list[dict[str, Any]] = []

        used_chars = 0

        for rank, (document, score) in enumerate(
            results,
            start=1,
        ):

            metadata = dict(document.metadata)

            filename = str(
                metadata.get("filename")
                or Path(
                    str(
                        metadata.get(
                            "source",
                            "unknown",
                        )
                    )
                ).name
                or "unknown"
            )

            page_number = metadata.get(
                "page_number",
                "unknown",
            )

            content_type = str(
                metadata.get(
                    "content_type",
                    "unknown",
                )
            )

            source_label = self._source_label(
                filename=filename,
                page_number=page_number,
            )

            content = document.page_content.strip()

            if not content:
                continue

            # ---------------------------------------------------------
            # Source header
            # ---------------------------------------------------------

            header = (
                f"SOURCE {rank}\n"
                f"Citation: {source_label}\n"
                f"File: {filename}\n"
                f"Page: {page_number}\n"
                f"Content type: {content_type}\n"
                f"Retrieval score: {score:.4f}\n"
            )

            remaining = (
                self.max_context_chars
                - used_chars
                - len(header)
            )

            if remaining <= 0:
                break

            # ---------------------------------------------------------
            # Context truncation
            # ---------------------------------------------------------

            if len(content) > remaining:

                content = (
                    content[:remaining]
                    + "\n[Context truncated]"
                )

            section = (
                f"{header}"
                f"Content:\n{content}"
            )

            context_sections.append(section)

            used_chars += len(section)

            # ---------------------------------------------------------
            # Source metadata
            # ---------------------------------------------------------

            source_record = {
                "rank": rank,
                "score": float(score),
                "filename": filename,
                "page_number": page_number,
                "content_type": content_type,
                "citation": source_label,
                "document_id": metadata.get(
                    "document_id"
                ),
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "chunk_index": metadata.get(
                    "chunk_index"
                ),
                "table_index": metadata.get(
                    "table_index"
                ),
                "image_index": metadata.get(
                    "image_index"
                ),
                "image_path": metadata.get(
                    "image_path"
                ),
            }

            sources.append(source_record)

        context = "\n\n---\n\n".join(
            context_sections
        )

        return context, sources

    # =================================================================
    # Build Prompt
    # =================================================================

    def _build_messages(
        self,
        *,
        query: str,
        context: str,
    ) -> list[Any]:

        user_prompt = format_multimodal_rag_user_prompt(
            query=query,
            context=context,
        )

        messages = [
            SystemMessage(
                content=MULTIMODAL_RAG_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=user_prompt
            ),
        ]

        return messages

    # =================================================================
    # Response Helpers
    # =================================================================

    @staticmethod
    def _extract_response_text(
        response: Any,
    ) -> str:

        # LangChain AIMessage.text
        text = getattr(
            response,
            "text",
            None,
        )

        if isinstance(text, str) and text.strip():
            return text.strip()

        # AIMessage.content
        content = getattr(
            response,
            "content",
            "",
        )

        if isinstance(content, str):
            return content.strip()

        # Structured content
        if isinstance(content, list):

            text_parts: list[str] = []

            for block in content:

                if isinstance(block, str):
                    text_parts.append(block)
                    continue

                if not isinstance(block, dict):
                    continue

                value = (
                    block.get("text")
                    or block.get("content")
                )

                if isinstance(value, str):
                    text_parts.append(value)

            return "\n".join(
                part
                for part in text_parts
                if part
            ).strip()

        return str(content).strip()

    # =================================================================
    # Usage Metadata
    # =================================================================

    @staticmethod
    def _usage_metadata(
        response: Any,
    ) -> dict[str, Any] | None:

        usage = getattr(
            response,
            "usage_metadata",
            None,
        )

        if isinstance(usage, dict):
            return usage

        return None

    # =================================================================
    # Main RAG Generation
    # =================================================================

    def generate(
        self,
        query: str,
        *,
        k: int = 6,
        min_score: float | None = None,
        filename: str | None = None,
        document_id: str | None = None,
        content_types: Iterable[str] | None = None,
        page_number: int | None = None,
        page_from: int | None = None,
        page_to: int | None = None,
    ) -> dict[str, Any]:

        try:

            # ---------------------------------------------------------
            # Validate query
            # ---------------------------------------------------------

            query = query.strip()

            if not query:
                raise ValueError(
                    "Query cannot be empty"
                )

            if k <= 0:
                raise ValueError(
                    "k must be greater than 0"
                )

            # ---------------------------------------------------------
            # Retrieve documents
            # ---------------------------------------------------------

            results = (
                self.retriever.retrieve_with_scores(
                    query=query,
                    k=k,
                    filename=filename,
                    document_id=document_id,
                    content_types=content_types,
                    page_number=page_number,
                    page_from=page_from,
                    page_to=page_to,
                )
            )

            # ---------------------------------------------------------
            # Minimum similarity score
            # ---------------------------------------------------------

            if min_score is not None:

                results = [
                    (document, score)
                    for document, score in results
                    if score >= min_score
                ]

            # ---------------------------------------------------------
            # No relevant documents
            # ---------------------------------------------------------

            if not results:

                logger.info(
                    "generation_no_retrieval_results",
                    query=query,
                )

                return {
                    "query": query,
                    "answer": (
                        "I could not find enough relevant "
                        "information in the indexed documents "
                        "to answer this question."
                    ),
                    "sources": [],
                    "retrieval_count": 0,
                    "model_name": self.model_name,
                    "collection_name": (
                        self.retriever.collection_name
                    ),
                    "usage": None,
                }

            # ---------------------------------------------------------
            # Prepare context
            # ---------------------------------------------------------

            context, sources = (
                self._prepare_context(
                    results
                )
            )

            # ---------------------------------------------------------
            # Build messages
            # ---------------------------------------------------------

            messages = self._build_messages(
                query=query,
                context=context,
            )

            logger.info(
                "generation_started",
                query=query,
                retrieved_documents=len(
                    results
                ),
                model_name=self.model_name,
            )

            # ---------------------------------------------------------
            # Call Groq
            # ---------------------------------------------------------

            response = self.llm.invoke(
                messages
            )

            # ---------------------------------------------------------
            # Extract answer
            # ---------------------------------------------------------

            answer = (
                self._extract_response_text(
                    response
                )
            )

            # ---------------------------------------------------------
            # Final result
            # ---------------------------------------------------------

            # ---------------------------------------------------------
            # Retrieved image metadata for UI compatibility
            #
            # The current Groq generation pipeline sends OCR/text
            # context to the LLM. Images are exposed to the Streamlit
            # UI as retrieved evidence, but are not sent as native
            # image inputs to the model.
            # ---------------------------------------------------------

            used_images = [
                source
                for source in sources
                if (
                    str(source.get("content_type", "")).lower()
                    == "image"
                    and source.get("image_path")
                )
            ][: self.max_images]

            result = {
                "query": query,
                "answer": answer,
                "sources": sources,
                "used_images": used_images,
                "retrieval_count": len(
                    results
                ),
                "model_name": self.model_name,
                "collection_name": (
                    self.retriever.collection_name
                ),
                "usage": self._usage_metadata(
                    response
                ),
            }

            logger.info(
                "generation_completed",
                query=query,
                retrieved_documents=len(
                    results
                ),
            )

            return result

        except Exception as exc:

            logger.exception(
                "generation_failed",
                query=query,
                error=str(exc),
            )

            raise DocumentPortalException(
                "Failed to generate MM-RAG answer",
                exc,
            ) from exc

    # =================================================================
    # Friendly Alias
    # =================================================================

    def answer_question(
        self,
        query: str,
        **kwargs: Any,
    ) -> dict[str, Any]:

        return self.generate(
            query=query,
            **kwargs,
        )


# =====================================================================
# Manual Test
# =====================================================================

if __name__ == "__main__":

    generator = MultimodalRAGGenerator()

    result = generator.answer_question(
        GENERATION_SMOKE_TEST_QUERY,
        k=6,
    )

    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(result["answer"])

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    for source in result["sources"]:

        print(
            source["citation"],
            "| type:",
            source["content_type"],
            "| score:",
            round(
                source["score"],
                4,
            ),
        )

    print("\n" + "=" * 70)
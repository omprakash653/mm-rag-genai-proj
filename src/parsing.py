import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf as fitz
import pandas as pd
import pdfplumber
import pytesseract

from PIL import Image
from langchain_core.documents import Document


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger


logger = CustomLogger().get_logger(__name__)


class ComplexPDFParser:
    """
    Multimodal PDF parser for RAG.

    Handles:

    1. Normal/selectable PDF text
    2. Scanned PDF pages
    3. OCR text
    4. Embedded images
    5. OCR on embedded images
    6. Tables

    Output:
        LangChain Document objects containing:
        - page text
        - OCR text
        - tables
        - image OCR
        - useful metadata
    """

    def __init__(
        self,
        pdf_path: str,
        output_dir: str,
        tesseract_path: Optional[str] = None,
        render_scale: float = 2.0,
    ):

        self.pdf_path = Path(pdf_path)

        self.output_dir = Path(output_dir)

        self.image_dir = (
            self.output_dir / "extracted_images"
        )

        self.page_image_dir = (
            self.output_dir / "page_images"
        )

        self.render_scale = render_scale

        # Parsed records
        self.page_records: List[Dict[str, Any]] = []

        self.image_records: List[Dict[str, Any]] = []

        self.table_records: List[Dict[str, Any]] = []

        self.langchain_docs: List[Document] = []

        try:

            self._validate_pdf()

            self._create_output_directories()

            self._configure_tesseract(
                tesseract_path
            )

            logger.info(
                "pdf_parser_initialized",
                pdf_path=str(self.pdf_path),
                output_dir=str(self.output_dir),
            )

        except Exception as exc:

            logger.exception(
                "pdf_parser_initialization_failed",
                pdf_path=str(self.pdf_path),
                error=str(exc),
            )

            raise DocumentPortalException(
                "Failed to initialize PDF parser",
                exc,
            ) from exc

    # ============================================================
    # VALIDATE PDF
    # ============================================================

    def _validate_pdf(self) -> None:

        if not self.pdf_path.exists():

            raise FileNotFoundError(
                f"PDF file not found: {self.pdf_path}"
            )

        if self.pdf_path.suffix.lower() != ".pdf":

            raise ValueError(
                f"Expected PDF file, got: {self.pdf_path.suffix}"
            )

    # ============================================================
    # CREATE DIRECTORIES
    # ============================================================

    def _create_output_directories(self) -> None:

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.page_image_dir.mkdir(
            parents=True,
            exist_ok=True
        )

    # ============================================================
    # CONFIGURE TESSERACT
    # ============================================================

    def _configure_tesseract(
        self,
        tesseract_path: Optional[str]
    ) -> None:

        if not tesseract_path:

            logger.info(
                "tesseract_path_not_provided_using_system_path"
            )

            return

        path = Path(tesseract_path)

        if not path.exists():

            raise FileNotFoundError(
                f"Tesseract executable not found: {path}"
            )

        pytesseract.pytesseract.tesseract_cmd = str(
            path
        )

        logger.info(
            "tesseract_configured",
            path=str(path),
        )

    # ============================================================
    # OCR
    # ============================================================

    def run_ocr_on_image(
        self,
        image_path: str | Path
    ) -> str:

        try:

            with Image.open(image_path) as image:

                # Convert to RGB
                if image.mode != "RGB":
                    image = image.convert("RGB")

                text = pytesseract.image_to_string(
                    image
                )

            return text.strip()

        except Exception as exc:

            logger.warning(
                "image_ocr_failed",
                image_path=str(image_path),
                error=str(exc),
            )

            return ""

    # ============================================================
    # EXTRACT PDF TEXT + IMAGES + OCR
    # ============================================================

    def extract_text_and_images(
        self,
    ) -> tuple[
        List[Dict[str, Any]],
        List[Dict[str, Any]]
    ]:

        page_records: List[
            Dict[str, Any]
        ] = []

        image_records: List[
            Dict[str, Any]
        ] = []

        pdf = fitz.open(
            str(self.pdf_path)
        )

        try:

            for page_index in range(
                len(pdf)
            ):

                page = pdf[page_index]

                page_number = (
                    page_index + 1
                )

                # ==================================================
                # SELECTABLE TEXT
                # ==================================================

                selectable_text = (
                    page.get_text("text")
                    .strip()
                )

                # ==================================================
                # EMBEDDED IMAGES
                # ==================================================

                images = page.get_images(
                    full=True
                )

                # ==================================================
                # RENDER COMPLETE PAGE
                # ==================================================

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(
                        self.render_scale,
                        self.render_scale
                    )
                )

                page_image_path = (
                    self.page_image_dir
                    / f"page_{page_number:03d}.png"
                )

                pixmap.save(
                    str(page_image_path)
                )

                # ==================================================
                # OCR COMPLETE PAGE
                # ==================================================

                page_ocr_text = (
                    self.run_ocr_on_image(
                        page_image_path
                    )
                )

                # ==================================================
                # PAGE RECORD
                # ==================================================

                page_info = {

                    "page_number":
                        page_number,

                    "text":
                        selectable_text,

                    "ocr_text":
                        page_ocr_text,

                    "image_count":
                        len(images),

                    "width":
                        page.rect.width,

                    "height":
                        page.rect.height,

                    "page_image_path":
                        str(page_image_path),
                }

                page_records.append(
                    page_info
                )

                # ==================================================
                # EXTRACT EMBEDDED IMAGES
                # ==================================================

                for image_index, image_info in enumerate(
                    images,
                    start=1
                ):

                    xref = image_info[0]

                    try:

                        base_image = (
                            pdf.extract_image(
                                xref
                            )
                        )

                        image_bytes = (
                            base_image["image"]
                        )

                        image_ext = (
                            base_image["ext"]
                        )

                        image_path = (
                            self.image_dir
                            / (
                                f"page_{page_number:03d}_"
                                f"image_{image_index}."
                                f"{image_ext}"
                            )
                        )

                        with open(
                            image_path,
                            "wb"
                        ) as file:

                            file.write(
                                image_bytes
                            )

                        image_ocr_text = (
                            self.run_ocr_on_image(
                                image_path
                            )
                        )

                        image_records.append(

                            {
                                "page_number":
                                    page_number,

                                "image_index":
                                    image_index,

                                "image_path":
                                    str(image_path),

                                "image_ext":
                                    image_ext,

                                "image_ocr_text":
                                    image_ocr_text,
                            }
                        )

                    except Exception as exc:

                        logger.warning(
                            "embedded_image_extraction_failed",
                            page_number=page_number,
                            image_index=image_index,
                            error=str(exc),
                        )

        finally:

            pdf.close()

        self.page_records = (
            page_records
        )

        self.image_records = (
            image_records
        )

        return (
            page_records,
            image_records
        )

    # ============================================================
    # EXTRACT TABLES
    # ============================================================

    def extract_tables(
        self
    ) -> List[Dict[str, Any]]:

        table_records: List[
            Dict[str, Any]
        ] = []

        with pdfplumber.open(
            str(self.pdf_path)
        ) as pdf:

            for page_index, page in enumerate(
                pdf.pages
            ):

                page_number = (
                    page_index + 1
                )

                try:

                    tables = (
                        page.extract_tables()
                        or []
                    )

                except Exception as exc:

                    logger.warning(
                        "table_extraction_failed",
                        page_number=page_number,
                        error=str(exc),
                    )

                    tables = []

                for table_index, table in enumerate(
                    tables,
                    start=1
                ):

                    if not table:
                        continue

                    cleaned_table = []

                    for row in table:

                        cleaned_row = [

                            cell.strip()
                            if isinstance(
                                cell,
                                str
                            )
                            else cell

                            for cell in row
                        ]

                        cleaned_table.append(
                            cleaned_row
                        )

                    if not cleaned_table:
                        continue

                    # ==================================================
                    # CONVERT TO DATAFRAME
                    # ==================================================

                    try:

                        dataframe = pd.DataFrame(

                            cleaned_table[1:],

                            columns=cleaned_table[0]
                        )

                    except Exception:

                        dataframe = pd.DataFrame(
                            cleaned_table
                        )

                    # ==================================================
                    # MARKDOWN
                    # ==================================================

                    try:

                        markdown = (
                            dataframe.to_markdown(
                                index=False
                            )
                        )

                    except Exception:

                        markdown = (
                            dataframe.to_string(
                                index=False
                            )
                        )

                    # ==================================================
                    # CSV
                    # ==================================================

                    csv_data = (
                        dataframe.to_csv(
                            index=False
                        )
                    )

                    table_records.append(

                        {
                            "page_number":
                                page_number,

                            "table_index":
                                table_index,

                            "raw_table":
                                cleaned_table,

                            "markdown":
                                markdown,

                            "csv":
                                csv_data,
                        }
                    )

        self.table_records = (
            table_records
        )

        return table_records

    # ============================================================
    # CREATE LANGCHAIN DOCUMENTS
    # ============================================================

    def create_langchain_documents(
        self
    ) -> List[Document]:

        documents: List[
            Document
        ] = []

        # ========================================================
        # PAGE TEXT + OCR
        # ========================================================

        for page in self.page_records:

            page_number = (
                page["page_number"]
            )

            selectable_text = (
                page.get("text", "")
                .strip()
            )

            ocr_text = (
                page.get("ocr_text", "")
                .strip()
            )

            # Don't create completely empty documents
            if not selectable_text and not ocr_text:
                continue

            combined_text = (

                f"PAGE {page_number}\n\n"

                f"SELECTABLE TEXT:\n"
                f"{selectable_text}\n\n"

                f"OCR TEXT:\n"
                f"{ocr_text}"
            )

            documents.append(

                Document(

                    page_content=combined_text,

                    metadata={

                        "source":
                            str(self.pdf_path),

                        "page_number":
                            page_number,

                        "content_type":
                            "page_text_plus_ocr",

                        "image_count":
                            page["image_count"],

                        "page_image_path":
                            page["page_image_path"],
                    }
                )
            )

        # ========================================================
        # TABLES
        # ========================================================

        for table in self.table_records:

            page_number = (
                table["page_number"]
            )

            table_index = (
                table["table_index"]
            )

            table_text = (

                f"TABLE FOUND ON PAGE "
                f"{page_number}\n\n"

                f"TABLE INDEX: "
                f"{table_index}\n\n"

                f"TABLE DATA:\n"

                f"{table['markdown']}"
            )

            documents.append(

                Document(

                    page_content=table_text,

                    metadata={

                        "source":
                            str(self.pdf_path),

                        "page_number":
                            page_number,

                        "content_type":
                            "table",

                        "table_index":
                            table_index,
                    }
                )
            )

        # ========================================================
        # IMAGE OCR
        # ========================================================

        for image in self.image_records:

            page_number = (
                image["page_number"]
            )

            image_index = (
                image["image_index"]
            )

            image_ocr_text = (
                image.get(
                    "image_ocr_text",
                    ""
                ).strip()
            )

            # Skip image if OCR produced nothing
            if not image_ocr_text:
                continue

            image_text = (

                f"IMAGE FOUND ON PAGE "
                f"{page_number}\n\n"

                f"IMAGE INDEX: "
                f"{image_index}\n\n"

                f"IMAGE OCR TEXT:\n"

                f"{image_ocr_text}"
            )

            documents.append(

                Document(

                    page_content=image_text,

                    metadata={

                        "source":
                            str(self.pdf_path),

                        "page_number":
                            page_number,

                        "content_type":
                            "image",

                        "image_index":
                            image_index,

                        "image_path":
                            image["image_path"],

                        "image_ext":
                            image["image_ext"],
                    }
                )
            )

        self.langchain_docs = (
            documents
        )

        return documents

    # ============================================================
    # SAVE JSON
    # ============================================================

    @staticmethod
    def _save_json(
        path: Path,
        data: Any
    ) -> None:

        with open(
            path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(

                data,

                file,

                indent=2,

                ensure_ascii=False,

                default=str,
            )

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================

    def save_outputs(self) -> None:

        self._save_json(

            self.output_dir
            / "page_records.json",

            self.page_records
        )

        self._save_json(

            self.output_dir
            / "image_records.json",

            self.image_records
        )

        self._save_json(

            self.output_dir
            / "table_records.json",

            self.table_records
        )

        # ========================================================
        # RAG READY DOCUMENTS
        # ========================================================

        rag_documents_path = (

            self.output_dir
            / "rag_ready_documents.md"
        )

        with open(
            rag_documents_path,
            "w",
            encoding="utf-8"
        ) as file:

            for index, document in enumerate(
                self.langchain_docs,
                start=1
            ):

                file.write(
                    f"\n\n# Document {index}\n"
                )

                file.write(
                    "\nMetadata:\n"
                )

                file.write(
                    "```json\n"
                )

                file.write(

                    json.dumps(

                        document.metadata,

                        indent=2,

                        ensure_ascii=False,

                        default=str,
                    )
                )

                file.write(
                    "\n```\n"
                )

                file.write(
                    "\nContent:\n"
                )

                file.write(
                    document.page_content
                )

                file.write(
                    "\n\n---\n"
                )

        # ========================================================
        # TABLES
        # ========================================================

        tables_path = (

            self.output_dir
            / "extracted_tables.md"
        )

        with open(
            tables_path,
            "w",
            encoding="utf-8"
        ) as file:

            for table in self.table_records:

                file.write(

                    f"\n\n## Page "
                    f"{table['page_number']} "
                    f"- Table "
                    f"{table['table_index']}\n\n"
                )

                file.write(
                    table["markdown"]
                )

                file.write(
                    "\n\n---\n"
                )

    # ============================================================
    # COMPLETE PARSING PIPELINE
    # ============================================================

    def parse(
        self,
        save_output: bool = True
    ) -> Dict[str, Any]:

        logger.info(

            "pdf_parsing_started",

            pdf_path=str(
                self.pdf_path
            )
        )

        try:

            # ==================================================
            # STEP 1: TEXT + IMAGES + OCR
            # ==================================================

            self.extract_text_and_images()

            logger.info(

                "pdf_content_extracted",

                pages=len(
                    self.page_records
                ),

                images=len(
                    self.image_records
                )
            )

            # ==================================================
            # STEP 2: TABLES
            # ==================================================

            self.extract_tables()

            logger.info(

                "pdf_tables_extracted",

                tables=len(
                    self.table_records
                )
            )

            # ==================================================
            # STEP 3: LANGCHAIN DOCUMENTS
            # ==================================================

            self.create_langchain_documents()

            logger.info(

                "langchain_documents_created",

                documents=len(
                    self.langchain_docs
                )
            )

            # ==================================================
            # STEP 4: SAVE OUTPUT
            # ==================================================

            if save_output:

                self.save_outputs()

                logger.info(

                    "parser_outputs_saved",

                    output_dir=str(
                        self.output_dir
                    )
                )

            # ==================================================
            # FINAL RESULT
            # ==================================================

            result = {

                "pages":
                    self.page_records,

                "images":
                    self.image_records,

                "tables":
                    self.table_records,

                "documents":
                    self.langchain_docs,

                "output_dir":
                    str(self.output_dir),
            }

            logger.info(
                "pdf_parsing_completed"
            )

            return result

        except DocumentPortalException:

            raise

        except Exception as exc:

            logger.exception(

                "pdf_parsing_failed",

                pdf_path=str(
                    self.pdf_path
                ),

                error=str(exc),
            )

            raise DocumentPortalException(

                f"Failed to parse PDF: "
                f"{self.pdf_path}",

                exc,
            ) from exc
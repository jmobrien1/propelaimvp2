import io
import os
import json
import shutil
import logging
import re
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

# --- FastAPI and Pydantic Imports ---
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError, Field

# --- LangChain and AI Imports ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# --- Document Processing ---
from unstructured.partition.auto import partition

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('propelai.log'),
              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# =============================================================================
# 1. APPLICATION SETUP
# =============================================================================

app = FastAPI(
    title="PropelAI MVP-Plus (Cross-Reference Engine)",
    description=
    "A production-ready RAG system for government RFP compliance matrix generation with a cross-reference resolution engine.",
    version="1.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Configuration ---
VECTOR_STORE_PATH = "vector_store"
PROJECT_DOCS_PATH = "project_documents.json"
TRACE_LOG_PATH = "trace.jsonl"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_RETRIEVAL_DOCS = 3  # Retrieve top 3 chunks per reference

# Global state
vector_store: Optional[FAISS] = None
embeddings: Optional[OpenAIEmbeddings] = None
llm: Optional[ChatOpenAI] = None

# =============================================================================
# 2. PYDANTIC MODELS
# =============================================================================


class DocumentInfo(BaseModel):
    filename: str
    status: str
    chunks_created: int
    timestamp: str


class ProjectStatus(BaseModel):
    total_documents: int
    indexed_documents: int
    total_chunks: int
    vector_store_ready: bool
    last_updated: Optional[str] = None


class ComplianceMatrixEntry(BaseModel):
    section_l_reference: str = Field(
        ..., description="Reference to Section L requirement")
    requirement: str = Field(..., description="The actual requirement text")
    section_m_reference: str = Field(
        default="TBD",
        description="Reference to Section M evaluation criteria")
    proposal_section: str = Field(
        default="TBD",
        description="Proposed section for addressing this requirement")
    proposal_theme: str = Field(
        default="TBD", description="Theme or approach for this requirement")
    notes_approach: str = Field(
        default="", description="Notes on how to approach this requirement")
    responsible_person: str = Field(
        default="TBD", description="Person responsible for this requirement")
    source_trace: List[str] = Field(
        default_factory=list,
        description="Source documents and page references")
    confidence_score: float = Field(
        default=0.0, description="Confidence in the requirement extraction")


class ComplianceMatrixResponse(BaseModel):
    compliance_matrix: List[ComplianceMatrixEntry]
    generation_metadata: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str


# =============================================================================
# 3. INITIALIZATION AND UTILITY FUNCTIONS
# =============================================================================


def initialize_services():
    """Initialize OpenAI services with error handling"""
    global embeddings, llm

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                      request_timeout=30)
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            request_timeout=60,
            max_retries=3,
            model_kwargs={"response_format": {
                "type": "json_object"
            }})
        logger.info("Successfully initialized OpenAI services")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI services: {e}")
        raise


def log_trace(trace_data: Dict[str, Any]):
    """Log structured trace data for analysis"""
    try:
        trace_entry = {"timestamp": datetime.now().isoformat(), **trace_data}
        with open(TRACE_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(trace_entry) + '\n')
    except Exception as e:
        logger.error(f"Failed to write trace log: {e}")


def load_project_documents() -> Dict[str, DocumentInfo]:
    """Load project document metadata"""
    try:
        if os.path.exists(PROJECT_DOCS_PATH):
            with open(PROJECT_DOCS_PATH, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                return {
                    filename: DocumentInfo(**doc_info)
                    for filename, doc_info in raw_data.items()
                }
        return {}
    except Exception as e:
        logger.error(f"Error loading project documents: {e}")
        return {}


def save_project_documents(docs: Dict[str, DocumentInfo]):
    """Save project document metadata"""
    try:
        serializable_docs = {
            filename: doc_info.model_dump()
            for filename, doc_info in docs.items()
        }
        with open(PROJECT_DOCS_PATH, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving project documents: {e}")


# =============================================================================
# 4. DOCUMENT PROCESSING PIPELINE
# =============================================================================


async def process_document_safe(file_name: str,
                                file_bytes: bytes) -> List[Document]:
    """Process a document with comprehensive error handling"""
    try:
        logger.info(f"Processing document with 'hi_res' strategy: {file_name}")
        elements = partition(
            file=io.BytesIO(file_bytes),
            file_filename=file_name,
            strategy=
            "hi_res",  # Use high-resolution strategy for better accuracy
            include_page_breaks=True)

        if not elements:
            logger.warning(f"No elements extracted from {file_name}")
            return []

        documents = []
        for i, element in enumerate(elements):
            try:
                content = str(element).strip()
                if content and len(content) > 10:
                    metadata = {
                        "source":
                        file_name,
                        "element_id":
                        i,
                        "element_type":
                        getattr(element, 'category', 'unknown'),
                        "page_number":
                        getattr(element.metadata, 'page_number', 0) if hasattr(
                            element, 'metadata') else 0
                    }
                    documents.append(
                        Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.warning(
                    f"Error processing element {i} in {file_name}: {e}")
                continue

        logger.info(f"Extracted {len(documents)} documents from {file_name}")
        return documents

    except Exception as e:
        logger.error(f"Error processing document {file_name}: {e}")
        return []


async def chunk_documents_safe(documents: List[Document]) -> List[Document]:
    """Chunk documents with error handling"""
    try:
        if not documents:
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""])
        chunks = text_splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(
            f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking documents: {e}")
        return []


# =============================================================================
# 5. REQUIREMENT EXTRACTION AND CONTEXT ASSEMBLY
# =============================================================================


def extract_section_l_regex(text: str) -> List[Dict[str, str]]:
    """Extract Section L requirements using regex patterns"""
    try:
        patterns = [
            r'(?:^|\n)(L\.\d+(?:\.\d+)*)\s*[:\-\.]?\s*([^\n]+(?:\n(?!\s*L\.\d+)[^\n]*)*)',
            r'(?:^|\n)(\d+\.\d+(?:\.\d+)*)\s*[:\-\.]?\s*([^\n]+(?:\n(?!\s*\d+\.\d+)[^\n]*)*)',
            r'(?:^|\n)([A-Z]\.\d+(?:\.\d+)*)\s*[:\-\.]?\s*([^\n]+(?:\n(?!\s*[A-Z]\.\d+)[^\n]*)*)'
        ]

        requirements = []
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                ref, req_text = match.groups()
                req_text = re.sub(r'\s+', ' ', req_text.strip())
                if len(req_text) > 20:
                    requirements.append({
                        "reference": ref.strip(),
                        "text": req_text,
                        "extraction_method": "regex"
                    })

        seen_refs = set()
        unique_requirements = []
        for req in requirements:
            if req["reference"] not in seen_refs:
                seen_refs.add(req["reference"])
                unique_requirements.append(req)

        logger.info(
            f"Extracted {len(unique_requirements)} requirements using regex")
        return unique_requirements

    except Exception as e:
        logger.error(f"Error in regex extraction: {e}")
        return []


async def extract_section_l_llm(text: str) -> List[Dict[str, str]]:
    """Extract Section L requirements using LLM as fallback"""
    try:
        if not llm:
            logger.error("LLM not initialized for Section L parsing")
            return []

        prompt = ChatPromptTemplate.from_template("""
You are an expert at analyzing government RFP documents. Extract all proposal submission requirements from the following Section L text.
For each requirement, provide:
1. A reference number (e.g., "L.1.1" or "Para 1.2") - use "REQ-N" if no number exists
2. The complete requirement text
Return a JSON array of objects with "reference" and "text" fields.
Text to analyze:
{text}
Return only valid JSON:
""")
        parser = JsonOutputParser()
        chain = prompt | llm | parser

        if len(text) > 8000:
            text = text[:8000] + "..."

        result = await chain.ainvoke({"text": text})

        if isinstance(result, list):
            requirements = []
            for i, item in enumerate(result):
                if isinstance(item, dict) and "text" in item:
                    req = {
                        "reference": item.get("reference", f"REQ-{i+1}"),
                        "text": str(item["text"]).strip(),
                        "extraction_method": "llm"
                    }
                    requirements.append(req)

            logger.info(
                f"Extracted {len(requirements)} requirements using LLM")
            return requirements

        return []

    except Exception as e:
        logger.error(f"Error in LLM extraction: {e}")
        return []


def extract_cross_references(text: str) -> List[str]:
    """Extracts cross-references like 'Section C.1.2' or 'PWS 3.4' from text."""
    patterns = [
        r"section\s+([A-Z](?:\.\d+)*)",
        r"PWS\s+(?:paragraph|section)?\s*([\d\.]+)",
        r"(?:SOW|SOO)\s+(?:paragraph|section)?\s*([\d\.]+)",
        r"attachment\s+([\w\-\d]+)",
    ]
    all_refs = []
    for pattern in patterns:
        all_refs.extend(re.findall(pattern, text, re.IGNORECASE))

    return list(set(all_refs))


async def assemble_progressive_context(requirement: Dict[str, str],
                                       retriever: Any) -> str:
    """Assembles a structured context by resolving cross-references."""
    context_str = f"PRIMARY REQUIREMENT (from {requirement['reference']}):\n{requirement['text']}\n"

    cross_refs = extract_cross_references(requirement['text'])

    if cross_refs:
        context_str += "\n--- REFERENCED CONTEXT ---\n"
        for ref in cross_refs:
            retrieved_docs = await retriever.ainvoke(
                f"Find the full text for reference: '{ref}'")
            if retrieved_docs:
                ref_text = "\n\n".join([
                    doc.page_content for doc in retrieved_docs
                    if doc.page_content
                ])
                context_str += f"\n--- Context for reference '{ref}' ---\n{ref_text}\n"

    return context_str.strip()


# =============================================================================
# 6. API ENDPOINTS
# =============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        initialize_services()
        logger.info("PropelAI MVP-Plus started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.2.0"
    }


@app.post("/project/reset")
async def reset_project():
    """Reset all project data"""
    global vector_store

    try:
        vector_store = None
        for path in [VECTOR_STORE_PATH, PROJECT_DOCS_PATH, TRACE_LOG_PATH]:
            if os.path.exists(path):
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

        logger.info("Project reset completed")
        return {"message": "Project has been completely reset"}

    except Exception as e:
        logger.error(f"Error during project reset: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.get("/project/status", response_model=ProjectStatus)
async def get_project_status():
    """Get current project status"""
    try:
        project_docs = load_project_documents()
        total_chunks = sum(doc.chunks_created for doc in project_docs.values())

        return ProjectStatus(
            total_documents=len(project_docs),
            indexed_documents=len(
                [d for d in project_docs.values() if d.status == "indexed"]),
            total_chunks=total_chunks,
            vector_store_ready=vector_store is not None,
            last_updated=max([doc.timestamp for doc in project_docs.values()],
                             default=None))

    except Exception as e:
        logger.error(f"Error getting project status: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Status check failed: {str(e)}")


@app.post("/project/index")
async def index_documents(files: List[UploadFile] = File(...)):
    """Index uploaded documents"""
    global vector_store

    if not embeddings:
        raise HTTPException(status_code=500,
                            detail="OpenAI services not initialized")

    try:
        project_docs = load_project_documents()
        all_chunks = []
        newly_processed = 0

        for file in files:
            if not file.filename or file.filename in project_docs:
                continue

            file_bytes = await file.read()
            documents = await process_document_safe(file.filename, file_bytes)

            if not documents:
                logger.warning(f"No content extracted from {file.filename}")
                continue

            chunks = await chunk_documents_safe(documents)
            all_chunks.extend(chunks)

            project_docs[file.filename] = DocumentInfo(
                filename=file.filename,
                status="indexed",
                chunks_created=len(chunks),
                timestamp=datetime.now().isoformat())
            newly_processed += 1

        if not all_chunks:
            return {
                "message":
                "No new documents to process or no content extracted"
            }

        if vector_store is None and os.path.exists(VECTOR_STORE_PATH):
            vector_store = FAISS.load_local(
                VECTOR_STORE_PATH,
                embeddings,
                allow_dangerous_deserialization=True)
            vector_store.add_documents(all_chunks)
        elif vector_store is None:
            vector_store = FAISS.from_documents(all_chunks, embeddings)
        else:
            vector_store.add_documents(all_chunks)

        vector_store.save_local(VECTOR_STORE_PATH)
        save_project_documents(project_docs)

        logger.info(
            f"Successfully indexed {newly_processed} documents with {len(all_chunks)} chunks"
        )

        return {
            "message":
            f"Successfully indexed {newly_processed} new document(s)",
            "documents_processed":
            newly_processed,
            "chunks_created":
            len(all_chunks),
            "total_chunks":
            sum(doc.chunks_created for doc in project_docs.values())
        }

    except Exception as e:
        logger.error(f"Error during indexing: {e}")
        raise HTTPException(status_code=500,
                            detail=f"Indexing failed: {str(e)}")


@app.post("/debug/get-rfp-text")
async def debug_get_rfp_text():
    """Debug endpoint to examine extracted RFP text"""
    global vector_store
    try:
        if not vector_store:
            if os.path.exists(VECTOR_STORE_PATH) and embeddings:
                vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True)
            else:
                raise HTTPException(status_code=400,
                                    detail="No vector store found")

        project_docs = load_project_documents()
        rfp_files = [
            f for f in project_docs.keys()
            if any(keyword in f.upper()
                   for keyword in ["RFP", "SOLICITATION", "SECTION L"])
        ]

        if not rfp_files:
            raise HTTPException(status_code=404,
                                detail="No RFP/Solicitation document found")

        all_docs = []
        for doc_id in vector_store.index_to_docstore_id.values():
            try:
                doc = vector_store.docstore.search(doc_id)
                if doc and doc.page_content and doc.metadata.get(
                        'source') in rfp_files:
                    all_docs.append(doc.page_content)
            except Exception:
                continue

        if not all_docs:
            raise HTTPException(status_code=404,
                                detail="No content found for RFP documents")

        full_text = "\n\n".join(all_docs)

        return {
            "rfp_files_found":
            rfp_files,
            "content_blocks":
            len(all_docs),
            "total_length":
            len(full_text),
            "extracted_text":
            full_text[:5000] + "..." if len(full_text) > 5000 else full_text
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")


@app.post("/generate-compliance-matrix",
          response_model=ComplianceMatrixResponse)
async def generate_compliance_matrix():
    """Generate compliance matrix from indexed documents"""
    global vector_store

    try:
        if not vector_store:
            if os.path.exists(VECTOR_STORE_PATH) and embeddings:
                vector_store = FAISS.load_local(
                    VECTOR_STORE_PATH,
                    embeddings,
                    allow_dangerous_deserialization=True)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=
                    "No vector store found. Please index documents first.")

        if not llm:
            raise HTTPException(status_code=500, detail="LLM not initialized")

        project_docs = load_project_documents()
        rfp_files = [
            f for f in project_docs.keys()
            if any(keyword in f.upper()
                   for keyword in ["RFP", "SOLICITATION", "SECTION L"])
        ]
        if not rfp_files:
            raise HTTPException(status_code=404,
                                detail="No RFP/Solicitation document found")

        all_rfp_docs = []
        for doc_id in vector_store.index_to_docstore_id.values():
            try:
                doc = vector_store.docstore.search(doc_id)
                if doc and doc.page_content and doc.metadata.get(
                        'source') in rfp_files:
                    all_rfp_docs.append(doc.page_content)
            except Exception:
                continue

        if not all_rfp_docs:
            raise HTTPException(status_code=404,
                                detail="No content found for RFP documents")

        full_rfp_text = "\n\n".join(all_rfp_docs)

        requirements = extract_section_l_regex(full_rfp_text)
        if not requirements:
            logger.info("Regex extraction failed, trying LLM extraction")
            requirements = await extract_section_l_llm(full_rfp_text)

        if not requirements:
            raise HTTPException(
                status_code=404,
                detail="Could not extract any requirements from Section L")

        logger.info(f"Found {len(requirements)} requirements to process")

        synthesis_prompt = ChatPromptTemplate.from_template("""
You are an expert proposal manager. From the structured context below, generate a single, complete JSON object for a compliance matrix entry.
**Complete Requirement Context:**
{complete_context}
**Your Task:**
Create a JSON object matching the Pydantic model exactly.
Return only the valid JSON object.
""")
        parser = JsonOutputParser(pydantic_object=ComplianceMatrixEntry)
        synthesis_chain = synthesis_prompt | llm | parser

        retriever = vector_store.as_retriever(
            search_kwargs={'k': MAX_RETRIEVAL_DOCS})
        matrix_entries = []

        for requirement in requirements:
            try:
                complete_context = await assemble_progressive_context(
                    requirement, retriever)

                log_trace({
                    "action":
                    "context_assembled",
                    "requirement_ref":
                    requirement.get("reference", "N/A"),
                    "assembled_context":
                    complete_context
                })

                entry = await synthesis_chain.ainvoke(
                    {"complete_context": complete_context})

                if entry:
                    matrix_entries.append(entry)

            except ValidationError as e:
                logger.error(
                    f"Pydantic validation failed for requirement {requirement.get('reference', 'unknown')}: {e}"
                )
                # Log the malformed data for debugging
                try:
                    raw_response = await (
                        synthesis_prompt | llm).ainvoke({
                            "complete_context": complete_context
                        })
                    logger.warning(
                        f"Malformed JSON from LLM for requirement {requirement.get('reference', 'unknown')}: {raw_response.content}"
                    )
                except Exception as llm_e:
                    logger.error(
                        f"Could not even get raw response from LLM after validation failure: {llm_e}"
                    )
                continue
            except Exception as e:
                logger.error(
                    f"General error processing requirement {requirement.get('reference', 'unknown')}: {e}",
                    exc_info=True)
                continue

        if not matrix_entries:
            raise HTTPException(
                status_code=500,
                detail=
                "Failed to generate any compliance matrix entries after processing all requirements."
            )

        metadata = {
            "total_requirements_found":
            len(requirements),
            "matrix_entries_generated":
            len(matrix_entries),
            "rfp_files_processed":
            rfp_files,
            "generation_timestamp":
            datetime.now().isoformat(),
            "extraction_methods_used":
            list(
                set(
                    req.get("extraction_method", "unknown")
                    for req in requirements))
        }

        log_trace({
            "action":
            "compliance_matrix_generated",
            "requirements_processed":
            len(requirements),
            "entries_generated":
            len(matrix_entries),
            "success_rate":
            len(matrix_entries) / len(requirements) if requirements else 0
        })

        return ComplianceMatrixResponse(compliance_matrix=matrix_entries,
                                        generation_metadata=metadata)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating compliance matrix: {e}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"Matrix generation failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500,
                        content=ErrorResponse(
                            error="Internal server error",
                            details=str(exc),
                            timestamp=datetime.now().isoformat()).model_dump())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

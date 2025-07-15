import os
import io
import json
import logging
import shutil
from datetime import datetime
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Local Module Imports ---
from app.models import schemas
from app.processing import document, extraction
from app.utils import file_io
from app.rag import vector_store, matrix_generator

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('propelai.log'),
              logging.StreamHandler()])
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="PropelAI Complete v7 - Modular",
    description=
    "Handles large documents for 100% PWS requirement capture and compliance matrix generation.",
    version="7.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/health", response_model=schemas.HealthResponse)
async def health_check():
    return schemas.HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        message="PropelAI Complete v7 - Batch Processing Enabled",
        services={
            "google_document_ai":
            all([
                os.getenv("GCLOUD_PROJECT_ID"),
                os.getenv("GCLOUD_LOCATION"),
                os.getenv("GCLOUD_PROCESSOR_ID")
            ]),
            "google_storage":
            bool(os.getenv("GCLOUD_STORAGE_BUCKET")),
            "unstructured_fallback":
            document.UNSTRUCTURED_AVAILABLE,
            "openai_integration":
            bool(os.getenv("OPENAI_API_KEY"))
        })


@app.get("/debug/config")
async def debug_config():
    """Debug endpoint to check configuration without exposing sensitive data"""
    try:
        config_status = {
            "google_cloud": {
                "project_id_set": bool(os.getenv("GCLOUD_PROJECT_ID")),
                "location_set": bool(os.getenv("GCLOUD_LOCATION")),
                "processor_id_set": bool(os.getenv("GCLOUD_PROCESSOR_ID")),
                "storage_bucket_set": bool(os.getenv("GCLOUD_STORAGE_BUCKET")),
            },
            "openai": {
                "api_key_set": bool(os.getenv("OPENAI_API_KEY")),
            },
            "environment": {
                "google_application_credentials_set":
                bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
            }
        }

        return {
            "status": "debug_info",
            "timestamp": datetime.now().isoformat(),
            "configuration": config_status
        }

    except Exception as e:
        logger.error(f"Debug config error: {e}")
        return {"error": str(e)}


@app.get("/")
async def root():
    return {
        "message":
        "PropelAI Complete v7 API - Modular & Batch Processing Enabled",
        "version":
        "7.0.0",
        "features": [
            "Batch processing for large documents (>30 pages)",
            "Section 5 PWS extraction",
            "Traditional Section 2.0 PWS extraction",
            "Enhanced Section L volume requirements",
            "Google Document AI with credential security",
            "RAG-based compliance matrix generation",
        ],
        "endpoints": [
            "/health", "/project/status", "/project/reset", "/project/process",
            "/project/analyze", "/project/build-vector-store",
            "/project/generate-compliance-matrix"
        ]
    }


@app.post("/project/reset")
async def reset_project():
    try:
        if os.path.exists(file_io.PROJECT_DOCS_PATH):
            os.remove(file_io.PROJECT_DOCS_PATH)
        if os.path.exists(file_io.PROCESSED_CONTENT_DIR):
            shutil.rmtree(file_io.PROCESSED_CONTENT_DIR)

        vector_files = [
            "compliance_vector_store.faiss",
            "compliance_vector_store.faiss.pkl"
        ]
        for file_path in vector_files:
            if os.path.exists(file_path):
                os.remove(file_path)

        logger.info("Project reset completed - including vector store cleanup")
        return {
            "message": "Project reset successfully.",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/project/status", response_model=schemas.ProjectStatus)
async def get_project_status():
    try:
        project_docs = file_io.load_project_documents()

        pws_count = len([
            doc for doc in project_docs.values()
            if doc.document_category == 'PWS'
        ])
        rfp_count = len([
            doc for doc in project_docs.values()
            if doc.document_category in ['RFP', 'Section L']
        ])
        total_requirements = sum(doc.requirements_found
                                 for doc in project_docs.values())

        vector_store_exists = os.path.exists("compliance_vector_store.faiss")

        return schemas.ProjectStatus(
            total_documents=len(project_docs),
            pws_documents=pws_count,
            rfp_documents=rfp_count,
            processing_status="ready",
            last_updated=max([doc.timestamp for doc in project_docs.values()],
                             default=None),
            available_services={
                "google_document_ai":
                True,
                "vector_store_ready":
                vector_store_exists,
                "openai_integration":
                bool(os.getenv("OPENAI_API_KEY")),
                "compliance_matrix":
                vector_store_exists and bool(os.getenv("OPENAI_API_KEY"))
            },
            total_requirements=total_requirements)
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/project/process-with-sample")
async def process_with_sample_content():
    """Process a sample RFP for testing when document extraction fails"""
    try:
        sample_rfp_content = """
DEPARTMENT OF LABOR
REQUEST FOR PROPOSALS
RFP-DOL-2025-001
SECTION L - INSTRUCTIONS TO OFFERORS
For Volume I, the offeror shall:
a. Provide a comprehensive technical approach describing how the contractor will meet all requirements
b. Submit detailed project management plans including timelines and milestones
c. Include organizational charts showing key personnel and reporting relationships
d. Demonstrate understanding of all performance requirements outlined in Section 5
e. Provide past performance information for similar projects
"""
        filename = "Sample_DOL_RFP.txt"
        document_category = extraction.detect_document_category(
            filename, sample_rfp_content)

        file_io.save_processed_content(filename, sample_rfp_content)

        requirements = extraction.extract_section_l_requirements_enhanced(
            sample_rfp_content, filename)

        doc_info = schemas.DocumentInfo(filename=filename,
                                        status="processed",
                                        processing_method="sample_content",
                                        content_length=len(sample_rfp_content),
                                        timestamp=datetime.now().isoformat(),
                                        file_type="text",
                                        document_category=document_category,
                                        requirements_found=len(requirements))

        project_docs = file_io.load_project_documents()
        project_docs[filename] = doc_info
        file_io.save_project_documents(project_docs)

        return {
            "message": "Processed sample RFP content for testing",
            "files": [doc_info.model_dump()],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Sample processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/project/process")
async def process_files(files: List[UploadFile] = File(...)):
    project_docs = file_io.load_project_documents()
    processed_files = []

    for file in files:
        if not file.filename:
            continue

        try:
            file_bytes = await file.read()
            text_content = await document.process_with_document_ai(
                file.filename, file_bytes)

            if text_content.startswith("Error:") or text_content.startswith(
                    "Configuration Error:"):
                raise Exception(text_content)

            file_io.save_processed_content(file.filename, text_content)
            document_category = extraction.detect_document_category(
                file.filename, text_content)

            # --- UPDATED LOGIC ---
            # Now, we accumulate requirements from all possible functions
            all_requirements = []

            # Always try to extract Section L requirements
            all_requirements.extend(
                extraction.extract_section_l_requirements_enhanced(
                    text_content, file.filename))
            all_requirements.extend(
                extraction.extract_section_l_requirements(
                    text_content, file.filename))

            # Always try to extract PWS requirements (Part 5 is a PWS)
            all_requirements.extend(
                extraction.extract_pws_section_5_requirements(
                    text_content, file.filename))
            all_requirements.extend(
                extraction.extract_pws_section_2_requirements(
                    text_content, file.filename))

            # Remove duplicate requirements if any are found by multiple methods
            unique_requirements = list(
                {req.text: req
                 for req in all_requirements}.values())

            doc_info = schemas.DocumentInfo(
                filename=file.filename,
                status="processed",
                processing_method="google_document_ai_batch",
                content_length=len(text_content),
                timestamp=datetime.now().isoformat(),
                file_type=file_io.detect_file_type(file.filename),
                document_category=document_category,
                requirements_found=len(
                    unique_requirements))  # Use count of unique requirements

            project_docs[file.filename] = doc_info
            processed_files.append(doc_info.model_dump())

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            processed_files.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })

    file_io.save_project_documents(project_docs)

    return {
        "message":
        f"Processed {len(processed_files)} files using Batch Processing.",
        "files": processed_files,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/project/analyze", response_model=schemas.AnalysisResponse)
async def analyze_requirements():
    try:
        project_docs = file_io.load_project_documents()

        if not project_docs:
            raise HTTPException(status_code=400,
                                detail="No documents processed yet")

        all_requirements = []
        extraction_summary = {}
        section_breakdown = {}
        validation_status = {}

        for filename, doc_info in project_docs.items():
            try:
                content = file_io.load_processed_content(filename)

                if content:
                    # Rerun extraction to get latest requirements
                    current_doc_reqs = []
                    current_doc_reqs.extend(
                        extraction.extract_section_l_requirements_enhanced(
                            content, filename))
                    current_doc_reqs.extend(
                        extraction.extract_section_l_requirements(
                            content, filename))
                    current_doc_reqs.extend(
                        extraction.extract_pws_section_5_requirements(
                            content, filename))
                    current_doc_reqs.extend(
                        extraction.extract_pws_section_2_requirements(
                            content, filename))

                    unique_requirements = list(
                        {req.text: req
                         for req in current_doc_reqs}.values())
                    all_requirements.extend(unique_requirements)

                    for req in unique_requirements:
                        extraction_summary[
                            req.extraction_method] = extraction_summary.get(
                                req.extraction_method, 0) + 1
                        section_breakdown[
                            req.document_section] = section_breakdown.get(
                                req.document_section, 0) + 1

                validation_status[filename] = {
                    "status":
                    f"Analyzed and found {len(unique_requirements)} requirements."
                }

            except Exception as e:
                logger.error(f"Error analyzing {filename}: {e}")
                validation_status[filename] = {"status": f"ERROR: {str(e)}"}
                continue

        all_requirements.sort(key=lambda x: (x.priority, -x.confidence))
        pws_reqs = len([
            req for req in all_requirements
            if req.requirement_type == "performance"
        ])
        section_l_reqs = len([
            req for req in all_requirements
            if req.requirement_type == "submission"
        ])
        summary_message = f"Found {len(all_requirements)} total unique requirements ({pws_reqs} PWS performance, {section_l_reqs} Section L submission)"

        return schemas.AnalysisResponse(
            message=summary_message,
            requirements=all_requirements,
            analysis_timestamp=datetime.now().isoformat(),
            documents_analyzed=len(project_docs),
            extraction_summary=extraction_summary,
            section_breakdown=section_breakdown,
            validation_status=validation_status)

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/project/build-vector-store")
async def build_vector_store():
    """Build vector store from all processed documents for RAG compliance matrix"""
    try:
        project_docs = file_io.load_project_documents()
        if not project_docs:
            raise HTTPException(status_code=400,
                                detail="No documents processed yet")

        vector_manager = vector_store.VectorStoreManager()
        vector_manager.initialize_embeddings()

        total_chunks = 0
        for filename, doc_info in project_docs.items():
            try:
                content = file_io.load_processed_content(filename)
                if content and len(content) > 100:
                    chunks = vector_manager.create_chunks(
                        content, filename, doc_info.document_category)
                    vector_manager.add_to_vector_store(chunks)
                    total_chunks += len(chunks)
                    logger.info(f"Added {len(chunks)} chunks from {filename}")
            except Exception as e:
                logger.error(
                    f"Error processing {filename} for vector store: {e}")
                continue

        vector_manager.save_vector_store()

        return {
            "message":
            f"Vector store built successfully with {total_chunks} chunks",
            "documents_processed": len(project_docs),
            "total_chunks": total_chunks,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Vector store build error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/project/generate-compliance-matrix",
          response_model=schemas.ComplianceMatrixResponse)
async def generate_full_compliance_matrix():
    """Generate 100% complete compliance matrix from all RFP requirements"""
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise HTTPException(status_code=500,
                                detail="OPENAI_API_KEY not configured")

        project_docs = file_io.load_project_documents()
        if not project_docs:
            raise HTTPException(status_code=400,
                                detail="No documents processed yet")

        vector_manager = vector_store.VectorStoreManager()
        if not vector_manager.load_vector_store():
            raise HTTPException(
                status_code=400,
                detail=
                "Vector store not found. Run /project/build-vector-store first"
            )

        all_section_l_requirements = []
        doc_types = matrix_generator.identify_document_types(project_docs)
        rfp_sources = doc_types['RFP'] + doc_types['Section_L']

        if not rfp_sources:
            raise HTTPException(status_code=400,
                                detail="No RFP or Section L documents found")

        for filename in rfp_sources:
            content = file_io.load_processed_content(filename)
            if content:
                requirements = extraction.extract_section_l_requirements_enhanced(
                    content, filename)
                all_section_l_requirements.extend(requirements)

        unique_requirements = list(
            {req.text: req
             for req in all_section_l_requirements}.values())

        if not unique_requirements:
            raise HTTPException(
                status_code=404,
                detail="No Section L requirements found in RFP documents")

        compliance_matrix = await matrix_generator.generate_compliance_matrix_comprehensive(
            unique_requirements, project_docs, vector_manager)

        matrix_summary = {
            "total_requirements":
            len(compliance_matrix),
            "pws_coverage":
            len([
                entry for entry in compliance_matrix if entry.pws_references
                and entry.pws_references[0] != "No PWS content found"
            ]),
        }

        return schemas.ComplianceMatrixResponse(
            compliance_matrix=compliance_matrix,
            matrix_summary=matrix_summary,
            generation_timestamp=datetime.now().isoformat(),
            total_requirements=len(compliance_matrix),
            document_sources=list(project_docs.keys()))

    except Exception as e:
        logger.error(f"Compliance matrix generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

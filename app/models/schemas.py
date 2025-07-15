from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime


# Models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    message: str
    services: Dict[str, bool]


class DocumentInfo(BaseModel):
    filename: str
    status: str
    processing_method: str
    content_length: int
    timestamp: str
    file_type: str
    document_category: str
    requirements_found: int = 0


class ProjectStatus(BaseModel):
    total_documents: int
    pws_documents: int
    rfp_documents: int
    processing_status: str
    last_updated: Optional[str] = None
    available_services: Dict[str, bool]
    total_requirements: int = 0


class RequirementEntry(BaseModel):
    reference: str
    text: str
    extraction_method: str
    source: str
    document_section: str
    requirement_type: str
    confidence: float = 0.0
    priority: int = 1


class AnalysisResponse(BaseModel):
    message: str
    requirements: List[RequirementEntry]
    analysis_timestamp: str
    documents_analyzed: int
    extraction_summary: Dict[str, int]
    section_breakdown: Dict[str, int]
    validation_status: Dict[str, Any] = {}


# Enhanced Compliance Matrix Models
class ComplianceMatrixEntry(BaseModel):
    rfp_requirement_summary: str
    rfp_location: str
    proposal_section: str
    pws_references: List[str]
    sow_references: List[str] = []
    soo_references: List[str] = []
    section_l_context: str
    compliance_approach: str
    responsible_person: str = "TBD"
    priority: str = "Medium"
    extraction_confidence: float = 0.0


class ComplianceMatrixResponse(BaseModel):
    compliance_matrix: List[ComplianceMatrixEntry]
    matrix_summary: Dict[str, Any]
    generation_timestamp: str
    total_requirements: int
    document_sources: List[str]

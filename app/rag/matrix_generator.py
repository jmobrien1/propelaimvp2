import logging
from typing import List, Dict, Any
from fastapi import HTTPException

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import from your new modules
from app.models.schemas import DocumentInfo, RequirementEntry, ComplianceMatrixEntry
from app.rag.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


def identify_document_types(
        project_docs: Dict[str, DocumentInfo]) -> Dict[str, List[str]]:
    """Identify and categorize documents by type for comprehensive analysis"""
    doc_types = {
        'PWS': [],
        'SOW': [],
        'SOO': [],
        'RFP': [],
        'Section_L': [],
        'Other': []
    }

    for filename, doc_info in project_docs.items():
        filename_lower = filename.lower()
        category = doc_info.document_category

        # Enhanced detection
        if category == 'PWS' or any(
                term in filename_lower
                for term in ['pws', 'performance work statement']):
            doc_types['PWS'].append(filename)
        elif 'sow' in filename_lower or 'statement of work' in filename_lower:
            doc_types['SOW'].append(filename)
        elif 'soo' in filename_lower or 'statement of objectives' in filename_lower:
            doc_types['SOO'].append(filename)
        elif category in ['RFP', 'Section L'] or any(
                term in filename_lower
                for term in ['rfp', 'request for proposal', 'solicitation']):
            if 'section_l' in filename_lower or 'instructions' in filename_lower:
                doc_types['Section_L'].append(filename)
            else:
                doc_types['RFP'].append(filename)
        else:
            doc_types['Other'].append(filename)

    return doc_types


async def generate_compliance_matrix_comprehensive(
        section_l_requirements: List[RequirementEntry],
        project_docs: Dict[str, DocumentInfo],
        vector_manager: VectorStoreManager) -> List[ComplianceMatrixEntry]:
    """Generate comprehensive compliance matrix using RAG"""

    if not vector_manager.vector_store:
        raise HTTPException(status_code=400,
                            detail="Vector store not initialized")

    llm = ChatOpenAI(model="gpt-4",
                     temperature=0,
                     model_kwargs={"response_format": {
                         "type": "json_object"
                     }})

    doc_types = identify_document_types(project_docs)
    compliance_matrix = []

    # Create comprehensive prompt for compliance analysis
    compliance_prompt = ChatPromptTemplate.from_template("""
    You are an expert proposal compliance analyst. Create a detailed compliance matrix entry for the given Section L requirement.

    Analyze the requirement and find supporting content from PWS, SOW, and SOO documents using the provided context.

    Return a JSON object with these exact keys:
    - "rfp_requirement_summary": Clear, actionable summary of the requirement
    - "rfp_location": Location in the RFP/Section L
    - "proposal_section": Suggested proposal section to address this
    - "pws_references": List of specific PWS content that addresses this requirement
    - "sow_references": List of specific SOW content that addresses this requirement  
    - "soo_references": List of specific SOO content that addresses this requirement
    - "section_l_context": Additional context from Section L
    - "compliance_approach": Recommended approach to address this requirement
    - "responsible_person": Set to "TBD"
    - "priority": "High", "Medium", or "Low"
    - "extraction_confidence": Float between 0.0 and 1.0

    Section L Requirement: {requirement_text}
    Requirement Reference: {requirement_ref}

    Supporting Context from Documents:
    {supporting_context}

    Available Document Types: {doc_types}
    """)

    compliance_chain = compliance_prompt | llm | JsonOutputParser()

    for req in section_l_requirements:
        try:
            # Search for supporting content in all document types
            search_query = f"{req.text} {req.reference}"

            # Get relevant documents from vector store
            retriever = vector_manager.vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 10})

            supporting_docs = retriever.invoke(search_query)

            # Organize supporting content by document type
            pws_content = []
            sow_content = []
            soo_content = []
            other_content = []

            for doc in supporting_docs:
                source = doc.metadata.get('source', '')
                content = doc.page_content
                doc_category = doc.metadata.get('document_category', '')

                if any(pws_file in source for pws_file in
                       doc_types['PWS']) or doc_category == 'PWS':
                    pws_content.append(
                        f"Source: {source} | Content: {content[:300]}...")
                elif any(sow_file in source for sow_file in doc_types['SOW']):
                    sow_content.append(
                        f"Source: {source} | Content: {content[:300]}...")
                elif any(soo_file in source for soo_file in doc_types['SOO']):
                    soo_content.append(
                        f"Source: {source} | Content: {content[:300]}...")
                else:
                    other_content.append(
                        f"Source: {source} | Content: {content[:300]}...")

            supporting_context = f"""
            PWS Content: {' | '.join(pws_content[:3])}
            SOW Content: {' | '.join(sow_content[:3])}
            SOO Content: {' | '.join(soo_content[:3])}
            Other Supporting Content: {' | '.join(other_content[:2])}
            """

            # Generate compliance matrix entry
            matrix_entry = await compliance_chain.ainvoke({
                "requirement_text":
                req.text,
                "requirement_ref":
                req.reference,
                "supporting_context":
                supporting_context,
                "doc_types":
                str(doc_types)
            })

            # Convert to Pydantic model
            compliance_entry = ComplianceMatrixEntry(**matrix_entry)
            compliance_matrix.append(compliance_entry)

            logger.info(f"Generated compliance entry for {req.reference}")

        except Exception as e:
            logger.error(
                f"Error generating compliance entry for {req.reference}: {e}")
            # Create fallback entry
            fallback_entry = ComplianceMatrixEntry(
                rfp_requirement_summary=req.text[:200] + "...",
                rfp_location=f"Section L - {req.reference}",
                proposal_section="TBD",
                pws_references=["No PWS content found"],
                sow_references=["No SOW content found"],
                soo_references=["No SOO content found"],
                section_l_context=req.document_section,
                compliance_approach="Manual review required",
                responsible_person="TBD",
                priority="Medium",
                extraction_confidence=0.3)
            compliance_matrix.append(fallback_entry)

    return compliance_matrix

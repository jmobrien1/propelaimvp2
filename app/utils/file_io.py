import os
import json
import logging
import shutil
import re
from typing import Dict

from app.models.schemas import DocumentInfo

logger = logging.getLogger(__name__)

# Configuration
PROJECT_DOCS_PATH = "project_documents.json"
PROCESSED_CONTENT_DIR = "processed_content"


# Utility functions
def ensure_content_dir():
    if not os.path.exists(PROCESSED_CONTENT_DIR):
        os.makedirs(PROCESSED_CONTENT_DIR)


def get_content_file_path(filename: str) -> str:
    ensure_content_dir()
    safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
    return os.path.join(PROCESSED_CONTENT_DIR, f"{safe_filename}.txt")


def save_processed_content(filename: str, content: str):
    try:
        content_path = get_content_file_path(filename)
        with open(content_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved processed content for {filename}")
    except Exception as e:
        logger.error(f"Error saving content for {filename}: {e}")


def load_processed_content(filename: str) -> str:
    try:
        content_path = get_content_file_path(filename)
        if os.path.exists(content_path):
            with open(content_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    except Exception as e:
        logger.error(f"Error loading content for {filename}: {e}")
        return ""


def load_project_documents() -> Dict[str, DocumentInfo]:
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
    try:
        serializable_docs = {
            filename: doc_info.model_dump()
            for filename, doc_info in docs.items()
        }
        with open(PROJECT_DOCS_PATH, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving project documents: {e}")


def detect_file_type(filename: str) -> str:
    ext = filename.lower().split('.')[-1] if '.' in filename else ""

    if ext in ['pdf']:
        return 'pdf'
    elif ext in ['doc', 'docx']:
        return 'word'
    elif ext in ['md', 'markdown']:
        return 'markdown'
    elif ext in ['txt']:
        return 'text'
    elif ext in ['xlsx', 'xls']:
        return 'excel'
    else:
        return 'unknown'

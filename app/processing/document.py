import os
import io
import re
import mimetypes
import asyncio
import logging
from uuid import uuid4

# --- Google Cloud Imports ---
from google.cloud import documentai, storage
from google.api_core import exceptions

# Document processing imports
try:
    from unstructured.partition.auto import partition
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False

logger = logging.getLogger(__name__)


# SECURITY: Enhanced credential cleaning function
def clean_document_ai_output(text_content: str) -> str:
    """Remove Google Cloud credentials and other sensitive data from extracted text"""
    # If the entire content looks like credentials, return error message
    if text_content.strip().startswith(
            '{') and 'service_account' in text_content[:500]:
        logger.error("Entire content appears to be credentials - rejecting")
        return "Error: Content appears to be credential data instead of document text"

    # Remove JSON credential blocks
    text_content = re.sub(
        r'"type":\s*"service_account".*?"universe_domain":\s*"googleapis\.com"\s*}',
        '[CREDENTIALS REMOVED FOR SECURITY]',
        text_content,
        flags=re.DOTALL)

    # Remove entire JSON objects that look like credentials
    text_content = re.sub(r'\{\s*"type":\s*"service_account".*?\}',
                          '[CREDENTIAL OBJECT REMOVED]',
                          text_content,
                          flags=re.DOTALL)

    # Remove private key blocks
    text_content = re.sub(
        r'-----BEGIN PRIVATE KEY-----.*?-----END PRIVATE KEY-----',
        '[PRIVATE KEY REMOVED FOR SECURITY]',
        text_content,
        flags=re.DOTALL)

    # Remove base64-encoded credential data (long strings)
    text_content = re.sub(r'[A-Za-z0-9+/]{200,}={0,2}',
                          '[BASE64 CREDENTIALS REMOVED]', text_content)

    # Remove service account emails
    text_content = re.sub(r'[\w\-\.]+@[\w\-\.]+\.iam\.gserviceaccount\.com',
                          '[SERVICE ACCOUNT EMAIL REMOVED]', text_content)

    # Remove specific credential fields
    text_content = re.sub(r'"client_id":\s*"[^"]*"',
                          '"client_id": "[REMOVED]"', text_content)
    text_content = re.sub(r'"auth_uri":\s*"[^"]*"', '"auth_uri": "[REMOVED]"',
                          text_content)
    text_content = re.sub(r'"token_uri":\s*"[^"]*"',
                          '"token_uri": "[REMOVED]"', text_content)
    text_content = re.sub(r'"private_key_id":\s*"[^"]*"',
                          '"private_key_id": "[REMOVED]"', text_content)
    text_content = re.sub(r'"client_email":\s*"[^"]*"',
                          '"client_email": "[REMOVED]"', text_content)

    # Remove file paths that might contain credentials
    text_content = re.sub(r'File\s+\{[^}]*service_account[^}]*\}',
                          '[CREDENTIAL FILE REFERENCE REMOVED]', text_content)

    return text_content


def _process_document_sync(file_name: str, file_bytes: bytes,
                           mime_type: str) -> str:
    """
    Synchronous helper to call Document AI, automatically handling large
    documents via batch processing in Google Cloud Storage.
    """
    try:
        project_id = os.getenv("GCLOUD_PROJECT_ID")
        location = os.getenv("GCLOUD_LOCATION")
        processor_id = os.getenv("GCLOUD_PROCESSOR_ID")
        gcs_bucket_name = os.getenv("GCLOUD_STORAGE_BUCKET")

        if not all([project_id, location, processor_id, gcs_bucket_name]):
            error_msg = "Google Cloud credentials or Storage Bucket not configured"
            logger.error(error_msg)
            return f"Configuration Error: {error_msg}"

        docai_client = documentai.DocumentProcessorServiceClient(
            client_options={
                "api_endpoint": f"{location}-documentai.googleapis.com"
            })
        storage_client = storage.Client()

        processor_path = docai_client.processor_path(project_id, location,
                                                     processor_id)

        # Use a unique prefix for each job to avoid collisions
        gcs_prefix = f"batch-processing/{uuid4()}"

        # 1. Upload file to Google Cloud Storage
        bucket = storage_client.get_bucket(gcs_bucket_name)
        input_blob = bucket.blob(f"{gcs_prefix}/input/{file_name}")
        input_blob.upload_from_string(file_bytes, content_type=mime_type)
        gcs_input_uri = f"gs://{gcs_bucket_name}/{input_blob.name}"
        logger.info(
            f"Uploaded '{file_name}' to {gcs_input_uri} for batch processing.")

        # 2. Set up batch processing request
        gcs_document = documentai.GcsDocument(gcs_uri=gcs_input_uri,
                                              mime_type=mime_type)
        input_config = documentai.BatchDocumentsInputConfig(
            gcs_documents=documentai.GcsDocuments(documents=[gcs_document]))

        # 3. Define the GCS output location
        gcs_output_uri = f"gs://{gcs_bucket_name}/{gcs_prefix}/output/"
        output_config = documentai.DocumentOutputConfig(
            gcs_output_config={"gcs_uri": gcs_output_uri})

        # 4. Call the batch processing API
        request = documentai.BatchProcessRequest(
            name=processor_path,
            input_documents=input_config,
            document_output_config=output_config,
        )
        operation = docai_client.batch_process_documents(request)
        logger.info(
            f"Waiting for batch operation '{operation.operation.name}' to complete..."
        )
        operation.result(timeout=1200)  # Increased timeout for large docs
        logger.info("Batch operation completed.")

        # 5. Get the results from the output location in GCS
        match = re.match(r"gs://([^/]+)/(.+)", gcs_output_uri)
        output_bucket_name = match.group(1)
        output_prefix = match.group(2)

        output_bucket = storage_client.get_bucket(output_bucket_name)

        full_text = []
        for blob in output_bucket.list_blobs(prefix=output_prefix):
            if ".json" in blob.name:
                json_string = blob.download_as_string()
                document = documentai.Document.from_json(
                    json_string, ignore_unknown_fields=True)
                full_text.append(document.text)
                logger.info(f"Processed results from {blob.name}")

        # 6. Clean up the files in GCS
        for blob in output_bucket.list_blobs(prefix=gcs_prefix):
            blob.delete()
        logger.info(f"Cleaned up GCS files under prefix: {gcs_prefix}")

        if not full_text:
            raise Exception(
                "Batch processing finished but no text was extracted from output files."
            )

        return "\n\n".join(full_text)

    except exceptions.InvalidArgument as e:
        logger.error(f"Invalid Argument during processing: {e}")
        return f"Error: Document has an issue. {e}"
    except Exception as e:
        logger.error(
            f"Document AI batch processing error for '{file_name}': {e}",
            exc_info=True)
        return f"Error processing {file_name} with Document AI Batch: {str(e)}"


async def process_with_document_ai(file_name: str, file_bytes: bytes) -> str:
    """Async wrapper for Document AI processing with robust fallback"""
    try:
        mime_type, _ = mimetypes.guess_type(file_name)
        if not mime_type:
            mime_type = "application/pdf"

        # First, try Google Document AI
        try:
            loop = asyncio.get_event_loop()
            text_content = await loop.run_in_executor(None,
                                                      _process_document_sync,
                                                      file_name, file_bytes,
                                                      mime_type)

            # Check if we got an error message
            if text_content.startswith("Error") or text_content.startswith(
                    "Configuration Error") or text_content.startswith(
                        "Document AI service error"):
                logger.warning(
                    f"Document AI failed for {file_name}: {text_content[:200]}..."
                )
                raise Exception("Document AI authentication or service issue")

            # Check for meaningful content
            if len(text_content.strip()) < 50:
                logger.warning(
                    f"Very short content from Document AI for {file_name}")
                raise Exception("Insufficient content extracted")

            logger.info(f"Document AI successful for {file_name}")
            return text_content

        except Exception as doc_ai_error:
            logger.warning(
                f"Document AI failed for {file_name}: {doc_ai_error}")
            # Fall through to fallback processing

        # Fallback to unstructured processing
        logger.info(f"Attempting fallback processing for {file_name}")
        return await process_with_fallback(file_name, file_bytes)

    except Exception as e:
        logger.error(f"All processing methods failed for '{file_name}': {e}")
        return f"Error: Could not process {file_name}. Please verify the file is a valid document and try again."


def clean_extracted_text(text_content: str) -> str:
    """Enhanced cleaning for heavily corrupted OCR text"""
    # Remove security credential artifacts first
    text_content = clean_document_ai_output(text_content)

    # Stage 1: Remove obvious OCR artifacts
    # Remove (cid:X) patterns
    text_content = re.sub(r'\(cid:\d+\)', '', text_content)

    # Remove strings of random characters (likely OCR errors)
    # Pattern: sequences of 10+ mixed case letters/numbers without spaces
    text_content = re.sub(r'\b[a-zA-Z0-9]{10,}\b', ' ', text_content)

    # Remove lines that are mostly garbage
    lines = text_content.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 2:
            cleaned_lines.append('')
            continue

        # Calculate various quality metrics
        alpha_chars = sum(c.isalpha() for c in line)
        space_chars = sum(c.isspace() for c in line)
        digit_chars = sum(c.isdigit() for c in line)
        total_chars = len(line)

        if total_chars > 0:
            alpha_ratio = alpha_chars / total_chars
            word_count = len(line.split())
            avg_word_length = sum(len(word) for word in line.split()) / max(
                word_count, 1)

            # Keep lines that look like real text
            keep_line = False

            # High-quality text indicators
            if alpha_ratio > 0.7 and avg_word_length > 2 and avg_word_length < 12:
                keep_line = True

            # Lines with important keywords (even if corrupted)
            important_keywords = [
                'section', 'volume', 'page', 'requirements', 'proposal',
                'contractor', 'offeror', 'shall', 'must', 'provide',
                'department', 'labor', 'rfp', 'solicitation', 'request',
                'performance', 'work', 'statement', 'submission', 'due'
            ]

            line_lower = line.lower()
            if any(keyword in line_lower for keyword in important_keywords):
                keep_line = True

            # Lines that look like headers or titles
            if len(line) < 100 and alpha_ratio > 0.5 and line.isupper():
                keep_line = True

            # Lines with reasonable punctuation patterns
            if re.search(r'[.!?:;]\s+[A-Z]', line):
                keep_line = True

            if keep_line:
                cleaned_lines.append(line)

    # Stage 2: Reconstruct and clean further
    cleaned_text = '\n'.join(cleaned_lines)

    # Remove excessive whitespace
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    cleaned_text = re.sub(r' +', ' ', cleaned_text)

    # Try to identify and extract any readable sections
    readable_sections = []
    paragraphs = cleaned_text.split('\n\n')

    for para in paragraphs:
        para = para.strip()
        if len(para) > 20:
            words = para.split()
            if len(words) > 3:
                # Check if this paragraph has reasonable word structure
                avg_word_len = sum(len(word) for word in words) / len(words)
                if 2 < avg_word_len < 15:  # Reasonable word lengths
                    readable_sections.append(para)

    if readable_sections:
        final_text = '\n\n'.join(readable_sections)
    else:
        final_text = cleaned_text

    final_text = final_text.strip()

    # If we still have very little readable content, provide a helpful message
    if len(final_text) < 200:
        corruption_msg = f"""
This document appears to be severely corrupted or is a poorly scanned PDF. 

Extracted fragments:
{final_text[:500]}

RECOMMENDATION: Please try one of the following:
1. Upload a different version of the document (preferably text-based PDF)
2. Convert the document to text format before uploading
3. Use a different PDF file if available

The system detected this as an RFP document but cannot extract meaningful requirements due to text corruption.
"""
        return corruption_msg

    return final_text


def _process_with_unstructured_sync(file_name: str, file_bytes: bytes):
    """Enhanced unstructured processing with multiple strategies"""
    try:
        file_like_object = io.BytesIO(file_bytes)

        # Try different unstructured strategies
        try:
            # Strategy 1: Default partition
            elements = partition(file=file_like_object,
                                 file_filename=file_name)
            if elements and len(str(elements[0])) > 10:
                return elements
        except Exception as e:
            logger.warning(f"Default partition failed: {e}")

        # Strategy 2: Try with specific settings for PDFs
        try:
            file_like_object.seek(0)  # Reset file pointer
            elements = partition(
                file=file_like_object,
                file_filename=file_name,
                strategy="fast",  # Use fast strategy for corrupted docs
                include_page_breaks=True)
            if elements:
                return elements
        except Exception as e:
            logger.warning(f"Fast strategy partition failed: {e}")

        # Strategy 3: Extract whatever we can
        file_like_object.seek(0)
        elements = partition(file=file_like_object, file_filename=file_name)
        return elements if elements else []

    except Exception as e:
        logger.error(f"All unstructured strategies failed: {e}")
        return []


async def process_with_fallback(file_name: str, file_bytes: bytes) -> str:
    """Enhanced fallback document processing using unstructured with better text cleaning"""
    try:
        if not UNSTRUCTURED_AVAILABLE:
            return f"Error: Both Document AI and fallback processing unavailable for {file_name}. Please check your environment configuration."

        logger.info(f"Using unstructured fallback processing for {file_name}")

        # Use unstructured for fallback processing with enhanced options
        loop = asyncio.get_event_loop()
        elements = await loop.run_in_executor(None,
                                              _process_with_unstructured_sync,
                                              file_name, file_bytes)

        # Extract and clean text from elements
        raw_texts = []
        for element in elements:
            element_text = str(element).strip()
            if element_text and len(
                    element_text) > 5:  # Filter out very short fragments
                raw_texts.append(element_text)

        # Join and clean the content
        raw_content = "\n\n".join(raw_texts)
        cleaned_content = clean_extracted_text(raw_content)

        if len(cleaned_content.strip()) < 100:
            return f"Error: Could not extract meaningful content from {file_name}. The file may be corrupted, password-protected, or heavily corrupted from scanning."

        logger.info(
            f"Fallback processing successful for {file_name} ({len(cleaned_content)} chars after cleaning)"
        )
        return cleaned_content

    except Exception as e:
        logger.error(f"Fallback processing error for {file_name}: {e}")
        return f"Error: All processing methods failed for {file_name}. File may be corrupted or in unsupported format: {str(e)}"

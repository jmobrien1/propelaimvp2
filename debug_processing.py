import os
import asyncio
import logging
from main import clean_extracted_text  # We only import the cleaner now

# --- Direct Google Cloud Imports ---
from google.cloud import documentai

# --- Configure logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Enhanced Debugging Version of the Processing Function ---
def detailed_document_ai_debug(file_name: str, file_bytes: bytes):
    """
    A highly verbose version of the Document AI processing function
    to inspect the raw API response.
    """
    logger.info("--- 🕵️ VERBOSE DEBUG MODE INITIATED ---")
    try:
        project_id = os.getenv("GCLOUD_PROJECT_ID")
        location = os.getenv("GCLOUD_LOCATION")
        processor_id = os.getenv("GCLOUD_PROCESSOR_ID")

        if not all([project_id, location, processor_id]):
            logger.error(
                "🔴 Critical Error: Google Cloud environment variables are missing."
            )
            return

        logger.info(
            f"Attempting to connect with Project: {project_id}, Location: {location}, Processor: {processor_id}"
        )

        client_options = {
            "api_endpoint": f"{location}-documentai.googleapis.com"
        }
        client = documentai.DocumentProcessorServiceClient(
            client_options=client_options)
        name = client.processor_path(project_id, location, processor_id)

        raw_document = documentai.RawDocument(content=file_bytes,
                                              mime_type="application/pdf")
        request = documentai.ProcessRequest(name=name,
                                            raw_document=raw_document)

        logger.info("Sending request to Google Document AI...")
        result = client.process_document(request=request)
        logger.info("--- ✅ Request Complete. Raw API Result Received ---")

        # --- THIS IS THE MOST IMPORTANT PART ---
        # Print the entire raw result object from Google
        print("\n\n=============== RAW GOOGLE API RESPONSE ===============\n")
        print(result)
        print("\n=======================================================\n\n")

        # Now we analyze the result
        gcloud_document = result.document
        if gcloud_document.text:
            logger.info(
                f"✅ Document object CONTAINS text. Length: {len(gcloud_document.text)}"
            )
        else:
            logger.warning(
                "🔴 Document object contains NO text. The 'document.text' field is empty."
            )

        if gcloud_document.error and gcloud_document.error.message:
            logger.error(
                f"🔴🔴🔴 Google reported an error inside the response: {gcloud_document.error.message}"
            )

        # Continue with original logic for comparison
        cleaned = clean_extracted_text(gcloud_document.text)
        logger.info("--- Final Cleaned Text Preview ---")
        print(cleaned[:500])

    except Exception as e:
        logger.error(f"🔴🔴🔴 A hard exception occurred during the API call: {e}",
                     exc_info=True)


# --- Main execution block ---
if __name__ == "__main__":
    PDF_FILENAME = "DOLRFP.pdf"
    logger.info(f"--- Starting verbose debug for {PDF_FILENAME} ---")
    if not os.path.exists(PDF_FILENAME):
        logger.error(
            f"🔴 ERROR: '{PDF_FILENAME}' not found. Please ensure it is uploaded."
        )
    else:
        with open(PDF_FILENAME, "rb") as f:
            pdf_bytes = f.read()
        detailed_document_ai_debug(PDF_FILENAME, pdf_bytes)

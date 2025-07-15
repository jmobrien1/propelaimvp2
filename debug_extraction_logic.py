import logging
from main import (load_processed_content, extract_pws_section_5_requirements,
                  extract_section_l_requirements_enhanced)

# Configure logging to see output in the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# The name of the file that was successfully processed by the batch job
PROCESSED_FILENAME = "DOLRFP.pdf"


def test_logic():
    """Tests the regex extraction logic on the processed text file."""

    logger.info(
        f"--- 🕵️ Testing Regex Extraction Logic on {PROCESSED_FILENAME} ---")

    # Load the text content we already extracted successfully
    content = load_processed_content(PROCESSED_FILENAME)

    if not content or len(content) < 500:
        logger.error(
            f"🔴 Processed content file ('processed_content/{PROCESSED_FILENAME}.txt') is missing or empty. Please process a file first."
        )
        return

    # --- Test PWS Section 5 Extraction ---
    logger.info("\n--- Testing PWS Section 5 Extraction ---")
    pws_reqs = extract_pws_section_5_requirements(content, PROCESSED_FILENAME)
    if pws_reqs:
        logger.info(f"✅ SUCCESS: Found {len(pws_reqs)} PWS requirements.")
        for i, req in enumerate(pws_reqs[:2]):  # Print first 2
            logger.info(
                f"  - PWS Req {i+1}: Ref='{req.reference}', Text='{req.text[:80]}...'"
            )
    else:
        logger.warning(
            "⚠️ No PWS requirements found. The patterns in `extract_pws_section_5_requirements` may need improvement."
        )

    # --- Test Section L Extraction ---
    logger.info("\n--- Testing Section L Extraction ---")
    sec_l_reqs = extract_section_l_requirements_enhanced(
        content, PROCESSED_FILENAME)
    if sec_l_reqs:
        logger.info(
            f"✅ SUCCESS: Found {len(sec_l_reqs)} Section L requirements.")
        for i, req in enumerate(sec_l_reqs[:2]):  # Print first 2
            logger.info(
                f"  - Sec L Req {i+1}: Ref='{req.reference}', Text='{req.text[:80]}...'"
            )
    else:
        logger.warning(
            "⚠️ No Section L requirements found. The patterns in `extract_section_l_requirements_enhanced` may need improvement."
        )


if __name__ == "__main__":
    test_logic()

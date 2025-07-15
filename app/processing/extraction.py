import re
import logging
from typing import List

# Import from your new modules
from app.models.schemas import RequirementEntry

logger = logging.getLogger(__name__)


# Enhanced Document Category Detection
def detect_document_category(filename: str, content: str) -> str:
    """Enhanced detection for RFP vs PWS documents"""
    filename_lower = filename.lower()
    content_lower = content.lower()

    logger.info(f"Analyzing document category for {filename}")

    # Debug: Check key indicators
    section_l_present = 'section l' in content_lower or 'instructions to offeror' in content_lower
    volume_requirements = 'volume i' in content_lower or 'volume ii' in content_lower
    rfp_indicators = content_lower.count(
        'request for quote') + content_lower.count(
            'rfp') + content_lower.count('solicitation')
    pws_indicators = content_lower.count(
        'performance work statement') + content_lower.count('contractor shall')
    section_5_present = 'section 5' in content_lower or 'section v' in content_lower

    logger.info(
        f"Section L: {section_l_present}, Volumes: {volume_requirements}, RFP indicators: {rfp_indicators}, PWS indicators: {pws_indicators}"
    )

    # Priority 1: Check for Section L content (definitive RFP indicator)
    section_l_patterns = [
        'section l', 'instructions to offeror', 'instructions to quoter',
        'proposal submission', 'volume i', 'volume ii', 'cover letter must',
        'offeror shall submit', 'quoter shall submit'
    ]

    section_l_matches = sum(1 for pattern in section_l_patterns
                            if pattern in content_lower)

    if section_l_matches >= 2:
        logger.info(
            f"✅ Strong Section L indicators ({section_l_matches}) - treating as RFP"
        )
        return 'RFP'

    # Priority 2: Check filename patterns
    if 'rfp' in filename_lower or 'request' in filename_lower or 'solicitation' in filename_lower:
        # Even if it has PWS content, if it's named as RFP and has submission requirements, treat as RFP
        submission_indicators = [
            'shall submit', 'must provide', 'include in proposal',
            'cover letter', 'volume', 'proposal format'
        ]
        submission_count = sum(1 for indicator in submission_indicators
                               if indicator in content_lower)

        if submission_count > 0:
            logger.info(
                f"✅ RFP filename with submission indicators ({submission_count}) - treating as RFP"
            )
            return 'RFP'

    # Priority 3: PWS-specific patterns
    if 'pws' in filename_lower or 'performance work statement' in filename_lower:
        logger.info(f"✅ PWS filename - treating as PWS")
        return 'PWS'

    # Priority 4: Content analysis for PWS
    pws_content_indicators = [
        'performance work statement', 'blanket purchase agreement',
        'scope of work', 'statement of work'
    ]

    pws_section_patterns = [
        'section c', 'section 5', 'section v', 'attachment', '1.0 scope',
        '2.0 background'
    ]

    has_pws_content = any(indicator in content_lower
                          for indicator in pws_content_indicators)
    has_pws_structure = any(pattern in content_lower
                            for pattern in pws_section_patterns)
    shall_count = len(
        re.findall(r'(?:contractor|offeror)\s+shall', content_lower))

    # If significant PWS content and structure
    if has_pws_content and (has_pws_structure or shall_count > 10):
        logger.info(
            f"✅ Strong PWS content: content={has_pws_content}, structure={has_pws_structure}, shall_count={shall_count}"
        )
        return 'PWS'

    # Priority 5: Evaluation criteria and submission format (RFP indicators)
    rfp_specific_patterns = [
        'evaluation criteria', 'technical approach', 'past performance',
        'price evaluation', 'award criteria', 'evaluation factors'
    ]

    rfp_specific_count = sum(1 for pattern in rfp_specific_patterns
                             if pattern in content_lower)

    if rfp_specific_count >= 2:
        logger.info(
            f"✅ RFP evaluation patterns found ({rfp_specific_count}) - treating as RFP"
        )
        return 'RFP'

    # Priority 6: Final decision based on content balance
    if rfp_indicators > pws_indicators and section_l_matches > 0:
        logger.info(f"✅ RFP indicators outweigh PWS - treating as RFP")
        return 'RFP'
    elif pws_indicators > rfp_indicators or shall_count > 5:
        logger.info(f"✅ PWS indicators outweigh RFP - treating as PWS")
        return 'PWS'

    # Default fallback
    if 'rfp' in filename_lower or 'request' in filename_lower:
        logger.info(f"⚠️ Defaulting to RFP based on filename")
        return 'RFP'
    elif 'pws' in filename_lower or 'statement' in filename_lower:
        logger.info(f"⚠️ Defaulting to PWS based on filename")
        return 'PWS'

    logger.warning(f"❌ Could not determine document category for {filename}")
    return 'Unknown'


# Debug function for Section 5
def debug_section_5_content(text: str, filename: str):
    """Debug function to understand Section 5 structure"""
    logger.info(f"=== DEBUGGING SECTION 5 in {filename} ===")

    # Look for Section 5
    section_5_patterns = [
        r'(?:^|\n)(section\s+5[^\n]*)\n(.*?)(?=\n(?:section\s+6|$))',
        r'(?:^|\n)(section\s+v[^\n]*)\n(.*?)(?=\n(?:section\s+v[i]|$))',
        r'(?:^|\n)(5\.?\s+[^\n]*requirements[^\n]*)\n(.*?)(?=\n(?:6\.|$))'
    ]

    for i, pattern in enumerate(section_5_patterns):
        section_5_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if section_5_match:
            title = section_5_match.group(1)
            content = section_5_match.group(2)
            logger.info(f"Found Section 5 with pattern {i+1}: {title}")
            logger.info(f"Content length: {len(content)} characters")
            logger.info(f"First 500 chars: {content[:500]}")

            # Count patterns
            shall_count = len(
                re.findall(r'(?:contractor|offeror)\s+shall', content,
                           re.IGNORECASE))
            task_count = len(re.findall(r'task\s+\d+', content, re.IGNORECASE))
            numbered_count = len(
                re.findall(r'(?:^|\n)\d+\.', content, re.MULTILINE))

            logger.info(
                f"Patterns found - Shall statements: {shall_count}, Tasks: {task_count}, Numbered items: {numbered_count}"
            )
            return content

    logger.warning("❌ Section 5 not found in document")
    logger.info("=== END DEBUG ===")
    return None


# Section 5 Specific PWS Extraction
def extract_pws_section_5_requirements(
        text: str, source_filename: str) -> List[RequirementEntry]:
    """Extract PWS requirements specifically from Section 5 format"""
    requirements = []

    logger.info(f"Starting Section 5 PWS extraction for {source_filename}")

    # Debug the content first
    section_5_content = debug_section_5_content(text, source_filename)

    # STEP 1: Find Section 5 content with multiple patterns
    section_5_patterns = [
        # Section 5 - Requirements and Tasks
        r'(?:^|\n)((?:section\s+)?5\.?\s*[-–]*\s*(?:requirements and tasks|requirements|tasks).*?)\n(.*?)(?=\n(?:section\s+[6-9]|6\.|appendix|attachment|$))',
        # Section V (Roman numeral)
        r'(?:^|\n)((?:section\s+)?v\.?\s*[-–]*\s*(?:requirements and tasks|requirements|tasks).*?)\n(.*?)(?=\n(?:section\s+v[i]|appendix|attachment|$))',
        # Very broad Section 5 capture
        r'(?:^|\n)((?:section\s+)?5\.?\s*[^\n]*)\n(.*?)(?=\n(?:section\s+[6-9]|6\.|appendix|attachment|$))',
        # Alternative broad capture
        r'(?:^|\n)((?:section\s+)?v\.?\s*[^\n]*)\n(.*?)(?=\n(?:section\s+v[i]|appendix|attachment|$))'
    ]

    section_content = ""
    section_title = ""

    for i, pattern in enumerate(section_5_patterns):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            potential_title = match.group(1).strip()
            potential_content = match.group(2).strip()

            # Verify this contains substantial PWS content
            shall_count = len(
                re.findall(r'(?:contractor|offeror)\s+shall',
                           potential_content, re.IGNORECASE))
            task_count = len(
                re.findall(r'task\s+\d+', potential_content, re.IGNORECASE))

            logger.info(
                f"Section 5 candidate {i+1} - Title: {potential_title[:100]}")
            logger.info(
                f"Content length: {len(potential_content)}, Shall statements: {shall_count}, Tasks: {task_count}"
            )

            if len(potential_content) > 1000 and (shall_count > 3
                                                  or task_count > 1):
                section_title = potential_title
                section_content = potential_content
                logger.info(f"✅ Found Section 5 PWS using pattern {i+1}")
                break

    # STEP 2: If Section 5 not found, try other approaches
    if not section_content:
        logger.info("Section 5 not found, trying other PWS patterns...")

        # Try traditional patterns
        traditional_patterns = [
            r'(?:^|\n)(2\.0\s+REQUIREMENTS\.?.*?)\n(.*?)(?=\n(?:3\.0|[3-9]\.0|$))',
            r'(?:^|\n)((?:section\s+)?c\.?\s*[-–]*\s*(?:performance work statement|statement of work).*?)\n(.*?)(?=\n(?:section\s+[d-z]|appendix|attachment|$))'
        ]

        for i, pattern in enumerate(traditional_patterns):
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                section_title = match.group(1).strip()
                section_content = match.group(2).strip()
                logger.info(f"✅ Found PWS using traditional pattern {i+1}")
                break

    if not section_content:
        logger.warning("❌ No PWS section found, trying general fallback")
        return extract_pws_general_fallback(text, source_filename)

    # STEP 3: Extract requirements from Section 5
    requirements = []

    # Pattern 1: Look for numbered tasks (common in Section 5)
    task_pattern = r'(?:^|\n)((?:task\s+)?(\d+)\.?\d*\.?)\s+([^\n]+(?:\n(?!(?:task\s+)?\d+\.)[^\n]*)*)'
    task_matches = list(
        re.finditer(task_pattern, section_content,
                    re.MULTILINE | re.IGNORECASE))

    if task_matches:
        logger.info(f"Found {len(task_matches)} numbered tasks")
        for match in task_matches:
            full_ref = match.group(1).strip()
            task_num = match.group(2)
            task_content = match.group(3).strip()
            task_content = re.sub(r'\s+', ' ', task_content)

            if len(task_content) > 50:
                requirements.append(
                    RequirementEntry(
                        reference=f"5.{task_num}",
                        text=task_content[:500] +
                        "..." if len(task_content) > 500 else task_content,
                        extraction_method="section_5_tasks",
                        source=source_filename,
                        document_section=f"Section 5 - Task {task_num}",
                        requirement_type="performance",
                        confidence=0.95,
                        priority=1))

                logger.info(
                    f"✅ Extracted Task {task_num}: {task_content[:100]}...")

    # Pattern 2: Look for lettered subsections (A., B., C., etc.)
    if not requirements:
        logger.info("No numbered tasks found, trying lettered subsections...")
        letter_pattern = r'(?:^|\n)([A-Z]\.)\s+([^\n]+(?:\n(?!\s*[A-Z]\.)[^\n]*)*)'
        letter_matches = list(
            re.finditer(letter_pattern, section_content, re.MULTILINE))

        if letter_matches:
            logger.info(f"Found {len(letter_matches)} lettered subsections")
            for match in letter_matches:
                ref = match.group(1).strip()
                content = match.group(2).strip()
                content = re.sub(r'\s+', ' ', content)

                if len(content) > 50 and is_meaningful_requirement(content):
                    requirements.append(
                        RequirementEntry(
                            reference=f"5.{ref}",
                            text=content[:500] +
                            "..." if len(content) > 500 else content,
                            extraction_method="section_5_lettered",
                            source=source_filename,
                            document_section=f"Section 5 - Item {ref}",
                            requirement_type="performance",
                            confidence=0.9,
                            priority=1))

    # Pattern 3: Look for "contractor shall" statements
    if not requirements:
        logger.info(
            "No structured requirements found, extracting 'shall' statements..."
        )
        shall_pattern = r'(?:The\s+)?(?:contractor|offeror)\s+shall\s+([^.]+\.)'
        shall_matches = list(
            re.finditer(shall_pattern, section_content,
                        re.MULTILINE | re.IGNORECASE))

        if shall_matches:
            logger.info(f"Found {len(shall_matches)} 'shall' statements")
            for i, match in enumerate(shall_matches, 1):
                req_text = match.group(1).strip()
                req_text = re.sub(r'\s+', ' ', req_text)

                if len(req_text) > 40:
                    requirements.append(
                        RequirementEntry(
                            reference=f"5.REQ-{i}",
                            text=f"The contractor shall {req_text}",
                            extraction_method="section_5_shall",
                            source=source_filename,
                            document_section=f"Section 5 - Requirement {i}",
                            requirement_type="performance",
                            confidence=0.85,
                            priority=1))

                    if len(requirements
                           ) >= 50:  # Reasonable limit for large sections
                        break

    # Pattern 4: Look for any substantial paragraphs that seem like requirements
    if not requirements:
        logger.info(
            "No 'shall' statements found, trying paragraph extraction...")
        paragraph_pattern = r'(?:^|\n\n)([A-Z][^.]*\.(?:\s+[A-Z][^.]*\.)*)'
        paragraph_matches = list(
            re.finditer(paragraph_pattern, section_content, re.MULTILINE))

        meaningful_paragraphs = []
        for match in paragraph_matches:
            para_text = match.group(1).strip()
            para_text = re.sub(r'\s+', ' ', para_text)

            if len(para_text) > 60 and is_meaningful_requirement(para_text):
                meaningful_paragraphs.append(para_text)

        if meaningful_paragraphs:
            logger.info(
                f"Found {len(meaningful_paragraphs)} meaningful paragraphs")
            for i, para_text in enumerate(meaningful_paragraphs[:30],
                                          1):  # Limit to 30
                requirements.append(
                    RequirementEntry(
                        reference=f"5.PARA-{i}",
                        text=para_text[:500] +
                        "..." if len(para_text) > 500 else para_text,
                        extraction_method="section_5_paragraphs",
                        source=source_filename,
                        document_section=f"Section 5 - Paragraph {i}",
                        requirement_type="performance",
                        confidence=0.75,
                        priority=1))

    if not requirements:
        logger.error(
            "❌ No requirements extracted from Section 5 - falling back to general extraction"
        )
        return extract_pws_general_fallback(section_content, source_filename)

    logger.info(
        f"✅ Successfully extracted {len(requirements)} requirements from Section 5"
    )
    return requirements


# Standard PWS extraction (for Section 2.0 format)
def extract_pws_section_2_requirements(
        text: str, source_filename: str) -> List[RequirementEntry]:
    """Extract requirements from standard PWS Section 2.0 format"""
    requirements = []

    logger.info(
        f"Starting standard PWS Section 2.0 extraction for {source_filename}")

    # Check if this might be a Section 5 format instead
    if 'section 5' in text.lower() and len(
            re.findall(r'(?:contractor|offeror)\s+shall', text,
                       re.IGNORECASE)) > 5:
        logger.info(
            "Detected potential Section 5 format, switching extraction method..."
        )
        return extract_pws_section_5_requirements(text, source_filename)

    # Traditional Section 2.0 patterns
    section_2_patterns = [
        r'(?:^|\n)(2\.0\s+REQUIREMENTS\.?.*?)\n(.*?)(?=\n(?:3\.0|[3-9]\.0|##\s*[3-9]|$))',
        r'(?:^|\n)(2\.0\s*REQUIREMENTS\.?.*?)\n(.*?)(?=\n(?:3\.0|[3-9]\.0|$))',
        r'(?:^|\n)(##\s*2\.0.*?REQUIREMENTS.*?)\n(.*?)(?=\n##\s*[3-9]|$)',
        r'(?:^|\n)(2\.\s+REQUIREMENTS\.?.*?)\n(.*?)(?=\n(?:3\.|[3-9]\.|$))'
    ]

    section_2_content = ""
    section_2_title = ""

    for i, pattern in enumerate(section_2_patterns):
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            section_2_title = match.group(1).strip()
            section_2_content = match.group(2).strip()
            logger.info(
                f"Found Section 2.0 using pattern {i+1}: {section_2_title}")
            break

    if not section_2_content:
        logger.warning(
            "Could not find Section 2.0, trying Section 5 format...")
        return extract_pws_section_5_requirements(text, source_filename)

    # Extract 2.X subsections (existing logic)
    subsection_patterns = [
        r'(?:^|\n)(2\.\d+)\s+([A-Z][A-Z\s&,\-\(\)]+\.)\s*(.*?)(?=\n2\.\d+|\n[3-9]\.\d+|$)',
        r'(?:^|\n)(2\.\d+)\s+([A-Z][A-Z\s&,\-\(\)]+[^\.]\s*)\n(.*?)(?=\n2\.\d+|\n[3-9]\.\d+|$)',
        r'(?:^|\n)(2\.\d+)\s+([^\n]+)\n(.*?)(?=\n2\.\d+|\n[3-9]\.\d+|$)',
        r'(?:^|\n)(2\.\d+)\s*([^\n]*.*?)(?=\n2\.\d+|\n[3-9]\.\d+|$)'
    ]

    all_subsection_matches = []

    for pattern_index, pattern in enumerate(subsection_patterns):
        matches = list(
            re.finditer(pattern, section_2_content, re.DOTALL | re.MULTILINE))
        logger.info(
            f"Pattern {pattern_index + 1} found {len(matches)} matches")

        for match in matches:
            subsection_num = match.group(1).strip()

            if len(match.groups()) >= 3:
                subsection_title = match.group(2).strip()
                subsection_content = match.group(3).strip()
            elif len(match.groups()) == 2:
                combined_content = match.group(2).strip()
                lines = combined_content.split('\n', 1)
                if len(lines) >= 2:
                    subsection_title = lines[0].strip()
                    subsection_content = lines[1].strip()
                else:
                    subsection_title = combined_content
                    subsection_content = ""
            else:
                subsection_title = f"Section {subsection_num}"
                subsection_content = ""

            if not any(existing[0] == subsection_num
                       for existing in all_subsection_matches):
                all_subsection_matches.append(
                    (subsection_num, subsection_title, subsection_content))

    all_subsection_matches.sort(key=lambda x: float(x[0].replace('2.', '')))

    for subsection_num, subsection_title, subsection_content in all_subsection_matches:
        if len(subsection_content) > 10 or len(subsection_title) > 10:
            cleaned_content = re.sub(
                r'\s+', ' ', subsection_content) if subsection_content else ""
            full_title = subsection_title.rstrip('.')
            full_title = re.sub(r'^[^\w]*', '', full_title)
            full_title = re.sub(r'\s+', ' ', full_title).strip()

            if cleaned_content and len(cleaned_content) > 30:
                sentences = cleaned_content.split('.')
                first_sentence = sentences[
                    0] + '.' if sentences else cleaned_content
                if len(first_sentence) > 400:
                    first_sentence = first_sentence[:400] + "..."
                requirement_text = f"{full_title}: {first_sentence}"
            else:
                requirement_text = full_title

            requirements.append(
                RequirementEntry(
                    reference=subsection_num,
                    text=requirement_text,
                    extraction_method="pws_section_2_structured",
                    source=source_filename,
                    document_section=f"Section {subsection_num} - {full_title}",
                    requirement_type="performance",
                    confidence=0.95,
                    priority=1))

    if not requirements:
        logger.warning(
            "Section 2.0 extraction failed, trying Section 5 as fallback...")
        return extract_pws_section_5_requirements(text, source_filename)

    logger.info(
        f"Successfully extracted {len(requirements)} Section 2.0 requirements")
    return requirements


def extract_pws_general_fallback(
        text: str, source_filename: str) -> List[RequirementEntry]:
    """Fallback extraction for PWS when structured approach fails"""
    requirements = []

    logger.info("Using PWS general fallback extraction")

    shall_patterns = [
        r'(?:The\s+)?Contractor\s+shall\s+([^.]+\.)',
        r'(?:The\s+)?Offeror\s+shall\s+([^.]+\.)',
    ]

    for pattern in shall_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for i, match in enumerate(matches):
            req_text = match.group(1).strip()
            req_text = re.sub(r'\s+', ' ', req_text)

            if len(req_text) > 50 and is_meaningful_requirement(
                    f"Contractor shall {req_text}"):
                requirements.append(
                    RequirementEntry(
                        reference=f"PWS-FALLBACK-{i+1}",
                        text=f"The Contractor shall {req_text}",
                        extraction_method="pws_fallback",
                        source=source_filename,
                        document_section="PWS General Requirements",
                        requirement_type="performance",
                        confidence=0.6,
                        priority=2))

                if len(requirements) >= 15:
                    break

    logger.info(f"Fallback extraction found {len(requirements)} requirements")
    return requirements


def extract_section_l_requirements_enhanced(
        text: str, source_filename: str) -> List[RequirementEntry]:
    """Enhanced Section L extraction with compliance matrix focus"""
    requirements = []
    logger.info(f"Enhanced Section L extraction for {source_filename}")

    # NEW PATTERN to find requirements under "The Proposal shall:"
    proposal_shall_pattern = r'The\s+Proposal\s+shall:\s*(.*?)(?=\n\n[A-Z])'
    proposal_match = re.search(proposal_shall_pattern, text,
                               re.DOTALL | re.IGNORECASE)

    if proposal_match:
        content = proposal_match.group(1)
        # Find bullet points (-, *, etc.)
        bullet_pattern = r'^\s*[-•*]\s*(.*)'
        bullet_matches = re.finditer(bullet_pattern, content, re.MULTILINE)
        for i, match in enumerate(bullet_matches, 1):
            req_text = match.group(1).strip()
            if len(req_text) > 10:
                requirements.append(
                    RequirementEntry(
                        reference=f"L-PS-{i}",
                        text=req_text,
                        extraction_method="section_l_proposal_shall",
                        source=source_filename,
                        document_section="Section L - Proposal Shall",
                        requirement_type="submission",
                        confidence=0.92,
                        priority=1))
                logger.info(
                    f"✅ Extracted 'Proposal Shall' Req {i}: {req_text[:50]}..."
                )

    # Pattern 1: Volume-based requirements
    volume_pattern = r'For\s+(Volume\s+[IVX]+),?\s+the\s+(?:offeror|quoter|contractor)\s+(?:shall|must|will):\s*(.*?)(?=\n\nFor\s+Volume\s+[IVX]+|Cover Letter|SECTION|$)'
    volume_matches = re.finditer(volume_pattern, text,
                                 re.DOTALL | re.IGNORECASE)

    for volume_match in volume_matches:
        volume_ref = volume_match.group(1)
        volume_content = volume_match.group(2)

        logger.info(
            f"Found {volume_ref} with content length: {len(volume_content)}")

        # Extract lettered sub-requirements
        sub_pattern = r'(?:^|\n)([a-z]\.)\s*([^\n]+(?:\n(?!\s*[a-z]\.)[^\n]*)*)'
        sub_matches = re.finditer(sub_pattern, volume_content, re.MULTILINE)

        for sub_match in sub_matches:
            sub_ref = sub_match.group(1)
            sub_text = re.sub(r'\s+', ' ', sub_match.group(2).strip())

            if len(sub_text) > 10:
                requirements.append(
                    RequirementEntry(
                        reference=f"{volume_ref}.{sub_ref}",
                        text=sub_text,
                        extraction_method="section_l_volume_enhanced",
                        source=source_filename,
                        document_section=f"Section L - {volume_ref}",
                        requirement_type="submission",
                        confidence=0.95,
                        priority=1))
                logger.info(
                    f"✅ Extracted {volume_ref}.{sub_ref}: {sub_text[:50]}...")

    # Pattern 2: Cover letter requirements
    cover_pattern = r'Cover Letter Requirements?:?\s*(.*?)(?=\n\n[A-Z]|SECTION|$)'
    cover_match = re.search(cover_pattern, text, re.DOTALL | re.IGNORECASE)

    if cover_match:
        cover_content = cover_match.group(1)
        logger.info(
            f"Found cover letter section with content length: {len(cover_content)}"
        )

        # Extract numbered or lettered items
        item_pattern = r'(?:^|\n)([A-Z]\.)\s*([^\n]+)'
        item_matches = re.finditer(item_pattern, cover_content, re.MULTILINE)

        for item_match in item_matches:
            ref = item_match.group(1)
            item_text = item_match.group(2).strip()

            if len(item_text) > 5:
                requirements.append(
                    RequirementEntry(
                        reference=f"Cover-{ref}",
                        text=item_text,
                        extraction_method="section_l_cover",
                        source=source_filename,
                        document_section="Section L - Cover Letter",
                        requirement_type="submission",
                        confidence=0.9,
                        priority=1))
                logger.info(f"✅ Extracted Cover-{ref}: {item_text[:50]}...")

    # Pattern 3: Section 5 performance requirements (for comprehensive extraction)
    section_5_pattern = r'SECTION 5[^\n]*\n(.*?)(?=SUBMISSION REQUIREMENTS|$)'
    section_5_match = re.search(section_5_pattern, text,
                                re.DOTALL | re.IGNORECASE)

    if section_5_match:
        section_5_content = section_5_match.group(1)
        logger.info(
            f"Found Section 5 content with length: {len(section_5_content)}")

        # Extract tasks
        task_pattern = r'Task\s+(\d+):\s*([^\n]+)\n([^T]*?)(?=Task\s+\d+|$)'
        task_matches = re.finditer(task_pattern, section_5_content,
                                   re.MULTILINE | re.IGNORECASE)

        for task_match in task_matches:
            task_num = task_match.group(1)
            task_title = task_match.group(2).strip()
            task_content = task_match.group(3).strip()

            # Clean up task content
            task_content = re.sub(r'\s+', ' ', task_content)
            full_task_text = f"{task_title}. {task_content[:300]}..." if len(
                task_content) > 300 else f"{task_title}. {task_content}"

            requirements.append(
                RequirementEntry(
                    reference=f"Section5-Task{task_num}",
                    text=full_task_text,
                    extraction_method="section_l_section5_tasks",
                    source=source_filename,
                    document_section=f"Section 5 - Task {task_num}",
                    requirement_type="performance",
                    confidence=0.85,
                    priority=1))
            logger.info(f"✅ Extracted Section 5 Task {task_num}: {task_title}")

    # Pattern 4: Submission requirements
    submission_pattern = r'SUBMISSION REQUIREMENTS?:?\s*(.*?)'
    submission_match = re.search(submission_pattern, text,
                                 re.DOTALL | re.IGNORECASE)

    if submission_match:
        submission_content = submission_match.group(1)
        logger.info(
            f"Found submission requirements with length: {len(submission_content)}"
        )

        # Split into individual requirements
        submission_lines = [
            line.strip() for line in submission_content.split('\n')
            if line.strip()
        ]

        for i, line in enumerate(submission_lines, 1):
            if len(line) > 10:
                requirements.append(
                    RequirementEntry(
                        reference=f"SUBMIT-{i}",
                        text=line,
                        extraction_method="section_l_submission_req",
                        source=source_filename,
                        document_section="Submission Requirements",
                        requirement_type="submission",
                        confidence=0.9,
                        priority=1))
                logger.info(f"✅ Extracted Submission Req {i}: {line[:50]}...")

    # Pattern 5: Direct submission requirements
    submission_patterns = [
        r'(?:The\s+)?(?:offeror|quoter|contractor)\s+(?:shall|must|will)\s+submit\s+([^.]+\.)',
        r'(?:The\s+)?(?:proposal|quotation|response)\s+(?:shall|must|will)\s+include\s+([^.]+\.)',
        r'Provide\s+([^.]+\.)', r'Include\s+([^.]+\.)', r'Submit\s+([^.]+\.)'
    ]

    for pattern in submission_patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        for i, match in enumerate(matches):
            req_text = re.sub(r'\s+', ' ', match.group(1).strip())

            if len(req_text) > 10 and is_submission_requirement(req_text):
                requirements.append(
                    RequirementEntry(
                        reference=f"L-DIRECT-{i+1}",
                        text=f"Submit {req_text}",
                        extraction_method="section_l_direct_submission",
                        source=source_filename,
                        document_section="Section L - Direct Requirements",
                        requirement_type="submission",
                        confidence=0.8,
                        priority=1))

    logger.info(
        f"Enhanced Section L extraction found {len(requirements)} total requirements"
    )
    return requirements


def extract_section_l_requirements(
        text: str, source_filename: str) -> List[RequirementEntry]:
    """Extract submission requirements from Section L with enhanced patterns"""
    requirements = []

    logger.info(f"Starting Section L extraction for {source_filename}")

    # Volume requirements
    volume_pattern = r'(Volume\s+[IVX]+)[^\n]*\n\nFor\s+(Volume\s+[IVX]+),\s+the\s+(?:offeror|quoter)\s+shall:\s*(.*?)(?=\n\nVolume\s+[IVX]+|\n\n[A-Z]+\.|$)'
    volume_matches = re.finditer(volume_pattern, text,
                                 re.DOTALL | re.IGNORECASE)

    for volume_match in volume_matches:
        volume_ref = volume_match.group(2)
        volume_content = volume_match.group(3)

        logger.info(f"Found {volume_ref} content")

        sub_pattern = r'(?:^|\n)([a-f]\.)\s*([^\n]+(?:\n(?!\s*[a-f]\.)[^\n]*)*)'
        sub_matches = re.finditer(sub_pattern, volume_content, re.MULTILINE)

        for sub_match in sub_matches:
            sub_ref, sub_text = sub_match.groups()
            sub_text = re.sub(r'\s+', ' ', sub_text.strip())

            if len(sub_text) > 25 and is_submission_requirement(sub_text):
                requirements.append(
                    RequirementEntry(reference=f"{volume_ref}.{sub_ref}",
                                     text=sub_text,
                                     extraction_method="section_l_volume",
                                     source=source_filename,
                                     document_section="Section L Instructions",
                                     requirement_type="submission",
                                     confidence=0.9,
                                     priority=1))

    # Traditional L.X.X patterns
    l_pattern = r'(L\.?\d+(?:\.\d+)*)\s*[\:\-\.]?\s*([^\n]+(?:\n(?!\s*L\.?\d+)[^\n]*)*)'
    l_matches = re.finditer(l_pattern, text, re.MULTILINE | re.IGNORECASE)

    for l_match in l_matches:
        ref, req_text = l_match.groups()
        req_text = re.sub(r'\s+', ' ', req_text.strip())

        if len(req_text) > 25 and is_submission_requirement(req_text):
            requirements.append(
                RequirementEntry(reference=ref.strip(),
                                 text=req_text,
                                 extraction_method="section_l_traditional",
                                 source=source_filename,
                                 document_section="Section L Instructions",
                                 requirement_type="submission",
                                 confidence=0.85,
                                 priority=1))

    # Cover letter requirements
    cover_pattern = r'Signed cover letters must provide the following information:\s*(.*?)(?=\n\n[A-Z]|$)'
    cover_match = re.search(cover_pattern, text, re.DOTALL | re.IGNORECASE)

    if cover_match:
        cover_content = cover_match.group(1)
        cover_item_pattern = r'([A-R]\.?\)?\s*)\s*([^\n]+)'
        cover_items = re.finditer(cover_item_pattern, cover_content)

        for cover_item in cover_items:
            ref, req_text = cover_item.groups()
            req_text = req_text.strip()

            clean_ref = re.sub(r'[^\w]', '', ref) + '.'

            if len(req_text
                   ) > 10 and not req_text.lower().startswith('submissions'):
                requirements.append(
                    RequirementEntry(
                        reference=f"Cover-{clean_ref}",
                        text=req_text,
                        extraction_method="section_l_cover_letter",
                        source=source_filename,
                        document_section="Cover Letter Requirements",
                        requirement_type="submission",
                        confidence=0.85,
                        priority=1))

    logger.info(f"Section L extraction found {len(requirements)} requirements")
    return requirements


def is_meaningful_requirement(text: str) -> bool:
    """Enhanced check for meaningful requirements"""
    text_lower = text.lower()

    requirement_indicators = [
        'shall', 'must', 'will', 'should', 'require', 'provide', 'deliver',
        'develop', 'implement', 'maintain', 'support', 'ensure', 'perform',
        'assist', 'facilitate', 'coordinate', 'manage', 'conduct'
    ]

    if not any(indicator in text_lower
               for indicator in requirement_indicators):
        return False

    exclusions = [
        'table of contents', 'page number', 'header', 'footer', 'attachment',
        'appendix', 'see attached', 'refer to', 'as follows', 'page', 'of',
        'bpa performance work statement', 'contractor personnel',
        'years of experience', 'must possess', 'shall have at least',
        'certification'
    ]

    if any(exclusion in text_lower for exclusion in exclusions):
        return False

    words = text.split()
    return len(words) >= 6 and len(text) >= 40


def is_meaningful_sub_requirement(text: str) -> bool:
    """Check for meaningful sub-requirements"""
    text_lower = text.lower()

    action_indicators = [
        'support', 'assist', 'develop', 'provide', 'implement', 'maintain',
        'facilitate', 'coordinate', 'manage', 'conduct', 'ensure', 'perform',
        'create', 'establish', 'deliver', 'execute', 'review', 'update',
        'analyze', 'identify', 'determine', 'define', 'configure'
    ]

    if not any(indicator in text_lower for indicator in action_indicators):
        return False

    exclusions = [
        'years of experience', 'certification', 'must possess', 'page',
        'attachment', 'appendix'
    ]

    if any(exclusion in text_lower for exclusion in exclusions):
        return False

    words = text.split()
    return len(words) >= 5 and len(text) >= 30


def is_submission_requirement(text: str) -> bool:
    """Check if text represents a submission/proposal requirement"""
    text_lower = text.lower()

    submission_indicators = [
        'submit', 'provide', 'include', 'demonstrate', 'describe', 'explain',
        'proposal', 'quotation', 'response', 'documentation', 'resume',
        'experience', 'approach', 'plan', 'volume', 'complete', 'name'
    ]

    return any(indicator in text_lower for indicator in submission_indicators)

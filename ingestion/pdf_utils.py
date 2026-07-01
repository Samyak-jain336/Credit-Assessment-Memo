"""
pdf_utils.py — PDF page extraction and preprocessing utilities for
Indian SME Credit Appraisal Memo (CAM) generation.

Uses pdfplumber exclusively. No LLM, database, or vector store calls.
Pure computation for page classification, two-column layout detection,
table serialization, and text chunking.
"""

import pdfplumber

__all__ = [
    "detect_two_column_layout",
    "split_two_column_page",
    "classify_page",
    "serialize_table_to_text",
    "extract_table_page",
    "extract_narrative_page",
    "extract_mixed_page",
    "chunk_narrative_text",
    "chunk_table_text",
]

# ---------------------------------------------------------------------------
# Validated thresholds (tested against three real Indian SME annual reports)
# ---------------------------------------------------------------------------

MIN_CHARS_THRESHOLD = 200
"""Pages with fewer than 200 extractable characters are blank/image-only."""

TABLE_DENSITY_THRESHOLD = 0.18
"""Numeric density above which a page is considered a table page
(provided it also has <= TABLE_MAX_CHARS)."""

TABLE_MAX_CHARS = 5000
"""Pages exceeding this character count are not pure table pages even if
density is high — they are Notes to Accounts (mixed)."""

MIXED_DENSITY_THRESHOLD = 0.10
"""Numeric density above which a page is considered mixed (tables + prose),
when it does not meet the stricter table criteria."""

TWO_COLUMN_MIN_WIDTH_RATIO = 0.2
"""Each detected column must span at least 20% of the page width to be
considered a real column (filters out narrow sidebars)."""

NARRATIVE_CHUNK_SIZE = 1000
"""Narrative text is split into 1000-character chunks."""

NARRATIVE_CHUNK_OVERLAP = 150
"""Consecutive narrative chunks overlap by 150 characters."""


# ---------------------------------------------------------------------------
# Two-column layout detection and splitting
# ---------------------------------------------------------------------------

def detect_two_column_layout(page) -> bool:
    """Detect whether a pdfplumber page has a two-column side-by-side layout.

    Args:
        page: A pdfplumber page object.

    Returns:
        True if the page contains two substantial side-by-side text columns,
        False otherwise.

    Algorithm:
        Divides the page width into 20 vertical slices, counts word midpoints
        per slice, finds the sparsest slice in the middle 60% (slices 4–15)
        as the column gap, then checks that both the left and right clusters
        each span at least TWO_COLUMN_MIN_WIDTH_RATIO of the page width.
    """
    words = page.extract_words()
    if len(words) < 10:
        return False

    page_width = page.width
    num_slices = 20
    slice_width = page_width / num_slices

    # Count word x-midpoints per slice
    slice_counts = [0] * num_slices
    word_mids = []
    for w in words:
        x_mid = (w["x0"] + w["x1"]) / 2
        word_mids.append(x_mid)
        idx = int(x_mid / slice_width)
        if idx >= num_slices:
            idx = num_slices - 1
        slice_counts[idx] += 1

    # Find the sparsest slice in the middle 60% (indices 4–15 inclusive)
    middle_start = 4
    middle_end = 15
    min_count = None
    min_slice_idx = middle_start
    for i in range(middle_start, middle_end + 1):
        if min_count is None or slice_counts[i] < min_count:
            min_count = slice_counts[i]
            min_slice_idx = i

    # The gap x-coordinate is the centre of that sparsest slice
    gap_x = (min_slice_idx + 0.5) * slice_width

    # Partition words into left and right clusters
    left_xs = [xm for xm in word_mids if xm < gap_x]
    right_xs = [xm for xm in word_mids if xm > gap_x]

    if not left_xs or not right_xs:
        return False

    left_span = max(left_xs) - min(left_xs)
    right_span = max(right_xs) - min(right_xs)

    return (
        left_span >= TWO_COLUMN_MIN_WIDTH_RATIO * page_width
        and right_span >= TWO_COLUMN_MIN_WIDTH_RATIO * page_width
    )


def split_two_column_page(page) -> tuple[str, str]:
    """Split a two-column pdfplumber page into left and right text.

    Args:
        page: A pdfplumber page object that has been identified as
              having a two-column layout.

    Returns:
        A tuple (left_text, right_text) of plain strings extracted
        from the left and right halves of the page.
    """
    mid_x = page.width / 2
    left_crop = page.crop((0, 0, mid_x, page.height))
    right_crop = page.crop((mid_x, 0, page.width, page.height))

    left_text = left_crop.extract_text() or ""
    right_text = right_crop.extract_text() or ""

    return (left_text, right_text)


# ---------------------------------------------------------------------------
# Page classification
# ---------------------------------------------------------------------------

def classify_page(page_text: str) -> str:
    """Classify a page's extracted text into a processing category.

    Args:
        page_text: Plain text string extracted from a PDF page.

    Returns:
        One of:
        - "skip"      — blank or image-only page (< 200 chars)
        - "table"     — high numeric density, structured financial data
        - "mixed"     — tables interleaved with prose (e.g. Notes to Accounts)
        - "narrative" — mostly prose (Director's Report, MDA, etc.)
    """
    total = len(page_text)
    if total < MIN_CHARS_THRESHOLD:
        return "skip"

    digits = sum(1 for ch in page_text if ch.isdigit())
    density = digits / total

    if density > TABLE_DENSITY_THRESHOLD and total <= TABLE_MAX_CHARS:
        return "table"
    if density > MIXED_DENSITY_THRESHOLD:
        return "mixed"
    return "narrative"


# ---------------------------------------------------------------------------
# Table serialization helpers
# ---------------------------------------------------------------------------

def _forward_fill_headers(row: list) -> list:
    """Replace None values in a header row with the last seen non-None value.

    Handles merged header cells — pdfplumber returns None for cells that
    were merged in the original PDF table.

    Args:
        row: A list representing one row of a pdfplumber table
             (typically the header row).

    Returns:
        A new list with None values forward-filled.

    Example:
        >>> _forward_fill_headers(["FY2024", None, "FY2023", None])
        ['FY2024', 'FY2024', 'FY2023', 'FY2023']
    """
    filled = []
    last_value = None
    for cell in row:
        if cell is not None and str(cell).strip():
            last_value = str(cell).strip()
        filled.append(last_value)
    return filled


def serialize_table_to_text(table_2d: list) -> str:
    """Convert a 2D pdfplumber table into flat natural language text.

    Args:
        table_2d: A 2D list (list of lists) from pdfplumber's
                  extract_tables(). Row 0 is the header row.

    Returns:
        A newline-joined string of "<row_label> for <col_header> is <value>"
        statements, suitable for embedding. Returns "" if the table has
        fewer than 2 rows.

    Example output:
        Revenue for FY2024 is 5907.39
        Revenue for FY2023 is 3950.55
        PAT for FY2024 is 648.01
    """
    if len(table_2d) < 2:
        return ""

    raw_headers = table_2d[0]
    filled_headers = _forward_fill_headers(raw_headers)

    lines = []

    for row in table_2d[1:]:
        if not row:
            continue

        row_label = str(row[0]).strip() if row[0] is not None else ""
        if row_label is None or (isinstance(row_label, str) and row_label.strip() == ""):
            continue

        for col_idx in range(1, len(row)):
            cell = row[col_idx]

            # Determine the column header (if available)
            col_header = None
            if col_idx < len(filled_headers):
                col_header = filled_headers[col_idx]

            if cell is None or (isinstance(cell, str) and cell.strip() == ""):
                # Cell is empty — check if this is a merged-cell artifact
                raw_header = raw_headers[col_idx] if col_idx < len(raw_headers) else ""
                if raw_header is None:
                    # Merged cell artifact: report as "not reported"
                    if col_header is not None:
                        lines.append(
                            f"{row_label} for {col_header} is not reported"
                        )
                # Else: truly empty cell — skip silently
            else:
                # Cell has a value
                if col_header is not None:
                    lines.append(f"{row_label} for {col_header} is {cell}")
                else:
                    lines.append(f"{row_label}: {cell}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Page-level extraction by type
# ---------------------------------------------------------------------------

def extract_table_page(page) -> str:
    """Extract and serialize all tables from a pdfplumber page classified
    as a table page.

    Args:
        page: A pdfplumber page object.

    Returns:
        A string of serialized table text. Falls back to raw
        page.extract_text() if no tables are detected.
    """
    tables = page.extract_tables()

    if not tables:
        return page.extract_text() or ""

    serialized = []
    for table in tables:
        result = serialize_table_to_text(table)
        if result:
            serialized.append(result)

    if not serialized:
        return page.extract_text() or ""

    return "\n\n".join(serialized)


def extract_narrative_page(page_text: str) -> str:
    """Clean raw narrative text by removing blank lines and excess whitespace.

    Args:
        page_text: Raw text string from pdfplumber's extract_text().

    Returns:
        Cleaned text with empty lines removed and each line stripped
        of leading/trailing whitespace.
    """
    lines = page_text.split("\n")
    cleaned = [line.strip() for line in lines]
    cleaned = [line for line in cleaned if line]
    return "\n".join(cleaned)


def extract_mixed_page(page) -> str:
    """Extract content from a mixed page (tables + narrative prose).

    Serializes tables separately to preserve row-column relationships,
    then appends the cleaned raw page text for surrounding prose context.

    Args:
        page: A pdfplumber page object classified as "mixed".

    Returns:
        Combined string: serialized tables first, then cleaned narrative,
        separated by double newlines.
    """
    sections = []

    tables = page.extract_tables()
    if tables:
        for table in tables:
            result = serialize_table_to_text(table)
            if result:
                sections.append(result)

    raw_text = page.extract_text() or ""
    cleaned = extract_narrative_page(raw_text)
    if cleaned:
        cleaned_lines = [
            line for line in cleaned.splitlines()
            if not all(c.isdigit() or c in " ,.-()%" for c in line)
        ]
        cleaned = "\n".join(cleaned_lines)
    if cleaned:
        sections.append(cleaned)

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_narrative_text(text: str, page_num: int) -> list[dict]:
    """Split narrative text into overlapping fixed-size chunks.

    Args:
        text:     Cleaned narrative text string.
        page_num: The page number this text was extracted from.

    Returns:
        A list of dicts, each containing:
        - "text":         the chunk text
        - "chunk_index":  sequential int starting at 0
        - "page_number":  the source page number
    """
    chunks = []
    step = NARRATIVE_CHUNK_SIZE - NARRATIVE_CHUNK_OVERLAP
    idx = 0
    start = 0

    while start < len(text):
        chunk_text = text[start : start + NARRATIVE_CHUNK_SIZE]
        if chunk_text.strip() and len(chunk_text.strip()) > 50:
            chunks.append({
                "text": chunk_text,
                "chunk_index": idx,
                "page_number": page_num,
            })
            idx += 1
        start += step

    return chunks


def chunk_table_text(
    text: str, page_num: int, chunk_idx_start: int = 0
) -> list[dict]:
    """Wrap serialized table text into a single chunk (tables are never split).

    Splitting a table across chunks would destroy row-column relationships.

    Args:
        text:            Serialized table text string.
        page_num:        The page number this table was extracted from.
        chunk_idx_start: Starting chunk index (default 0).

    Returns:
        A list containing exactly one chunk dict, or an empty list if the
        input text is empty/whitespace. The dict contains:
        - "text":         the table text
        - "chunk_index":  chunk_idx_start
        - "page_number":  the source page number
    """
    if not text or not text.strip():
        return []

    return [{
        "text": text,
        "chunk_index": chunk_idx_start,
        "page_number": page_num,
    }]

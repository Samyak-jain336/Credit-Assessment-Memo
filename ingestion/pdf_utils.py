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
    """Detect whether a pdfplumber page has two side-by-side text columns.

    Takes a pdfplumber page object, divides its width into 20 vertical slices,
    and finds the sparsest slice in the middle 60% as the column gap. Returns
    True only if both the left and right word clusters each span at least
    TWO_COLUMN_MIN_WIDTH_RATIO (20%) of the total page width.
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
    """Split a two-column pdfplumber page into separate left and right text.

    Crops the page at the horizontal midpoint, extracts text from each half
    independently, and returns a tuple (left_text, right_text). Should only
    be called on pages where detect_two_column_layout() returned True.
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
    """Classify extracted page text as 'skip', 'table', 'mixed', or 'narrative'.

    Takes a plain text string and uses character count and digit-to-total density
    ratios against validated thresholds to determine the page type. Returns 'skip'
    for near-blank pages, 'table' for high-density short pages, 'mixed' for
    moderate density, and 'narrative' for low-density prose pages.
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
    """Forward-fill None gaps in a header row caused by merged PDF cells.

    Takes a list (typically row 0 of a pdfplumber table) and replaces each
    None or whitespace-only cell with the last seen non-empty value. Returns
    a new list, e.g. ["FY2024", None, "FY2023", None] becomes
    ["FY2024", "FY2024", "FY2023", "FY2023"].
    """
    filled = []
    last_value = None
    for cell in row:
        if cell is not None and str(cell).strip():
            last_value = str(cell).strip()
        filled.append(last_value)
    return filled


def serialize_table_to_text(table_2d: list) -> str:
    """Serialize a 2D pdfplumber table into flat natural-language statements.

    Takes a list-of-lists (row 0 = headers, remaining rows = data) and produces
    lines like "Revenue for FY2024 is 5907.39". Handles merged header cells via
    forward-fill and marks missing values under merged columns as "not reported".
    Returns an empty string if the table has fewer than 2 rows.
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
    """Extract and serialize all tables from a page classified as 'table'.

    Calls page.extract_tables(), serializes each table into natural-language
    text via serialize_table_to_text(), and joins them with double newlines.
    Falls back to raw page.extract_text() if no tables are detected.
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
    """Clean raw narrative text by stripping whitespace and removing blank lines.

    Takes the raw text string from pdfplumber's extract_text(), strips each
    line of leading/trailing whitespace, drops all empty lines, and returns
    the cleaned text rejoined with newlines.
    """
    lines = page_text.split("\n")
    cleaned = [line.strip() for line in lines]
    cleaned = [line for line in cleaned if line]
    return "\n".join(cleaned)


def extract_mixed_page(page) -> str:
    """Extract content from a mixed page containing both tables and prose.

    Serializes tables first to preserve row-column structure, then appends
    the cleaned page text (with pure-numeric lines stripped out) for prose
    context. All sections are joined with double newlines.
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
    """Split narrative text into overlapping fixed-size chunks for embedding.

    Slides a 1000-char window across the text, advancing by 850 chars each step
    (150-char overlap). Returns a list of dicts with keys 'text', 'chunk_index',
    and 'page_number'. Chunks shorter than 50 chars after stripping are skipped.
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
    """Wrap serialized table text into a single chunk — tables are never split.

    Splitting would destroy row-column relationships, so the entire table is
    kept as one chunk dict with keys 'text', 'chunk_index', and 'page_number'.
    Returns an empty list if the input text is empty or whitespace-only.
    """
    if not text or not text.strip():
        return []

    return [{
        "text": text,
        "chunk_index": chunk_idx_start,
        "page_number": page_num,
    }]

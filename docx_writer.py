"""
docx_writer.py

Creates the final Credit Assessment Memo (CAM) document.

Responsibilities
----------------
- Create a Word document.
- Add title and company metadata.
- Insert generated CAM sections.
- Save the document.
"""

from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT

from stubs import CAMSection


class DocumentWriter:
    """
    Responsible for writing the final CAM document.
    """

    def __init__(self):
        self.document = Document()

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _add_title(
        self,
        company_name: str,
        fiscal_year: int,
    ):

        heading = self.document.add_heading(
            "Credit Assessment Memo",
            level=0,
        )

        heading.alignment = WD_PARAGRAPH_ALIGNMENT.CENTER

        p = self.document.add_paragraph()

        p.add_run("Company: ").bold = True
        p.add_run(company_name)

        p.add_run("\nFinancial Year: ").bold = True
        p.add_run(str(fiscal_year))

        self.document.add_page_break()

    # ---------------------------------------------------------
    # Section Writer
    # ---------------------------------------------------------

    def _add_section(
        self,
        section: CAMSection,
    ):

        heading = self.document.add_heading(
            section.title,
            level=1,
        )

        heading.style.font.size = Pt(16)

        body = self.document.add_paragraph()

        body.style.font.size = Pt(11)

        body.add_run(section.content)

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------

    def write(
        self,
        company_name: str,
        fiscal_year: int,
        sections: list[CAMSection],
        output_path: str,
    ) -> str:
        """
        Generate the final CAM document.
        """

        self._add_title(
            company_name,
            fiscal_year,
        )

        for section in sections:

            self._add_section(section)

        self.document.save(output_path)

        return output_path
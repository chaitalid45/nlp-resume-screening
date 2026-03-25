"""
Extracts structured information from resume sections.
Parses experience entries, education records, and bullet points.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExperienceEntry:
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""
    duration_months: Optional[int] = None


@dataclass
class EducationEntry:
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    graduation_year: Optional[str] = None
    gpa: Optional[str] = None


class SectionExtractor:
    """
    Extracts structured data from raw section text.

    Usage:
        se = SectionExtractor()
        exp_entries = se.extract_experience(sections["experience"])
        edu_entries = se.extract_education(sections["education"])
        bullets     = se.extract_bullets(sections["skills"])
    """

    # Date patterns
    YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
    DATE_RE = re.compile(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
        r"\.?\s*(19|20)?\d{2}",
        re.IGNORECASE,
    )
    DURATION_RE = re.compile(r"(present|current|now)", re.IGNORECASE)
    GPA_RE = re.compile(r"GPA\s*[:\-]?\s*(\d\.\d+)", re.IGNORECASE)

    # Degree keywords
    DEGREE_KEYWORDS = [
        "bachelor", "b.sc", "b.s.", "b.eng", "b.tech",
        "master", "m.sc", "m.s.", "m.eng", "mba", "m.tech",
        "phd", "ph.d", "doctorate", "associate", "diploma",
    ]

    def extract_experience(self, text: str) -> list[ExperienceEntry]:
        """
        Parse experience section into a list of ExperienceEntry objects.
        Uses heuristics: company/title on consecutive lines, dates in line.
        """
        entries: list[ExperienceEntry] = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        i = 0
        while i < len(lines):
            line = lines[i]

            # A line with a date likely marks the start of an entry
            if self.DATE_RE.search(line) or self.YEAR_RE.search(line):
                dates = self._extract_dates(line)
                title_company = self._extract_title_company(lines, i)
                desc_lines: list[str] = []

                # Collect bullet points / description until next date-line
                j = i + 1
                while j < len(lines) and not (
                    self.DATE_RE.search(lines[j]) and j != i
                ):
                    desc_lines.append(lines[j])
                    j += 1

                entries.append(
                    ExperienceEntry(
                        company=title_company.get("company", ""),
                        title=title_company.get("title", ""),
                        start_date=dates.get("start"),
                        end_date=dates.get("end"),
                        description="\n".join(desc_lines),
                    )
                )
                i = j
            else:
                i += 1

        return entries

    def extract_education(self, text: str) -> list[EducationEntry]:
        """Parse education section into EducationEntry objects."""
        entries: list[EducationEntry] = []
        lines = [l.strip() for l in text.splitlines() if l.strip()]

        for line in lines:
            lower = line.lower()
            if any(deg in lower for deg in self.DEGREE_KEYWORDS):
                gpa_match = self.GPA_RE.search(line)
                year_match = self.YEAR_RE.search(line)
                entries.append(
                    EducationEntry(
                        institution=self._guess_institution(lines, line),
                        degree=self._extract_degree(line),
                        graduation_year=year_match.group() if year_match else None,
                        gpa=gpa_match.group(1) if gpa_match else None,
                    )
                )

        return entries

    def extract_bullets(self, text: str) -> list[str]:
        """
        Extract bullet points from a section.
        Handles •, -, *, >, and plain numbered lists.
        """
        bullet_re = re.compile(r"^[\•\-\*\>\◦\▸\▪]\s*")
        number_re = re.compile(r"^\d+[\.\)]\s*")
        bullets: list[str] = []

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            clean = bullet_re.sub("", line)
            clean = number_re.sub("", clean).strip()
            if clean and len(clean) > 3:
                bullets.append(clean)

        return bullets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_dates(self, line: str) -> dict[str, Optional[str]]:
        dates = self.DATE_RE.findall(line)
        is_present = bool(self.DURATION_RE.search(line))
        return {
            "start": " ".join(dates[0]) if dates else None,
            "end": "Present" if is_present else (" ".join(dates[1]) if len(dates) > 1 else None),
        }

    def _extract_title_company(
        self, lines: list[str], idx: int
    ) -> dict[str, str]:
        """
        Try to infer job title and company from lines around a date line.
        Common patterns:
            "Software Engineer | Acme Corp | Jan 2022 – Dec 2023"
            "Software Engineer\nAcme Corp\nJan 2022 – Dec 2023"
        """
        result = {"title": "", "company": ""}
        if idx == 0:
            return result

        prev = lines[idx - 1] if idx > 0 else ""
        curr = lines[idx]

        # Pattern: "Title | Company | date"
        parts = re.split(r"\s*[\|,]\s*", curr)
        if len(parts) >= 2 and not self.DATE_RE.match(parts[0]):
            result["title"] = parts[0].strip()
            result["company"] = parts[1].strip()
        elif prev:
            # Title on previous line, current line has dates
            title_company = re.split(r"\s*[@,at]\s*", prev, maxsplit=1)
            result["title"] = title_company[0].strip()
            result["company"] = title_company[1].strip() if len(title_company) > 1 else ""

        return result

    def _guess_institution(self, all_lines: list[str], degree_line: str) -> str:
        """Look for the institution name near the degree line."""
        idx = all_lines.index(degree_line) if degree_line in all_lines else -1
        if idx > 0:
            candidate = all_lines[idx - 1]
            # If it doesn't look like a degree, treat it as the institution
            if not any(d in candidate.lower() for d in self.DEGREE_KEYWORDS):
                return candidate
        return ""

    def _extract_degree(self, line: str) -> Optional[str]:
        lower = line.lower()
        for deg in self.DEGREE_KEYWORDS:
            if deg in lower:
                return deg.capitalize()
        return None
"""
Normalises resume and job description text before vectorisation.
Removes noise, standardises whitespace, and optionally applies lemmatisation.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional


class TextCleaner:
    """
    Cleans and normalises raw resume / JD text.

    Usage:
        cleaner = TextCleaner()
        clean = cleaner.clean("  Hello,   World!\n\n  ")
        # → "hello world"
    """

    # Characters to strip (keep alphanumeric + basic punctuation)
    _KEEP_RE = re.compile(r"[^a-zA-Z0-9\s.,;:()\-+#/&@]")
    # Collapse multiple whitespace / newlines
    _WS_RE = re.compile(r"\s+")
    # Remove URLs
    _URL_RE = re.compile(r"https?://\S+|www\.\S+")
    # Remove email addresses (already extracted; don't pollute embeddings)
    _EMAIL_RE = re.compile(r"\S+@\S+\.\S+")

    def clean(
        self,
        text: str,
        lowercase: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        normalize_unicode: bool = True,
    ) -> str:
        """Return a cleaned version of `text`."""
        if normalize_unicode:
            text = unicodedata.normalize("NFKD", text)
            text = text.encode("ascii", "ignore").decode("ascii")

        if remove_urls:
            text = self._URL_RE.sub(" ", text)

        if remove_emails:
            text = self._EMAIL_RE.sub(" ", text)

        # Strip non-essential characters
        text = self._KEEP_RE.sub(" ", text)

        if lowercase:
            text = text.lower()

        # Collapse whitespace
        text = self._WS_RE.sub(" ", text).strip()
        return text

    def clean_for_embedding(self, text: str, max_tokens: int = 512) -> str:
        """
        Minimal cleaning suitable for sentence-transformer input.
        Preserves case and casing-sensitive terms like acronyms.
        Truncates to `max_tokens` word-tokens to avoid exceeding model limit.
        """
        text = self._URL_RE.sub("", text)
        text = self._EMAIL_RE.sub("", text)
        text = self._WS_RE.sub(" ", text).strip()

        words = text.split()
        if len(words) > max_tokens:
            text = " ".join(words[:max_tokens])

        return text

    def clean_section(self, section_text: str) -> str:
        """Light cleaning for individual resume sections."""
        text = self._URL_RE.sub("", section_text)
        text = self._WS_RE.sub(" ", text).strip()
        return text
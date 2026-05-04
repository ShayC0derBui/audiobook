"""Tests for EPUB parser and chapter extraction."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from epub_audiobook.epub.extractor import _clean_html_to_text, _generate_chapter_id


class TestCleanHtmlToText:
    def test_basic_paragraph_extraction(self):
        html = b"""
        <html><body>
            <p>First paragraph of the chapter.</p>
            <p>Second paragraph with more text.</p>
        </body></html>
        """
        result = _clean_html_to_text(html)
        assert "First paragraph" in result
        assert "Second paragraph" in result

    def test_removes_script_and_style(self):
        html = b"""
        <html><body>
            <script>var x = 1;</script>
            <style>.foo { color: red; }</style>
            <p>Readable content here.</p>
        </body></html>
        """
        result = _clean_html_to_text(html)
        assert "var x" not in result
        assert "color: red" not in result
        assert "Readable content" in result

    def test_removes_nav_and_footer(self):
        html = b"""
        <html><body>
            <nav><a href="toc.html">Table of Contents</a></nav>
            <p>Chapter text.</p>
            <footer>Page 42</footer>
        </body></html>
        """
        result = _clean_html_to_text(html)
        assert "Table of Contents" not in result
        assert "Page 42" not in result
        assert "Chapter text" in result

    def test_converts_footnote_references(self):
        html = b"""
        <html><body>
            <p>Important claim<sup><a href="#fn1">1</a></sup> in the text.</p>
        </body></html>
        """
        result = _clean_html_to_text(html)
        assert "(note 1)" in result

    def test_normalizes_whitespace(self):
        html = b"""
        <html><body>
            <p>Text   with    extra     spaces.</p>
        </body></html>
        """
        result = _clean_html_to_text(html)
        assert "  " not in result


class TestChapterIdGeneration:
    def test_stable_id(self):
        id1 = _generate_chapter_id(0, "chapter1.xhtml")
        id2 = _generate_chapter_id(0, "chapter1.xhtml")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = _generate_chapter_id(0, "chapter1.xhtml")
        id2 = _generate_chapter_id(1, "chapter2.xhtml")
        assert id1 != id2

    def test_id_length(self):
        id1 = _generate_chapter_id(0, "chapter1.xhtml")
        assert len(id1) == 12

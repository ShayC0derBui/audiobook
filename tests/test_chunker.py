"""Tests for sentence-aware text chunking."""

import pytest
from epub_audiobook.tts.chunker import chunk_text, _split_sentences


class TestSplitSentences:
    def test_basic_splitting(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_preserves_question_marks(self):
        text = "Is this working? Yes it is. Great!"
        sentences = _split_sentences(text)
        assert any("working?" in s for s in sentences)

    def test_handles_empty_text(self):
        sentences = _split_sentences("")
        assert sentences == []


class TestChunkText:
    def test_single_short_text_one_chunk(self):
        text = "A short sentence."
        chunks = chunk_text(text, "ch001", max_chars=500)
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_respects_max_chars(self):
        text = "First sentence is here. " * 50  # ~1200 chars
        chunks = chunk_text(text, "ch001", max_chars=200)
        for chunk in chunks:
            # Allow some overflow for sentence boundaries
            assert chunk.char_count <= 250  # some tolerance

    def test_chunk_ids_are_sequential(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        chunks = chunk_text(text, "ch001", max_chars=30)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i
            assert f"c{i:04d}" in chunk.id

    def test_chunk_ids_include_chapter(self):
        text = "Some text here."
        chunks = chunk_text(text, "abc123")
        assert all(c.chapter_id == "abc123" for c in chunks)

    def test_long_sentence_force_split(self):
        text = "word " * 200  # 1000 chars, single "sentence"
        chunks = chunk_text(text, "ch001", max_chars=100)
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk.char_count <= 105  # small tolerance for word boundaries

    def test_empty_text_no_chunks(self):
        chunks = chunk_text("", "ch001")
        assert chunks == []

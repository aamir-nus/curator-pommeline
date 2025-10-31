"""
Semantic chunking for markdown documents with respect for structure.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

from ..utils.logger import get_logger
from ..config import settings

logger = get_logger("chunker")


@dataclass
class DocumentChunk:
    """A chunk of a document with metadata."""
    content: str
    chunk_id: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict


class SemanticChunker:
    """Chunks documents semantically while respecting markdown structure."""

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 min_chunk_size: int = 50):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size

        # Markdown patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|(.+)\|$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'^```[\w]*\n.*?\n```$', re.MULTILINE | re.DOTALL)
        self.list_pattern = re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE)

    def chunk_document(self, file_path: Path) -> List[DocumentChunk]:
        """Chunk a single document file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

        return self.chunk_text(content, str(file_path))

    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """Chunk text content semantically."""
        if not text.strip():
            return []

        # Pre-process text
        text = self._preprocess_text(text)

        # Identify structural boundaries
        sections = self._identify_sections(text)

        # Chunk each section
        chunks = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(section, source, chunk_index)
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        logger.info(f"Created {len(chunks)} chunks from {source}")
        return chunks

    def _preprocess_text(self, text: str) -> str:
        """Pre-process text before chunking."""
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Remove excessive whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Ensure proper spacing around headers
        text = re.sub(r'\n(#{1,6})', r'\n\1', text)
        text = re.sub(r'(#{1,6})[^\n]', r'\1 ', text)

        return text.strip()

    def _identify_sections(self, text: str) -> List[Tuple[str, str, int, int]]:
        """Identify document sections based on headers and other boundaries."""
        sections = []

        # Find all headers
        headers = []
        for match in self.header_pattern.finditer(text):
            level = len(match.group(1))
            title = match.group(2)
            start_pos = match.start()
            headers.append((start_pos, level, title))

        # If no headers, treat entire text as one section
        if not headers:
            return [(text, "document", 0, len(text))]

        # Create sections based on headers
        for i, (start_pos, level, title) in enumerate(headers):
            end_pos = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            section_text = text[start_pos:end_pos].strip()
            section_type = f"header_{level}"

            # Skip empty sections
            if len(section_text) >= self.min_chunk_size:
                sections.append((section_text, section_type, start_pos, end_pos))

        return sections

    def _chunk_section(self, section: Tuple[str, str, int, int], source: str, start_index: int) -> List[DocumentChunk]:
        """Chunk a single section."""
        content, section_type, start_char, end_char = section

        # Check for special content types that should be preserved
        if self._is_table(content):
            return self._chunk_table(content, source, start_index, start_char)
        elif self._is_code_block(content):
            return self._chunk_code_block(content, source, start_index, start_char)
        elif self._is_list(content):
            return self._chunk_list(content, source, start_index, start_char)
        else:
            return self._chunk_paragraphs(content, source, start_index, start_char)

    def _is_table(self, text: str) -> bool:
        """Check if text is a markdown table."""
        lines = text.split('\n')
        table_lines = 0
        for line in lines:
            if line.strip().startswith('|') and line.strip().endswith('|'):
                table_lines += 1
        return table_lines >= 2 and table_lines / len(lines) > 0.5

    def _is_code_block(self, text: str) -> bool:
        """Check if text is a code block."""
        return text.startswith('```') and text.endswith('```')

    def _is_list(self, text: str) -> bool:
        """Check if text is primarily a list."""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        list_lines = sum(1 for line in lines if re.match(r'^[-*+]\s+', line))
        return list_lines >= 2 and list_lines / len(lines) > 0.5

    def _chunk_table(self, content: str, source: str, start_index: int, start_char: int) -> List[DocumentChunk]:
        """Chunk table content - preserve as single chunk."""
        chunk = DocumentChunk(
            content=content,
            chunk_id=f"{source}_table_{start_index}",
            source_file=source,
            chunk_index=start_index,
            start_char=start_char,
            end_char=start_char + len(content),
            metadata={"type": "table", "section_type": "table"}
        )
        return [chunk]

    def _chunk_code_block(self, content: str, source: str, start_index: int, start_char: int) -> List[DocumentChunk]:
        """Chunk code block content - preserve as single chunk."""
        # Extract language if specified
        lines = content.split('\n')
        language = lines[0].replace('```', '').strip() if lines else ""

        chunk = DocumentChunk(
            content=content,
            chunk_id=f"{source}_code_{start_index}",
            source_file=source,
            chunk_index=start_index,
            start_char=start_char,
            end_char=start_char + len(content),
            metadata={
                "type": "code_block",
                "language": language,
                "section_type": "code"
            }
        )
        return [chunk]

    def _chunk_list(self, content: str, source: str, start_index: int, start_char: int) -> List[DocumentChunk]:
        """Chunk list content."""
        # Split list items
        list_items = re.split(r'\n(?=[\s]*[-*+]\s+)', content)
        chunks = []

        current_chunk = ""
        chunk_start = start_char

        for i, item in enumerate(list_items):
            item = item.strip()
            if not item:
                continue

            # Check if adding this item would exceed chunk size
            if len(current_chunk) + len(item) > self.chunk_size and current_chunk:
                # Create chunk from accumulated items
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    source,
                    start_index + len(chunks),
                    chunk_start,
                    chunk_start + len(current_chunk),
                    {"type": "list", "section_type": "list"}
                )
                chunks.append(chunk)

                # Start new chunk
                current_chunk = item
                chunk_start += len(current_chunk)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n" + item
                else:
                    current_chunk = item

        # Add remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                source,
                start_index + len(chunks),
                chunk_start,
                chunk_start + len(current_chunk),
                {"type": "list", "section_type": "list"}
            )
            chunks.append(chunk)

        return chunks

    def _chunk_paragraphs(self, content: str, source: str, start_index: int, start_char: int) -> List[DocumentChunk]:
        """Chunk paragraph content with semantic boundaries."""
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        chunks = []

        current_chunk = ""
        chunk_start = start_char

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph would exceed chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                # Create chunk from accumulated paragraphs
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    source,
                    start_index + len(chunks),
                    chunk_start,
                    chunk_start + len(current_chunk),
                    {"type": "paragraphs", "section_type": "content"}
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.chunk_overlap//5:] if len(words) > self.chunk_overlap//5 else []
                current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                chunk_start += len(current_chunk) - len(" ".join(overlap_words)) - 2
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                source,
                start_index + len(chunks),
                chunk_start,
                chunk_start + len(current_chunk),
                {"type": "paragraphs", "section_type": "content"}
            )
            chunks.append(chunk)

        return chunks

    def _create_chunk(self, content: str, source: str, index: int, start_char: int, end_char: int, metadata: dict) -> DocumentChunk:
        """Create a document chunk."""
        return DocumentChunk(
            content=content,
            chunk_id=f"{source}_chunk_{index}",
            source_file=source,
            chunk_index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata
        )

    def chunk_directory(self, directory: Path, pattern: str = "*.md") -> List[DocumentChunk]:
        """Chunk all markdown files in a directory."""
        chunks = []
        file_paths = list(directory.glob(pattern))

        logger.info(f"Found {len(file_paths)} files to chunk in {directory}")

        for file_path in file_paths:
            if file_path.is_file():
                file_chunks = self.chunk_document(file_path)
                chunks.extend(file_chunks)

        logger.info(f"Created total of {len(chunks)} chunks from {len(file_paths)} files")
        return chunks
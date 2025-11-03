"""
Semantic chunking for markdown documents with respect for structure.
"""

import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

from ..utils.logger import get_logger
from ..config import settings
from ..utils.llm_pipeline import LLM

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
    """
    Chunks documents semantically while respecting markdown structure.

    This class provides intelligent document chunking that:
    - Preserves markdown structure (headers, tables, code blocks)
    - Implements smart section merging for small sections
    - Uses sentence boundary completion instead of truncation
    - Maintains context through strategic overlap
    """

    def __init__(self,
                 chunk_size: int = None,
                 chunk_overlap: int = None,
                 min_chunk_size: int = 50,
                 context_retention:bool=False):
        """
        Initialize the semantic chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            min_chunk_size: Minimum size for a chunk to be considered valid
            context_retention: Whether to enable contextual retrieval summaries
        """
        
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.min_chunk_size = min_chunk_size

        #set a flag for context retention.
        # If true, use Anthropic's contextual retrieval approach to add a 'summary' of this chunk
        # Contextual Retrieval: placing chunk and what it does in the context of the previous 5 + next 5 chunks
        self.context_retention = context_retention
        self.llm = LLM(system_prompt="You are a helpful assistant that summarizes document chunks for context retention." \
                                    "You add keywords and a brief summary of the chunk content.",
                        model="glm-4.5-air")

        # Markdown patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.table_pattern = re.compile(r'^\|(.+)\|$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'^```[\w]*\n.*?\n```$', re.MULTILINE | re.DOTALL)
        self.list_pattern = re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE)

        # Sentence boundary pattern for smart completion
        self.sentence_pattern = re.compile(r'(?<=[.!?])\s+')
        self.word_pattern = re.compile(r'\w+')

    def chunk_document(self, file_path: Path) -> List[DocumentChunk]:
        """
        Chunk a single markdown document file.

        Args:
            file_path: Path to the markdown file to chunk

        Returns:
            List of DocumentChunk objects containing the chunked content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

        return self.chunk_text(content, str(file_path))

    def chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        Chunk text content semantically while preserving structure.

        Args:
            text: The text content to chunk
            source: Source identifier for the text

        Returns:
            List of DocumentChunk objects containing the chunked content
        """
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
        """
        Identify document sections based on headers and merge small sections intelligently.

        Args:
            text: The text to analyze for sections

        Returns:
            List of tuples containing (section_text, section_type, start_pos, end_pos)
        """
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
            sections.append((section_text, section_type, start_pos, end_pos))

        # Merge small sections with adjacent sections
        sections = self._merge_small_sections(sections)

        return sections

    def _merge_small_sections(self, sections: List[Tuple[str, str, int, int]]) -> List[Tuple[str, str, int, int]]:
        """
        Merge sections that are too small with adjacent sections to maintain semantic coherence.

        Args:
            sections: List of sections to potentially merge

        Returns:
            List of merged sections where small sections have been combined
        """
        if not sections:
            return sections

        merged_sections = []
        i = 0
        n = len(sections)

        while i < n:
            current_text, current_type, current_start, current_end = sections[i]

            # If current section is too small, try to merge with adjacent sections
            if len(current_text) < self.min_chunk_size:
                merged_text = current_text
                merged_start = current_start
                merged_end = current_end
                merged_type = current_type

                # Try to merge with next section first (prefer forward merging)
                if i + 1 < n:
                    next_text, next_type, _, next_end = sections[i + 1]
                    combined_text = current_text + "\n\n" + next_text

                    # Check if combined size is reasonable (not too large)
                    if len(combined_text) <= self.chunk_size * 1.5:  # Allow some flexibility
                        merged_text = combined_text
                        merged_end = next_end
                        merged_type = f"{current_type}_merged_{next_type}"
                        i += 1  # Skip the next section as it's now merged
                    elif i > 0:
                        # If can't merge forward, try to merge with previous section
                        prev_text, prev_type, prev_start, _ = merged_sections[-1]
                        combined_text = prev_text + "\n\n" + current_text

                        if len(combined_text) <= self.chunk_size * 1.5:
                            # Update the last merged section
                            merged_sections[-1] = (
                                combined_text,
                                f"{prev_type}_merged_{current_type}",
                                prev_start,
                                current_end
                            )
                            i += 1
                            continue

                # Add the (potentially merged) section if it's still too small to stand alone
                if len(merged_text) >= self.min_chunk_size or not merged_sections:
                    merged_sections.append((merged_text, merged_type, merged_start, merged_end))
            else:
                merged_sections.append((current_text, current_type, current_start, current_end))

            i += 1

        return merged_sections

    def _find_sentence_boundary(self, text: str, start_pos: int, max_length: int) -> int:
        """
        Find the nearest sentence boundary within the length limit to avoid mid-sentence truncation.

        Args:
            text: The text to search for sentence boundaries
            start_pos: Starting position in the text
            max_length: Maximum allowed length from start position

        Returns:
            Position of the nearest sentence boundary within the limit
        """
        if start_pos >= len(text):
            return len(text)

        # Calculate the search end position
        search_end = min(start_pos + max_length, len(text))
        search_text = text[start_pos:search_end]

        # Find all sentence boundaries in the search text
        sentence_matches = list(self.sentence_pattern.finditer(search_text))

        if sentence_matches:
            # Return the position of the last sentence boundary within the limit
            last_match = sentence_matches[-1]
            return start_pos + last_match.end()

        # If no sentence boundaries found, try to find word boundaries
        word_matches = list(self.word_pattern.finditer(search_text))
        if word_matches:
            # Return the position of the last word boundary within the limit
            last_word_match = word_matches[-1]
            return start_pos + last_word_match.end()

        # If no boundaries found, return the max length
        return search_end

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
        """
        Chunk paragraph content with sentence boundary completion to avoid mid-sentence truncation.

        Args:
            content: The paragraph content to chunk
            source: Source identifier for the content
            start_index: Starting index for chunks
            start_char: Starting character position

        Returns:
            List of DocumentChunk objects containing the chunked paragraphs
        """
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
                # Create chunk from accumulated paragraphs with sentence boundary completion
                if len(current_chunk) > self.chunk_size:
                    # Find sentence boundary to avoid truncation
                    boundary_pos = self._find_sentence_boundary(current_chunk, 0, self.chunk_size)
                    chunk_content = current_chunk[:boundary_pos].strip()

                    chunk = self._create_chunk(
                        chunk_content,
                        source,
                        start_index + len(chunks),
                        chunk_start,
                        chunk_start + len(chunk_content),
                        {"type": "paragraphs", "section_type": "content"}
                    )
                    chunks.append(chunk)

                    # Start new chunk with overlap from remaining content
                    remaining_content = current_chunk[boundary_pos:].strip()
                    overlap_size = min(self.chunk_overlap, len(remaining_content))

                    if overlap_size > 0:
                        # Find word boundary for overlap
                        overlap_boundary = self._find_sentence_boundary(remaining_content, 0, overlap_size)
                        overlap_content = remaining_content[:overlap_boundary].strip()
                        current_chunk = overlap_content + "\n\n" + paragraph
                        chunk_start += boundary_pos
                    else:
                        current_chunk = paragraph
                        chunk_start += boundary_pos
                else:
                    # Normal chunk creation without truncation
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
                    overlap_content = " ".join(overlap_words)
                    current_chunk = overlap_content + "\n\n" + paragraph
                    chunk_start += len(current_chunk) - len(overlap_content)
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add remaining content with sentence boundary completion if needed
        if current_chunk.strip():
            final_content = current_chunk.strip()
            if len(final_content) > self.chunk_size:
                # Find sentence boundary for final chunk
                boundary_pos = self._find_sentence_boundary(final_content, 0, self.chunk_size)
                final_content = final_content[:boundary_pos].strip()

            chunk = self._create_chunk(
                final_content,
                source,
                start_index + len(chunks),
                chunk_start,
                chunk_start + len(final_content),
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
        """
        Chunk all markdown files in a directory.

        Args:
            directory: Directory path containing markdown files
            pattern: File pattern to match (default: "*.md")

        Returns:
            List of DocumentChunk objects from all files
        """
        chunks = []
        file_paths = list(directory.glob(pattern))

        logger.info(f"Found {len(file_paths)} files to chunk in {directory}")

        for file_path in file_paths:
            if file_path.is_file():
                file_chunks = self.chunk_document(file_path)
                chunks.extend(file_chunks)

        logger.info(f"Created total of {len(chunks)} chunks from {len(file_paths)} files")
        return chunks
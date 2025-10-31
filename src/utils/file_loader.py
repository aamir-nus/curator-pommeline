"""
File loading utilities for documents.
"""

import os
import pathlib
from typing import List, Dict, Any, Optional
import markdown

from .logger import get_logger

logger = get_logger("file_loader")


def load_markdown_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a markdown file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'source': file_path,
            'content': content,
            'metadata': {
                'file_type': 'markdown',
                'file_size': len(content),
                'file_name': os.path.basename(file_path)
            }
        }
    except Exception as e:
        logger.error(f"Failed to load markdown file {file_path}: {e}")
        return None


def load_text_file(file_path: str) -> Dict[str, Any]:
    """Load and parse a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return {
            'source': file_path,
            'content': content,
            'metadata': {
                'file_type': 'text',
                'file_size': len(content),
                'file_name': os.path.basename(file_path)
            }
        }
    except Exception as e:
        logger.error(f"Failed to load text file {file_path}: {e}")
        return None


def load_documents_from_directory(directory: str,
                                 recursive: bool = True,
                                 file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Load all supported documents from a directory.

    Args:
        directory: Directory path to load documents from
        recursive: Whether to search subdirectories
        file_extensions: List of file extensions to include (default: ['.md', '.txt'])

    Returns:
        List of document dictionaries
    """
    if file_extensions is None:
        file_extensions = ['.md', '.txt']

    directory_path = pathlib.Path(directory)
    if not directory_path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []

    documents = []
    pattern = "**/*" if recursive else "*"

    for file_path in directory_path.glob(pattern):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()

            if file_ext in file_extensions:
                logger.debug(f"Loading file: {file_path}")

                if file_ext == '.md':
                    doc = load_markdown_file(str(file_path))
                elif file_ext == '.txt':
                    doc = load_text_file(str(file_path))
                else:
                    logger.warning(f"Unsupported file type: {file_ext}")
                    continue

                if doc:
                    documents.append(doc)

    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def load_single_document(file_path: str) -> Optional[Dict[str, Any]]:
    """Load a single document file."""
    file_path = pathlib.Path(file_path)

    if not file_path.exists():
        logger.error(f"File does not exist: {file_path}")
        return None

    file_ext = file_path.suffix.lower()

    if file_ext == '.md':
        return load_markdown_file(str(file_path))
    elif file_ext == '.txt':
        return load_text_file(str(file_path))
    else:
        logger.error(f"Unsupported file type: {file_ext}")
        return None
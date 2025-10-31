"""
Link masking utilities to prevent hallucinated URLs in responses.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("link_masker")


@dataclass
class LinkMapping:
    """Mapping between masked link tokens and actual URLs."""
    token: str
    original_url: str
    context: str = ""
    link_id: int = 0


class LinkMasker:
    """Masks and unmasks URLs in text to prevent hallucination."""

    def __init__(self):
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?',
            re.IGNORECASE
        )
        self.link_counter = 0
        self.current_mappings: Dict[str, LinkMapping] = {}

    def mask_links(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Replace all URLs in text with <link_id> tokens.

        Args:
            text: Input text containing URLs

        Returns:
            Tuple of (masked_text, link_mapping_dict)
        """
        if not text:
            return text, {}

        masked_text = text
        link_mapping = {}
        matches = list(self.url_pattern.finditer(text))

        # Process matches in reverse order to maintain string positions
        for match in reversed(matches):
            original_url = match.group()
            start_pos = match.start()
            end_pos = match.end()

            # Get context around the link
            context_start = max(0, start_pos - 50)
            context_end = min(len(text), end_pos + 50)
            context = text[context_start:context_end]

            # Create link token
            link_token = f"<link_{self.link_counter}>"

            # Create mapping
            mapping = LinkMapping(
                token=link_token,
                original_url=original_url,
                context=context.strip(),
                link_id=self.link_counter
            )

            # Store mapping
            link_id_str = str(self.link_counter)
            link_mapping[link_id_str] = original_url
            self.current_mappings[link_token] = mapping

            # Replace URL with token
            masked_text = masked_text[:start_pos] + link_token + masked_text[end_pos:]

            self.link_counter += 1

        logger.debug(f"Masked {len(matches)} URLs in text")
        return masked_text, link_mapping

    def unmask_links(self, masked_text: str, link_mapping: Dict[str, str] = None) -> str:
        """
        Replace <link_id> tokens with actual URLs.

        Args:
            masked_text: Text containing masked link tokens
            link_mapping: Optional mapping dictionary (uses current mappings if not provided)

        Returns:
            Text with URLs restored
        """
        if not masked_text:
            return masked_text

        if link_mapping is None:
            link_mapping = {str(mapping.link_id): mapping.original_url
                          for mapping in self.current_mappings.values()}

        unmasked_text = masked_text

        # Find all link tokens
        link_token_pattern = re.compile(r'<link_(\d+)>')

        def replace_token(match):
            link_id = match.group(1)
            original_url = link_mapping.get(link_id, f"<MISSING_LINK_{link_id}>")
            return original_url

        unmasked_text = link_token_pattern.sub(replace_token, unmasked_text)

        logger.debug(f"Unmasked links in text")
        return unmasked_text

    def mask_links_in_dict(self, data: Dict) -> Tuple[Dict, Dict[str, str]]:
        """
        Mask links in all string values within a dictionary.

        Args:
            data: Dictionary potentially containing URLs

        Returns:
            Tuple of (masked_dict, link_mapping)
        """
        masked_data = {}
        combined_link_mapping = {}

        def mask_value(value):
            if isinstance(value, str):
                masked_value, link_mapping = self.mask_links(value)
                return masked_value, link_mapping
            elif isinstance(value, dict):
                masked_dict = {}
                dict_mapping = {}
                for k, v in value.items():
                    if isinstance(v, str):
                        masked_dict[k], dict_mapping = self.mask_value(v)
                    else:
                        masked_dict[k] = v
                return masked_dict, dict_mapping
            elif isinstance(value, list):
                masked_list = []
                list_mapping = {}
                for i, item in enumerate(value):
                    if isinstance(item, str):
                        masked_item, item_mapping = self.mask_value(item)
                        masked_list.append(masked_item)
                        list_mapping.update(item_mapping)
                    else:
                        masked_list.append(item)
                return masked_list, list_mapping
            else:
                return value, {}

        for key, value in data.items():
            if isinstance(value, str):
                masked_data[key], link_mapping = self.mask_value(value)
                combined_link_mapping.update(link_mapping)
            elif isinstance(value, (dict, list)):
                masked_data[key], link_mapping = self.mask_value(value)
                combined_link_mapping.update(link_mapping)
            else:
                masked_data[key] = value

        return masked_data, combined_link_mapping

    def extract_links_from_text(self, text: str) -> List[str]:
        """Extract all URLs from text without masking."""
        matches = self.url_pattern.findall(text)
        return matches

    def has_links(self, text: str) -> bool:
        """Check if text contains any URLs."""
        return bool(self.url_pattern.search(text))

    def count_links(self, text: str) -> int:
        """Count the number of URLs in text."""
        return len(self.url_pattern.findall(text))

    def validate_link_tokens(self, text: str, expected_mapping: Dict[str, str]) -> bool:
        """
        Validate that all link tokens in text have corresponding mappings.

        Args:
            text: Text containing link tokens
            expected_mapping: Expected link token to URL mapping

        Returns:
            True if all tokens have mappings, False otherwise
        """
        link_token_pattern = re.compile(r'<link_(\d+)>')
        found_tokens = set(link_token_pattern.findall(text))
        expected_tokens = set(expected_mapping.keys())

        return found_tokens.issubset(expected_tokens)

    def get_current_mappings(self) -> Dict[str, LinkMapping]:
        """Get all current link mappings."""
        return self.current_mappings.copy()

    def clear_mappings(self):
        """Clear all current link mappings."""
        self.current_mappings.clear()
        self.link_counter = 0
        logger.debug("Cleared all link mappings")

    def get_mapping_stats(self) -> Dict[str, int]:
        """Get statistics about current mappings."""
        return {
            "total_mappings": len(self.current_mappings),
            "link_counter": self.link_counter,
            "unique_domains": len(set(mapping.original_url.split('/')[2]
                                   for mapping in self.current_mappings.values()
                                   if '://' in mapping.original_url))
        }


# Global link masker instance
link_masker = LinkMasker()


def get_link_masker() -> LinkMasker:
    """Get the global link masker instance."""
    return link_masker


def mask_text_links(text: str) -> Tuple[str, Dict[str, str]]:
    """Mask links in text using global masker."""
    return link_masker.mask_links(text)


def unmask_text_links(masked_text: str, link_mapping: Dict[str, str]) -> str:
    """Unmask links in text using global masker."""
    return link_masker.unmask_links(masked_text, link_mapping)
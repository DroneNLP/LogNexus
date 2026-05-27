import re
import os
import argparse
from typing import List


def pretokenize_log_message(text: str) -> List[str]:
    """
    Pre-tokenize drone flight log messages for NER-based segmentation.

    Rules:
    1. Ensure text ends with punctuation (add period if none)
    2. Remove wrapping quotes (single/double) while preserving apostrophes
    3. Split on space
    4. Keep multiple dots (...) as a single token
    5. Separate specified punctuation (.,;:!?) from tokens

    Args:
        text: Raw log message string

    Returns:
        List of pre-tokenized strings
    """
    if not text or not text.strip():
        return []

    # Step 1: Ensure text ends with punctuation
    text = text.strip()
    if text and text[-1] not in '.!?;:':
        text += '.'

    # Step 2: Remove wrapping quotes
    # Remove leading/trailing quotes from the entire text
    text = text.strip('"\'')

    # Remove quotes that wrap individual words/phrases
    # But preserve apostrophes (contractions and possessives, including plural possessives)

    # Remove opening quotes: space or start followed by quote followed by alphanumeric
    text = re.sub(r'(^|\s)["\'](?=\w)', r'\1', text)

    # Remove closing quotes, but NOT apostrophes for possessives
    # Only remove quotes that are NOT preceded by 's' or other letters (to preserve possessives)
    # Pattern: Remove quotes after word characters, but only if NOT forming a possessive
    # We'll be more conservative: only remove quotes that are clearly wrapping (after non-letter or at specific positions)

    # Remove double quotes after word characters
    text = re.sub(r'(?<=\w)"(?=\s|[.,;:!?]|$)', '', text)

    # Remove single quotes only if they're NOT part of a possessive pattern
    # A possessive pattern is: letter + 's + quote OR letters + quote (plural possessive)
    # So we only remove single quotes that come after non-letter characters or are clearly wrapping
    # This is tricky, so let's only remove single quotes that are definitely wrapping quotes
    # (after space/punctuation or at start, before space/punctuation or at end)
    text = re.sub(r"(?<=[.,;:!?\s])'(?=\s|[.,;:!?]|$)", '', text)
    text = re.sub(r"(^)'(?=\s|[.,;:!?]|$)", '', text)

    # Step 3: Initial split on space
    tokens = text.split()

    # Step 4 & 5: Process each token
    final_tokens = []

    for token in tokens:
        # Check if token is just ellipsis or contains ellipsis
        if '...' in token:
            # Split around ellipsis while keeping it
            parts = token.split('...')
            for i, part in enumerate(parts):
                if i > 0:
                    final_tokens.append('...')
                if part:
                    # Process the part for other punctuation
                    final_tokens.extend(_separate_punctuation(part))
        else:
            # Separate specified punctuation
            final_tokens.extend(_separate_punctuation(token))

    # Remove any empty strings
    final_tokens = [t for t in final_tokens if t]

    return final_tokens


def _separate_punctuation(token: str) -> List[str]:
    """
    Separate specified punctuation marks (.,;:!?) from a token.
    Preserves apostrophes within words (including plural possessives).
    Preserves periods in decimal numbers.

    Args:
        token: Single token string

    Returns:
        List of tokens with punctuation separated
    """
    if not token:
        return []

    result = []
    current = ""

    for i, char in enumerate(token):
        if char == '.':
            # Check if this period is part of a decimal number
            # Look behind and ahead to see if we have digits
            is_decimal = False
            if i > 0 and i < len(token) - 1:
                if token[i-1].isdigit() and token[i+1].isdigit():
                    is_decimal = True

            if is_decimal:
                # Keep period as part of the number
                current += char
            else:
                # Separate period as punctuation
                if current:
                    result.append(current)
                    current = ""
                result.append(char)
        elif char in ',;:!?':
            # Add accumulated characters
            if current:
                result.append(current)
                current = ""
            # Add punctuation as separate token
            result.append(char)
        elif char == "'":
            # Always keep apostrophe with the word (handles contractions and possessives)
            # This includes singular possessive (drone's) and plural possessive (drones')
            current += char
        else:
            current += char

    # Add any remaining characters
    if current:
        result.append(current)

    return result


def download_model_cli():
    """
    CLI entry point for downloading models.
    """
    parser = argparse.ArgumentParser(description="Download the LogNexus NER model from Hugging Face.")
    parser.add_argument('--model_name', type=str, default="swardiantara/LogNexus-distilbert-base-uncased",
                        help='Hugging Face model ID (e.g., "swardiantara/LogNexus-distilbert-base-uncased")')
    parser.add_argument('--model_dir', type=str, default='./model',
                        help='Local directory to store the model files.')
    args = parser.parse_args()

    print(f"[-] Downloading model '{args.model_name}' to '{args.model_dir}'...")
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(args.model_dir, exist_ok=True)
        snapshot_download(repo_id=args.model_name, local_dir=args.model_dir, local_dir_use_symlinks=False)
        print(f"[+] Model successfully downloaded to {args.model_dir}")
    except ImportError as e:
        print(f"[!] Download failed: huggingface_hub is required. {e}")
        exit(1)
    except Exception as e:
        print(f"[!] Download failed: {e}")
        exit(1)

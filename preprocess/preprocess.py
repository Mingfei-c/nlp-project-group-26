from bs4 import BeautifulSoup, NavigableString
from pydantic import BaseModel, Field
from typing import List, Optional
import re
import hashlib
import asyncio
import json
from pathlib import Path

# ============================================================================
# Data Models
# ============================================================================

class TextBlock(BaseModel):
    """
    Represents a single block of text extracted from HTML.
    
    Attributes:
        tag (str): The HTML tag name (e.g., 'p', 'h1', 'div')
        text (str): The extracted text content from this block
    """
    tag: str = Field(..., description="HTML tag name")
    text: str = Field(..., description="Text content of the block")

class ParsedHTML(BaseModel):
    """
    Represents the complete parsed HTML document with metadata.
    
    Attributes:
        url (Optional[str]): URL extracted from <text id="..."> if present (CleanEval format)
        file_number (Optional[str]): File number extracted from the filename (characters before extension)
        total_blocks (int): Total number of block-level elements found in original HTML
        retained_blocks (int): Number of content blocks that were extracted and retained
        ignored_blocks (int): Number of blocks that were removed (script, style, etc.)
        blocks (List[TextBlock]): List of extracted text blocks with their tags
    """
    url: Optional[str] = Field(None, description="Source URL if available")
    file_number: Optional[str] = Field(None, description="File number from filename (before extension)")
    content_hash: str = Field(..., description="SHA-256 hash of the original HTML content")
    total_blocks: int = Field(..., description="Total block-level elements in HTML")
    retained_blocks: int = Field(..., description="Number of content blocks retained")
    ignored_blocks: int = Field(..., description="Number of blocks removed as noise")
    blocks: List[TextBlock] = Field(default_factory=list, description="List of retained text blocks")

# ============================================================================
# HTML Preprocessing Constants
# ============================================================================


# Define block-level elements (require newlines)
BLOCK_TAGS = {
    'address', 'article', 'aside', 'blockquote', 'canvas', 'dd', 'div', 'dl', 'dt', 
    'fieldset', 'figcaption', 'figure', 'footer', 'form', 'h1', 'h2', 'h3', 'h4', 
    'h5', 'h6', 'header', 'hr', 'main', 'nav', 'noscript', 'ol', 'p', 'pre', 
    'section', 'table', 'tfoot', 'ul', 'video',
    # Table-related elements (for handling old table-based layouts)
    'td', 'th', 'tr'
}

# Define tags to be removed
IGNORE_TAGS = {'script', 'style', 'noscript', 'meta', 'title'}

# Supported HTML file extensions
HTML_SUFFIXES = {'.html', '.htm'}

def preprocess_html(html_content):
    """
    Parse HTML string into DOM tree, merge inline tags, and preserve block-level structure.
    
    Args:
        html_content (str): Raw HTML string
        
    Returns:
        str: Preprocessed plain text with paragraphs separated by newlines
    """
    if not html_content:
        return ""
    
    # CleanEval special format handling: extract content from <text> wrapper if present
    html_content, _ = _extract_from_text_wrapper(html_content)  # Ignore URL for plain text extraction
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # 1. Remove noise tags
    for tag in soup(IGNORE_TAGS):
        tag.decompose()
        
    # 2. Recursively extract text
    text_parts = []
    _traverse_and_extract(soup, text_parts)
    
    # 3. Merge and clean
    full_text = "".join(text_parts)
    
    # Normalize whitespace: merge multiple spaces into one
    full_text = re.sub(r' +', ' ', full_text)
    
    # Fix split words where single capital letter is separated from rest of word
    # Pattern: single capital letter + space + lowercase letter (e.g., "E ncourage" -> "Encourage")
    full_text = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', full_text)
    
    # Clean excessive empty lines (keep single newlines)
    lines = [line.strip() for line in full_text.split('\n')]
    clean_text = '\n'.join(line for line in lines if line)
    
    return clean_text

def _extract_from_text_wrapper(html_content):
    """
    Handle CleanEval special format: <text id="URL">...</text>
    Extract content from <text> tag and the URL if this format is detected.
    
    Returns:
        tuple: (html_content, url) where url is None if not found
    """
    # Check if wrapped by <text> tag and extract URL
    match = re.search(r'<text[^>]*id=["\']([^"\']+)["\'][^>]*>(.*)</text>', html_content, re.DOTALL | re.IGNORECASE)
    if match:
        url = match.group(1)
        content = match.group(2)
        return content, url
    
    # If no <text> tag, just return content with None URL
    return html_content, None

def extract_blocks(html_content, return_stats=False):
    """
    Extract block-level elements and their text content (merge inline tags) from HTML string.
    Uses leaf-node extraction method to properly handle old table-based layouts.
    
    Args:
        html_content (str): Raw HTML string
        return_stats (bool): If True, return tuple (blocks, stats_dict) with statistics
        
    Returns:
        list or tuple: If return_stats=False, returns list of {'tag': tag_name, 'text': text_content}
                      If return_stats=True, returns (blocks_list, {'total': int, 'retained': int, 'ignored': int})
    """
    if not html_content:
        if return_stats:
            return [], {'total': 0, 'retained': 0, 'ignored': 0}
        return []
    
    # CleanEval special format handling
    html_content, _ = _extract_from_text_wrapper(html_content)  # Ignore URL for now (will be handled in parse_html_file)
        
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Count total block-level elements before removal
    total_blocks = len(soup.find_all(list(BLOCK_TAGS)))
    
    # 1. Remove noise tags
    for tag in soup(IGNORE_TAGS):
        tag.decompose()
    
    # 2. Extract all text leaf nodes
    leaves = _get_text_leaves(soup, [])
    
    # 3. Group by semantic parent tag
    blocks = _group_leaves_by_semantic_parent(leaves)
    
    retained_blocks = len(blocks)
    
    # Calculate ignored blocks (blocks not retained - either removed or merged)
    ignored_blocks = total_blocks - retained_blocks
    
    if return_stats:
        stats = {
            'total': total_blocks,
            'retained': retained_blocks,
            'ignored': ignored_blocks
        }
        return blocks, stats
    
    return blocks

# Define semantic tags (tags with meaningful content)
SEMANTIC_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'dt', 'dd', 'blockquote', 'pre', 'article', 'section', 'div'}

# Define layout tags (tags used only for layout, with minimal content significance)
LAYOUT_TAGS = {'table', 'tr', 'td', 'th', 'tbody', 'thead', 'tfoot', 'html', 'body', 'center', 'font'}

def _get_text_leaves(node, tag_path):
    """
    Recursively extract all text leaf nodes and their tag paths (similar to BoilerNet's get_leaves)
    
    Args:
        node: BeautifulSoup node
        tag_path: Current tag path list
        
    Returns:
        list: [(text_string, tag_path), ...]
    """
    from bs4 import NavigableString
    
    result = []
    
    # If it's a text node
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            result.append((text, tag_path))
        return result
    
    # If it's an element node, update tag path
    new_tag_path = tag_path + [node.name] if hasattr(node, 'name') and node.name else tag_path
    
    # Recursively process child nodes
    for child in node.children:
        result.extend(_get_text_leaves(child, new_tag_path))
    
    return result

def _find_semantic_parent(tag_path):
    """
    Find the nearest semantic tag from the tag path
    Skip layout tags (table, tr, td, etc.)
    
    Args:
        tag_path: Tag path list, e.g., ['html', 'body', 'table', 'tr', 'td', 'p']
        
    Returns:
        str: Semantic tag name, e.g., 'p'; returns 'div' if not found
    """
    # Search backwards for the first semantic tag
    for tag in reversed(tag_path):
        if tag in SEMANTIC_TAGS:
            return tag
    
    # If no semantic tag is found, return 'div' as default
    return 'div'

def _group_leaves_by_semantic_parent(leaves):
    """
    Group text leaf nodes by semantic parent tag
    Consecutive leaf nodes with the same semantic tag are merged into one block
    
    Args:
        leaves: [(text, tag_path), ...] list
        
    Returns:
        list: [{'tag': tag_name, 'text': text_content}, ...]
    """
    if not leaves:
        return []
    
    blocks = []
    current_tag = None
    current_texts = []
    
    for text, tag_path in leaves:
        semantic_tag = _find_semantic_parent(tag_path)
        
        if semantic_tag == current_tag:
            # Same tag, continue accumulating
            current_texts.append(text)
        else:
            # Different tag, save previous block and start new block
            if current_tag is not None and current_texts:
                combined_text = ' '.join(current_texts).strip()
                # Fix split words (e.g., "A dvertise" -> "Advertise")
                combined_text = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', combined_text)
                if combined_text:
                    blocks.append({
                        'tag': current_tag,
                        'text': combined_text
                    })
            
            current_tag = semantic_tag
            current_texts = [text]
    
    # Save the last block
    if current_tag is not None and current_texts:
        combined_text = ' '.join(current_texts).strip()
        # Fix split words (e.g., "A dvertise" -> "Advertise")
        combined_text = re.sub(r'\b([A-Z])\s+([a-z])', r'\1\2', combined_text)
        if combined_text:
            blocks.append({
                'tag': current_tag,
                'text': combined_text
            })
    
    return blocks

def _traverse_and_extract(node, text_parts):
    """
    Recursively traverse DOM tree
    """
    # If it's a text node
    if isinstance(node, NavigableString):
        text = str(node)
        if text.strip():  # Only add if there's actual content
            # Don't automatically add spaces - preserve original spacing
            text_parts.append(text) 
        return

    # If it's an element node
    if node.name == 'br':
        text_parts.append('\n')
        return

    # Recursively process child nodes
    for child in node.children:
        _traverse_and_extract(child, text_parts)
        
    # If it's a block-level element, add newline after it
    if node.name in BLOCK_TAGS:
        text_parts.append('\n')


def _collect_html_files(directory: Path):
    """
    Recursively collect all HTML files within a directory.
    
    Args:
        directory (Path): Directory to search.
        
    Returns:
        List[Path]: Sorted list of HTML file paths.
    """
    html_files = [
        path for path in directory.rglob('*')
        if path.is_file() and path.suffix.lower() in HTML_SUFFIXES
    ]
    html_files.sort()
    return html_files


def _parsed_html_to_dict(parsed_html: ParsedHTML):
    """
    Safely convert ParsedHTML into a serializable dictionary.
    
    Args:
        parsed_html (ParsedHTML): Parsed HTML data.
        
    Returns:
        dict: Serializable representation of the parsed data.
    """
    if hasattr(parsed_html, "model_dump"):
        return parsed_html.model_dump()
    return parsed_html.dict()


def _parse_html_single(file_path: Path, method='blocks'):
    """
    Parse a single HTML file and return the requested representation.
    
    Args:
        file_path (Path): Path to the HTML file.
        method (str): Extraction method - 'blocks' (default) or 'text'.
        
    Returns:
        ParsedHTML or str: Structured data when method='blocks', plain text otherwise.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"HTML file not found: {file_path}")
    
    # Extract file number from filename (characters before extension)
    file_number = file_path.stem  # Gets filename without extension
    
    with open(file_path, 'rb') as f:
        raw_content = f.read()
    
    content_hash = hashlib.sha256(raw_content).hexdigest()
    html_content = raw_content.decode('utf-8', errors='ignore')
    _, url = _extract_from_text_wrapper(html_content)
    
    if method == 'text':
        return preprocess_html(html_content)
    
    blocks_data, stats = extract_blocks(html_content, return_stats=True)
    text_blocks = [TextBlock(**block) for block in blocks_data]
    return ParsedHTML(
        url=url,
        file_number=file_number,
        content_hash=content_hash,
        total_blocks=stats['total'],
        retained_blocks=stats['retained'],
        ignored_blocks=stats['ignored'],
        blocks=text_blocks
    )


def _parse_html_directory(directory: Path, output_dir: Path, method: str):
    """
    Parse all HTML files within a directory and write results to disk.
    
    Args:
        directory (Path): Directory containing HTML files.
        output_dir (Path): Directory to store JSON outputs.
        method (str): Extraction method. Only 'blocks' is supported for directories.
        
    Returns:
        List[Path]: List of output file paths that were written.
    """
    if method != 'blocks':
        raise ValueError("Directory input only supports method='blocks'.")
    
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    html_files = _collect_html_files(directory)
    output_paths = []
    
    for html_file in html_files:
        parsed_html = _parse_html_single(html_file, method=method)
        if not isinstance(parsed_html, ParsedHTML):
            raise ValueError("Directory processing expects ParsedHTML output.")
        
        relative_path = html_file.relative_to(directory)
        output_path = (output_dir / relative_path).with_suffix('.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        serialized = _parsed_html_to_dict(parsed_html)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serialized, f, ensure_ascii=False, indent=2)
        
        output_paths.append(output_path)
    
    return output_paths


def parse_html_file(file_path, method='blocks', output_dir=None):
    """
    Parse an HTML file or directory and extract structured text blocks.
    This is the main public API function for this module.
    
    Args:
        file_path (str or Path): Path to the HTML file or directory to parse.
        method (str): Extraction method - 'blocks' (default) or 'text'
                     - 'blocks': Returns ParsedHTML object with structured data
                     - 'text': Returns plain text string (backward compatibility)
        output_dir (str or Path, optional): Destination for JSON dumps when file_path is a directory.
        
    Returns:
        ParsedHTML, str, or List[Path]:
            - ParsedHTML when parsing a single file with method='blocks'
            - str when parsing a single file with method='text'
            - List[Path] of generated JSON files when parsing a directory
                          
    Raises:
        FileNotFoundError: If the input path does not exist
        ValueError: If directory input is provided without output_dir
    
    Example:
        >>> result = parse_html_file('example.html')
        >>> print(f"URL: {result.url}")
        >>> print(f"Total blocks: {result.total_blocks}")
        >>> print(f"Retained: {result.retained_blocks}, Ignored: {result.ignored_blocks}")
        >>> generated = parse_html_file('html_directory', output_dir='parsed_output')
        >>> print(f"Generated {len(generated)} JSON files")
    """
    input_path = Path(file_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {file_path}")
    
    if input_path.is_dir():
        if output_dir is None:
            raise ValueError("output_dir must be provided when processing a directory.")
        return _parse_html_directory(input_path, Path(output_dir), method)
    
    return _parse_html_single(input_path, method=method)


async def parse_html_file_async(file_path, method='blocks'):
    """
    Async version of parse_html_file for concurrent processing.
    
    Args:
        file_path (str): Path to the HTML file to parse
        method (str): Extraction method - 'blocks' (default) or 'text'
        
    Returns:
        ParsedHTML or str: Same as parse_html_file
        
    Raises:
        FileNotFoundError: If the HTML file does not exist
        ValueError: If a directory path is provided
    """
    input_path = Path(file_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"HTML file not found: {file_path}")
    if input_path.is_dir():
        raise ValueError("parse_html_file_async does not support directory inputs. Use parse_html_file instead.")
    
    # Extract file number from filename (characters before extension)
    file_number = input_path.stem  # Gets filename without extension
    
    # Run file I/O in executor to make it truly async
    loop = asyncio.get_event_loop()
    
    # Read file asynchronously
    def _read_file():
        with open(input_path, 'rb') as f:
            return f.read()
    
    raw_content = await loop.run_in_executor(None, _read_file)
    
    # Calculate SHA-256 hash
    content_hash = hashlib.sha256(raw_content).hexdigest()
    
    # Decode content
    html_content = raw_content.decode('utf-8', errors='ignore')
    
    # Extract URL from CleanEval format
    _, url = _extract_from_text_wrapper(html_content)
    
    # Process HTML based on method
    if method == 'text':
        return preprocess_html(html_content)
    else:  # method == 'blocks' (default)
        # Extract blocks with statistics (CPU-bound, run in executor)
        def _extract():
            return extract_blocks(html_content, return_stats=True)
        
        blocks_data, stats = await loop.run_in_executor(None, _extract)
        
        # Convert to Pydantic models
        text_blocks = [TextBlock(**block) for block in blocks_data]
        
        # Create ParsedHTML object
        return ParsedHTML(
            url=url,
            file_number=file_number,
            content_hash=content_hash,
            total_blocks=stats['total'],
            retained_blocks=stats['retained'],
            ignored_blocks=stats['ignored'],
            blocks=text_blocks
        )


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Preprocess HTML file or directory and extract structured text')
    parser.add_argument('input_path', type=str, help='HTML file or directory path to process')
    parser.add_argument('output_dir', type=str, help='Directory to store parsed outputs')
    parser.add_argument('--method', type=str, choices=['text', 'blocks'], default='blocks',
                       help='Output method: text (plain text) or blocks (structured, default)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    try:
        if input_path.is_dir():
            written_files = parse_html_file(input_path, method=args.method, output_dir=output_dir)
        else:
            result_data = parse_html_file(input_path, method=args.method)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        exit(1)
    
    if input_path.is_dir():
        processed_count = len(written_files)
        print("✓ Processing complete")
        print(f"  Input directory: {input_path}")
        print(f"  Output directory: {output_dir}")
        print(f"  Files processed: {processed_count}")
        if processed_count == 0:
            print("  Note: No HTML files were found under the input directory.")
        else:
            sample_path = written_files[0]
            print(f"  Example output: {sample_path}")
        exit(0)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"html_parsing_{timestamp}.txt"
    output_path = output_dir / output_filename
    
    if args.method == 'text':
        result = result_data  # String
        output_format = "Plain Text"
        url_info = None
        file_number_info = None
        content_hash_info = None
        stats_info = None
    else:  # blocks - returns ParsedHTML object
        result_lines = [f"<{block.tag}> {block.text}" for block in result_data.blocks]
        result = "\n".join(result_lines)
        output_format = "Structured"
        url_info = result_data.url
        file_number_info = result_data.file_number
        content_hash_info = result_data.content_hash
        stats_info = {
            'total': result_data.total_blocks,
            'retained': result_data.retained_blocks,
            'ignored': result_data.ignored_blocks
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"HTML File: {input_path}\n")
        f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Format: {output_format}\n")
        if file_number_info:
            f.write(f"File Number: {file_number_info}\n")
        if url_info:
            f.write(f"Source URL: {url_info}\n")
        if stats_info and content_hash_info:
            f.write(f"Content Hash: {content_hash_info}\n")
            f.write(f"Total Blocks: {stats_info['total']}\n")
            f.write(f"Retained Blocks: {stats_info['retained']}\n")
            f.write(f"Ignored Blocks: {stats_info['ignored']}\n")
        f.write("=" * 80 + "\n\n")
        f.write(result)
    
    print("✓ Processing complete")
    print(f"  Input file: {input_path}")
    print(f"  Output file: {output_path}")
    print(f"  Output format: {output_format}")
    if file_number_info:
        print(f"  File Number: {file_number_info}")
    if url_info:
        print(f"  Source URL: {url_info}")
    if stats_info and content_hash_info:
        print(f"  Content Hash: {content_hash_info}")
        print(f"  Total blocks: {stats_info['total']}")
        print(f"  Retained blocks: {stats_info['retained']}")
        print(f"  Ignored blocks: {stats_info['ignored']}")
    else:
        print(f"  Character count: {len(result)}")



from bs4 import BeautifulSoup, NavigableString
from pydantic import BaseModel, Field
from typing import List, Optional
import re

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
        total_blocks (int): Total number of block-level elements found in original HTML
        retained_blocks (int): Number of content blocks that were extracted and retained
        ignored_blocks (int): Number of blocks that were removed (script, style, etc.)
        blocks (List[TextBlock]): List of extracted text blocks with their tags
    """
    url: Optional[str] = Field(None, description="Source URL if available")
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
    'h5', 'h6', 'header', 'hr', 'li', 'main', 'nav', 'noscript', 'ol', 'p', 'pre', 
    'section', 'table', 'tfoot', 'ul', 'video',
    # Table-related elements (for handling old table-based layouts)
    'td', 'th', 'tr'
}

# Define tags to be removed
IGNORE_TAGS = {'script', 'style', 'noscript', 'meta', 'head', 'title'}

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
SEMANTIC_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'dt', 'dd', 'blockquote', 'pre', 'article', 'section', 'div'}

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
        text = str(node).strip()
        if text:
            # Note: Adding space here for general logic like Resiliparse to prevent English word concatenation
            # For Chinese text, more refined handling might be needed
            text_parts.append(text + " ") 
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

def _extract_blocks_recursive(node, blocks):
    """
    Recursively extract block-level elements and their text content
    """
    # If it's a text node, skip
    if isinstance(node, NavigableString):
        return
    
    # If it's a block-level element
    if node.name in BLOCK_TAGS:
        # Check child element situation
        block_children = [
            child for child in node.children 
            if hasattr(child, 'name') and child.name in BLOCK_TAGS
        ]
        
        if block_children:
            # Check if only contains table-related child elements
            table_tags = {'table', 'tr', 'td', 'th', 'tbody', 'thead', 'tfoot'}
            only_table_children = all(child.name in table_tags for child in block_children)
            
            if only_table_children and node.name in table_tags:
                # If it's a table element with only table children, extract all text (don't split)
                text_parts = []
                _extract_text_from_block(node, text_parts)
                text = "".join(text_parts).strip()
                
                if text:
                    blocks.append({
                        'tag': node.name,
                        'text': text
                    })
                return
            else:
                # If there are non-table block children, or not a table element, use original logic
                # First extract current block's "direct content" (not in child blocks)
                direct_text_parts = []
                for child in node.children:
                    if isinstance(child, NavigableString):
                        text = str(child).strip()
                        if text:
                            direct_text_parts.append(text + " ")
                    elif hasattr(child, 'name') and child.name not in BLOCK_TAGS:
                        # Inline element, extract its text
                        _extract_text_from_block(child, direct_text_parts)
                
                direct_text = "".join(direct_text_parts).strip()
                if direct_text:
                    blocks.append({
                        'tag': node.name,
                        'text': direct_text
                    })
                
                # Then recursively process block-level child elements
                for child in block_children:
                    _extract_blocks_recursive(child, blocks)
        else:
            # If no block-level children, extract all text from current block
            text_parts = []
            _extract_text_from_block(node, text_parts)
            text = "".join(text_parts).strip()
            
            if text:  # Only keep blocks with non-empty text
                blocks.append({
                    'tag': node.name,
                    'text': text
                })
        return
    
    # If not a block-level element, continue recursion
    for child in node.children:
        _extract_blocks_recursive(child, blocks)

def _extract_text_from_block(node, text_parts):
    """
    Extract all text from a block-level element (merge inline elements)
    """
    for child in node.children:
        if isinstance(child, NavigableString):
            text = str(child).strip()
            if text:
                text_parts.append(text + " ")
        elif child.name == 'br':
            text_parts.append(' ')  # br converted to space within block
        elif child.name not in BLOCK_TAGS:  # If it's an inline element, recursively extract
            _extract_text_from_block(child, text_parts)
        
    # If it's a block-level element, add newline after it
    if node.name in BLOCK_TAGS:
        text_parts.append('\n')

def parse_html_file(file_path, method='blocks'):
    """
    Parse an HTML file and extract structured text blocks.
    This is the main public API function for this module.
    
    Args:
        file_path (str): Path to the HTML file to parse
        method (str): Extraction method - 'blocks' (default) or 'text'
                     - 'blocks': Returns ParsedHTML object with structured data
                     - 'text': Returns plain text string (backward compatibility)
        
    Returns:
        ParsedHTML or str: If method='blocks', returns ParsedHTML object with url, block_count, and blocks.
                          If method='text', returns plain text string.
                          
    Raises:
        FileNotFoundError: If the HTML file does not exist
        
    Example:
        >>> result = parse_html_file('example.html')
        >>> print(f"URL: {result.url}")
        >>> print(f"Total blocks: {result.total_blocks}")
        >>> print(f"Retained: {result.retained_blocks}, Ignored: {result.ignored_blocks}")
        >>> for block in result.blocks:
        ...     print(f"<{block.tag}> {block.text[:50]}...")
    """
    import os
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"HTML file not found: {file_path}")
    
    # Read HTML file
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    
    # Extract URL from CleanEval format
    _, url = _extract_from_text_wrapper(html_content)
    
    # Process HTML based on method
    if method == 'text':
        # Backward compatibility: return plain text string
        return preprocess_html(html_content)
    else:  # method == 'blocks' (default)
        # Extract blocks with statistics
        blocks_data, stats = extract_blocks(html_content, return_stats=True)
        
        # Convert to Pydantic models
        text_blocks = [TextBlock(**block) for block in blocks_data]
        
        # Create ParsedHTML object with statistics
        return ParsedHTML(
            url=url,
            total_blocks=stats['total'],
            retained_blocks=stats['retained'],
            ignored_blocks=stats['ignored'],
            blocks=text_blocks
        )


if __name__ == "__main__":
    import argparse
    import os
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Preprocess HTML file and extract structured text')
    parser.add_argument('html_file', type=str, help='HTML file path to process')
    parser.add_argument('--output-dir', type=str, 
                       default='../../trace',
                       help='Output directory (default: ../../trace)')
    parser.add_argument('--method', type=str, choices=['text', 'blocks'], default='blocks',
                       help='Output method: text (plain text) or blocks (structured, default)')
    
    args = parser.parse_args()
    
    # Parse HTML file using the public API function
    try:
        result_data = parse_html_file(args.html_file, method=args.method)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Format output based on method
    if args.method == 'text':
        result = result_data  # String
        output_format = "Plain Text"
        url_info = None
        block_count_info = None
    else:  # blocks - returns ParsedHTML object
        # Format blocks for output
        result_lines = [f"<{block.tag}> {block.text}" for block in result_data.blocks]
        result = "\n".join(result_lines)
        output_format = "Structured"
        url_info = result_data.url
        stats_info = {
            'total': result_data.total_blocks,
            'retained': result_data.retained_blocks,
            'ignored': result_data.ignored_blocks
        }
    
    # Create output directory
    output_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), args.output_dir
    ))
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"html_parsing_{timestamp}.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"HTML File: {args.html_file}\n")
        f.write(f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Output Format: {output_format}\n")
        if url_info:
            f.write(f"Source URL: {url_info}\n")
        if args.method == 'blocks':
            f.write(f"Total Blocks: {stats_info['total']}\n")
            f.write(f"Retained Blocks: {stats_info['retained']}\n")
            f.write(f"Ignored Blocks: {stats_info['ignored']}\n")
        f.write("=" * 80 + "\n\n")
        f.write(result)
    
    print(f"✓ Processing complete")
    print(f"  Input file: {args.html_file}")
    print(f"  Output file: {output_path}")
    print(f"  Output format: {output_format}")
    if url_info:
        print(f"  Source URL: {url_info}")
    if args.method == 'blocks':
        print(f"  Total blocks: {stats_info['total']}")
        print(f"  Retained blocks: {stats_info['retained']}")
        print(f"  Ignored blocks: {stats_info['ignored']}")
    else:
        print(f"  Character count: {len(result)}")



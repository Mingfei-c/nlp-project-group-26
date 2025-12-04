# HTML Preprocessing Module

A lightweight, standalone HTML content extraction tool with structured output using Pydantic models.

## Installation

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

There are two ways to use this HTML preprocessor:

### Method 1: Command-Line Interface (with trace file output)

Parse HTML files directly from the terminal. Results are **automatically saved** to the `trace/` folder with a timestamp.

```bash
# Basic usage - saves to trace/html_parsing_YYYYMMDD_HHMMSS.txt
python preprocess/preprocess.py path/to/file.html

# Extract plain text instead of blocks
python preprocess/preprocess.py path/to/file.html --method text

# Custom output directory
python preprocess/preprocess.py path/to/file.html --output-dir ./custom_output
```

**Console output:**
```
✓ Processing complete
  Input file: path/to/file.html
  Output file: trace/html_parsing_20251204_113224.txt
  Output format: Structured
  Source URL: http://example.com
  Content Hash: a1b2c3d4e5f6...
  Total blocks: 91
  Retained blocks: 12
  Ignored blocks: 79
```

**Trace file contents:**
```
HTML File: path/to/file.html
Processing Time: 2025-12-04 11:32:24
Output Format: Structured
Source URL: http://example.com
Content Hash: a1b2c3d4e5f6...
Total Blocks: 91
Retained Blocks: 12
Ignored Blocks: 79
================================================================================

<h1> Page Title
<p> First paragraph content...
<li> List item content...
...
```

### Method 2: Function Call (programmatic usage)

Import and use as a Python library. Returns a `ParsedHTML` object with structured data. **Does not save to trace folder** - you have full control over the output.

```python
from preprocess import parse_html_file, ParsedHTML, TextBlock

# Parse HTML file and get structured result
result: ParsedHTML = parse_html_file('path/to/file.html')

# Access metadata
print(f"URL: {result.url}")                    # Source URL (if available)
print(f"Hash: {result.content_hash}")          # SHA-256 hash of original content
print(f"Total blocks: {result.total_blocks}")  # All blocks in HTML
print(f"Retained: {result.retained_blocks}")   # Content blocks extracted
print(f"Ignored: {result.ignored_blocks}")     # Blocks removed/merged

# Iterate through content blocks
for block in result.blocks:
    print(f"<{block.tag}> {block.text[:50]}...")
    
# Export to JSON for storage/API
json_data = result.model_dump_json(indent=2)

# Save manually if needed
with open('output.json', 'w') as f:
    f.write(json_data)
```

**Example output:**
```python
>>> result = parse_html_file('example.html')
>>> print(result.url)
'http://example.com'
>>> print(result.content_hash)
'a1b2c3d4e5f6...'
>>> print(result.retained_blocks)
12
>>> print(result.blocks[0])
TextBlock(tag='h1', text='Page Title')
```

## Features

- 🎯 **Leaf-node extraction** - Inspired by BoilerNet for accurate content extraction
- 📊 **Detailed statistics** - Track total, retained, and ignored blocks
- 🔗 **URL extraction** - Automatically extracts URLs from CleanEval format
- 🔒 **Content Hashing** - Calculates SHA-256 hash of original HTML content
- 📦 **Pydantic models** - Type-safe, validated data structures
- 🚀 **Standalone** - Only 2 dependencies, works anywhere

## Output

The tool generates a trace file with:
- Source file path
- Processing timestamp
- Source URL (if available)
- Content Hash (SHA-256)
- Block statistics (total/retained/ignored)
- All extracted content blocks with tags

## Data Models

### `TextBlock`
```python
class TextBlock(BaseModel):
    tag: str   # HTML tag name (e.g., 'p', 'h1', 'div')
    text: str  # Text content
```

### `ParsedHTML`
```python
class ParsedHTML(BaseModel):
    url: Optional[str]        # Source URL
    content_hash: str         # SHA-256 hash of original content
    total_blocks: int         # Total blocks found
    retained_blocks: int      # Content blocks retained
    ignored_blocks: int       # Blocks removed/merged
    blocks: List[TextBlock]   # All content blocks
```

## Dependencies

- `beautifulsoup4>=4.12.0` - HTML parsing
- `pydantic>=2.0.0` - Data validation

## Requirements

- Python 3.7+
- Compatible with Python 3.12

## License

MIT

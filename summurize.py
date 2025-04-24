import os
import time
import re
from transformers import pipeline
import fitz  # PyMuPDF
import torch
import numpy as np
from collections import Counter

def create_safe_summarizer():
    """
    Create a summarization pipeline configured to work reliably on CPU
    with optimized parameters for stable operation.
    
    Returns:
        pipeline: HuggingFace summarization pipeline
    """
    print("\nInitializing BART-large-CNN summarization model...")
    print("Using CPU for processing to avoid CUDA errors")
    
    # Force CPU device to avoid CUDA errors
    device = -1  # Use CPU
    
    try:
        # Create a summarization pipeline with conservative parameters
        summarizer = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn",
            device=device,
            # More conservative length settings
            max_length=200,
            min_length=50,
            # Parameters for stability
            temperature=1.0,  # Default temperature
            do_sample=False,  # Deterministic generation
            num_beams=2,      # Simpler beam search
        )
        
        print("Summarization model loaded successfully!")
        return summarizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def extract_text_and_metadata_from_pdf(pdf_path, verbose=True):
    """
    Extract text and metadata from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        verbose (bool): Whether to print progress information
    
    Returns:
        tuple: (extracted_text, metadata_dict)
    """
    try:
        if verbose:
            print(f"Processing PDF: {pdf_path}")
        
        # Open the PDF file
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        
        if verbose:
            print(f"PDF has {total_pages} pages")
        
        # Extract metadata from PDF
        metadata = {
            'title': os.path.basename(pdf_path).replace('.pdf', ''),  # Default to filename
            'authors': None,
            'pages': total_pages,
            'publication_date': None
        }
        
        # Try to get metadata from the document
        pdf_metadata = doc.metadata
        if pdf_metadata:
            if pdf_metadata.get('title'):
                metadata['title'] = pdf_metadata.get('title')
            if pdf_metadata.get('author'):
                metadata['authors'] = pdf_metadata.get('author')
            # Try to get date if available
            for date_field in ['creationDate', 'modDate']:
                if pdf_metadata.get(date_field):
                    date_str = pdf_metadata.get(date_field)
                    # Format is usually like "D:20201231235959"
                    if date_str.startswith('D:'):
                        try:
                            year = date_str[2:6]
                            metadata['publication_date'] = year
                        except:
                            pass
        
        # Get all text from the PDF
        all_text = ""
        first_page_text = ""
        
        for page_num, page in enumerate(doc, 1):
            if verbose and page_num % 5 == 0:  # Only log every 5 pages to reduce clutter
                print(f"Extracting text from page {page_num}/{total_pages}")
            
            text = page.get_text("text")
            if text.strip():
                all_text += text + "\n\n"
                
                # Save first page text separately for author extraction
                if page_num == 1:
                    first_page_text = text
        
        # Try to extract author information from first page if not in metadata
        if not metadata['authors']:
            authors = extract_authors_from_first_page(first_page_text)
            if authors:
                metadata['authors'] = authors
        
        doc.close()
        return all_text, metadata
    
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        raise

def extract_authors_from_first_page(text):
    """
    Extract author information from the first page of a scholarly paper.
    
    Args:
        text (str): Text from the first page
    
    Returns:
        str: Extracted author information or None
    """
    # Common patterns for author sections in research papers
    author_patterns = [
        # Look for lines between title and abstract with common author indicators
        r'(?i)(?:by|author[s]?:?)[\s*]*(.*?)(?=abstract|\n\n|\s{2,})',
        # Look for typical academic affiliations
        r'(?m)^.*(?:University|Institute|College|Laboratory|Department).*$',
        # Look for email patterns that might indicate authors
        r'(?i)(?:[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)',
        # Look for numbered or bullet-pointed authors
        r'(?m)^(?:\d+\s+|\*\s+|•\s+)([A-Z][a-z]+ [A-Z][a-z]+).*$',
    ]
    
    potential_authors = []
    
    # Apply each pattern
    for pattern in author_patterns:
        matches = re.findall(pattern, text)
        if matches:
            for match in matches:
                if isinstance(match, tuple):  # Some regex groups return tuples
                    match = ' '.join(match)
                
                # Clean up the match
                match = match.strip()
                
                # Skip if too short or too long
                if len(match) < 4 or len(match) > 200:
                    continue
                    
                potential_authors.append(match)
    
    # If we found potential authors, return the most common/likely one
    if potential_authors:
        # If there are multiple candidates, try to select the most reasonable one
        if len(potential_authors) > 1:
            # Prefer entries with multiple names (likely a proper author list)
            author_entries = [a for a in potential_authors if ',' in a or 'and' in a.lower()]
            if author_entries:
                return max(author_entries, key=len)
            
            # Otherwise return the longest potential match
            return max(potential_authors, key=len)
        
        return potential_authors[0]
    
    return None

def process_text_for_summarization(text, max_chunk_size=800, overlap=100):
    """
    Process text into smaller, manageable chunks for summarization.
    Using smaller chunks to avoid memory issues.
    
    Args:
        text (str): Text to process
        max_chunk_size (int): Maximum size of each chunk
        overlap (int): Overlap between chunks in characters
    
    Returns:
        list: List of text chunks
    """
    # Clean text: remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # If text is shorter than max chunk size, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split into chunks with overlap
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of this chunk
        end = min(start + max_chunk_size, len(text))
        
        # If we're not at the end of the text, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence boundaries
            sentence_break = max(
                text.rfind('. ', start, end),
                text.rfind('? ', start, end),
                text.rfind('! ', start, end)
            )
            
            # If found a good breaking point, use it
            if sentence_break != -1 and sentence_break > start + (max_chunk_size // 2):
                end = sentence_break + 2  # Include the period and space
        
        # Extract this chunk
        chunk = text[start:end].strip()
        chunks.append(chunk)
        
        # Set the start point for the next chunk, with overlap
        start = end - overlap if end < len(text) else end
    
    return chunks

def summarize_text_safely(summarizer, text, verbose=False, retry_count=0):
    """
    Summarize text with error handling and multiple fallbacks.
    
    Args:
        summarizer: Summarization pipeline
        text (str): Text to summarize
        verbose (bool): Whether to print progress information
        retry_count (int): Current retry count for recursive calls
    
    Returns:
        str: Summary text or error message
    """
    if retry_count > 3:  # Prevent infinite recursion
        return "Summary generation failed after multiple attempts."
    
    if not text or len(text.strip()) < 50:
        return None  # Text too short to summarize
    
    if verbose:
        print(f"Summarizing text ({len(text)} characters)...")
    
    # Clean the text - remove problematic characters that might cause issues
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII chars
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()  # Normalize whitespace
    
    # If text was significantly cleaned (lost > 30% of content), log a warning
    if len(cleaned_text) < 0.7 * len(text):
        print(f"Warning: Text was significantly cleaned (from {len(text)} to {len(cleaned_text)} chars)")
    
    try:
        # First, try with conservative truncation
        if len(cleaned_text) > 1000:
            # Start with just the first part of the text for efficiency
            try_text = cleaned_text[:1000]
            if verbose:
                print(f"First attempting with truncated text ({len(try_text)} chars)")
            
            summary = summarizer(try_text, truncation=True, max_length=150, min_length=40)
            if summary and len(summary) > 0 and len(summary[0]['summary_text']) > 30:
                return summary[0]['summary_text']
            else:
                # First attempt failed, try with more text
                if verbose:
                    print("Short text summarization failed, trying with more content")
        
        # Primary summarization attempt with full content (or fallback from above)
        summary = summarizer(cleaned_text, truncation=True)
        if summary and len(summary) > 0:
            return summary[0]['summary_text']
        else:
            if verbose:
                print("Summarizer returned empty result")
            
            # Try alternative approach - break into smaller chunks and summarize each
            if len(cleaned_text) > 500 and retry_count == 0:
                chunks = []
                for i in range(0, len(cleaned_text), 500):
                    chunk = cleaned_text[i:i+500]
                    if len(chunk) > 100:  # Only process substantial chunks
                        chunks.append(chunk)
                
                if chunks:
                    chunk_summaries = []
                    for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks
                        if verbose:
                            print(f"Summarizing chunk {i+1}/{min(len(chunks), 3)}")
                        
                        chunk_summary = summarize_text_safely(summarizer, chunk, verbose, retry_count + 1)
                        if chunk_summary:
                            chunk_summaries.append(chunk_summary)
                    
                    if chunk_summaries:
                        return " ".join(chunk_summaries)
            
            return "No summary could be generated."
    
    except Exception as e:
        print(f"Error during summarization: {e}")
        
        # Try a fallback approach with smaller text
        if len(cleaned_text) > 500:
            # Take first chunk for a partial summary
            try:
                if verbose:
                    print("Attempting fallback with shorter text...")
                
                # Try progressively smaller chunks
                for size in [800, 500, 300]:
                    if len(cleaned_text) > size:
                        partial_text = cleaned_text[:size]
                        
                        try:
                            summary = summarizer(partial_text, truncation=True)
                            if summary and len(summary) > 0:
                                return f"{summary[0]['summary_text']}"
                        except:
                            if verbose:
                                print(f"Fallback with {size} chars failed, trying smaller...")
            except Exception as inner_e:
                print(f"All fallbacks failed: {inner_e}")
        
        # Super-fallback: return a simple extract from the beginning
        if len(cleaned_text) > 100:
            # Find first sentence break after 100 chars
            first_sentence_end = -1
            for end_char in ['. ', '? ', '! ']:
                pos = cleaned_text.find(end_char, 100)
                if pos > 0 and (first_sentence_end == -1 or pos < first_sentence_end):
                    first_sentence_end = pos + 1
            
            if first_sentence_end > 0:
                return cleaned_text[:first_sentence_end] + " (...)"
            else:
                return cleaned_text[:150] + " (...)"
        
        # Last resort: return a message about the error
        return f"Summary generation failed. Text may be too complex for automated summarization."

def identify_sections(text):
    """
    Enhanced section identification for research papers.
    
    Args:
        text (str): Full document text
    
    Returns:
        dict: Dictionary of identified sections
    """
    # Improved section patterns in research papers
    section_patterns = {
        'title': r'(?im)^(.{10,150})\n+(?:by|author|abstract|\d{4})',  # Title is usually at the start, followed by authors or abstract
        'abstract': r'(?i)(abstract[:\s]*)(.*?)(?=(introduction|keywords|^\d+[\.\s]+|^[I1][\.\s]+))',
        'introduction': r'(?i)(\d+[\.\s]+|\b|^)introduction\s*(.*?)(?=(\d+[\.\s]+\w+|\b(background|related work|literature|methods|methodology|results|discussion)\b))',
        'methods': r'(?i)(\d+[\.\s]+|\b|^)(methodology|methods|materials and methods|experimental setup|study design)\s*(.*?)(?=(\d+[\.\s]+\w+|\b(results|findings|data|evaluation|discussion)\b))',
        'results': r'(?i)(\d+[\.\s]+|\b|^)(results|findings|data analysis|evaluation)\s*(.*?)(?=(\d+[\.\s]+\w+|\b(discussion|conclusion|limitations|future work)\b))',
        'discussion': r'(?i)(\d+[\.\s]+|\b|^)(discussion)\s*(.*?)(?=(\d+[\.\s]+\w+|\b(conclusion|future work|limitations|references|bibliography)\b))',
        'conclusion': r'(?i)(\d+[\.\s]+|\b|^)(conclusion|conclusions|concluding remarks|summary and conclusions|final remarks)\s*(.*?)(?=(\d+[\.\s]+\w+|\b(future work|limitations|references|bibliography|acknowledgements)\b))',
        'references': r'(?i)(\d+[\.\s]+|\b|^)(references|bibliography|works cited|literature cited)\s*(.*)'
    }
    
    sections = {}
    
    # Extract each section based on patterns
    for section_name, pattern in section_patterns.items():
        matches = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if matches:
            if section_name == 'title':
                # For title, we just want the matched title text
                sections[section_name] = matches.group(1).strip()
            elif len(matches.groups()) >= 3:
                # If we have heading + content in separate groups
                sections[section_name] = matches.group(3).strip()
            elif len(matches.groups()) >= 2:
                # If we have heading + content
                sections[section_name] = matches.group(2).strip()
    
    # If we didn't find sections using patterns, fall back to simpler approach
    if len(sections) <= 2:  # If we found 2 or fewer sections, try alternate method
        fallback_patterns = {
            'abstract': r'(?i)abstract\s*\n(.*?)(?=\n\n|\Z)',
            'introduction': r'(?i)(?:^|\n\n)(?:\d+[\.\s]+)?introduction\s*\n(.*?)(?=\n\n|\Z)',
            'methods': r'(?i)(?:^|\n\n)(?:\d+[\.\s]+)?(?:methods|methodology)\s*\n(.*?)(?=\n\n|\Z)',
            'results': r'(?i)(?:^|\n\n)(?:\d+[\.\s]+)?results\s*\n(.*?)(?=\n\n|\Z)',
            'discussion': r'(?i)(?:^|\n\n)(?:\d+[\.\s]+)?discussion\s*\n(.*?)(?=\n\n|\Z)',
            'conclusion': r'(?i)(?:^|\n\n)(?:\d+[\.\s]+)?conclusion\s*\n(.*?)(?=\n\n|\Z)',
            'references': r'(?i)(?:^|\n\n)(?:\d+[\.\s]+)?references\s*\n(.*?)(?=\n\n|\Z)'
        }
        
        for section_name, pattern in fallback_patterns.items():
            if section_name not in sections:
                matches = re.search(pattern, text, re.DOTALL)
                if matches:
                    sections[section_name] = matches.group(1).strip()
    
    # If still no luck, try chunking by headings
    if len(sections) <= 2:
        lines = text.split('\n')
        current_section = 'body'
        current_content = []
        
        for line in lines:
            line = line.strip()
            # Look for likely headings (short lines in their own paragraph)
            if len(line) > 0 and len(line) < 50 and line.isupper() or re.match(r'^\d+\.\s+\w+', line):
                # Found potential heading
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                    current_content = []
                
                # Determine section type
                lower_line = line.lower()
                if 'abstract' in lower_line:
                    current_section = 'abstract'
                elif 'introduction' in lower_line or 'background' in lower_line:
                    current_section = 'introduction'
                elif 'method' in lower_line or 'setup' in lower_line:
                    current_section = 'methods'
                elif 'result' in lower_line or 'finding' in lower_line:
                    current_section = 'results'
                elif 'discussion' in lower_line:
                    current_section = 'discussion'
                elif 'conclusion' in lower_line:
                    current_section = 'conclusion'
                elif 'reference' in lower_line or 'bibliography' in lower_line:
                    current_section = 'references'
                else:
                    current_section = line  # Use the heading itself
            else:
                current_content.append(line)
        
        # Add the last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
    
    return sections

def summarize_pdf(pdf_path, verbose=True):
    """
    Enhanced PDF summarization using the BART-large-CNN model with improved metadata extraction.
    Produces clearer, more organized summaries with better formatting.
    
    Args:
        pdf_path (str): Path to the PDF file
        verbose (bool): Whether to print progress information
    
    Returns:
        tuple: (summary_text, full_text, metadata)
    """
    try:
        # Extract document name
        doc_name = os.path.basename(pdf_path)
        
        # Extract text and metadata from PDF
        full_text, metadata = extract_text_and_metadata_from_pdf(pdf_path, verbose)
        
        if not full_text.strip():
            summary = f"Document: {doc_name}\n\nNo text could be extracted from the PDF."
            return summary, full_text, metadata
        
        # Check if the model is already initialized, otherwise initialize it
        summarizer = create_safe_summarizer()
        
        # Try to identify document sections
        sections = identify_sections(full_text)
        
        # If title wasn't found in metadata but was identified in sections, update metadata
        if sections.get('title') and (not metadata['title'] or metadata['title'] == doc_name.replace('.pdf', '')):
            metadata['title'] = sections.get('title')
        
        # Prepare the summary with clear formatting
        summary_parts = []
        
        # Add metadata to summary
        summary_parts.append(f"Document: {metadata['title']}")
        
        if metadata['authors']:
            summary_parts.append(f"Authors: {metadata['authors']}")
        
        summary_parts.append(f"Pages: {metadata['pages']}")
        
        if metadata['publication_date']:
            summary_parts.append(f"Publication Date: {metadata['publication_date']}")
        
        summary_parts.append(f"Type: Research Paper")
        summary_parts.append("-" * 40)  # Add a separator line for clarity
        
        # If we found sections, summarize each one
        if sections and len(sections) >= 2:  # At least 2 sections to be considered structured
            if verbose:
                print(f"Found {len(sections)} document sections")
            
            section_summaries_added = False
            
            # First, prioritize the abstract if available (it's often the best overview)
            if 'abstract' in sections and len(sections['abstract']) > 100:
                if verbose:
                    print(f"Processing ABSTRACT section... ({len(sections['abstract'])} chars)")
                
                abstract_summary = summarize_text_safely(summarizer, sections['abstract'], verbose)
                if abstract_summary and len(abstract_summary) > 50:
                    summary_parts.append("OVERVIEW")
                    summary_parts.append(abstract_summary)
                    summary_parts.append("-" * 40)  # Add a separator line
                    section_summaries_added = True
            
            # Process main sections in logical order
            section_order = ['introduction', 'methods', 'results', 'discussion', 'conclusion']
            
            for section_name in section_order:
                if section_name in sections:
                    section_text = sections[section_name]
                    
                    if len(section_text.strip()) < 100:  # Skip very small sections
                        continue
                    
                    if verbose:
                        print(f"Summarizing {section_name.upper()} section... ({len(section_text)} chars)")
                    
                    # Process section into smaller chunks
                    chunks = process_text_for_summarization(section_text)
                    
                    section_summaries = []
                    for i, chunk in enumerate(chunks):
                        if verbose and len(chunks) > 1:
                            print(f"  Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
                        
                        summary = summarize_text_safely(summarizer, chunk, verbose)
                        if summary and len(summary) > 10:  # Only add meaningful summaries
                            section_summaries.append(summary)
                    
                    if section_summaries:
                        # Map section names to more readable titles
                        section_titles = {
                            'introduction': 'BACKGROUND & INTRODUCTION',
                            'methods': 'METHODOLOGY',
                            'results': 'KEY FINDINGS',
                            'discussion': 'DISCUSSION & IMPLICATIONS',
                            'conclusion': 'CONCLUSION'
                        }
                        
                        # Add section with clearer formatting
                        title = section_titles.get(section_name, section_name.upper())
                        summary_parts.append(title)
                        
                        # Join summaries with better spacing and formatting
                        # Remove redundancy between chunks
                        joined_summary = " ".join(section_summaries)
                        # Remove repeated phrases that often appear in summarization
                        joined_summary = re.sub(r'(?i)(the paper|the study|the authors)(\s+\w+){0,3}\s+(the paper|the study|the authors)', r'\1', joined_summary)
                        summary_parts.append(joined_summary)
                        
                        # Add separator
                        summary_parts.append("-" * 40)
                        section_summaries_added = True
            
            # If all section summarization failed, fall back to whole document
            if not section_summaries_added:
                if verbose:
                    print("Section summarization produced no results, falling back to whole document summarization.")
                perform_whole_document_summarization = True
            else:
                perform_whole_document_summarization = False
        else:
            # If sections weren't found, process the whole document
            if verbose:
                print("No clear sections found. Summarizing document as a whole.")
            perform_whole_document_summarization = True
        
        # Whole document summarization if needed
        if perform_whole_document_summarization:
            # First try the abstract if available (it's often the best summary)
            if 'abstract' in sections and len(sections['abstract']) > 200:
                if verbose:
                    print("Using abstract as main summary source")
                
                abstract_summary = summarize_text_safely(summarizer, sections['abstract'], verbose)
                if abstract_summary and len(abstract_summary) > 50:
                    summary_parts.append("ABSTRACT")
                    summary_parts.append(abstract_summary)
                    summary_parts.append("-" * 40)
            
            # Then process the beginning of the document (first ~4000 chars)
            # This often contains the most important information
            intro_text = full_text[:4000]
            if verbose:
                print(f"Summarizing document introduction ({len(intro_text)} chars)")
            
            intro_chunks = process_text_for_summarization(intro_text, max_chunk_size=1000)
            intro_summaries = []
            
            for i, chunk in enumerate(intro_chunks):
                if verbose:
                    print(f"  Processing intro chunk {i+1}/{len(intro_chunks)}")
                
                summary = summarize_text_safely(summarizer, chunk, verbose)
                if summary and len(summary) > 10:
                    intro_summaries.append(summary)
            
            if intro_summaries:
                summary_parts.append("OVERVIEW")
                joined_intro = " ".join(intro_summaries)
                # Clean up the summary text
                joined_intro = re.sub(r'(?i)(the paper|the study|the authors)(\s+\w+){0,3}\s+(the paper|the study|the authors)', r'\1', joined_intro)
                summary_parts.append(joined_intro)
                summary_parts.append("-" * 40)
            
            # Now process the full document in chunks
            if verbose:
                print(f"Summarizing main document content ({len(full_text)} chars)")
            
            # Use smaller chunks for better processing
            chunks = process_text_for_summarization(full_text, max_chunk_size=800, overlap=100)
            
            # Take a reasonable number of chunks to summarize
            # For very long documents, we'll sample chunks throughout the document
            if len(chunks) > 10:
                if verbose:
                    print(f"Document is very long ({len(chunks)} chunks). Sampling throughout.")
                
                # Sample chunks from beginning, middle, and end
                sampled_indices = []
                
                # Beginning chunks (first 30%)
                begin_count = min(3, int(len(chunks) * 0.3))
                sampled_indices.extend(range(begin_count))
                
                # Middle chunks (middle 40%)
                mid_start = int(len(chunks) * 0.3)
                mid_end = int(len(chunks) * 0.7)
                mid_count = min(4, mid_end - mid_start)
                mid_step = (mid_end - mid_start) // mid_count if mid_count > 0 else 1
                sampled_indices.extend(range(mid_start, mid_end, mid_step)[:mid_count])
                
                # End chunks (last 30%)
                end_count = min(3, int(len(chunks) * 0.3))
                end_start = max(mid_end, len(chunks) - end_count)
                sampled_indices.extend(range(end_start, len(chunks)))
                
                # Use the sampled chunks
                sampled_chunks = [chunks[i] for i in sampled_indices]
                chunks = sampled_chunks
            
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                if verbose:
                    print(f"  Processing main content chunk {i+1}/{len(chunks)}")
                
                summary = summarize_text_safely(summarizer, chunk, verbose)
                if summary and len(summary) > 10:  # Only add meaningful summaries
                    chunk_summaries.append(summary)
            
            if chunk_summaries:
                # Combine summaries into a single document summary
                summary_parts.append("KEY POINTS & FINDINGS")
                
                # Join and clean up the summaries
                joined_summary = " ".join(chunk_summaries)
                
                # Clean up repetitive phrases
                joined_summary = re.sub(r'(?i)(the paper|the study|the authors)(\s+\w+){0,3}\s+(the paper|the study|the authors)', r'\1', joined_summary)
                
                # Break long summary into bullet points for readability if it's long
                if len(joined_summary) > 300:
                    # Convert to bullet points
                    sentences = re.split(r'(?<=[.!?])\s+', joined_summary)
                    formatted_summary = ""
                    
                    for sentence in sentences:
                        if len(sentence.strip()) > 10:  # Skip very short sentences
                            formatted_summary += f"• {sentence.strip()}\n"
                    
                    summary_parts.append(formatted_summary)
                else:
                    summary_parts.append(joined_summary)
        
        # If no summary content was generated, add a fallback generic summary
        if len(summary_parts) <= 6:  # Only metadata and separator were added
            print("WARNING: Failed to generate summary content through normal means.")
            
            # Try one last direct summarization of the beginning of the document
            try:
                # Take the first 1500 characters
                start_text = full_text[:1500].strip()
                if len(start_text) > 200:
                    fallback_summary = summarizer(start_text, truncation=True)[0]['summary_text']
                    if fallback_summary and len(fallback_summary) > 50:
                        summary_parts.append("CONTENT OVERVIEW")
                        summary_parts.append(fallback_summary)
                        summary_parts.append("-" * 40)
            except Exception as fallback_error:
                print(f"Fallback summarization also failed: {fallback_error}")
                summary_parts.append("Could not generate automatic summary content. The document may be in a format that is difficult to process.")
        
        # Join all parts with proper spacing
        summary = "\n\n".join(summary_parts)
        
        return summary, full_text, metadata
        
    except Exception as e:
        print(f"Error during summarization process: {e}")
        import traceback
        traceback.print_exc()  # Print full error stack trace
        summary = f"Document: {os.path.basename(pdf_path)}\n\nSummarization failed due to an error: {e}"
        return summary, "", {"title": os.path.basename(pdf_path), "authors": None, "pages": "Unknown"}

def create_qa_system():
    """
    Create a question-answering pipeline with a better model for scientific content.
    
    Returns:
        pipeline: HuggingFace QA pipeline
    """
    print("\nInitializing question-answering model...")
    
    try:
        # Create a QA pipeline using a model better suited for scientific papers
        qa_pipeline = pipeline(
            "question-answering", 
            model="deepset/roberta-base-squad2",  # More robust QA model
            device=-1  # Use CPU
        )
        
        print("Question-answering model loaded successfully!")
        return qa_pipeline
    except Exception as e:
        print(f"Error loading QA model: {e}")
        raise

def ask_question(question, summary, full_text, metadata, verbose=False):
    """
    Enhanced question answering that properly distinguishes between metadata
    and content questions, with improved content searching.
    
    Args:
        question (str): The question to ask
        summary (str): The summary text
        full_text (str): The full document text
        metadata (dict): Document metadata
        verbose (bool): Whether to print progress information
    
    Returns:
        str: Answer to the question
    """
    question_lower = question.lower()
    
    # Check for common metadata-specific questions first
    metadata_keywords = {
        'author': ['who wrote', 'who authored', 'who is the author', 'authors of', 'written by'],
        'title': ['title', 'what is this paper called', 'paper name', 'what is the name'],
        'date': ['when', 'year', 'date', 'published', 'publication year'],
        'pages': ['how many pages', 'page count', 'length of document', 'how long is']
    }
    
    # Check if this is a metadata question
    is_metadata_question = False
    for metadata_type, keywords in metadata_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            is_metadata_question = True
            if metadata_type == 'author' and metadata.get('authors'):
                return metadata['authors']
            elif metadata_type == 'title' and metadata.get('title'):
                return metadata['title']
            elif metadata_type == 'date' and metadata.get('publication_date'):
                return f"This paper was published in {metadata['publication_date']}."
            elif metadata_type == 'pages' and metadata.get('pages'):
                return f"This document has {metadata['pages']} pages."
    
    # If it matched metadata patterns but we don't have the info, continue to content search
    
    # For non-metadata questions, use the QA model on the content
    try:
        qa_pipeline = create_qa_system()
        
        if verbose:
            print(f"Processing content question: {question}")
            print("This is NOT a metadata question - searching document content")
        
        # First, try to find the most relevant section for the question
        if verbose:
            print("Searching for relevant content...")
        
        # Find most relevant context by checking term overlap
        question_terms = set(re.findall(r'\b\w{4,}\b', question_lower))  # Words of 4+ chars
        
        # The full text is too large, so let's segment it and search
        segments = []
        
        # First add the summary as a high-priority segment
        segments.append((summary, 2.0))  # Weight summary higher
        
        # Break full text into manageable paragraphs
        paragraphs = re.split(r'\n\s*\n', full_text)
        
        for para in paragraphs:
            if len(para.strip()) < 50:  # Skip very short paragraphs
                continue
                
            # Calculate relevance score based on term overlap
            para_lower = para.lower()
            matches = sum(1 for term in question_terms if term in para_lower)
            
            # Higher score for paragraphs with exact phrase matches
            for phrase in re.findall(r'\b(\w+\s+\w+\s+\w+)\b', question_lower):
                if phrase in para_lower:
                    matches += 2
            
            # Check for specific numbers mentioned in the question
            question_numbers = set(re.findall(r'\d+\.?\d*%?', question))
            for num in question_numbers:
                if num in para:
                    matches += 3  # Strongly prioritize paragraphs with matching numbers
            
            # Only include paragraphs with some relevance
            if matches > 0:
                segments.append((para, matches))
        
        # Sort segments by relevance score (descending)
        segments.sort(key=lambda x: x[1], reverse=True)
        
        # Take top segments that fit within context window (about 4000 chars)
        relevant_text = ""
        total_chars = 0
        
        for segment, score in segments:
            if total_chars + len(segment) <= 4000:
                if relevant_text:
                    relevant_text += "\n\n"
                relevant_text += segment
                total_chars += len(segment)
            
            if total_chars >= 3000:  # Get at least 3000 chars if available
                break
        
        if not relevant_text:
            # Fallback if no relevant segments found
            relevant_text = summary
            
            # If summary is too short, add beginning of document
            if len(relevant_text) < 500 and full_text:
                # Find the first few paragraphs
                first_paras = "\n\n".join(paragraphs[:3])
                if first_paras:
                    relevant_text += "\n\n" + first_paras
        
        if verbose:
            print(f"Using {len(relevant_text)} chars of relevant text for question answering")
        
        # Get answer using the QA pipeline
        answer = qa_pipeline(question=question, context=relevant_text)
        
        # Verify answer quality
        if answer['score'] < 0.1 or len(answer['answer'].strip()) < 5:
            # Try with a different context compilation approach if confidence is low
            if verbose:
                print(f"Low confidence answer (score: {answer['score']:.2f}), trying alternative approach")
            
            # Extract sections from the text that might contain answers
            section_texts = []
            
            # Look for sections that typically contain findings/results
            section_patterns = {
                'results': r'(?i)results?.*?\n(.*?)(?=\n\s*\n|$)',
                'findings': r'(?i)findings.*?\n(.*?)(?=\n\s*\n|$)',
                'discussion': r'(?i)discussion.*?\n(.*?)(?=\n\s*\n|$)',
                'conclusion': r'(?i)conclusion.*?\n(.*?)(?=\n\s*\n|$)',
            }
            
            for section_name, pattern in section_patterns.items():
                matches = re.search(pattern, full_text, re.DOTALL)
                if matches:
                    section_texts.append(matches.group(1)[:1000])  # Take first 1000 chars
            
            # If we found relevant sections, use them
            if section_texts:
                alt_context = "\n\n".join(section_texts)
                alt_answer = qa_pipeline(question=question, context=alt_context)
                
                if alt_answer['score'] > answer['score']:
                    answer = alt_answer
            
            # If still low confidence, try with specific paragraphs that contain key terms
            if answer['score'] < 0.15:
                # Extract paragraphs with key question terms
                key_terms = [term for term in question_terms if len(term) > 5]
                
                if key_terms:
                    term_paragraphs = []
                    
                    for para in paragraphs:
                        para_lower = para.lower()
                        if any(term in para_lower for term in key_terms) and len(para) > 100:
                            term_paragraphs.append(para)
                    
                    if term_paragraphs:
                        term_context = "\n\n".join(term_paragraphs[:5])  # Use top 5 paragraphs
                        term_answer = qa_pipeline(question=question, context=term_context)
                        
                        if term_answer['score'] > answer['score']:
                            answer = term_answer
        
        # Format the answer for better readability
        final_answer = answer['answer'].strip()
        
        # If the answer is too short or seems incomplete, provide more context
        if len(final_answer) < 50 or not final_answer.endswith('.'):
            # Find the sentence containing the answer in the context
            answer_context = ""
            sentences = re.split(r'(?<=[.!?])\s+', relevant_text)
            
            for sentence in sentences:
                if final_answer in sentence and len(sentence) > len(final_answer):
                    answer_context = sentence
                    break
            
            if answer_context and len(answer_context) > len(final_answer) * 1.5:
                final_answer = answer_context
        
        # Ensure answer is properly formatted
        final_answer = final_answer.strip()
        if not final_answer.endswith(('.', '!', '?')):
            final_answer += '.'
        
        # If answer seems to be just the author name and nothing else, 
        # it's likely incorrect - provide a fallback response
        if final_answer.lower() == metadata.get('authors', '').lower():
            return "I couldn't find specific information about that in the paper. The document doesn't seem to directly address this question in the available text."
        
        # Add confidence qualifier if answer seems uncertain
        if answer['score'] < 0.3:
            final_answer += " (Note: This information may be incomplete based on the available text.)"
        
        return final_answer
    
    except Exception as e:
        if verbose:
            print(f"Error in question answering: {e}")
            import traceback
            traceback.print_exc()
        
        # Provide a graceful fallback
        return "I'm sorry, I couldn't find a clear answer to that question in the document. The specific information may not be present in the extracted text, or it might require more context to answer accurately."

def find_relevant_context(question, summary, full_text):
    """
    Find the most relevant context for a given question.
    
    Args:
        question (str): The question
        summary (str): The document summary
        full_text (str): The full document text
    
    Returns:
        str: The most relevant context
    """
    question_lower = question.lower()
    
    # Map question types to likely relevant sections
    context_mapping = {
        'abstract': ['what is the paper about', 'main idea', 'summary', 'overview'],
        'introduction': ['background', 'problem statement', 'why', 'purpose'],
        'methods': ['how did they', 'methodology', 'approach', 'data collection', 'experiment'],
        'results': ['what did they find', 'findings', 'outcome', 'data show'],
        'discussion': ['implications', 'significance', 'interpretation', 'meaning'],
        'conclusion': ['conclude', 'future work', 'recommendation', 'final']
    }
    
    # Try to find a section match
    for section, keywords in context_mapping.items():
        if any(keyword in question_lower for keyword in keywords):
            # Try to find the section in the summary
            section_pattern = re.compile(f"{section.upper()}:(.*?)(?=\n\n|$)", re.DOTALL)
            section_match = section_pattern.search(summary)
            
            if section_match:
                return section_match.group(1)
    
    # If no specific section match, check for entity-type questions
    entity_keywords = {
        'who': ['researcher', 'scientist', 'professor', 'doctor', 'team'],
        'where': ['university', 'institute', 'lab', 'country', 'region'],
        'when': ['year', 'date', 'period', 'time', 'during']
    }
    
    for entity, keywords in entity_keywords.items():
        if entity in question_lower or any(keyword in question_lower for keyword in keywords):
            # For entity questions, check first few paragraphs
            first_few_para = "\n".join(full_text.split("\n\n")[:3])
            if len(first_few_para) > 200:
                return first_few_para
    
    # By default, use the summary as context
    return summary

def interactive_mode():
    """
    Run the enhanced PDF summarizer in interactive mode.
    """
    print("\n===== Enhanced PDF Summarizer with BART-large-CNN =====")
    print("This tool summarizes PDF documents with improved metadata extraction and QA capabilities.")
    
    # Ask for PDF path
    while True:
        pdf_path = input("\nEnter the path to your PDF file: ")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"Error: File '{pdf_path}' not found. Please enter a valid path.")
            continue
            
        # Check if it's a PDF
        if not pdf_path.lower().endswith('.pdf'):
            print(f"Error: '{pdf_path}' is not a PDF file. Please enter a valid PDF path.")
            continue
            
        break
    
    # Process the PDF
    print("\nStarting summarization process...")
    start_time = time.time()
    
    try:
        summary, full_text, metadata = summarize_pdf(pdf_path)
        
        end_time = time.time()
        print(f"\nSummarization completed in {end_time - start_time:.2f} seconds.")
        
        # Check if summary contains actual content beyond metadata
        metadata_line_count = 5  # Approx. number of lines used for metadata
        summary_lines = [line for line in summary.split('\n') if line.strip()]
        
        if len(summary_lines) <= metadata_line_count:
            print("\nWARNING: The summarization process did not generate content summary.")
            print("Attempting emergency fallback summarization...")
            
            try:
                # Create a fresh summarizer
                summarizer = create_safe_summarizer()
                
                # Get the first 2000 characters of the document for emergency summary
                start_text = full_text[:2000].strip()
                if len(start_text) > 200:
                    # Simple direct summarization as last resort
                    emergency_summary = summarizer(start_text, truncation=True)[0]['summary_text']
                    if emergency_summary and len(emergency_summary) > 50:
                        summary += "\n\nEMERGENCY CONTENT SUMMARY:\n" + emergency_summary
                        print("Emergency summary generated successfully.")
                    else:
                        print("Emergency summarization produced no meaningful results.")
                else:
                    print("Document content too short for emergency summarization.")
            except Exception as fallback_error:
                print(f"Emergency summarization failed: {fallback_error}")
        
        # Print the summary
        print("\n===== SUMMARY =====\n")
        print(summary)
        
        # Option to save the summary
        save_option = input("\nWould you like to save this summary to a file? (y/n): ")
        if save_option.lower() == 'y':
            output_file = input("Enter the output file name: ")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            print(f"Summary saved to {output_file}")
        
        # Enter Q&A mode
        print("\n===== QUESTION & ANSWER MODE =====")
        print("You can now ask questions about the document.")
        print("Try questions like 'Who wrote this paper?', 'What is the main finding?', etc.")
        print("Type 'exit' to quit the program.")
        
        while True:
            question = input("\nEnter your question (or 'exit' to quit): ")
            
            if question.lower() == 'exit':
                print("\nExiting program. Thank you for using the Enhanced PDF Summarizer!")
                break
            
            answer = ask_question(question, summary, full_text, metadata, verbose=True)
            print("\n===== ANSWER =====")
            print(answer)
    
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()  # Print full error stack trace
        print("\nTips to resolve common issues:")
        print("1. Make sure you have installed PyMuPDF correctly: pip install PyMuPDF")
        print("2. Make sure you have installed transformers correctly: pip install transformers torch")
        print("3. Make sure the PDF file is not corrupted or password-protected")

if __name__ == "__main__":
    try:
        interactive_mode()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("\nPlease make sure you have installed all required packages:")
        print("pip install transformers torch PyMuPDF")
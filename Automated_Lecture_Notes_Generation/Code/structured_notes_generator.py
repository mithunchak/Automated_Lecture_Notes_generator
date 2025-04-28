#!/usr/bin/env python
import os
import sys
import json
import time
import re
from pathlib import Path
import torch
import subprocess
from transformers import pipeline

# Try to import Groq (install if needed)
try:
    from groq import Groq, RateLimitError, APIError
    GROQ_AVAILABLE = True
except ImportError:
    # Install Groq library
    print("Installing Groq library...")
    install_groq_lib = subprocess.run(['pip', 'install', '-q', 'groq>=0.4.0'], capture_output=True, text=True)
    if install_groq_lib.returncode == 0:
        print("Groq library installed successfully.")
        try:
            from groq import Groq, RateLimitError, APIError
            GROQ_AVAILABLE = True
        except ImportError:
            print("Failed to import Groq library after installation.")
            GROQ_AVAILABLE = False
    else:
        print("Error installing Groq library:")
        print(install_groq_lib.stderr)
        GROQ_AVAILABLE = False

# Initialize Groq client if available
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_client = None
llm_model_name = "llama3-8b-8192"  # Default to Llama 3 8B model

if GROQ_AVAILABLE and groq_api_key:
    try:
        groq_client = Groq(api_key=groq_api_key)
        print(f"Groq client initialized with model: {llm_model_name}")
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        groq_client = None
elif GROQ_AVAILABLE:
    print("Groq API key not found in environment variables. Set GROQ_API_KEY to use Groq.")
else:
    print("Groq library not available. Will use Hugging Face models only.")

# Try to import tiktoken for token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    tokenizer = tiktoken.get_encoding("cl100k_base")
except ImportError:
    # Install tiktoken library
    print("Installing tiktoken library for token counting...")
    try:
        subprocess.run(['pip', 'install', '-q', 'tiktoken'], check=True)
        import tiktoken
        TIKTOKEN_AVAILABLE = True
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except:
        print("Failed to install or import tiktoken library. Using basic token estimation.")
        TIKTOKEN_AVAILABLE = False
        tokenizer = None
except Exception as e:
    print(f"Error initializing tiktoken: {e}")
    TIKTOKEN_AVAILABLE = False
    tokenizer = None

# Define base directories
BASE_DIR = Path("./data")
PROCESSED_DIR = BASE_DIR / "processed"
TRANSCRIPTS_DIR = PROCESSED_DIR / "transcripts"
OUTPUT_DIR = BASE_DIR / "output"
NOTES_DIR = OUTPUT_DIR / "lecture_notes"

# Ensure directories exist
for directory in [PROCESSED_DIR, OUTPUT_DIR, NOTES_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

def load_text_generation_model(model_id="google/flan-t5-large"):
    """Load a text generation pipeline from Hugging Face"""
    try:
        print(f"Loading text generation model: {model_id}")
        
        # Set device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create a text generation pipeline
        text_generator = pipeline(
            "text2text-generation",
            model=model_id,
            device=device
        )
        
        print(f"Text generation model loaded: {model_id}")
        return text_generator
    except Exception as e:
        print(f"Error loading text generation model: {e}")
        return None

def extract_title_from_filename(filename):
    """Extract a clean title from the filename"""
    # Remove file extension
    name = filename.split('.')[0]
    
    # If it's a transcript file, remove _transcript suffix
    name = name.replace('_transcript', '')
    
    # Handle common patterns in filenames
    if '_' in name:
        parts = name.split('_')
        # If it starts with a topic prefix, keep it meaningful
        if any(topic in parts[0].lower() for topic in ['heap', 'bst', 'tree', 'expn', 'tbt', 'binary', 'agentic', 'lora', 'multimodal']):
            topic = parts[0].capitalize()
            # Try to extract a more descriptive part from the rest
            for part in parts[1:]:
                if len(part) > 3 and not part.isdigit() and not part.startswith('20'):  # Skip dates, numbers
                    return f"{topic} - {part.replace('-', ' ')}"
            return topic
    
    # Default to the full name with spaces instead of underscores
    return name.replace('_', ' ').replace('-', ' ')

def get_slide_text_content(slide_data):
    """Extract plain text content from slide data"""
    if not slide_data:
        return ""
    
    content_parts = []
    
    # Add metadata
    content_parts.append(f"Presentation: {slide_data['metadata']['title']}")
    content_parts.append(f"Total Slides: {slide_data['metadata']['slide_count']}")
    content_parts.append("")
    
    # Process each slide
    for slide in slide_data["slides"]:
        # Add slide header
        content_parts.append(f"Slide {slide['slide_number']}: {slide['title']}")
        
        # Add content
        for item in slide["content"]:
            if item["type"] == "text":
                content_parts.append(item["text"])
        
        # Add notes
        if slide["notes"]:
            content_parts.append(f"Notes: {slide['notes']}")
        
        # Add separator between slides
        content_parts.append("-" * 40)
    
    return "\n".join(content_parts)

def format_structured_notes(generated_dict):
    """
    Convert the generated sections into a well-formatted markdown document
    following the requested structure
    """
    lecture_title = generated_dict.get("lecture_title", "Untitled Lecture")
    main_topic = generated_dict.get("main_topic", lecture_title)
    
    # Format the notes as a markdown document with the specific structure
    formatted_notes = f"# Lecture Notes: {lecture_title}\n\n"
    formatted_notes += f"# {main_topic}\n\n"
    
    # Add sections with proper formatting
    for section in generated_dict.get("sections", []):
        section_title = section.get("title", "")
        formatted_notes += f"### {section_title}\n\n"
        
        # Format content as bullet points if needed
        content = section.get("content", "")
        if isinstance(content, list):
            for point in content:
                formatted_notes += f"* {point}\n"
            formatted_notes += "\n"
        else:
            # Check if content already has bullet points
            if "* " in content or "- " in content:
                formatted_notes += f"{content}\n\n"
            else:
                # Split into paragraphs and format
                paragraphs = content.split("\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        formatted_notes += f"{paragraph}\n\n"
    
    return formatted_notes

def chunk_text(text, max_tokens_per_chunk=2000, overlap_tokens=200):
    """Splits text into chunks based on estimated token count with overlap."""
    global tokenizer, TIKTOKEN_AVAILABLE
    
    if not TIKTOKEN_AVAILABLE or not tokenizer:
        # Fallback to character-based chunking if tokenizer failed
        print("Using character-based chunking due to tokenizer issue.")
        max_len = max_tokens_per_chunk * 4 # Rough estimate: 4 chars per token
        overlap_len = overlap_tokens * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_len, len(text))
            chunks.append(text[start:end])
            start += max_len - overlap_len
            if start >= len(text): break # Avoid infinite loop on short overlap
        return chunks

    # Token-based chunking
    tokens = tokenizer.encode(text)
    chunks = []
    start_token = 0
    while start_token < len(tokens):
        end_token = min(start_token + max_tokens_per_chunk, len(tokens))
        # Decode the chunk tokens back to text
        chunk_text = tokenizer.decode(tokens[start_token:end_token])
        chunks.append(chunk_text)

        next_start_token = start_token + max_tokens_per_chunk - overlap_tokens
        # Ensure we don't get stuck if overlap is too large or chunk is small
        if next_start_token <= start_token:
             next_start_token = start_token + 1 # Force progression

        start_token = next_start_token
        if start_token >= len(tokens): break

    return chunks

def generate_notes_with_groq(prompt, max_retries=2, initial_delay=5):
    """Calls the Groq API to generate notes, handles rate limits with retry."""
    global groq_client, llm_model_name, tokenizer
    
    if not groq_client:
        print("  Error: Groq client not initialized.")
        return None

    delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            # Estimate prompt tokens BEFORE sending (optional but good practice)
            prompt_tokens = 0
            if TIKTOKEN_AVAILABLE and tokenizer:
                try:
                    prompt_tokens = len(tokenizer.encode(prompt))
                    # Check against model limit (leave room for completion tokens)
                    # Llama3 8k limit (8192). Let's set a safer threshold like 7000.
                    token_threshold = 7000
                    if prompt_tokens > token_threshold:
                        print(f"  Error: Estimated prompt tokens ({prompt_tokens}) exceed safety margin ({token_threshold}) for {llm_model_name}. Skipping API call.")
                        return "## Error: Content Too Long\n\nThe text chunk combined with presentation data exceeded the safe input length for this model. Notes could not be generated for this segment."
                except Exception as e:
                    print(f"  Warning: Token estimation failed. Proceeding without check. Error: {e}")

            # Actual API Call
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=llm_model_name,
                temperature=0.5,
            )

            # Process Response
            if chat_completion.choices and chat_completion.choices[0].message:
                response_content = chat_completion.choices[0].message.content
                if response_content and isinstance(response_content, str) and response_content.strip():
                    # Success case
                    return response_content
                else:
                    # Handle empty response from LLM
                    print(f"  Warning: Received empty content from LLM (Attempt {attempt + 1}/{max_retries + 1}).")
                    if attempt < max_retries:
                        print(f"   Retrying in {delay} seconds...")
                        time.sleep(delay)
                        delay *= 2
                        continue # Go to next attempt
                    else:
                        return None # Failed after retries
            else:
                # Handle invalid response structure
                print(f"  Warning: Received invalid response structure from LLM (Attempt {attempt + 1}/{max_retries + 1}).")
                if attempt < max_retries:
                    print(f"   Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2
                    continue # Go to next attempt
                else:
                    return None # Failed after retries

        except RateLimitError as e:
            # Handle rate limit errors
            if attempt < max_retries:
                print(f"  Warning: Rate limit hit (Attempt {attempt + 1}/{max_retries + 1}). Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                print(f"  Error: Rate limit exceeded after {max_retries} retries. {e}")
                return None # Failed after retries
                
        except APIError as e:
            # Handle API errors (including context length)
            error_message = str(e).lower()
            if "request too large" in error_message or \
               "context_length_exceeded" in error_message or \
               "maximum context length" in error_message or \
               "prompt is too long" in error_message:
                print(f"  Error: Input context length exceeded for model {llm_model_name}. Prompt is too long.")
                return "## Error: Content Too Long\n\nThe combined transcript and presentation content exceeded the maximum length for this model. Notes could not be generated for this segment."
            else:
                # Handle other API errors
                print(f"  Error: Groq API error (Attempt {attempt + 1}/{max_retries + 1}): {e}")
                # Consider retrying certain API errors? For now, fail immediately.
                return None
                
        except Exception as e:
            # Handle unexpected errors during the call
            print(f"  Error: An unexpected error occurred during LLM call (Attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                 print(f"   Retrying in {delay} seconds...")
                 time.sleep(delay)
                 delay *= 2
                 continue # Go to next attempt
            else:
                 return None # Failed after retries

    # Fallback if all retries fail
    return None

def generate_structured_lecture_notes(transcript_file, slide_content_file=None):
    """
    Generate structured lecture notes from transcript and slide content
    Using either Groq API (if available) or Hugging Face models as fallback
    """
    try:
        # 1. Load transcript data
        with open(transcript_file, 'r') as f:
            transcript_data = json.load(f)
        
        transcript_text = transcript_data.get("transcript_text", "")
        lecture_title = extract_title_from_filename(os.path.basename(transcript_file))
        
        # 2. Load slide data if available
        slide_data = None
        if slide_content_file and os.path.exists(slide_content_file):
            try:
                with open(slide_content_file, 'r') as f:
                    slide_data = json.load(f)
            except Exception as e:
                print(f"Error loading slide content: {e}")
        
        # 3. Prepare slide content string once
        slide_text = ""
        if slide_data:
            slide_text = get_slide_text_content(slide_data)
            max_slide_len = 3000
            if len(slide_text) > max_slide_len:
                slide_text = slide_text[:max_slide_len] + "... [slide content truncated]"
        
        # 4. Determine which method to use (Groq or Hugging Face)
        use_groq = GROQ_AVAILABLE and groq_client is not None
        
        if use_groq:
            print(f"Using Groq API with model {llm_model_name} for note generation")
            return generate_structured_notes_with_groq(transcript_text, slide_text, lecture_title)
        else:
            print("Using Hugging Face models for note generation (Groq not available)")
            return generate_structured_notes_with_huggingface(transcript_text, slide_text, lecture_title)
            
    except Exception as e:
        print(f"Error generating structured notes: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def generate_structured_notes_with_groq(transcript_text, slide_text, lecture_title):
    """Generate notes using Groq API with chunking for long content"""
    
    # Create a Groq-specific prompt template for main topic
    groq_title_prompt = f"""
    Based on this lecture transcript and slide content, provide a concise but descriptive title
    for this lecture (as a single phrase or short sentence). The title should clearly indicate
    the main subject area covered.
    
    Lecture: {lecture_title}
    
    Transcript excerpt:
    {transcript_text[:1500]}
    
    Slide Content excerpt:
    {slide_text[:1000] if slide_text else "No slide content available"}
    
    Return only the title, nothing else.
    """.strip()
    
    # Get main topic using Groq
    main_topic = generate_notes_with_groq(groq_title_prompt)
    if not main_topic:
        main_topic = lecture_title  # Fallback to the file-derived title
    main_topic = main_topic.strip()
    
    # Split transcript into manageable chunks to avoid context limit issues
    transcript_chunks = chunk_text(transcript_text, max_tokens_per_chunk=1800, overlap_tokens=150)
    print(f"Transcript split into {len(transcript_chunks)} chunks")
    
    # For outline generation, use the first transcript chunk (beginning of the lecture)
    # which likely has the most introductory/overview material
    outline_context = transcript_chunks[0]
    if len(transcript_chunks) > 1:
        # Add beginning of second chunk for better coverage
        outline_context += "\n\n" + transcript_chunks[1][:500]
    
    # Create Groq-specific outline generation prompt
    groq_outline_prompt = f"""
    You are an expert educator creating lecture notes. Based on the transcript and slides, 
    identify the 3-6 most important sections that should appear in well-structured lecture notes 
    on {main_topic}.
    
    Each section should be a key concept, definition, or topic area. Format your response as a
    simple list of section headings only.
    
    Transcript excerpt:
    {outline_context}
    
    Slide Content:
    {slide_text if slide_text else "No slide content available"}
    
    List only the section headings, one per line.
    """.strip()
    
    # Get outline using Groq
    outline_text = generate_notes_with_groq(groq_outline_prompt)
    if not outline_text:
        # Fallback headings if API call fails
        outline_text = "Introduction\nMain Concepts\nExamples\nSummary"
    
    # Parse sections from the outline
    section_titles = [line.strip() for line in outline_text.split('\n') if line.strip()]
    # Filter out any non-title-like entries (too long, etc.)
    section_titles = [title for title in section_titles if len(title) < 100]
    
    print(f"Generated outline with {len(section_titles)} section titles")
    
    # For each section, generate content using the most relevant transcript chunks
    sections = []
    all_section_content = []
    
    # Generate content for each section using appropriate chunks
    chunk_per_section = max(1, len(transcript_chunks) // len(section_titles))
    
    for i, section_title in enumerate(section_titles):
        # Determine which chunks to use for this section
        start_chunk = min(i * chunk_per_section, len(transcript_chunks) - 1)
        end_chunk = min(start_chunk + chunk_per_section, len(transcript_chunks))
        
        # Combine relevant chunks with overlap
        section_transcript = "\n\n".join(transcript_chunks[start_chunk:end_chunk])
        
        # Create the section prompt
        section_prompt = f"""
        You are creating detailed lecture notes on {main_topic}. Write the content for the section: "{section_title}".
        
        Based on this transcript segment:
        {section_transcript[:5000]}
        
        And using these slides for reference:
        {slide_text[:2000] if slide_text else "No slide content available"}
        
        Generate well-structured notes for this section. Include:
        1. Clear definitions of key concepts
        2. Important points formatted as bullet points
        3. Any formulas, algorithms, or examples mentioned
        
        Write in a clear academic style. Use bullet points for lists.
        Limit your response to just the content for this section.
        """.strip()
        
        # Generate content using Groq
        section_content = generate_notes_with_groq(section_prompt)
        
        if not section_content:
            section_content = f"*Content generation failed for this section. Please refer to the original lecture materials.*"
        
        # Add to sections list
        sections.append({
            "title": section_title,
            "content": section_content
        })
        
        # Keep track of all generated content
        all_section_content.append(section_content)
    
    # Assemble the full notes structure
    notes_structure = {
        "lecture_title": lecture_title,
        "main_topic": main_topic,
        "sections": sections,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Format the notes into the desired structure
    formatted_notes = format_structured_notes(notes_structure)
    
    # Save the notes - using lecture_title to create filenames
    # Instead of transcript_file which is not passed to this function
    safe_filename = "".join(c for c in lecture_title if c.isalnum() or c in " _-").rstrip()
    safe_filename = safe_filename.replace(" ", "_")
    output_filename = f"{safe_filename}_notes.txt"
    output_path = os.path.join(NOTES_DIR, output_filename)
    
    with open(output_path, 'w') as f:
        f.write(formatted_notes)
    
    print(f"Notes generated and saved to: {output_path}")
    
    # Also save the structured data for potential future use
    structure_path = os.path.join(NOTES_DIR, output_filename.replace('.txt', '_structure.json'))
    with open(structure_path, 'w') as f:
        json.dump(notes_structure, f, indent=2)
    
    return {"success": True, "output_path": output_path}

def generate_structured_notes_with_huggingface(transcript_text, slide_text, lecture_title):
    """Generate notes using Hugging Face models (original implementation)"""
    
    # 3. Load text generation model
    generator = load_text_generation_model("google/flan-t5-large")
    if generator is None:
        # Fallback to a smaller model
        generator = load_text_generation_model("google/flan-t5-base")
        if generator is None:
            return {"error": "Failed to load language model"}
    
    # 4. Prepare context from transcript and slides
    # Truncate transcript to manage context length
    max_transcript_len = 5000
    if len(transcript_text) > max_transcript_len:
        truncated_transcript = transcript_text[:max_transcript_len] + "... [transcript truncated]"
    else:
        truncated_transcript = transcript_text
    
    context = f"Lecture: {lecture_title}\n\nTranscript:\n{truncated_transcript}"
    
    # Add slide content if available
    if slide_text:
        context += f"\n\nSlide Content:\n{slide_text}"
    
    # 5. Generate main topic and title
    title_prompt = f"""
    Based on this lecture transcript and slide content, provide a concise but descriptive title
    for this lecture (as a single phrase or short sentence). The title should clearly indicate
    the main subject area covered.
    
    Context:
    {context[:2000]}
    
    Return only the title, nothing else.
    """.strip()
    
    main_topic_result = generator(title_prompt, max_length=50, do_sample=True, temperature=0.5)
    main_topic = main_topic_result[0]['generated_text'] if isinstance(main_topic_result, list) else main_topic_result
    main_topic = main_topic.strip()
    
    # 6. Generate an outline of key sections
    outline_prompt = f"""
    You are an expert educator creating lecture notes. Based on the transcript and slides, 
    identify the 3-6 most important sections that should appear in well-structured lecture notes 
    on {main_topic}.
    
    Each section should be a key concept, definition, or topic area. Format your response as a
    simple list of section headings only.
    
    Context:
    {context[:3000]}
    
    List only the section headings, one per line.
    """.strip()
    
    outline_result = generator(outline_prompt, max_length=300, do_sample=True, temperature=0.7)
    outline_text = outline_result[0]['generated_text'] if isinstance(outline_result, list) else outline_result
    
    # Parse sections from the outline
    section_titles = [line.strip() for line in outline_text.split('\n') if line.strip()]
    # Filter out any non-title-like entries (too long, etc.)
    section_titles = [title for title in section_titles if len(title) < 100]
    
    # 7. Generate content for each section
    sections = []
    
    # Determine how to split the context
    context_segments = []
    if len(context) > 6000:
        # Split into thirds with some overlap
        third = len(context) // 3
        context_segments = [
            context[:third*2],
            context[third:third*2 + third],
            context[third*2:]
        ]
    else:
        context_segments = [context]
    
    # Use appropriate context segment for each section
    for i, section_title in enumerate(section_titles):
        # Select context segment based on position
        if len(context_segments) > 1:
            segment_idx = min(i // 2, len(context_segments) - 1)
            current_context = context_segments[segment_idx]
        else:
            current_context = context_segments[0]
        
        section_prompt = f"""
        You are creating detailed lecture notes on {main_topic}. Write the content for the section: "{section_title}".
        
        Based on this context:
        {current_context[:4000]}
        
        Generate well-structured notes for this section. Include:
        1. Clear definitions of key concepts
        2. Important points formatted as bullet points
        3. Any formulas, algorithms, or examples mentioned
        
        Write in a clear academic style. Use bullet points for lists. 
        Limit your response to just the content for this section.
        """.strip()
        
        section_result = generator(section_prompt, max_length=800, do_sample=True, temperature=0.7)
        section_content = section_result[0]['generated_text'] if isinstance(section_result, list) else section_result
        
        sections.append({
            "title": section_title,
            "content": section_content
        })
    
    # 8. Assemble the full notes structure
    notes_structure = {
        "lecture_title": lecture_title,
        "main_topic": main_topic,
        "sections": sections,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 9. Format the notes into the desired structure
    formatted_notes = format_structured_notes(notes_structure)
    
    # 10. Save the notes
    output_filename = os.path.basename(transcript_file).replace('_transcript.json', '_notes.txt')
    output_path = os.path.join(NOTES_DIR, output_filename)
    
    with open(output_path, 'w') as f:
        f.write(formatted_notes)
    
    print(f"Notes generated and saved to: {output_path}")
    
    # Also save the structured data for potential future use
    structure_path = os.path.join(NOTES_DIR, output_filename.replace('.txt', '_structure.json'))
    with open(structure_path, 'w') as f:
        json.dump(notes_structure, f, indent=2)
    
    return {"success": True, "output_path": output_path}

def process_all_transcripts():
    """Process all transcript files and generate structured notes"""
    results = []
    
    # Get all transcript files
    transcript_files = list(TRANSCRIPTS_DIR.glob("*_transcript.json"))
    if not transcript_files:
        print("No transcript files found in", TRANSCRIPTS_DIR)
        return []
    
    print(f"Found {len(transcript_files)} transcript files. Generating structured notes...")
    
    for i, transcript_file in enumerate(transcript_files):
        try:
            transcript_name = transcript_file.stem.replace('_transcript', '')
            print(f"[{i+1}/{len(transcript_files)}] Processing {transcript_name}...")
            
            # Find matching slide content if available
            slide_content_file = None
            
            # First, look for exact match with transcript name
            potential_slide_content = PROCESSED_DIR / f"{transcript_name}_content.json"
            if potential_slide_content.exists():
                slide_content_file = potential_slide_content
            else:
                # Look for topic prefix match
                topic_prefix = transcript_name.split('_')[0]  # Get topic prefix
                slide_content_files = list(PROCESSED_DIR.glob(f"{topic_prefix}*_content.json"))
                if slide_content_files:
                    slide_content_file = slide_content_files[0]
            
            if slide_content_file:
                print(f"  Found matching slide content: {slide_content_file.name}")
            
            # Generate structured notes
            result = generate_structured_lecture_notes(transcript_file, slide_content_file)
            
            if "error" not in result:
                results.append({
                    "transcript": str(transcript_file),
                    "slide_content": str(slide_content_file) if slide_content_file else None,
                    "output": result.get("output_path"),
                    "status": "success"
                })
                print(f"  ✓ Successfully generated structured notes for {transcript_name}")
            else:
                results.append({
                    "transcript": str(transcript_file),
                    "slide_content": str(slide_content_file) if slide_content_file else None,
                    "error": result.get("error"),
                    "status": "error"
                })
                print(f"  ✗ Failed to generate structured notes for {transcript_name}: {result.get("error")}")
        
        except Exception as e:
            print(f"  ✗ Error processing {transcript_file.name}: {e}")
            results.append({
                "transcript": str(transcript_file),
                "error": str(e),
                "status": "error"
            })
    
    # Save results log
    log_path = PROCESSED_DIR / "structured_notes_generation_log.json"
    log_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(transcript_files),
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "results": results
    }
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nStructured notes generation complete! Successfully processed {log_data['success_count']}/{len(transcript_files)} files.")
    print(f"Log saved to: {log_path}")
    
    return results

def generate_structured_notes_for_file(transcript_file, slide_content_file=None):
    """Generate structured notes for a specific transcript file"""
    if not os.path.exists(transcript_file):
        print(f"Transcript file not found: {transcript_file}")
        return {"error": "Transcript file not found"}
    
    if slide_content_file and not os.path.exists(slide_content_file):
        print(f"Slide content file not found: {slide_content_file}")
        slide_content_file = None
    
    return generate_structured_lecture_notes(transcript_file, slide_content_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate structured lecture notes from transcripts and slide content")
    parser.add_argument("--process-all", action="store_true", help="Process all available transcripts")
    parser.add_argument("--transcript", type=str, help="Path to a specific transcript file to process")
    parser.add_argument("--slides", type=str, help="Path to a specific slide content file to use (optional)")
    
    args = parser.parse_args()
    
    if args.process_all:
        process_all_transcripts()
    elif args.transcript:
        generate_structured_notes_for_file(args.transcript, args.slides)
    else:
        print("Please specify either --process-all or --transcript")
        parser.print_help()
import os
import json
import re
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import difflib
from tqdm import tqdm
import time

# Model path - update this to your local model path
MODEL_PATH = "D:/pdf/pythonProject/model"

# Global variables for model and tokenizer
tokenizer = None
model = None


def load_model_if_needed():
    """Load the model and tokenizer with 8-bit quantization if not already loaded"""
    global model, tokenizer

    if model is None or tokenizer is None:
        print("Loading model and tokenizer...")
        start_time = time.time()

        # Configure 8-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # Load model with 8-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Optimize for inference
        model.eval()  # Set to evaluation mode

        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")


def compare_documents(doc1_content, doc2_content):
    """Compare documents using local Llama 2 with fallback to difflib"""
    # Load model if not already loaded
    load_model_if_needed()

    # Process document content in chunks to handle larger documents
    chunks = create_chunks_for_comparison(doc1_content, doc2_content)
    print(f"Document split into {len(chunks)} chunks for processing")

    comparison_results = []
    llm_failed = False

    # Calculate total text to process for progress bar
    total_chunks = min(len(chunks), 5)  # Limit to 5 chunks

    # Process each chunk with progress bar
    for i, chunk in tqdm(enumerate(chunks[:total_chunks]), total=total_chunks, desc="Processing document chunks"):
        # Skip empty chunks
        if not chunk["doc1"].strip() and not chunk["doc2"].strip():
            continue

        if not llm_failed:  # Only try LLM if previous attempts haven't failed
            try:
                # Clean up memory before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Generate result using local model
                result = generate_comparison_with_local_model(chunk["doc1"], chunk["doc2"])
                comparison_results.append(result)
                print(f"✓ Successfully processed chunk {i + 1}/{total_chunks}")

            except Exception as e:
                print(f"Error using LLM for chunk {i + 1}: {str(e)}")
                print("Switching to difflib for this and remaining chunks")
                llm_failed = True  # Mark as failed to avoid further attempts

                # Fallback to difflib for this chunk
                fallback = perform_difflib_comparison_for_chunk(chunk["doc1"], chunk["doc2"])
                comparison_results.append(fallback)
        else:
            # Use difflib if LLM already failed
            fallback = perform_difflib_comparison_for_chunk(chunk["doc1"], chunk["doc2"])
            comparison_results.append(fallback)

    # If LLM failed entirely or we didn't get any results, use full difflib comparison
    if llm_failed or not comparison_results:
        print("Using full difflib comparison")
        return perform_difflib_comparison(doc1_content, doc2_content)

    # Combine results from all chunks
    return combine_comparison_results(comparison_results)


def generate_comparison_with_local_model(doc1, doc2):
    """Generate comparison using local Llama 2 model"""
    global model, tokenizer

    # Create optimized prompt (more concise to save tokens)
    prompt = create_optimized_prompt(doc1, doc2)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response with optimized parameters
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=512,  # Reduced from 1024 to save memory
            temperature=0.1,  # Lower temperature for deterministic output
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=False,  # Greedy decoding for faster inference
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode response
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the response
    response_only = response_text[len(prompt):]

    # Parse the response into structured comparison result
    return parse_comparison_result(response_only, {"doc1": doc1, "doc2": doc2})


def create_optimized_prompt(doc1, doc2):
    """Create an optimized prompt for Llama 2 that uses fewer tokens"""
    return f"""<s>[INST] Compare these documents and identify all differences as JSON:

DOC1:
{doc1}

DOC2:
{doc2}

Format as:
{{
  "differences": [
    {{
      "type": "ADDED|REMOVED|MODIFIED",
      "doc1Text": "text in doc1 or empty if added",
      "doc2Text": "text in doc2 or empty if removed"
    }}
  ],
  "summary": {{
    "additions": number,
    "deletions": number,
    "modifications": number
  }}
}}
[/INST]"""


def create_chunks_for_comparison(doc1_content, doc2_content):
    """Create optimized chunks for comparison to handle large documents"""
    # If documents are small, process them as a single chunk
    doc1_len = len(doc1_content)
    doc2_len = len(doc2_content)

    if doc1_len < 1500 and doc2_len < 1500:
        return [{"doc1": doc1_content, "doc2": doc2_content}]

    # For larger documents, create smarter chunks with minimal overlap
    chunks = []

    # Determine chunk size based on document length
    if max(doc1_len, doc2_len) > 10000:
        chunk_size = 1000  # Smaller chunks for very large documents
    else:
        chunk_size = 1500  # Larger chunks for moderate size documents

    # Split both documents into paragraphs
    doc1_paragraphs = re.split(r'\n\s*\n', doc1_content)
    doc1_paragraphs = [p.strip() for p in doc1_paragraphs if p.strip()]

    doc2_paragraphs = re.split(r'\n\s*\n', doc2_content)
    doc2_paragraphs = [p.strip() for p in doc2_paragraphs if p.strip()]

    # Calculate paragraph similarity for better matching
    similarity_matrix = calculate_paragraph_similarity(doc1_paragraphs, doc2_paragraphs)

    # Create chunks with similar content
    doc1_idx = 0
    doc2_idx = 0

    while doc1_idx < len(doc1_paragraphs) or doc2_idx < len(doc2_paragraphs):
        # Get paragraphs for current chunk
        doc1_end = min(doc1_idx + 3, len(doc1_paragraphs))
        doc2_end = min(doc2_idx + 3, len(doc2_paragraphs))

        doc1_chunk = "\n\n".join(doc1_paragraphs[doc1_idx:doc1_end])
        doc2_chunk = "\n\n".join(doc2_paragraphs[doc2_idx:doc2_end])

        # Ensure chunks don't exceed the maximum size
        if len(doc1_chunk) + len(doc2_chunk) > chunk_size * 2:
            # If too large, reduce size
            doc1_end = doc1_idx + max(1, (doc1_end - doc1_idx) // 2)
            doc2_end = doc2_idx + max(1, (doc2_end - doc2_idx) // 2)

            doc1_chunk = "\n\n".join(doc1_paragraphs[doc1_idx:doc1_end])
            doc2_chunk = "\n\n".join(doc2_paragraphs[doc2_idx:doc2_end])

        chunks.append({"doc1": doc1_chunk, "doc2": doc2_chunk})

        # Update indices
        doc1_idx = doc1_end
        doc2_idx = doc2_end

    return chunks


def calculate_paragraph_similarity(paragraphs1, paragraphs2):
    """Calculate similarity between paragraphs for better chunking"""
    # This is a simplified implementation - could be more sophisticated
    similarity_matrix = []

    for p1 in paragraphs1:
        row = []
        for p2 in paragraphs2:
            # Simple similarity - ratio of shared words
            words1 = set(p1.lower().split())
            words2 = set(p2.lower().split())

            if not words1 or not words2:
                row.append(0)
                continue

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            similarity = intersection / union if union > 0 else 0
            row.append(similarity)

        similarity_matrix.append(row)

    return similarity_matrix


def parse_comparison_result(llm_response, chunk):
    """Parse LLM response with error handling and optimized for local model output"""
    try:
        # Extract JSON from response (might be surrounded by additional text)
        json_match = re.search(r'\{[\s\S]*\}', llm_response)
        if not json_match:
            raise Exception("No valid JSON found in LLM response")

        parsed_response = json.loads(json_match.group(0))

        # Validate and clean up the response
        differences = []
        for diff in parsed_response.get("differences", []):
            if not isinstance(diff, dict):
                continue

            # Clean up and standardize the difference
            differences.append({
                "type": diff.get("type", "UNKNOWN"),
                "doc1Text": diff.get("doc1Text", "").strip(),
                "doc2Text": diff.get("doc2Text", "").strip(),
                "context": diff.get("context", "").strip() if "context" in diff else ""
            })

        # Get summary counts or calculate them from differences
        summary = parsed_response.get("summary", {})
        additions = summary.get("additions", 0)
        deletions = summary.get("deletions", 0)
        modifications = summary.get("modifications", 0)

        # If summary values are missing, calculate from differences
        if additions == 0 and deletions == 0 and modifications == 0:
            for diff in differences:
                if diff["type"] == "ADDED":
                    additions += 1
                elif diff["type"] == "REMOVED":
                    deletions += 1
                elif diff["type"] == "MODIFIED":
                    modifications += 1

        return {
            "differences": differences,
            "summary": {
                "additions": additions,
                "deletions": deletions,
                "modifications": modifications
            }
        }
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        # Return fallback using difflib
        return perform_difflib_comparison_for_chunk(chunk["doc1"], chunk["doc2"])


def perform_difflib_comparison_for_chunk(doc1_content, doc2_content):
    """Perform difflib comparison for a single chunk - optimized for performance"""
    # For small chunks, use line-by-line comparison
    doc1_lines = doc1_content.splitlines()
    doc2_lines = doc2_content.splitlines()

    differences = []
    additions = 0
    deletions = 0
    modifications = 0

    # Use sequence matcher for more accurate grouping of changes
    matcher = difflib.SequenceMatcher(None, doc1_lines, doc2_lines)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # This is a modification
            differences.append({
                "type": "MODIFIED",
                "doc1Text": '\n'.join(doc1_lines[i1:i2]),
                "doc2Text": '\n'.join(doc2_lines[j1:j2]),
                "context": f"Lines {i1 + 1}-{i2}" if i1 != i2 - 1 else f"Line {i1 + 1}"
            })
            modifications += 1

        elif tag == 'delete':
            # Content was removed
            differences.append({
                "type": "REMOVED",
                "doc1Text": '\n'.join(doc1_lines[i1:i2]),
                "doc2Text": "",
                "context": f"After line {j1} in doc2"
            })
            deletions += 1

        elif tag == 'insert':
            # Content was added
            differences.append({
                "type": "ADDED",
                "doc1Text": "",
                "doc2Text": '\n'.join(doc2_lines[j1:j2]),
                "context": f"After line {i1} in doc1"
            })
            additions += 1

    return {
        "differences": differences,
        "summary": {
            "additions": additions,
            "deletions": deletions,
            "modifications": modifications
        }
    }


def perform_difflib_comparison(doc1_content, doc2_content):
    """Perform full difflib comparison for entire documents - optimized for performance"""
    # For larger documents, compare section by section
    doc1_sections = re.split(r'\n\s*\n', doc1_content)
    doc1_sections = [s.strip() for s in doc1_sections if s.strip()]

    doc2_sections = re.split(r'\n\s*\n', doc2_content)
    doc2_sections = [s.strip() for s in doc2_sections if s.strip()]

    # Compare section by section
    matcher = difflib.SequenceMatcher(None, doc1_sections, doc2_sections)

    differences = []
    additions = 0
    deletions = 0
    modifications = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            # This is a modification
            doc1_text = '\n\n'.join(doc1_sections[i1:i2])
            doc2_text = '\n\n'.join(doc2_sections[j1:j2])

            differences.append({
                "type": "MODIFIED",
                "doc1Text": doc1_text,
                "doc2Text": doc2_text,
                "context": f"Section {i1 + 1}"
            })
            modifications += 1

        elif tag == 'delete':
            # Section was removed
            differences.append({
                "type": "REMOVED",
                "doc1Text": '\n\n'.join(doc1_sections[i1:i2]),
                "doc2Text": "",
                "context": f"After section {j1} in doc2"
            })
            deletions += 1

        elif tag == 'insert':
            # Section was added
            differences.append({
                "type": "ADDED",
                "doc1Text": "",
                "doc2Text": '\n\n'.join(doc2_sections[j1:j2]),
                "context": f"After section {i1} in doc1"
            })
            additions += 1

    # Limit number of differences to avoid overwhelming the UI
    max_diffs = 15
    limited_diffs = differences[:max_diffs]

    return {
        "differences": limited_diffs,
        "summary": {
            "additions": additions,
            "deletions": deletions,
            "modifications": modifications
        }
    }


def combine_comparison_results(chunk_results):
    """Combine results from multiple chunks - optimized for memory efficiency"""
    combined_result = {
        "differences": [],
        "summary": {
            "additions": 0,
            "deletions": 0,
            "modifications": 0
        }
    }

    # Process each chunk result
    for result in chunk_results:
        if not result:
            continue

        # Add differences
        combined_result["differences"].extend(result.get("differences", []))

        # Update summary counts
        summary = result.get("summary", {})
        combined_result["summary"]["additions"] += summary.get("additions", 0)
        combined_result["summary"]["deletions"] += summary.get("deletions", 0)
        combined_result["summary"]["modifications"] += summary.get("modifications", 0)

    # Deduplicate differences
    unique_differences = []
    seen = set()

    for diff in combined_result["differences"]:
        key = f"{diff['type']}:{diff['doc1Text'][:50]}:{diff['doc2Text'][:50]}"
        if key not in seen:
            seen.add(key)
            unique_differences.append(diff)

    # Limit to 15 differences max
    combined_result["differences"] = unique_differences[:15]

    return combined_result


def unload_model():
    """Unload model to free up memory"""
    global model, tokenizer

    if model is not None:
        del model
        model = None

    if tokenizer is not None:
        del tokenizer
        tokenizer = None

    # Force garbage collection
    gc.collect()

    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Model unloaded and memory cleared")


# Example usage
if __name__ == "__main__":
    # Test the comparison function with sample documents
    doc1 = """This is a sample document.
It contains several paragraphs.
This paragraph will be modified.
This paragraph will be deleted.
This paragraph remains unchanged."""

    doc2 = """This is a sample document.
It contains several paragraphs.
This paragraph has been changed significantly.
This paragraph remains unchanged.
This is a new paragraph that was added."""

    # Compare the documents
    print("Comparing sample documents...")
    result = compare_documents(doc1, doc2)

    # Display the results
    print("\nComparison Results:")
    print(f"Additions: {result['summary']['additions']}")
    print(f"Deletions: {result['summary']['deletions']}")
    print(f"Modifications: {result['summary']['modifications']}")

    print("\nDetailed Differences:")
    for diff in result['differences']:
        print(f"Type: {diff['type']}")
        print(f"Doc1: {diff['doc1Text']}")
        print(f"Doc2: {diff['doc2Text']}")
        print("---")

    # Unload model to free memory
    unload_model()
import PyPDF2
import re


def extract_text_from_pdf(file_path):
    """Extract text from PDF while preserving structure"""
    try:
        pages = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)

            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()

                # Split text into paragraphs
                paragraphs = group_into_paragraphs(text)

                pages.append({
                    'pageNumber': page_num + 1,
                    'paragraphs': paragraphs
                })

        return pages

    except Exception as e:
        print(f"Error extracting PDF text: {str(e)}")
        raise Exception(f"Failed to extract text from PDF: {str(e)}")


def group_into_paragraphs(text):
    """Group text into paragraphs based on blank lines and formatting"""
    if not text:
        return []

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Split by double newlines or similar paragraph markers
    paragraphs = re.split(r'\n\s*\n|\r\n\s*\r\n', text)

    # Clean up paragraphs
    clean_paragraphs = []
    for p in paragraphs:
        # Remove leading/trailing whitespace and normalize internal spaces
        clean_p = re.sub(r'\s+', ' ', p).strip()
        if clean_p:  # Only add non-empty paragraphs
            clean_paragraphs.append(clean_p)

    # If no paragraphs were detected but we have text, treat the whole text as one paragraph
    if not clean_paragraphs and text.strip():
        clean_paragraphs = [text.strip()]

    return clean_paragraphs
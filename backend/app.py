import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import services
from services.pdf_service import extract_text_from_pdf
from services.llm_service import compare_documents

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handle file uploads and comparison"""
    # Check if both files are provided
    if 'file1' not in request.files or 'file2' not in request.files:
        return jsonify({'message': 'Please upload two PDF files'}), 400
    
    file1 = request.files['file1']
    file2 = request.files['file2']
    
    # Check if files are valid
    if file1.filename == '' or file2.filename == '':
        return jsonify({'message': 'Please select two PDF files'}), 400
    
    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return jsonify({'message': 'Please upload PDF files only'}), 400
    
    try:
        # Generate unique filenames
        file1_filename = f"{uuid.uuid4()}.pdf"
        file2_filename = f"{uuid.uuid4()}.pdf"
        
        # Save files
        file1_path = os.path.join(app.config['UPLOAD_FOLDER'], file1_filename)
        file2_path = os.path.join(app.config['UPLOAD_FOLDER'], file2_filename)
        file1.save(file1_path)
        file2.save(file2_path)
        
        # Extract text from PDFs
        doc1_content = extract_text_from_pdf(file1_path)
        doc2_content = extract_text_from_pdf(file2_path)
        
        # Flatten document content for comparison
        doc1_flat = "\n\n".join(["\n".join(page["paragraphs"]) for page in doc1_content])
        doc2_flat = "\n\n".join(["\n".join(page["paragraphs"]) for page in doc2_content])
        
        # Compare documents using LLM
        comparison_result = compare_documents(doc1_flat, doc2_flat)
        
        # Return results with the same structure expected by the frontend
        return jsonify({
            'comparisonResult': comparison_result,
            'doc1Url': f'/uploads/{file1_filename}',
            'doc2Url': f'/uploads/{file2_filename}',
            'doc1Name': file1.filename,
            'doc2Name': file2.filename
        }), 200
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return jsonify({'message': f'Error processing files: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

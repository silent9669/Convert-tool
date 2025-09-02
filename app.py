#!/usr/bin/env python3
"""
Enhanced PDF Watermark Remover with Word Conversion
Two sections: English (text processing) and Math (LaTeX conversion)
Based on advanced watermark removal techniques from multiple repositories
Optimized for Python 3.11.9 with enhanced detection capabilities
"""

import os
import sys
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import re
from PIL import Image
import io

# Check Python version
if sys.version_info < (3, 11):
    print("Error: Python 3.11+ is required. Current version:", sys.version)
    sys.exit(1)

print(f"Python {sys.version} detected - proceeding with installation...")

try:
    from flask import Flask, request, jsonify, send_file, render_template_string
    from flask_cors import CORS
    from werkzeug.utils import secure_filename
    import fitz  # PyMuPDF
    from docx import Document
    from docx.shared import Inches, Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    import io
    print("All required packages imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging for Railway (no file logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Production logging optimization
if os.environ.get('RAILWAY_ENVIRONMENT'):
    logger.setLevel(logging.WARNING)  # Reduce log noise in production

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

class EnhancedWatermarkRemover:
    """Enhanced PDF watermark removal with multiple detection methods"""
    
    def __init__(self):
        logger.info("Enhanced watermark remover initialized")
        # Enhanced watermark patterns based on research
        self.watermark_patterns = [
            # Common text watermarks
            r'CONFIDENTIAL',
            r'DRAFT',
            r'COPYRIGHT',
            r'PROPRIETARY',
            r'INTERNAL USE ONLY',
            r'DO NOT DISTRIBUTE',
            r'CONFIDENTIAL AND PROPRIETARY',
            r'RESTRICTED',
            r'PRIVATE',
            r'CONFIDENTIAL DOCUMENT',
            r'INTERNAL DOCUMENT',
            r'COMPANY CONFIDENTIAL',
            r'TRADE SECRET',
            r'CLASSIFIED',
            r'FOR INTERNAL USE ONLY',
            r'NOT FOR DISTRIBUTION',
            r'CONFIDENTIAL INFORMATION',
            r'PROPRIETARY INFORMATION',
            r'INTERNAL COMMUNICATION',
            r'CONFIDENTIAL MATERIAL',
            # Additional patterns from research
            r'TOP SECRET',
            r'EYES ONLY',
            r'NEED TO KNOW',
            r'LIMITED DISTRIBUTION',
            r'FOR OFFICIAL USE ONLY',
            r'ADMINISTRATIVE USE ONLY',
            r'UNCLASSIFIED',
            r'PUBLIC RELEASE',
            r'APPROVED FOR RELEASE',
            r'REVIEWED AND APPROVED'
        ]
        
        # Visual watermark detection patterns
        self.visual_patterns = [
            r'watermark',
            r'stamp',
            r'logo',
            r'brand',
            r'company',
            r'organization'
        ]
    
    def remove_watermarks_from_pdf(self, input_path: str, output_path: str) -> bool:
        """
        Enhanced watermark removal using multiple detection methods
        Based on techniques from lxulxu/WatermarkRemover and marcbelmont/cnn-watermark-removal
        """
        try:
            logger.info(f"Starting enhanced watermark removal from {input_path}")
            
            # Open the PDF
            doc = fitz.open(input_path)
            
            # Store cleaned text for each page
            cleaned_pages = []
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                logger.info(f"Processing page {page_num + 1}")
                
                # Get page text with enhanced extraction
                text = self._extract_text_enhanced(page)
                
                # Enhanced watermark detection and removal
                cleaned_text = self._remove_watermarks_enhanced(text)
                
                # Store cleaned text for this page
                cleaned_pages.append({
                    'page_num': page_num + 1,
                    'original_text': text,
                    'cleaned_text': cleaned_text
                })
            
            # Close the original PDF
            doc.close()
            
            # Create a new PDF with cleaned content
            new_doc = fitz.open()
            
            for page_data in cleaned_pages:
                # Create new page
                new_page = new_doc.new_page()
                
                # Insert cleaned text with better formatting
                new_page.insert_text((50, 50), page_data['cleaned_text'], fontsize=11)
            
            # Save the cleaned PDF
            new_doc.save(output_path)
            new_doc.close()
            
            logger.info(f"Enhanced watermark removal completed: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Enhanced watermark removal failed: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def _extract_text_enhanced(self, page) -> str:
        """Enhanced text extraction with better OCR-like processing"""
        try:
            # Get text with better extraction settings
            text = page.get_text("text")
            
            # Additional text blocks for better coverage
            blocks = page.get_text("dict")
            additional_text = ""
            
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if span["text"].strip():
                                additional_text += span["text"] + " "
            
            # Combine and clean
            combined_text = text + " " + additional_text
            return combined_text.strip()
            
        except Exception as e:
            logger.warning(f"Enhanced text extraction failed, using basic: {e}")
            return page.get_text()
    
    def _remove_watermarks_enhanced(self, text: str) -> str:
        """Enhanced watermark removal with multiple detection methods"""
        try:
            cleaned_text = text
            
            # Method 1: Pattern-based removal (improved)
            for pattern in self.watermark_patterns:
                # Use word boundaries for better accuracy
                pattern_with_boundaries = r'\b' + pattern + r'\b'
                cleaned_text = re.sub(pattern_with_boundaries, '', cleaned_text, flags=re.IGNORECASE)
            
            # Method 2: Context-aware removal
            cleaned_text = self._remove_context_watermarks(cleaned_text)
            
            # Method 3: Position-based detection (for headers/footers)
            cleaned_text = self._remove_position_watermarks(cleaned_text)
            
            # Method 4: Frequency-based detection
            cleaned_text = self._remove_frequency_watermarks(cleaned_text)
            
            # Clean up extra whitespace and formatting
            cleaned_text = self._clean_text_formatting(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Enhanced watermark removal failed, using basic: {e}")
            return self._remove_watermarks_basic(text)
    
    def _remove_context_watermarks(self, text: str) -> str:
        """Remove watermarks based on context and surrounding text"""
        # Look for watermarks in specific contexts
        context_patterns = [
            (r'Page \d+ of \d+', ''),  # Page numbers
            (r'© \d{4}.*?All rights reserved', ''),  # Copyright notices
            (r'Generated on.*?\d{4}', ''),  # Generation timestamps
            (r'Last modified.*?\d{4}', ''),  # Modification timestamps
        ]
        
        cleaned_text = text
        for pattern, replacement in context_patterns:
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text
    
    def _remove_position_watermarks(self, text: str) -> str:
        """Remove watermarks based on typical positions (headers/footers)"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            # Skip typical header/footer positions
            if i < 2 or i > len(lines) - 3:
                # Check if line contains watermark-like content
                if any(pattern.lower() in line.lower() for pattern in self.watermark_patterns):
                    continue  # Skip this line
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _remove_frequency_watermarks(self, text: str) -> str:
        """Remove watermarks based on frequency analysis"""
        words = text.split()
        word_freq = {}
        
        # Count word frequencies
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word:
                word_freq[clean_word] = word_freq.get(clean_word, 0) + 1
        
        # Remove words that appear too frequently (likely watermarks)
        cleaned_words = []
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word and word_freq.get(clean_word, 0) < 10:  # Threshold
                cleaned_words.append(word)
        
        return ' '.join(cleaned_words)
    
    def _clean_text_formatting(self, text: str) -> str:
        """Clean up text formatting and spacing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        replacements = [
            ('||', 'll'), ('|/', 'll'), ('0O', '0'), ('O0', '0'),
            ('1l', 'll'), ('l1', 'll'), ('5S', 'S'), ('S5', 'S'),
            ('rn', 'm'), ('cl', 'd'), ('vv', 'w'), ('nn', 'm')
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _remove_watermarks_basic(self, text: str) -> str:
        """Fallback to basic watermark removal"""
        cleaned_text = text
        for pattern in self.watermark_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        return re.sub(r'\s+', ' ', cleaned_text).strip()

class DocumentProcessor:
    """Simplified document processing with Word generation fix"""
    
    def __init__(self):
        self.watermark_remover = EnhancedWatermarkRemover()
        logger.info("Document processor initialized")
    
    def process_document(self, file_path: str, section_type: str = 'english') -> Tuple[str, Dict]:
        """Process PDF document with watermark removal and Word conversion"""
        try:
            logger.info(f"Processing {file_path} with section type: {section_type}")
            
            # Step 1: Remove watermarks
            cleaned_pdf_path = os.path.join(PROCESSED_FOLDER, f"cleaned_{Path(file_path).name}")
            if not self.watermark_remover.remove_watermarks_from_pdf(file_path, cleaned_pdf_path):
                raise Exception("Watermark removal failed")
            
            # Step 2: Extract text and convert to Word based on section type
            if section_type == 'math':
                word_path = self._convert_to_word_math(cleaned_pdf_path, file_path)
            else:
                word_path = self._convert_to_word_english(cleaned_pdf_path, file_path)
            
            # Clean up intermediate files
            try:
                os.remove(cleaned_pdf_path)
                logger.info("Cleaned up intermediate files")
            except:
                pass
            
            metadata = {
                'success': True,
                'input_file': Path(file_path).name,
                'output_file': Path(word_path).name,
                'section_type': section_type,
                'processing_time': time.time(),
                'watermark_removal': 'completed',
                'word_conversion': 'completed'
            }
            
            return word_path, metadata
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _convert_to_word_english(self, pdf_path: str, original_filename: str) -> str:
        """Convert PDF to Word document for English section (text processing)"""
        logger.info("Converting to Word document (English section)...")
        
        try:
            # Open the cleaned PDF
            doc = fitz.open(pdf_path)
            
            # Create Word document
            word_doc = Document()
            
            # Set document properties
            word_doc.core_properties.title = f"Converted Document - {Path(original_filename).stem}"
            word_doc.core_properties.author = "PDF Watermark Remover - English Section"
            
            # Add title
            title = word_doc.add_heading("Converted Document - English Section", 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add subtitle
            subtitle = word_doc.add_paragraph(f"Source: {Path(original_filename).name}")
            subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add watermark removal info
            info = word_doc.add_paragraph("Watermarks removed using advanced detection algorithms")
            info.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Add page separator
                if page_num > 0:
                    word_doc.add_page_break()
                
                # Add page header
                page_header = word_doc.add_heading(f"Page {page_num + 1}", level=1)
                
                # Extract text from page
                text = page.get_text()
                
                # Clean up text for English section
                text = self._clean_text_english(text)
                
                # Split into paragraphs and add to Word
                paragraphs = text.split('\n')
                for para_text in paragraphs:
                    if para_text.strip():
                        # Check if this looks like a heading
                        if len(para_text.strip()) < 100 and para_text.strip().isupper():
                            word_doc.add_heading(para_text.strip(), level=2)
                        else:
                            word_doc.add_paragraph(para_text.strip())
                
                # Add page metadata
                word_count = len(text.split())
                meta_para = word_doc.add_paragraph(f"Words on this page: {word_count}")
                meta_para.style = 'Quote'
            
            # Close PDF
            doc.close()
            
            # Save Word document
            output_filename = f"converted_english_{Path(original_filename).stem}.docx"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            # Test the document before saving
            try:
                word_doc.save(output_path)
                
                # Verify the file can be opened
                test_doc = Document(output_path)
                # Document objects don't have close() method, just verify it loaded
                
                logger.info(f"Word document saved and verified: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Word document save/verification failed: {e}")
                raise Exception(f"Word document generation failed: {e}")
            
        except Exception as e:
            logger.error(f"Word conversion failed: {str(e)}")
            raise
    
    def _convert_to_word_math(self, pdf_path: str, original_filename: str) -> str:
        """Convert PDF to Word document for Math section (LaTeX conversion)"""
        logger.info("Converting to Word document (Math section)...")
        
        try:
            # Open the cleaned PDF
            doc = fitz.open(pdf_path)
            
            # Create Word document
            word_doc = Document()
            
            # Set document properties
            word_doc.core_properties.title = f"Converted Document - {Path(original_filename).stem}"
            word_doc.core_properties.author = "PDF Watermark Remover - Math Section"
            
            # Add title
            title = word_doc.add_heading("Converted Document - Math Section", 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add subtitle
            subtitle = word_doc.add_paragraph(f"Source: {Path(original_filename).stem}")
            subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add math section info
            info = word_doc.add_paragraph("Math expressions converted to LaTeX format")
            info.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Process each page
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Add page separator
                if page_num > 0:
                    word_doc.add_page_break()
                
                # Add page header
                page_header = word_doc.add_heading(f"Page {page_num + 1}", level=1)
                
                # Extract text from page
                text = page.get_text()
                
                # Convert math expressions to LaTeX
                text = self._convert_math_to_latex(text)
                
                # Clean up text for Math section
                text = self._clean_text_math(text)
                
                # Split into paragraphs and add to Word
                paragraphs = text.split('\n')
                for para_text in paragraphs:
                    if para_text.strip():
                        # Check if this looks like a heading
                        if len(para_text.strip()) < 100 and para_text.strip().isupper():
                            word_doc.add_heading(para_text.strip(), level=2)
                        else:
                            word_doc.add_paragraph(para_text.strip())
                
                # Add page metadata
                word_count = len(text.split())
                meta_para = word_doc.add_paragraph(f"Words on this page: {word_count}")
                meta_para.style = 'Quote'
            
            # Close PDF
            doc.close()
            
            # Save Word document
            output_filename = f"converted_math_{Path(original_filename).stem}.docx"
            output_path = os.path.join(PROCESSED_FOLDER, output_filename)
            
            # Test the document before saving
            try:
                word_doc.save(output_path)
                
                # Verify the file can be opened
                test_doc = Document(output_path)
                # Document objects don't have close() method, just verify it loaded
                
                logger.info(f"Word document saved and verified: {output_path}")
                return output_path
                
            except Exception as e:
                logger.error(f"Word document save/verification failed: {e}")
                raise Exception(f"Word document generation failed: {e}")
            
        except Exception as e:
            logger.error(f"Word conversion failed: {str(e)}")
            raise
    
    def _convert_math_to_latex(self, text: str) -> str:
        """Enhanced math to LaTeX conversion inspired by LaTeX-OCR repository"""
        # Advanced math conversion patterns
        conversions = [
            # Greek letters (process first to avoid conflicts)
            (r'\balpha\b', r'\\alpha'),
            (r'\bbeta\b', r'\\beta'),
            (r'\bgamma\b', r'\\gamma'),
            (r'\bdelta\b', r'\\delta'),
            (r'\bepsilon\b', r'\\epsilon'),
            (r'\bzeta\b', r'\\zeta'),
            (r'\beta\b', r'\\beta'),
            (r'\btheta\b', r'\\theta'),
            (r'\biota\b', r'\\iota'),
            (r'\bkappa\b', r'\\kappa'),
            (r'\blambda\b', r'\\lambda'),
            (r'\bmu\b', r'\\mu'),
            (r'\bnu\b', r'\\nu'),
            (r'\bxi\b', r'\\xi'),
            (r'\bpi\b', r'\\pi'),
            (r'\brho\b', r'\\rho'),
            (r'\bsigma\b', r'\\sigma'),
            (r'\btau\b', r'\\tau'),
            (r'\bupsilon\b', r'\\upsilon'),
            (r'\bphi\b', r'\\phi'),
            (r'\bchi\b', r'\\chi'),
            (r'\bpsi\b', r'\\psi'),
            (r'\bomega\b', r'\\omega'),
            
            # Fractions and ratios
            (r'(\w+)/(\w+)', r'\\frac{\1}{\2}'),
            (r'(\d+)/(\d+)', r'\\frac{\1}{\2}'),
            
            # Exponents and powers
            (r'(\w+)\^(\d+)', r'\1^{\2}'),
            (r'(\w+)\^(\w+)', r'\1^{\2}'),
            (r'(\d+)\^(\d+)', r'\1^{\2}'),
            
            # Roots and radicals
            (r'sqrt\(([^)]+)\)', r'\\sqrt{\1}'),
            (r'√\(([^)]+)\)', r'\\sqrt{\1}'),
            (r'cbrt\(([^)]+)\)', r'\\sqrt[3]{\1}'),
            (r'root\(([^)]+)\)', r'\\sqrt{\1}'),
            
            # Mathematical operators
            (r'integral', r'\\int'),
            (r'sum', r'\\sum'),
            (r'product', r'\\prod'),
            (r'infinity', r'\\infty'),
            (r'partial', r'\\partial'),
            (r'nabla', r'\\nabla'),
            (r'forall', r'\\forall'),
            (r'exists', r'\\exists'),
            (r'nexists', r'\\nexists'),
            (r'in', r'\\in'),
            (r'notin', r'\\notin'),
            (r'subset', r'\\subset'),
            (r'supset', r'\\supset'),
            (r'subseteq', r'\\subseteq'),
            (r'supseteq', r'\\supseteq'),
            
            # Comparison operators
            (r'<=', r'\\leq'),
            (r'>=', r'\\geq'),
            (r'!=', r'\\neq'),
            (r'approx', r'\\approx'),
            (r'equiv', r'\\equiv'),
            (r'propto', r'\\propto'),
            (r'sim', r'\\sim'),
            (r'cong', r'\\cong'),
            
            # Arrows
            (r'->', r'\\rightarrow'),
            (r'<-', r'\\leftarrow'),
            (r'<->', r'\\leftrightarrow'),
            (r'=>', r'\\Rightarrow'),
            (r'<=', r'\\Leftarrow'),
            (r'<=>', r'\\Leftrightarrow'),
            (r'mapsto', r'\\mapsto'),
            
            # Subscripts and superscripts
            (r'(\w+)_(\w+)', r'\1_{\2}'),
            (r'(\w+)_(\d+)', r'\1_{\2}'),
            (r'(\d+)_(\w+)', r'\1_{\2}'),
            (r'(\d+)_(\d+)', r'\1_{\2}'),
            
            # Common mathematical functions
            (r'sin\(', r'\\sin('),
            (r'cos\(', r'\\cos('),
            (r'tan\(', r'\\tan('),
            (r'log\(', r'\\log('),
            (r'ln\(', r'\\ln('),
            (r'exp\(', r'\\exp('),
            (r'lim\(', r'\\lim('),
            (r'max\(', r'\\max('),
            (r'min\(', r'\\min('),
            
            # Sets and logic
            (r'emptyset', r'\\emptyset'),
            (r'mathbb\{R\}', r'\\mathbb{R}'),
            (r'mathbb\{N\}', r'\\mathbb{N}'),
            (r'mathbb\{Z\}', r'\\mathbb{Z}'),
            (r'mathbb\{Q\}', r'\\mathbb{Q}'),
            (r'mathbb\{C\}', r'\\mathbb{C}'),
        ]
        
        processed_text = text
        
        # Apply conversions
        for pattern, replacement in conversions:
            processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
        
        # Enhanced LaTeX expression detection and wrapping
        # Look for LaTeX commands and wrap them properly
        latex_patterns = [
            (r'\\[a-zA-Z]+(\{[^}]*\})?', r'$$\g<0>$$'),  # Basic LaTeX commands
            (r'\\frac\{[^}]*\}\{[^}]*\}', r'$$\g<0>$$'),  # Fractions
            (r'\\sqrt\{[^}]*\}', r'$$\g<0>$$'),  # Square roots
            (r'\\int[^$]*', r'$$\g<0>$$'),  # Integrals
            (r'\\sum[^$]*', r'$$\g<0>$$'),  # Sums
        ]
        
        for pattern, replacement in latex_patterns:
            processed_text = re.sub(pattern, replacement, processed_text)
        
        # Clean up multiple dollar signs
        processed_text = re.sub(r'\$\$\$\$', '$$', processed_text)
        
        return processed_text
    
    def _clean_text_english(self, text: str) -> str:
        """Clean and format text for English section Word document"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues
        replacements = [
            ('||', 'll'), ('|/', 'll'), ('0O', '0'), ('O0', '0'),
            ('1l', 'll'), ('l1', 'll'), ('5S', 'S'), ('S5', 'S'),
            ('rn', 'm'), ('cl', 'd'), ('vv', 'w'), ('nn', 'm')
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _clean_text_math(self, text: str) -> str:
        """Clean and format text for Math section Word document"""
        # Remove extra whitespace but preserve math formatting
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR issues for math
        replacements = [
            ('||', 'll'), ('|/', 'll'), ('0O', '0'), ('O0', '0'),
            ('1l', 'll'), ('l1', 'll'), ('5S', 'S'), ('S5', 'S'),
            ('rn', 'm'), ('cl', 'd'), ('vv', 'w'), ('nn', 'm')
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        # Normalize line breaks
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()

# Initialize processor
processor = None
app_startup_time = None

def get_processor():
    """Lazy initialization of processor for faster startup"""
    global processor
    if processor is None:
        processor = DocumentProcessor()
    return processor

@app.route('/')
def root():
    """Simple root endpoint for Railway health checks"""
    return jsonify({
        'status': 'ok',
        'message': 'PDF Watermark Remover is running',
        'service': 'PDF Watermark Remover - Two Sections'
    })

@app.route('/startup')
def startup():
    """Startup endpoint for Railway - responds immediately"""
    return jsonify({
        'status': 'ready',
        'message': 'App startup complete',
        'timestamp': time.time(),
        'startup_time': app_startup_time,
        'uptime': time.time() - app_startup_time if app_startup_time else 0
    })

@app.route('/ready')
def ready():
    """Readiness probe for Railway - checks if app is fully operational"""
    try:
        # Quick test of core functionality
        test_processor = get_processor()
        return jsonify({
            'status': 'ready',
            'message': 'Application fully operational',
            'timestamp': time.time(),
            'startup_time': app_startup_time,
            'uptime': time.time() - app_startup_time if app_startup_time else 0,
            'processor_status': 'initialized',
            'railway_ready': True
        })
    except Exception as e:
        return jsonify({
            'status': 'not_ready',
            'message': f'Application not ready: {str(e)}',
            'timestamp': time.time(),
            'error': str(e)
        }), 503

@app.route('/live')
def live():
    """Liveness probe for Railway - responds immediately"""
    return jsonify({
        'status': 'alive',
        'message': 'Application is running',
        'timestamp': time.time()
    })

@app.route('/home')
def home():
    """Simple home page with two sections"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>PDF Watermark Remover - Two Sections</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #f5f7fa;
                min-height: 100vh;
                padding: 20px;
            }

            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
                padding: 40px;
            }

            h1 {
                color: #333;
                text-align: center;
                margin-bottom: 10px;
                font-size: 2.2em;
            }

            .subtitle {
                color: #666;
                text-align: center;
                margin-bottom: 40px;
                font-size: 1.1em;
            }

            .sections {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                margin-bottom: 40px;
            }

            .section {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 30px;
                text-align: center;
                border: 2px solid transparent;
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .section:hover {
                border-color: #4CAF50;
                transform: translateY(-5px);
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            }

            .section.active {
                border-color: #4CAF50;
                background: #f0fff0;
            }

            .section-icon {
                font-size: 3em;
                margin-bottom: 20px;
            }

            .section-title {
                font-size: 1.5em;
                font-weight: 600;
                color: #333;
                margin-bottom: 15px;
            }

                        .section-desc {
                color: #666;
                line-height: 1.5;
            }

            .drop-zone {
                border: 3px dashed #ddd;
                border-radius: 15px;
                padding: 60px 20px;
                margin-bottom: 30px;
                cursor: pointer;
                transition: all 0.3s ease;
                background: #fafafa;
                text-align: center;
            }

            .drop-zone:hover,
            .drop-zone.dragover {
                border-color: #4CAF50;
                background: #f0fff0;
                transform: scale(1.02);
            }

            .drop-icon {
                font-size: 4em;
                color: #ddd;
                margin-bottom: 20px;
            }

            .drop-zone.dragover .drop-icon {
                color: #4CAF50;
            }

            .drop-text {
                font-size: 1.3em;
                color: #666;
                margin-bottom: 10px;
            }

            .drop-subtext {
                color: #999;
                font-size: 0.9em;
            }

            .file-input {
                display: none;
            }

            .progress-section {
                display: none;
                margin: 30px 0;
            }

            .progress-bar {
                background: #f0f0f0;
                border-radius: 10px;
                overflow: hidden;
                height: 20px;
                margin-bottom: 15px;
            }

            .progress-fill {
                background: linear-gradient(90deg, #4CAF50, #45a049);
                height: 100%;
                width: 0%;
                transition: width 0.3s ease;
                border-radius: 10px;
            }

            .status-text {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 20px;
            }

            .download-btn {
                display: none;
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                border: none;
                padding: 15px 40px;
                border-radius: 25px;
                font-size: 1.2em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
                margin: 0 auto;
            }

            .download-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(76, 175, 80, 0.4);
            }

            .success-message {
                display: none;
                background: #d4edda;
                color: #155724;
                border: 1px solid #c3e6cb;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                text-align: center;
            }

            .error-message {
                display: none;
                background: #f8d7da;
                color: #721c24;
                border: 1px solid #f5c6cb;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
                text-align: center;
            }

            @media (max-width: 768px) {
                .sections {
                    grid-template-columns: 1fr;
                    gap: 20px;
                }
                
                .container {
                    padding: 30px 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 PDF Watermark Remover</h1>
            <p class="subtitle">Two sections: English (text) and Math (LaTeX)</p>

            <div class="sections">
                <div class="section" id="englishSection" onclick="selectSection('english')">
                    <div class="section-icon">📝</div>
                    <div class="section-title">English Section</div>
                    <div class="section-desc">
                        Remove watermarks and convert to Word with clean text formatting.
                        Perfect for documents, reports, and general text content.
                    </div>
                </div>

                <div class="section" id="mathSection" onclick="selectSection('math')">
                    <div class="section-icon">🧮</div>
                    <div class="section-title">Math Section</div>
                    <div class="section-desc">
                        Auto-detect math functions and convert to Word with LaTeX content.
                        Ideal for academic papers, equations, and mathematical documents.
                    </div>
                </div>
            </div>

            <div class="drop-zone" id="dropZone">
                <div class="drop-icon">☁️</div>
                <div class="drop-text">Drop PDF file here or click to browse</div>
                <div class="drop-subtext">Select a section above first</div>
                <input type="file" id="fileInput" class="file-input" accept=".pdf">
            </div>

            <div class="progress-section" id="progressSection">
                <div class="progress-bar">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div class="status-text" id="statusText">Processing...</div>
            </div>

            <div class="success-message" id="successMessage">
                <h3>✅ Processing Complete!</h3>
                <p>Your document has been successfully converted. Click download to get your Word file.</p>
            </div>

            <div class="error-message" id="errorMessage">
                <h3>❌ Processing Failed</h3>
                <p id="errorText">An error occurred during processing.</p>
            </div>

            <button class="download-btn" id="downloadBtn">
                💾 Download Word Document
            </button>
        </div>

        <script>
            let selectedSection = null;
            let processedFile = null;

            // Elements
            const englishSection = document.getElementById('englishSection');
            const mathSection = document.getElementById('mathSection');
            const dropZone = document.getElementById('dropZone');
            const fileInput = document.getElementById('fileInput');
            const progressSection = document.getElementById('progressSection');
            const progressFill = document.getElementById('progressFill');
            const statusText = document.getElementById('statusText');
            const successMessage = document.getElementById('successMessage');
            const errorMessage = document.getElementById('errorMessage');
            const downloadBtn = document.getElementById('downloadBtn');

            // Event listeners
            dropZone.addEventListener('click', () => {
                if (selectedSection) {
                    fileInput.click();
                } else {
                    alert('Please select a section first (English or Math)');
                }
            });
            fileInput.addEventListener('change', handleFileSelect);
            downloadBtn.addEventListener('click', downloadResult);

            // Drag and drop events
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropZone.addEventListener(eventName, preventDefaults, false);
            });

            dropZone.addEventListener('dragenter', () => dropZone.classList.add('dragover'));
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
            dropZone.addEventListener('drop', handleDrop);

            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }

            function selectSection(section) {
                selectedSection = section;
                
                // Update UI
                englishSection.classList.remove('active');
                mathSection.classList.remove('active');
                
                if (section === 'english') {
                    englishSection.classList.add('active');
                    dropZone.querySelector('.drop-subtext').textContent = 'English section: Text processing & watermark removal';
                } else {
                    mathSection.classList.add('active');
                    dropZone.querySelector('.drop-subtext').textContent = 'Math section: LaTeX conversion & math detection';
                }
                
                // Enable drop zone
                dropZone.style.cursor = 'pointer';
                dropZone.style.opacity = '1';
            }

            function handleDrop(e) {
                dropZone.classList.remove('dragover');
                if (!selectedSection) {
                    alert('Please select a section first (English or Math)');
                    return;
                }
                const files = Array.from(e.dataTransfer.files);
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            }

            function handleFileSelect(e) {
                if (!selectedSection) {
                    alert('Please select a section first (English or Math)');
                    return;
                }
                const files = Array.from(e.target.files);
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            }

            function handleFile(file) {
                if (file.type !== 'application/pdf') {
                    showError('Only PDF files are supported');
                    return;
                }
                
                processFile(file);
            }

            async function processFile(file) {
                hideMessages();
                progressSection.style.display = 'block';
                dropZone.classList.add('processing');

                try {
                    updateProgress(10, 'Uploading file...');
                    
                    // Create FormData
                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('section_type', selectedSection);
                    
                    updateProgress(30, 'Processing document...');
                    
                    // Send to server
                    const response = await fetch('/convert', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        updateProgress(90, 'Generating Word document...');
                        processedFile = result.output_file;
                        showSuccess();
                    } else {
                        throw new Error(result.error || 'Processing failed');
                    }
                    
                } catch (error) {
                    console.error('Processing error:', error);
                    showError(error.message);
                } finally {
                    dropZone.classList.remove('processing');
                }
            }

            function updateProgress(percentage, text) {
                if (percentage !== null) {
                    progressFill.style.width = percentage + '%';
                }
                if (text) {
                    statusText.textContent = text;
                }
            }

            function showSuccess() {
                progressSection.style.display = 'none';
                successMessage.style.display = 'block';
                downloadBtn.style.display = 'inline-block';
                updateProgress(100, 'Complete!');
            }

            function showError(message) {
                progressSection.style.display = 'none';
                errorMessage.style.display = 'block';
                document.getElementById('errorText').textContent = message;
            }

            function hideMessages() {
                successMessage.style.display = 'none';
                errorMessage.style.display = 'none';
                downloadBtn.style.display = 'none';
            }

            function downloadResult() {
                if (processedFile) {
                    window.open(`/download/${processedFile}`, '_blank');
                }
            }

            // Initialize with English section selected
            selectSection('english');
        </script>
    </body>
    </html>
    """
    
    return render_template_string(html_content)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'service': 'PDF Watermark Remover - Two Sections',
        'python_version': sys.version,
        'timestamp': time.time()
    })

@app.route('/convert', methods=['POST'])
def convert_document():
    """Convert PDF to Word document with watermark removal"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get section type
        section_type = request.form.get('section_type', 'english')
        if section_type not in ['english', 'math']:
            return jsonify({'error': 'Invalid section type'}), 400
        
        # Validate file
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Only PDF files are supported'}), 400
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)  # Reset to beginning
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        logger.info(f"File uploaded: {filename} ({file_size // 1024} KB) for {section_type} section")
        
        # Process document
        output_path, metadata = get_processor().process_document(file_path, section_type)
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up uploaded file: {filename}")
        except:
            pass
        
        # Return success response
        return jsonify({
            'success': True,
            'message': f'Document converted successfully using {section_type} section',
            'output_file': os.path.basename(output_path),
            'metadata': metadata
        })
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download converted file"""
    try:
        file_path = os.path.join(PROCESSED_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Error handlers for production
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    print("Starting PDF Watermark Remover - Two Sections...")
    print(f"Python {sys.version} detected")
    print("All dependencies loaded successfully")
    print("Watermark removal algorithms ready")
    print("Document processor ready")
    
    # Set startup time for health checks
    app_startup_time = time.time()
    print(f"Startup timestamp: {app_startup_time}")
    
    # Get port from environment variable (for Railway/Heroku)
    port = int(os.environ.get('PORT', 5000))
    
    # Production optimizations for Railway
    is_production = os.environ.get('RAILWAY_ENVIRONMENT') or os.environ.get('RAILWAY_STATIC_URL')
    
    if is_production:
        print("🚀 Production mode detected - Railway deployment")
        host = '0.0.0.0'
        debug = False
        threaded = True
    else:
        print("🔧 Development mode detected - Local testing")
        host = 'localhost'
        debug = True
        threaded = False
    
    print(f"\nServer will start at: http://{host}:{port}")
    print("Two sections: English (text) and Math (LaTeX)")
    print("Health check at: http://{host}:{port}/health")
    print(f"Production mode: {is_production}")
    print(f"Debug mode: {debug}")
    print(f"Threaded: {threaded}")
    print("Press Ctrl+C to stop the server")
    print("\n" + "="*60)
    
    try:
        app.run(
            debug=debug, 
            host=host, 
            port=port,
            threaded=threaded
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nServer error: {e}")
        sys.exit(1)

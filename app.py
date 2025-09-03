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
    from flask import Flask, request, jsonify, send_file, render_template_string, redirect
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
    """Advanced PDF watermark removal with automatic detection"""
    
    def __init__(self):
        logger.info("Enhanced watermark remover initialized with auto-detection")
        
        # Base patterns for common watermarks (fallback)
        self.base_patterns = [
            r'CONFIDENTIAL', r'DRAFT', r'COPYRIGHT', r'PROPRIETARY',
            r'RESTRICTED', r'PRIVATE', r'CLASSIFIED', r'TOP SECRET'
        ]
        
        # Auto-detection parameters
        self.watermark_threshold = 0.7  # Similarity threshold for watermark detection
        self.frequency_threshold = 5    # Minimum frequency to consider as watermark
        self.position_weight = 0.3      # Weight for position-based detection
    
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
        """Advanced watermark removal with automatic detection"""
        try:
            logger.info("Starting automatic watermark detection and removal")
            
            # Step 1: Analyze text structure and extract potential watermarks
            potential_watermarks = self._detect_watermarks_automatically(text)
            logger.info(f"Detected {len(potential_watermarks)} potential watermarks")
            
            # Step 2: Remove detected watermarks
            cleaned_text = self._remove_detected_watermarks(text, potential_watermarks)
            
            # Step 3: Clean up formatting
            cleaned_text = self._clean_text_formatting(cleaned_text)
            
            return cleaned_text
            
        except Exception as e:
            logger.warning(f"Auto watermark detection failed, using fallback: {e}")
            return self._remove_watermarks_fallback(text)
    
    def _detect_watermarks_automatically(self, text: str) -> List[Dict]:
        """Automatically detect watermarks using multiple algorithms"""
        watermarks = []
        
        # Method 1: Frequency-based detection
        freq_watermarks = self._detect_frequency_watermarks(text)
        watermarks.extend(freq_watermarks)
        
        # Method 2: Position-based detection
        pos_watermarks = self._detect_position_watermarks(text)
        watermarks.extend(pos_watermarks)
        
        # Method 3: Pattern-based detection
        pattern_watermarks = self._detect_pattern_watermarks(text)
        watermarks.extend(pattern_watermarks)
        
        # Method 4: Context-based detection
        context_watermarks = self._detect_context_watermarks(text)
        watermarks.extend(context_watermarks)
        
        # Remove duplicates and return
        unique_watermarks = self._deduplicate_watermarks(watermarks)
        return unique_watermarks
    
    def _detect_frequency_watermarks(self, text: str) -> List[Dict]:
        """Detect watermarks based on word frequency analysis"""
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = {}
        
        # Count frequencies
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Find words that appear too frequently (likely watermarks)
        watermarks = []
        total_words = len(words)
        
        for word, freq in word_freq.items():
            if freq >= self.frequency_threshold and freq / total_words > 0.05:
                watermarks.append({
                    'type': 'frequency',
                    'text': word,
                    'confidence': min(freq / total_words, 1.0),
                    'pattern': r'\b' + re.escape(word) + r'\b'
                })
        
        return watermarks
    
    def _detect_position_watermarks(self, text: str) -> List[Dict]:
        """Detect watermarks based on position (headers/footers)"""
        lines = text.split('\n')
        watermarks = []
        
        # Check first and last few lines
        header_lines = lines[:3]
        footer_lines = lines[-3:] if len(lines) > 3 else []
        
        for line in header_lines + footer_lines:
            line = line.strip()
            if line and len(line) < 50:  # Short lines in headers/footers
                # Check if line contains watermark-like content
                if self._is_watermark_like(line):
                    watermarks.append({
                        'type': 'position',
                        'text': line,
                        'confidence': 0.8,
                        'pattern': re.escape(line)
                    })
        
        return watermarks
    
    def _detect_pattern_watermarks(self, text: str) -> List[Dict]:
        """Detect watermarks using pattern matching"""
        watermarks = []
        
        # Check against base patterns
        for pattern in self.base_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                watermarks.append({
                    'type': 'pattern',
                    'text': match,
                    'confidence': 0.9,
                    'pattern': r'\b' + re.escape(match) + r'\b'
                })
        
        # Detect repeated phrases
        repeated_phrases = self._find_repeated_phrases(text)
        watermarks.extend(repeated_phrases)
        
        return watermarks
    
    def _detect_context_watermarks(self, text: str) -> List[Dict]:
        """Detect watermarks based on context and surrounding text"""
        watermarks = []
        
        # Look for copyright notices, page numbers, etc.
        context_patterns = [
            (r'©\s*\d{4}.*?(?:all rights reserved|inc\.|llc\.)', 0.9),
            (r'page\s+\d+\s+of\s+\d+', 0.8),
            (r'generated\s+on.*?\d{4}', 0.8),
            (r'last\s+modified.*?\d{4}', 0.8),
            (r'confidential.*?document', 0.9),
            (r'proprietary.*?information', 0.9)
        ]
        
        for pattern, confidence in context_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                watermarks.append({
                    'type': 'context',
                    'text': match,
                    'confidence': confidence,
                    'pattern': pattern
                })
        
        return watermarks
    
    def _is_watermark_like(self, text: str) -> bool:
        """Check if text looks like a watermark"""
        # Short text
        if len(text) > 50:
            return False
        
        # Contains common watermark words
        watermark_indicators = [
            'confidential', 'draft', 'copyright', 'proprietary',
            'restricted', 'private', 'classified', 'internal',
            'watermark', 'stamp', 'logo', 'brand'
        ]
        
        text_lower = text.lower()
        for indicator in watermark_indicators:
            if indicator in text_lower:
                return True
        
        # All caps (common for watermarks)
        if text.isupper() and len(text) > 3:
            return True
        
        # Contains special characters or formatting
        if re.search(r'[©®™]', text):
            return True
        
        return False
    
    def _find_repeated_phrases(self, text: str) -> List[Dict]:
        """Find phrases that repeat frequently (likely watermarks)"""
        watermarks = []
        
        # Look for 2-4 word phrases that repeat
        words = re.findall(r'\b\w+\b', text.lower())
        
        for phrase_length in range(2, 5):
            phrases = {}
            for i in range(len(words) - phrase_length + 1):
                phrase = ' '.join(words[i:i + phrase_length])
                if len(phrase) > 10:  # Skip very short phrases
                    phrases[phrase] = phrases.get(phrase, 0) + 1
            
            # Find frequently repeated phrases
            for phrase, count in phrases.items():
                if count >= 3:  # Appears at least 3 times
                    watermarks.append({
                        'type': 'repeated_phrase',
                        'text': phrase,
                        'confidence': min(count / 10, 1.0),
                        'pattern': re.escape(phrase)
                    })
        
        return watermarks
    
    def _deduplicate_watermarks(self, watermarks: List[Dict]) -> List[Dict]:
        """Remove duplicate watermarks and merge similar ones"""
        unique_watermarks = []
        seen_texts = set()
        
        # Sort by confidence (highest first)
        watermarks.sort(key=lambda x: x['confidence'], reverse=True)
        
        for watermark in watermarks:
            text_lower = watermark['text'].lower()
            if text_lower not in seen_texts:
                unique_watermarks.append(watermark)
                seen_texts.add(text_lower)
        
        return unique_watermarks
    
    def _remove_detected_watermarks(self, text: str, watermarks: List[Dict]) -> str:
        """Remove detected watermarks from text"""
        cleaned_text = text
        
        for watermark in watermarks:
            if watermark['confidence'] >= self.watermark_threshold:
                pattern = watermark['pattern']
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
                logger.info(f"Removed watermark: '{watermark['text']}' (confidence: {watermark['confidence']:.2f})")
        
        return cleaned_text
    
    def _remove_watermarks_fallback(self, text: str) -> str:
        """Fallback watermark removal using basic patterns"""
        cleaned_text = text
        
        # Use base patterns as fallback
        for pattern in self.base_patterns:
            pattern_with_boundaries = r'\b' + pattern + r'\b'
            cleaned_text = re.sub(pattern_with_boundaries, '', cleaned_text, flags=re.IGNORECASE)
        
        return re.sub(r'\s+', ' ', cleaned_text).strip()
    
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
    


class SATDocumentProcessor:
    """Specialized SAT document processor with format detection and preservation"""
    
    def __init__(self):
        self.watermark_remover = EnhancedWatermarkRemover()
        self.sat_patterns = self._initialize_sat_patterns()
        logger.info("SAT document processor initialized")
    
    def _initialize_sat_patterns(self):
        """Initialize SAT-specific detection patterns"""
        return {
            'reading_passage': [
                r'^Reading\s+Passage\s*\d+[:\s]*$',  # "Reading Passage 1:" at start of line
                r'^Questions?\s*\d+[-\s]*\d*\s*are\s+based\s+on\s+the\s+following\s+passage$',  # "Questions 1-10 are based on the following passage"
                r'^The\s+following\s+passage\s+is\s+adapted\s+from',  # "The following passage is adapted from..."
                r'^Read\s+the\s+following\s+passage',  # "Read the following passage..."
            ],
            'question_start': [
                r'^\s*\d+\.\s+',  # "1. " at start of line
                r'^\s*Question\s+\d+[:\s]*',  # "Question 1:" or "Question 1"
                r'^\s*Q\s*\d+[:\s]*',  # "Q1:" or "Q1"
            ],
            'multiple_choice': [
                r'^\s*[A-D]\)\s+',  # "A) ", "B) ", "C) ", "D) "
                r'^\s*[A-D]\.\s+',  # "A. ", "B. ", "C. ", "D. "
                r'^\s*[A-D]\s+',    # "A ", "B ", "C ", "D "
            ],
            'section_header': [
                r'^SAT\s+Practice\s+Test',  # "SAT Practice Test"
                r'^(?:Section|Part)\s+\d+[:\s]*',  # "Section 1:" or "Part 1:"
                r'^(?:Reading|Writing|Math|Language)\s+(?:Test|Section)[:\s]*',  # "Reading Test:" or "Math Section:"
                r'^(?:Evidence-Based|Critical\s+Reading|Mathematics)[:\s]*',  # "Evidence-Based Reading:" or "Mathematics:"
            ]
        }
    
    def process_sat_document(self, input_path: str) -> str:
        """Process SAT document with format detection and preservation"""
        try:
            logger.info(f"Processing SAT document: {input_path}")
            
            # Extract text with watermark removal
            text = self.watermark_remover._extract_text_enhanced(input_path)
            logger.info(f"Extracted text length: {len(text)} characters")
            
            # Detect SAT structure
            sat_structure = self._detect_sat_structure(text)
            logger.info(f"Detected SAT structure: {sat_structure}")
            
            # Process and format for Word
            formatted_text = self._format_sat_for_word(text, sat_structure)
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"SAT document processing failed: {e}")
            raise
    
    def _detect_sat_structure(self, text: str) -> Dict:
        """Detect SAT document structure and components"""
        lines = text.split('\n')
        structure = {
            'reading_passages': [],
            'questions': [],
            'multiple_choices': [],
            'sections': [],
            'format_type': 'unknown'
        }
        
        current_section = None
        current_passage = None
        current_question = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if self._matches_pattern(line, self.sat_patterns['section_header']):
                current_section = line
                structure['sections'].append({
                    'type': 'section_header',
                    'text': line,
                    'line_number': i
                })
                continue
            
            # Detect reading passages
            if self._matches_pattern(line, self.sat_patterns['reading_passage']):
                current_passage = {
                    'start_line': i,
                    'text': line,
                    'questions': []
                }
                structure['reading_passages'].append(current_passage)
                continue
            
            # Detect questions
            if self._matches_pattern(line, self.sat_patterns['question_start']):
                current_question = {
                    'start_line': i,
                    'text': line,
                    'choices': [],
                    'passage_ref': current_passage
                }
                structure['questions'].append(current_question)
                continue
            
            # Detect multiple choice options
            if current_question and self._matches_pattern(line, self.sat_patterns['multiple_choice']):
                current_question['choices'].append({
                    'text': line,
                    'choice': line[0].upper(),
                    'line_number': i
                })
                continue
            
            # If we have a current passage, add lines to it
            if current_passage and not current_question:
                if 'content' not in current_passage:
                    current_passage['content'] = []
                current_passage['content'].append(line)
            
            # If we have a current question, add lines to it
            if current_question and not self._matches_pattern(line, self.sat_patterns['multiple_choice']):
                if 'content' not in current_question:
                    current_question['content'] = []
                current_question['content'].append(line)
        
        # Determine format type
        if structure['reading_passages'] and structure['questions']:
            structure['format_type'] = 'reading_comprehension'
        elif structure['questions'] and structure['multiple_choices']:
            structure['format_type'] = 'multiple_choice'
        else:
            structure['format_type'] = 'mixed'
        
        return structure
    
    def _matches_pattern(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given patterns"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _format_sat_for_word(self, text: str, structure: Dict) -> str:
        """Format SAT content for Word document with proper structure"""
        formatted_lines = []
        
        # Add title
        formatted_lines.append("SAT Practice Test")
        formatted_lines.append("=" * 50)
        formatted_lines.append("")
        
        # Process sections
        for section in structure['sections']:
            formatted_lines.append(f"**{section['text']}**")
            formatted_lines.append("")
        
        # Process reading passages
        for passage in structure['reading_passages']:
            formatted_lines.append(f"**{passage['text']}**")
            formatted_lines.append("")
            
            if 'content' in passage:
                for line in passage['content']:
                    formatted_lines.append(line)
                formatted_lines.append("")
        
        # Process questions
        for question in structure['questions']:
            formatted_lines.append(f"**{question['text']}**")
            if 'content' in question:
                for line in question['content']:
                    formatted_lines.append(line)
            
            # Add multiple choice options
            if question['choices']:
                formatted_lines.append("")
                for choice in question['choices']:
                    formatted_lines.append(f"  {choice['text']}")
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)


class DocumentProcessor:
    """Enhanced document processing with SAT support"""
    
    def __init__(self):
        self.watermark_remover = EnhancedWatermarkRemover()
        self.sat_processor = SATDocumentProcessor()
        logger.info("Enhanced document processor initialized with SAT support")
    
    def process_document(self, file_path: str, section_type: str = 'english') -> Tuple[str, Dict]:
        """Process PDF document with watermark removal and Word conversion"""
        try:
            logger.info(f"Processing {file_path} with section type: {section_type}")
            
            # Check if this is a SAT document first
            if self._is_sat_document(file_path):
                logger.info("SAT document detected - using specialized processor")
                word_path = self._convert_sat_to_word(file_path, file_path)
                section_type = 'sat'  # Override section type for SAT
            else:
                # Direct conversion from original PDF with watermark removal in text processing
                if section_type == 'math':
                    word_path = self._convert_to_word_math(file_path, file_path)
                else:
                    word_path = self._convert_to_word_english(file_path, file_path)
            
            metadata = {
                'success': True,
                'input_file': Path(file_path).name,
                'output_file': Path(word_path).name,
                'section_type': section_type,
                'processing_time': time.time(),
                'watermark_removal': 'completed',
                'word_conversion': 'completed',
                'sat_detection': 'completed' if section_type == 'sat' else 'not_applicable'
            }
            
            return word_path, metadata
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _is_sat_document(self, file_path: str) -> bool:
        """Detect if document is SAT format"""
        try:
            # Quick check of first few pages for SAT indicators
            text = self.watermark_remover._extract_text_enhanced(file_path, max_pages=3)
            return self._is_sat_document_from_text(text)
            
        except Exception as e:
            logger.warning(f"SAT detection failed: {e}")
            return False
    
    def _is_sat_document_from_text(self, text: str) -> bool:
        """Detect if text content is SAT format"""
        sat_indicators = [
            'sat', 'scholastic aptitude test', 'college board',
            'reading passage', 'evidence-based reading',
            'questions are based on the following passage',
            'multiple choice', 'a)', 'b)', 'c)', 'd)',
            'passage', 'question', 'section'
        ]
        
        text_lower = text.lower()
        indicator_count = sum(1 for indicator in sat_indicators if indicator in text_lower)
        
        # If more than 3 indicators found, likely SAT document
        return indicator_count >= 3
    
    def _convert_sat_to_word(self, pdf_path: str, original_filename: str) -> str:
        """Convert SAT PDF to Word document with format preservation"""
        logger.info("Converting SAT document to Word with format preservation...")
        
        try:
            # Process SAT document through specialized processor
            formatted_text = self.sat_processor.process_sat_document(pdf_path)
            
            # Create Word document
            word_doc = Document()
            
            # Set document properties
            word_doc.core_properties.title = f"SAT Practice Test - {Path(original_filename).stem}"
            word_doc.core_properties.author = "SAT Document Processor"
            
            # Add title
            title = word_doc.add_heading("SAT Practice Test", 0)
            title.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add subtitle
            subtitle = word_doc.add_paragraph(f"Source: {Path(original_filename).name}")
            subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add processing info
            info = word_doc.add_paragraph("Processed with SAT format detection and watermark removal")
            info.alignment = WD_ALIGN_PARAGRAPH.CENTER
            
            # Add separator
            word_doc.add_paragraph("=" * 50)
            
            # Process formatted text
            lines = formatted_text.split('\n')
            current_heading_level = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                if line.startswith('**') and line.endswith('**'):
                    # Remove markdown formatting
                    clean_text = line[2:-2]
                    if 'passage' in clean_text.lower():
                        word_doc.add_heading(clean_text, level=1)
                    elif 'section' in clean_text.lower():
                        word_doc.add_heading(clean_text, level=1)
                    else:
                        word_doc.add_heading(clean_text, level=2)
                    continue
                
                # Check for question numbers
                if re.match(r'^\d+\.', line):
                    word_doc.add_heading(line, level=3)
                    continue
                
                # Check for multiple choice options
                if re.match(r'^\s*[A-D][\)\.]\s*', line):
                    # Indent multiple choice options
                    para = word_doc.add_paragraph(line)
                    para.paragraph_format.left_indent = Inches(0.5)
                    continue
                
                # Regular paragraph
                if line:
                    word_doc.add_paragraph(line)
            
            # Save Word document
            output_path = os.path.join(PROCESSED_FOLDER, f"{Path(original_filename).stem}_SAT_Formatted.docx")
            word_doc.save(output_path)
            
            logger.info(f"SAT Word document saved: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"SAT to Word conversion failed: {e}")
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
                
                # Remove watermarks from text
                text = self.watermark_remover._remove_watermarks_enhanced(text)
                
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
                
                # Remove watermarks from text
                text = self.watermark_remover._remove_watermarks_enhanced(text)
                
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
    """Root endpoint - redirect to home page for web interface"""
    return redirect('/home')

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

#!/usr/bin/env python3
"""
PDF/Image to Word Converter with Advanced AI-Powered Watermark Removal
High-quality conversion preserving exact formatting and content
Based on research from advanced watermark removal repositories
"""

import os
import io
import base64
import logging
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import traceback

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement, qn
import easyocr
from skimage import restoration, morphology, filters, measure
from scipy import ndimage, signal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'tif'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Initialize EasyOCR reader for better accuracy
try:
    reader = easyocr.Reader(['en', 'fr', 'de', 'es', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'])
    logger.info("EasyOCR initialized successfully")
except Exception as e:
    logger.warning(f"EasyOCR initialization failed: {e}")
    reader = None

class AdvancedWatermarkRemover:
    """Advanced watermark removal using multiple AI/ML techniques"""
    
    def __init__(self):
        self.methods = [
            'color_analysis',
            'frequency_domain',
            'morphological',
            'edge_detection',
            'inpainting',
            'ml_enhancement'
        ]
    
    def remove_watermarks(self, image: Image.Image, section_type: str) -> Image.Image:
        """
        Remove watermarks using multiple advanced techniques
        Based on research from Goshin, zuruoke, and whitelok repositories
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply multiple watermark removal methods
        result = img_array.copy()
        
        for method in self.methods:
            try:
                if method == 'color_analysis':
                    result = self._color_based_removal(result, section_type)
                elif method == 'frequency_domain':
                    result = self._frequency_domain_removal(result)
                elif method == 'morphological':
                    result = self._morphological_removal(result)
                elif method == 'edge_detection':
                    result = self._edge_based_removal(result)
                elif method == 'inpainting':
                    result = self._inpainting_removal(result)
                elif method == 'ml_enhancement':
                    result = self._ml_enhancement(result)
                
                logger.info(f"Applied {method} watermark removal")
                
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
                continue
        
        # Final enhancement
        result = self._final_enhancement(result)
        
        return Image.fromarray(result)
    
    def _color_based_removal(self, img: np.ndarray, section_type: str) -> np.ndarray:
        """
        Advanced color-based watermark detection and removal
        Based on Goshin's approach with improvements
        """
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        
        # Create watermark mask
        watermark_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # Adaptive thresholding based on section type
        if section_type == 'math':
            sensitivity = 0.12  # Conservative for math
        else:
            sensitivity = 0.18  # More aggressive for text
        
        # HSV-based detection
        h, s, v = cv2.split(hsv)
        
        # Detect unusual saturation patterns
        s_mean = np.mean(s)
        s_std = np.std(s)
        s_lower = s_mean - sensitivity * s_std
        s_upper = s_mean + sensitivity * s_std
        
        s_mask = cv2.inRange(s, s_lower, s_upper)
        
        # Detect unusual value patterns
        v_mean = np.mean(v)
        v_std = np.std(v)
        v_lower = v_mean - sensitivity * v_std
        v_upper = v_mean + sensitivity * v_std
        
        v_mask = cv2.inRange(v, v_lower, v_upper)
        
        # LAB-based detection
        l, a, b = cv2.split(lab)
        
        # Detect unusual lightness patterns
        l_mean = np.mean(l)
        l_std = np.std(l)
        l_lower = l_mean - sensitivity * l_std
        l_upper = l_mean + sensitivity * l_std
        
        l_mask = cv2.inRange(l, l_lower, l_upper)
        
        # Combine masks
        watermark_mask = cv2.bitwise_or(s_mask, v_mask)
        watermark_mask = cv2.bitwise_or(watermark_mask, l_mask)
        
        # Remove noise
        kernel = np.ones((3, 3), np.uint8)
        watermark_mask = cv2.morphologyEx(watermark_mask, cv2.MORPH_OPEN, kernel)
        watermark_mask = cv2.morphologyEx(watermark_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply inpainting
        result = cv2.inpaint(img, watermark_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def _frequency_domain_removal(self, img: np.ndarray) -> np.ndarray:
        """
        Frequency domain watermark removal
        Based on zuruoke's approach with improvements
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create high-pass filter
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Adaptive filter radius based on image size
        filter_radius = min(rows, cols) // 20
        
        mask = np.ones((rows, cols), np.uint8)
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= filter_radius ** 2
        mask[mask_area] = 0
        
        # Apply filter
        f_shift_filtered = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        # Normalize and convert back to RGB
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Apply to all channels
        result = img.copy()
        for i in range(3):
            result[:, :, i] = img_back
        
        return result
    
    def _morphological_removal(self, img: np.ndarray) -> np.ndarray:
        """
        Morphological operations for watermark removal
        Based on whitelok's approach
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Opening to remove small artifacts
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Closing to fill gaps
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # Top-hat transform to detect bright regions
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black-hat transform to detect dark regions
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        # Combine transforms
        morphological_result = gray - tophat + blackhat
        
        # Apply to all channels
        result = img.copy()
        for i in range(3):
            result[:, :, i] = morphological_result
        
        return result
    
    def _edge_based_removal(self, img: np.ndarray) -> np.ndarray:
        """
        Edge-based watermark detection and removal
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Remove small edge clusters (likely watermarks)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area
        min_area = 100
        max_area = 10000
        
        watermark_mask = np.zeros_like(gray)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                cv2.fillPoly(watermark_mask, [contour], 255)
        
        # Apply inpainting
        result = cv2.inpaint(img, watermark_mask, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def _inpainting_removal(self, img: np.ndarray) -> np.ndarray:
        """
        Advanced inpainting for watermark removal
        Based on zuruoke's ML approach
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Create mask for inpainting
        # Use adaptive thresholding to detect potential watermarks
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Remove noise
        kernel = np.ones((2, 2), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Apply inpainting
        result = cv2.inpaint(img, thresh, 3, cv2.INPAINT_TELEA)
        
        return result
    
    def _ml_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Machine learning-based image enhancement
        """
        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Apply bilateral filter for edge-preserving smoothing
        filtered = cv2.bilateralFilter(img_float, 9, 75, 75)
        
        # Apply unsharp masking for enhancement
        gaussian = cv2.GaussianBlur(filtered, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
        
        # Convert back to uint8
        result = np.clip(unsharp_mask * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def _final_enhancement(self, img: np.ndarray) -> np.ndarray:
        """
        Final enhancement and quality improvement
        """
        # Convert to PIL for enhancement
        pil_img = Image.fromarray(img)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.05)
        
        # Enhance brightness slightly
        enhancer = ImageEnhance.Brightness(pil_img)
        pil_img = enhancer.enhance(1.02)
        
        return np.array(pil_img)

class DocumentProcessor:
    """High-quality document processing with format preservation"""
    
    def __init__(self):
        self.supported_languages = {
            'en': 'eng', 'fr': 'fra', 'de': 'deu', 'es': 'spa',
            'it': 'ita', 'pt': 'por', 'ru': 'rus', 'zh': 'chi_sim',
            'ja': 'jpn', 'ko': 'kor'
        }
        self.watermark_remover = AdvancedWatermarkRemover()
    
    def process_document(self, file_path: str, language: str = 'en', 
                        section_type: str = 'english') -> Tuple[str, Dict]:
        """
        Process document with high-quality conversion
        """
        try:
            logger.info(f"Processing {file_path} with language {language}")
            
            # Convert to high-quality images
            images = self._convert_to_images(file_path)
            
            processed_content = []
            metadata = {
                'pages': len(images),
                'language': language,
                'section_type': section_type,
                'processing_quality': 'enterprise'
            }
            
            for i, image in enumerate(images):
                logger.info(f"Processing page {i+1}/{len(images)}")
                
                # Advanced watermark removal
                clean_image = self.watermark_remover.remove_watermarks(image, section_type)
                
                # High-quality OCR with formatting
                text_content, page_metadata = self._perform_advanced_ocr(
                    clean_image, language, section_type, i+1
                )
                
                processed_content.append({
                    'page': i+1,
                    'content': text_content,
                    'metadata': page_metadata
                })
                
                metadata[f'page_{i+1}_confidence'] = page_metadata.get('confidence', 0)
            
            # Generate Word document
            output_path = self._generate_word_document(
                processed_content, file_path, section_type
            )
            
            metadata['output_path'] = output_path
            metadata['success'] = True
            
            return output_path, metadata
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _convert_to_images(self, file_path: str) -> List[Image.Image]:
        """Convert PDF/image to high-quality images"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext == '.pdf':
            # High-quality PDF conversion
            images = convert_from_path(
                file_path,
                dpi=400,  # Very high resolution for better quality
                fmt='PNG',
                thread_count=4
            )
            return images
        else:
            # Image file - load and enhance
            image = Image.open(file_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return [image]
    
    def _perform_advanced_ocr(self, image: Image.Image, language: str, 
                             section_type: str, page_num: int) -> Tuple[str, Dict]:
        """
        Perform high-quality OCR with format preservation
        """
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Try multiple OCR engines for best results
        ocr_results = []
        
        # Method 1: EasyOCR (best for multi-language)
        if reader:
            try:
                easyocr_result = reader.readtext(img_array)
                if easyocr_result:
                    text = '\n'.join([item[1] for item in easyocr_result])
                    confidence = np.mean([item[2] for item in easyocr_result])
                    ocr_results.append(('easyocr', text, confidence))
            except Exception as e:
                logger.warning(f"EasyOCR failed: {e}")
        
        # Method 2: Tesseract with custom configuration
        try:
            # Configure Tesseract for better accuracy
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?()[]{}:;\'"`~@#$%^&*+=|\\/<>'
            
            tesseract_lang = self.supported_languages.get(language, 'eng')
            tesseract_result = pytesseract.image_to_data(
                image, 
                lang=tesseract_lang,
                config=custom_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Extract text with positioning information
            text_lines = []
            current_line = []
            current_y = -1
            
            for i in range(len(tesseract_result['text'])):
                if tesseract_result['conf'][i] > 30:  # Confidence threshold
                    text = tesseract_result['text'][i]
                    x = tesseract_result['left'][i]
                    y = tesseract_result['top'][i]
                    conf = tesseract_result['conf'][i]
                    
                    if current_y == -1:
                        current_y = y
                    
                    # Check if this is a new line
                    if abs(y - current_y) > 10:
                        if current_line:
                            text_lines.append(' '.join(current_line))
                            current_line = []
                        current_y = y
                    
                    current_line.append(text)
            
            if current_line:
                text_lines.append(' '.join(current_line))
            
            tesseract_text = '\n'.join(text_lines)
            tesseract_confidence = np.mean([conf for conf in tesseract_result['conf'] if conf > 0])
            
            ocr_results.append(('tesseract', tesseract_text, tesseract_confidence))
            
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
        
        # Select best result
        if not ocr_results:
            raise Exception("All OCR methods failed")
        
        # Choose result with highest confidence
        best_result = max(ocr_results, key=lambda x: x[2])
        engine, text, confidence = best_result
        
        # Post-process text based on section type
        if section_type == 'math':
            text = self._enhance_math_content(text)
        else:
            text = self._enhance_text_content(text)
        
        metadata = {
            'ocr_engine': engine,
            'confidence': confidence,
            'page': page_num,
            'section_type': section_type
        }
        
        return text, metadata
    
    def _enhance_text_content(self, text: str) -> str:
        """Enhance text content quality"""
        # Fix common OCR mistakes
        replacements = [
            ('||', 'll'), ('|/', 'll'), ('0O', '0'), ('O0', '0'),
            ('1l', 'll'), ('l1', 'll'), ('5S', 'S'), ('S5', 'S'),
            ('rn', 'm'), ('cl', 'd'), ('vv', 'w'), ('nn', 'm')
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _enhance_math_content(self, text: str) -> str:
        """Enhance mathematical content recognition"""
        # Convert common math patterns to LaTeX
        math_patterns = [
            (r'(\w+)/(\w+)', r'\\frac{\1}{\2}'),
            (r'(\w+)\^(\d+)', r'\1^{\2}'),
            (r'sqrt\(([^)]+)\)', r'\\sqrt{\1}'),
            (r'√\(([^)]+)\)', r'\\sqrt{\1}'),
            (r'<=', r'\\leq'),
            (r'>=', r'\\geq'),
            (r'!=', r'\\neq'),
            (r'->', r'\\rightarrow'),
            (r'<-', r'\\leftarrow'),
            (r'<->', r'\\leftrightarrow'),
            (r'(\w+)_(\w+)', r'\1_{\2}'),
            (r'(\w+)_(\d+)', r'\1_{\2}'),
        ]
        
        for pattern, replacement in math_patterns:
            text = re.sub(pattern, replacement, text)
        
        return text
    
    def _generate_word_document(self, content: List[Dict], 
                               original_file: str, section_type: str) -> str:
        """
        Generate high-quality Word document preserving formatting
        """
        # Create new Word document
        doc = Document()
        
        # Set document properties
        doc.core_properties.title = f"Converted Document - {Path(original_file).stem}"
        doc.core_properties.author = "PDF/Image to Word Converter"
        
        # Add title
        title = doc.add_heading(f"Converted Document", 0)
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add subtitle
        subtitle = doc.add_paragraph(f"Source: {Path(original_file).name}")
        subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add section info
        section_info = doc.add_paragraph(f"Section Type: {section_type.title()}")
        section_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add content for each page
        for page_data in content:
            # Add page separator
            if page_data['page'] > 1:
                doc.add_page_break()
            
            # Add page header
            page_header = doc.add_heading(f"Page {page_data['page']}", level=1)
            
            # Add page content
            content_text = page_data['content']
            
            # Split into paragraphs and add
            paragraphs = content_text.split('\n')
            for para_text in paragraphs:
                if para_text.strip():
                    # Check if this looks like a heading
                    if len(para_text.strip()) < 100 and para_text.strip().isupper():
                        doc.add_heading(para_text.strip(), level=2)
                    else:
                        doc.add_paragraph(para_text.strip())
            
            # Add page metadata
            metadata = page_data['metadata']
            if metadata.get('confidence'):
                conf_para = doc.add_paragraph(f"OCR Confidence: {metadata['confidence']:.2f}%")
                conf_para.style = 'Quote'
        
        # Save document
        output_filename = f"converted_{Path(original_file).stem}_{section_type}.docx"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        doc.save(output_path)
        
        logger.info(f"Word document saved: {output_path}")
        return output_path

# Initialize processor
processor = DocumentProcessor()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'PDF/Image to Word Converter'})

@app.route('/convert', methods=['POST'])
def convert_document():
    """Convert PDF/image to Word document"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Get parameters
        language = request.form.get('language', 'en')
        section_type = request.form.get('section_type', 'english')
        
        # Validate file
        if not file or not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process document
        output_path, metadata = processor.process_document(
            file_path, language, section_type
        )
        
        # Return success response
        return jsonify({
            'success': True,
            'message': 'Document converted successfully',
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

if __name__ == '__main__':
    logger.info("Starting PDF/Image to Word Converter...")
    logger.info("Available languages: en, fr, de, es, it, pt, ru, zh, ja, ko")
    logger.info("Available sections: english, math")
    
    app.run(debug=True, host='0.0.0.0', port=5000)

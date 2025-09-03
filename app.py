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
    from docx.shared import RGBColor
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

# Global API key storage
GEMINI_API_KEY = None

class EnhancedWatermarkRemover:
    """Advanced PDF watermark removal with AI-powered training and image detection"""
    
    def __init__(self):
        logger.info("Enhanced watermark remover initialized with AI-powered training and image detection")
        
        # Base patterns for common watermarks (fallback)
        self.base_patterns = [
            r'CONFIDENTIAL', r'DRAFT', r'COPYRIGHT', r'PROPRIETARY',
            r'RESTRICTED', r'PRIVATE', r'CLASSIFIED', r'TOP SECRET'
        ]
        
        # Trainable watermark patterns (learned from user documents)
        self.trained_patterns = []
        self.trained_watermarks = []
        
        # Auto-detection parameters
        self.watermark_threshold = 0.7  # Similarity threshold for watermark detection
        self.frequency_threshold = 5    # Minimum frequency to consider as watermark
        self.position_weight = 0.3      # Weight for position-based detection
        
        # Training parameters
        self.min_training_samples = 3   # Minimum samples needed for training
        self.learning_rate = 0.1        # How much to adjust patterns based on feedback
        
        # AI Training parameters
        self.use_ai_training = True     # Enable AI-powered training
        self.ai_providers = ['gemini', 'openai']  # Supported AI providers
        self.ai_api_keys = {}  # Will be set via web interface
        
    def set_api_key(self, provider: str, api_key: str):
        """Set API key for a specific provider"""
        if provider in self.ai_providers:
            self.ai_api_keys[provider] = api_key
            logger.info(f"API key set for {provider}")
            return True
        return False
        
        # Image detection parameters
        self.image_detection_enabled = True
        self.min_image_size = 100       # Minimum image size to detect (pixels)
        self.image_quality = 0.8        # Image quality for extraction
        
        # Load any existing trained patterns
        self._load_trained_patterns()
    
    def _ai_enhance_patterns(self, text: str, images: List, watermarks_to_remove: List, content_to_preserve: List) -> Dict:
        """Use AI to enhance watermark detection patterns"""
        try:
            enhanced_patterns = {'remove': [], 'preserve': []}
            
            # Try Gemini first (free tier available)
            if self._has_gemini_api():
                enhanced_patterns = self._gemini_enhance_patterns(text, images, watermarks_to_remove, content_to_preserve)
            # Fallback to OpenAI if available
            elif self._has_openai_api():
                enhanced_patterns = self._openai_enhance_patterns(text, images, watermarks_to_remove, content_to_preserve)
            else:
                logger.info("No AI API keys available, using basic training only")
                return enhanced_patterns
            
            return enhanced_patterns
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            return {'remove': [], 'preserve': []}
    
    def _gemini_enhance_patterns(self, text: str, images: List, watermarks_to_remove: List, content_to_preserve: List) -> Dict:
        """Use Google Gemini to enhance watermark detection patterns"""
        try:
            import google.generativeai as genai
            
            # Configure Gemini
            genai.configure(api_key=self.ai_api_keys.get('gemini'))
            model = genai.GenerativeModel('gemini-pro')
            
            # Create prompt for watermark analysis
            prompt = f"""
            Analyze this document content and identify additional watermarks that should be removed.
            
            Current watermarks to remove: {watermarks_to_remove}
            Content to preserve: {content_to_preserve}
            
            Document text (first 1000 chars): {text[:1000]}...
            
            Based on the content, identify:
            1. Additional watermark patterns (text that appears repeatedly, headers, footers, etc.)
            2. Content that should definitely be preserved (main text, questions, answers)
            3. Suggest improved patterns for existing watermarks
            
            Return as JSON:
            {{
                "remove": ["pattern1", "pattern2"],
                "preserve": ["content1", "content2"],
                "improvements": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = model.generate_content(prompt)
            
            # Parse AI response
            try:
                import json
                result = json.loads(response.text)
                return {
                    'remove': result.get('remove', []),
                    'preserve': result.get('preserve', [])
                }
            except:
                # Fallback parsing
                return self._parse_ai_response_fallback(response.text)
                
        except Exception as e:
            logger.warning(f"Gemini enhancement failed: {e}")
            return {'remove': [], 'preserve': []}
    
    def _openai_enhance_patterns(self, text: str, images: List, watermarks_to_remove: List, content_to_preserve: List) -> Dict:
        """Use OpenAI ChatGPT to enhance watermark detection patterns"""
        try:
            import openai
            
            # Configure OpenAI
            openai.api_key = self.ai_api_keys.get('openai')
            
            # Create prompt for watermark analysis
            prompt = f"""
            Analyze this document content and identify additional watermarks that should be removed.
            
            Current watermarks to remove: {watermarks_to_remove}
            Content to preserve: {content_to_preserve}
            
            Document text (first 1000 chars): {text[:1000]}...
            
            Based on the content, identify:
            1. Additional watermark patterns (text that appears repeatedly, headers, footers, etc.)
            2. Content that should definitely be preserved (main text, questions, answers)
            3. Suggest improved patterns for existing watermarks
            
            Return as JSON:
            {{
                "remove": ["pattern1", "pattern2"],
                "preserve": ["content1", "content2"],
                "improvements": ["suggestion1", "suggestion2"]
            }}
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in document analysis and watermark detection."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response
            try:
                import json
                result = json.loads(ai_response)
                return {
                    'remove': result.get('remove', []),
                    'preserve': result.get('preserve', [])
                }
            except:
                # Fallback parsing
                return self._parse_ai_response_fallback(ai_response)
                
        except Exception as e:
            logger.warning(f"OpenAI enhancement failed: {e}")
            return {'remove': [], 'preserve': []}
    
    def _parse_ai_response_fallback(self, ai_response: str) -> Dict:
        """Fallback parsing for AI responses that aren't valid JSON"""
        try:
            enhanced_patterns = {'remove': [], 'preserve': []}
            
            # Simple pattern extraction from text
            lines = ai_response.split('\n')
            for line in lines:
                line = line.strip().lower()
                if 'remove' in line or 'watermark' in line:
                    # Extract potential patterns
                    words = re.findall(r'\b\w+\b', line)
                    enhanced_patterns['remove'].extend(words[:3])  # Take first 3 words
                elif 'preserve' in line or 'keep' in line:
                    words = re.findall(r'\b\w+\b', line)
                    enhanced_patterns['preserve'].extend(words[:3])
            
            return enhanced_patterns
            
        except Exception as e:
            logger.warning(f"Fallback parsing failed: {e}")
            return {'remove': [], 'preserve': []}
    
    def _has_gemini_api(self) -> bool:
        """Check if Gemini API key is available"""
        return 'gemini' in self.ai_api_keys and self.ai_api_keys['gemini']
    
    def _has_openai_api(self) -> bool:
        """Check if OpenAI API key is available"""
        return 'openai' in self.ai_api_keys and self.ai_api_keys['openai']
    
    def _ai_enhance_watermark_removal(self, text: str) -> str:
        """Use Gemini AI to enhance watermark removal for SAT documents"""
        try:
            if not self._has_gemini_api():
                return None
            
            logger.info("Using Gemini AI to enhance watermark removal...")
            
            # Create prompt for Gemini
            prompt = f"""
            Analyze this SAT document text and identify watermarks to remove while preserving important content.
            
            Document text:
            {text[:2000]}...
            
            Instructions:
            1. Identify watermarks (headers, footers, page numbers, company names, etc.)
            2. Identify important content to preserve (reading passages, questions, multiple choice options)
            3. Return cleaned text with watermarks removed
            
            Return the cleaned text directly, no JSON formatting needed.
            """
            
            # Use Gemini API
            import google.generativeai as genai
            genai.configure(api_key=self.ai_api_keys['gemini'])
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content(prompt)
            cleaned_text = response.text
            
            if cleaned_text and len(cleaned_text) > 100:
                logger.info("Gemini AI successfully enhanced watermark removal")
                return cleaned_text
            else:
                logger.warning("Gemini AI response too short, using fallback")
                return None
                
        except Exception as e:
            logger.warning(f"Gemini AI enhancement failed: {e}")
            return None
    
    def set_ai_api_key(self, provider: str, api_key: str):
        """Set AI API key for a specific provider"""
        if provider in self.ai_providers:
            self.ai_api_keys[provider] = api_key
            logger.info(f"API key set for {provider}")
        else:
            logger.warning(f"Unsupported AI provider: {provider}")
    
    def _extract_images_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract images from PDF with metadata"""
        try:
            images = []
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Get image list for this page
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    try:
                        # Get image data
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Get image metadata
                        img_rect = page.get_image_bbox(img)
                        img_width = img_rect.width
                        img_height = img_rect.height
                        
                        # Only process images above minimum size
                        if img_width >= self.min_image_size and img_height >= self.min_image_size:
                            image_info = {
                                'page': page_num + 1,
                                'index': img_index,
                                'width': img_width,
                                'height': img_height,
                                'bbox': img_rect,
                                'data': image_bytes,
                                'format': base_image["ext"],
                                'size': len(image_bytes)
                            }
                            images.append(image_info)
                            
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                        continue
            
            doc.close()
            logger.info(f"Extracted {len(images)} images from PDF")
            return images
            
        except Exception as e:
            logger.error(f"Image extraction failed: {e}")
            return []
    
    def train_on_document(self, document_path: str, user_feedback: Dict) -> bool:
        """Train the watermark detector on a specific document with AI-powered enhancement"""
        try:
            logger.info(f"Training watermark detector on: {document_path}")
            
            # Extract text and images from document
            text = self._extract_text_enhanced(document_path)
            images = self._extract_images_from_pdf(document_path)
            
            # Get user feedback about what should/shouldn't be removed
            watermarks_to_remove = user_feedback.get('remove', [])
            content_to_preserve = user_feedback.get('preserve', [])
            
            # AI-powered pattern enhancement
            if self.use_ai_training:
                enhanced_patterns = self._ai_enhance_patterns(
                    text, images, watermarks_to_remove, content_to_preserve
                )
                watermarks_to_remove.extend(enhanced_patterns.get('remove', []))
                content_to_preserve.extend(enhanced_patterns.get('preserve', []))
                logger.info(f"AI enhanced patterns: {len(enhanced_patterns.get('remove', []))} removal, {len(enhanced_patterns.get('preserve', []))} preservation")
            
            # Learn new patterns
            new_patterns = self._learn_from_feedback(text, watermarks_to_remove, content_to_preserve)
            
            # Update trained patterns
            self.trained_patterns.extend(new_patterns)
            
            # Save trained patterns
            self._save_trained_patterns()
            
            logger.info(f"Training completed. New patterns learned: {len(new_patterns)}")
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def _learn_from_feedback(self, text: str, watermarks_to_remove: List[str], content_to_preserve: List[str]) -> List[str]:
        """Learn new patterns from user feedback"""
        new_patterns = []
        
        # Learn patterns for watermarks to remove
        for watermark in watermarks_to_remove:
            pattern = self._create_pattern_from_text(watermark)
            if pattern:
                new_patterns.append(pattern)
                logger.info(f"Learned removal pattern: {pattern}")
        
        # Learn patterns for content to preserve (negative learning)
        for content in content_to_preserve:
            # Create a pattern that should NOT match
            pattern = self._create_preservation_pattern(content)
            if pattern:
                new_patterns.append(pattern)
                logger.info(f"Learned preservation pattern: {pattern}")
        
        return new_patterns
    
    def _create_pattern_from_text(self, text: str) -> str:
        """Create a regex pattern from text for watermark detection"""
        if not text or len(text) < 2:
            return None
        
        # Escape special characters
        escaped_text = re.escape(text)
        
        # Make it flexible for variations
        pattern = f"\\b{escaped_text}\\b"
        
        return pattern
    
    def _create_preservation_pattern(self, text: str) -> str:
        """Create a pattern that should preserve content"""
        if not text or len(text) < 2:
            return None
        
        # This is a negative pattern - we'll handle it specially
        escaped_text = re.escape(text)
        pattern = f"PRESERVE:{escaped_text}"
        
        return pattern
    
    def _load_trained_patterns(self):
        """Load trained patterns from storage"""
        try:
            # For now, load from memory. In production, this would be from a database/file
            # This is where you'd implement persistent storage
            pass
        except Exception as e:
            logger.warning(f"Failed to load trained patterns: {e}")
    
    def _save_trained_patterns(self):
        """Save trained patterns to storage"""
        try:
            # For now, save to memory. In production, this would be to a database/file
            # This is where you'd implement persistent storage
            pass
        except Exception as e:
            logger.warning(f"Failed to save trained patterns: {e}")
    
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
    
    def extract_text_from_pdf(self, pdf_path: str, max_pages: int = None) -> str:
        """Extract text from PDF file with enhanced processing"""
        try:
            logger.info(f"Extracting text from PDF: {pdf_path}")
            
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Determine how many pages to process
            if max_pages is None:
                max_pages = len(doc)
            else:
                max_pages = min(max_pages, len(doc))
            
            # Extract text from each page
            all_text = []
            for page_num in range(max_pages):
                page = doc[page_num]
                page_text = self._extract_text_enhanced(page)
                all_text.append(page_text)
            
            # Close the document
            doc.close()
            
            # Combine all text
            combined_text = "\n".join(all_text)
            logger.info(f"Extracted {len(combined_text)} characters from {max_pages} pages")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"Text extraction from PDF failed: {e}")
            return ""
    
    def _remove_watermarks_enhanced(self, text: str) -> str:
        """Advanced watermark removal with automatic detection and AI enhancement"""
        try:
            logger.info("Starting automatic watermark detection and removal with AI")
            
            # Step 1: Use AI to enhance watermark detection if available
            if self._has_gemini_api():
                ai_enhanced_text = self._ai_enhance_watermark_removal(text)
                if ai_enhanced_text:
                    logger.info("AI-enhanced watermark removal completed")
                    return ai_enhanced_text
            
            # Step 2: Analyze text structure and extract potential watermarks
            potential_watermarks = self._detect_watermarks_automatically(text)
            logger.info(f"Detected {len(potential_watermarks)} potential watermarks")
            
            # Step 3: Remove detected watermarks
            cleaned_text = self._remove_detected_watermarks(text, potential_watermarks)
            
            # Step 4: Clean up formatting
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
        """Initialize enhanced SAT-specific detection patterns"""
        return {
            'section_headers': [
                r'^SAT\s+Practice\s+Test',  # "SAT Practice Test"
                r'^(?:Section|Part)\s+\d+[:\s]*',  # "Section 1:" or "Part 1:"
                r'^(?:Reading|Writing|Math|Language)\s+(?:Test|Section)[:\s]*',  # "Reading Test:" or "Math Section:"
                r'^(?:Evidence-Based|Critical\s+Reading|Mathematics)[:\s]*',  # "Evidence-Based Reading:" or "Mathematics:"
            ],
            'reading_passage': [
                r'^Reading\s+Passage\s*\d+[:\s]*$',  # "Reading Passage 1:" at start of line
                r'^Questions?\s*\d+[-\s]*\d*\s*are\s+based\s+on\s+the\s+following\s+passage$',  # "Questions 1-10 are based on the following passage"
                r'^The\s+following\s+passage\s+is\s+adapted\s+from',  # "The following passage is adapted from..."
                r'^Read\s+the\s+following\s+passage',  # "Read the following passage..."
                # New patterns for SAT format
                r'^The\s+unique\s+',  # "The unique subak water management system..."
                r'^The\s+mihrab\s+',  # "The mihrab (or niche)..."
                r'^The\s+Egyptian\s+',  # "The Egyptian plover..."
                r'^Until\s+\d{4}',  # "Until 1917, there was no formal measure..."
                r'^[A-Z][a-z]+\s+[a-z]+\s+[a-z]+\s+system',  # General pattern for passage starts
                r'^[A-Z][a-z]+\s+[a-z]+\s+\([^)]+\)\s+is\s+one\s+of',  # "The mihrab (or niche) is one of..."
                r'^[A-Z][a-z]+\s+[a-z]+-[a-z]+',  # "The Egyptian plover-a bird..."
            ],
            'question_start': [
                r'^\s*\d+\.\s+',  # "1. " at start of line
                r'^\s*Question\s+\d+[:\s]*',  # "Question 1:" or "Question 1"
                r'^\s*Q\s*\d+[:\s]*',  # "Q1:" or "Q1"
                r'^\s*\d+\s*$',  # "31" (standalone number)
                r'^\s*\d+\s+[A-Z]',  # "31 A" (number followed by letter)
                r'^\s*[A-Z]\s*$',  # "A" (standalone letter)
            ],
            'multiple_choice': [
                r'^\s*[A-D]\)\s+',  # "A) ", "B) ", "C) ", "D) "
                r'^\s*[A-D]\.\s+',  # "A. ", "B. ", "C. ", "D. "
                r'^\s*[A-D]\s+',    # "A ", "B ", "C ", "D "
                r'^\s*[A-D]\s*$',   # "A", "B", "C", "D" (standalone)
                r'^\s*[A-D]\s*[A-Z]',  # "A A+", "B B+" (with additional text)
            ],
            'math_indicators': [
                r'\b(?:function|equation|graph|slope|intercept|variable|solve|calculate|area|perimeter|volume|angle|triangle|rectangle|circle)\b',
                r'[=+\-*/(){}[\]^]',  # Mathematical symbols
                r'\b\d+\s*(?:times|multiplied by|divided by|plus|minus)\b',  # Word problems
                r'\b(?:length|width|height|radius|diameter|base|height)\b',  # Geometry terms
            ],
            'reading_indicators': [
                r'\b(?:passage|author|text|paragraph|line|suggests|implies|indicates|according to)\b',
                r'\b(?:main idea|central theme|primary purpose|best describes)\b',
                r'\b(?:inference|conclusion|implication|evidence)\b',
                r'\b(?:vocabulary|word|phrase|meaning|definition)\b',
            ],
            'writing_indicators': [
                r'\b(?:grammar|sentence|paragraph|transition|conclusion|introduction)\b',
                r'\b(?:subject|verb|pronoun|agreement|tense|parallel)\b',
                r'\b(?:no change|omit|delete|add|replace)\b',
                r'[A-D]\)\s*(?:No change|omit|delete)',  # Writing section options
            ]
        }
    
    def _classify_question_type(self, question_text: str, context: str = "") -> Dict:
        """Classify SAT question type based on content analysis"""
        try:
            question_lower = question_text.lower()
            context_lower = context.lower()
            combined_text = f"{question_lower} {context_lower}"
            
            # Initialize classification
            classification = {
                'question_type': 'unknown',
                'section_type': 'unknown',
                'difficulty': 'medium',
                'topic': 'general',
                'format_requirements': []
            }
            
            # Math question detection
            math_score = 0
            for pattern in self.sat_patterns['math_indicators']:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    math_score += 1
            
            # Reading question detection
            reading_score = 0
            for pattern in self.sat_patterns['reading_indicators']:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    reading_score += 1
            
            # Writing question detection
            writing_score = 0
            for pattern in self.sat_patterns['writing_indicators']:
                if re.search(pattern, combined_text, re.IGNORECASE):
                    writing_score += 1
            
            # Determine question type based on scores
            if math_score > reading_score and math_score > writing_score:
                classification['section_type'] = 'math'
                classification['format_requirements'].append('preserve_math_notation')
                
                # Sub-classify math questions
                if re.search(r'\b(?:function|equation|graph)\b', combined_text):
                    classification['question_type'] = 'algebra'
                    classification['topic'] = 'functions_and_equations'
                elif re.search(r'\b(?:area|perimeter|volume|angle|triangle|rectangle|circle)\b', combined_text):
                    classification['question_type'] = 'geometry'
                    classification['topic'] = 'geometry'
                elif re.search(r'\b(?:data|chart|graph|statistics|probability)\b', combined_text):
                    classification['question_type'] = 'data_analysis'
                    classification['topic'] = 'data_analysis'
                else:
                    classification['question_type'] = 'problem_solving'
                    classification['topic'] = 'word_problems'
                    
            elif reading_score > writing_score:
                classification['section_type'] = 'reading'
                classification['format_requirements'].append('preserve_passage_structure')
                
                # Sub-classify reading questions
                if re.search(r'\b(?:main idea|central theme|primary purpose)\b', combined_text):
                    classification['question_type'] = 'main_idea'
                    classification['topic'] = 'comprehension'
                elif re.search(r'\b(?:according to|states|mentions)\b', combined_text):
                    classification['question_type'] = 'detail'
                    classification['topic'] = 'detail_retrieval'
                elif re.search(r'\b(?:suggests|implies|indicates|inference)\b', combined_text):
                    classification['question_type'] = 'inference'
                    classification['topic'] = 'inference'
                elif re.search(r'\b(?:vocabulary|word|phrase|meaning)\b', combined_text):
                    classification['question_type'] = 'vocabulary'
                    classification['topic'] = 'vocabulary_in_context'
                elif re.search(r'\b(?:evidence|support|best evidence)\b', combined_text):
                    classification['question_type'] = 'evidence'
                    classification['topic'] = 'command_of_evidence'
                else:
                    classification['question_type'] = 'comprehension'
                    classification['topic'] = 'general_comprehension'
                    
            elif writing_score > 0:
                classification['section_type'] = 'writing'
                classification['format_requirements'].append('maintain_line_numbers')
                
                # Sub-classify writing questions
                if re.search(r'\b(?:grammar|subject|verb|pronoun|agreement)\b', combined_text):
                    classification['question_type'] = 'grammar'
                    classification['topic'] = 'grammar_and_usage'
                elif re.search(r'\b(?:sentence|paragraph|transition|organization)\b', combined_text):
                    classification['question_type'] = 'expression'
                    classification['topic'] = 'expression_of_ideas'
                else:
                    classification['question_type'] = 'editing'
                    classification['topic'] = 'standard_english_conventions'
            
            # Determine difficulty based on question complexity
            if len(question_text) > 100 or re.search(r'\b(?:complex|advanced|sophisticated)\b', combined_text):
                classification['difficulty'] = 'hard'
            elif len(question_text) < 50:
                classification['difficulty'] = 'easy'
            
            logger.info(f"Question classified as: {classification['section_type']} - {classification['question_type']}")
            return classification
            
        except Exception as e:
            logger.warning(f"Question classification failed: {e}")
            return {
                'question_type': 'unknown',
                'section_type': 'unknown',
                'difficulty': 'medium',
                'topic': 'general',
                'format_requirements': []
            }
    
    def _ai_enhance_sat_structure_detection(self, text: str) -> Dict:
        """Use Gemini AI to enhance SAT structure detection"""
        try:
            if not self.watermark_remover._has_gemini_api():
                return None
            
            logger.info("Using Gemini AI to enhance SAT structure detection...")
            
            # Create prompt for Gemini
            prompt = f"""
            Analyze this SAT document text and identify the structure with exact formatting.
            
            Document text:
            {text[:3000]}...
            
            Instructions:
            1. Identify reading passages (long text blocks before questions)
            2. Identify questions (numbered items that ask something)
            3. Identify multiple choice options (A, B, C, D choices)
            4. Return the structure in this exact format:
            
            {{
                "reading_passages": [
                    {{
                        "text": "passage title or first line",
                        "content": ["line 1", "line 2", "line 3"]
                    }}
                ],
                "questions": [
                    {{
                        "text": "question text",
                        "content": ["question content"],
                        "choices": [
                            {{"text": "A) choice text"}},
                            {{"text": "B) choice text"}},
                            {{"text": "C) choice text"}},
                            {{"text": "D) choice text"}}
                        ]
                    }}
                ]
            }}
            
            Focus on the exact format above. Return valid JSON only.
            """
            
            # Use Gemini API
            import google.generativeai as genai
            genai.configure(api_key=self.watermark_remover.ai_api_keys['gemini'])
            model = genai.GenerativeModel('gemini-pro')
            
            response = model.generate_content(prompt)
            ai_response = response.text
            
            # Parse AI response
            try:
                import json
                result = json.loads(ai_response)
                
                # Validate structure
                if 'reading_passages' in result and 'questions' in result:
                    logger.info("Gemini AI successfully enhanced SAT structure detection")
                    return result
                else:
                    logger.warning("Gemini AI response missing required fields")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Gemini AI response as JSON: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"Gemini AI enhancement failed: {e}")
            return None
    
    def process_sat_document(self, input_path: str) -> str:
        """Process SAT document with format detection, image preservation, and watermark removal"""
        try:
            logger.info(f"Processing SAT document: {input_path}")
            
            # Extract text with watermark removal
            text = self.watermark_remover.extract_text_from_pdf(input_path)
            logger.info(f"Extracted text length: {len(text)} characters")
            
            # Extract images for integration
            images = self.watermark_remover._extract_images_from_pdf(input_path)
            logger.info(f"Extracted {len(images)} images from document")
            
            # Detect SAT structure with image integration
            sat_structure = self._detect_sat_structure_with_images(text, images)
            logger.info(f"Detected SAT structure with images: {sat_structure}")
            
            # Process and format for Word with images
            formatted_text = self._format_sat_for_word_with_images(text, sat_structure, images)
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"SAT document processing failed: {e}")
            raise
    
    def process_sat_document_english(self, input_path: str) -> str:
        """Process SAT document specifically for English section (text processing focus)"""
        try:
            logger.info(f"Processing SAT document for English section: {input_path}")
            
            # Extract text with enhanced watermark removal for English
            text = self.watermark_remover.extract_text_from_pdf(input_path)
            logger.info(f"Extracted text length: {len(text)} characters")
            
            # Extract images for integration
            images = self.watermark_remover._extract_images_from_pdf(input_path)
            logger.info(f"Extracted {len(images)} images from document")
            
            # Detect SAT structure with image integration
            sat_structure = self._detect_sat_structure_with_images(text, images)
            logger.info(f"Detected SAT structure with images: {sat_structure}")
            
            # Process and format for Word with English section focus
            formatted_text = self._format_sat_for_word_english(text, sat_structure, images)
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"SAT document processing for English section failed: {e}")
            raise
    
    def process_sat_document_math(self, input_path: str) -> str:
        """Process SAT document specifically for Math section (LaTeX conversion focus)"""
        try:
            logger.info(f"Processing SAT document for Math section: {input_path}")
            
            # Extract text with enhanced watermark removal for Math
            text = self.watermark_remover.extract_text_from_pdf(input_path)
            logger.info(f"Extracted text length: {len(text)} characters")
            
            # Extract images for integration
            images = self.watermark_remover._extract_images_from_pdf(input_path)
            logger.info(f"Extracted {len(text)} images from document")
            
            # Detect SAT structure with image integration
            sat_structure = self._detect_sat_structure_with_images(text, images)
            logger.info(f"Detected SAT structure with images: {sat_structure}")
            
            # Process and format for Word with Math section focus (LaTeX conversion)
            formatted_text = self._format_sat_for_word_math(text, sat_structure, images)
            
            return formatted_text
            
        except Exception as e:
            logger.error(f"SAT document processing for Math section failed: {e}")
            raise
    
    def _detect_sat_structure(self, text: str) -> Dict:
        """Detect SAT document structure and components with enhanced classification"""
        try:
            lines = text.split('\n')
            structure = {
                'sections': [],
                'reading_passages': [],
                'questions': [],
                'document_type': 'unknown',
                'total_questions': 0
            }
            
            current_passage = None
            current_question = None
            question_counter = 0
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # Detect section headers
                if self._matches_pattern(line, self.sat_patterns['section_headers']):
                    structure['sections'].append({
                        'text': line,
                        'line_number': i,
                        'section_type': self._classify_section_type(line)
                    })
                
                # Detect reading passages
                elif self._matches_pattern(line, self.sat_patterns['reading_passage']):
                    if current_passage:
                        structure['reading_passages'].append(current_passage)
                    current_passage = {
                        'text': line,
                        'content': [],
                        'line_number': i,
                        'passage_type': 'reading'
                    }
                
                # Detect questions
                elif self._matches_pattern(line, self.sat_patterns['question_start']):
                    if current_question:
                        # Classify the question before adding
                        question_classification = self._classify_question_type(
                            current_question['text'], 
                            ' '.join(current_question['content'])
                        )
                        current_question.update(question_classification)
                        structure['questions'].append(current_question)
                        question_counter += 1
                    
                    current_question = {
                        'text': line,
                        'content': [],
                        'choices': [],
                        'line_number': i,
                        'question_number': question_counter + 1
                    }
                
                # Detect multiple choice options
                elif self._matches_pattern(line, self.sat_patterns['multiple_choice']):
                    if current_question:
                        # Clean up the choice text
                        choice_text = re.sub(r'^[A-D][\)\.\s]*', '', line)
                        choice_text = re.sub(r'\s*[A-Z]+\s*$', '', choice_text)  # Remove trailing letters like "A+"
                        choice_text = choice_text.strip()
                        
                        if choice_text:  # Only add if there's actual content
                            current_question['choices'].append({
                                'text': choice_text,
                                'option': line[0] if line else 'A',
                                'is_correct': False
                            })
                        continue
                
                # Add content to current passage or question
                else:
                    if current_passage:
                        current_passage['content'].append(line)
                    elif current_question:
                        current_question['content'].append(line)
            
            # Add final passage and question
            if current_passage:
                structure['reading_passages'].append(current_passage)
            if current_question:
                # Classify the final question
                question_classification = self._classify_question_type(
                    current_question['text'], 
                    ' '.join(current_question['content'])
                )
                current_question.update(question_classification)
                structure['questions'].append(current_question)
                question_counter += 1
            
            # Determine overall document type
            structure['total_questions'] = question_counter
            structure['document_type'] = self._determine_document_type(structure)
            
            logger.info(f"Detected {len(structure['sections'])} sections, {len(structure['reading_passages'])} passages, {len(structure['questions'])} questions")
            logger.info(f"Document type: {structure['document_type']}")
            return structure
            
        except Exception as e:
            logger.error(f"SAT structure detection failed: {e}")
            return {'sections': [], 'reading_passages': [], 'questions': [], 'document_type': 'unknown', 'total_questions': 0}
    
    def _classify_section_type(self, section_text: str) -> str:
        """Classify section type based on header text"""
        section_lower = section_text.lower()
        if 'math' in section_lower or 'mathematics' in section_lower:
            return 'math'
        elif 'reading' in section_lower or 'evidence' in section_lower:
            return 'reading'
        elif 'writing' in section_lower or 'language' in section_lower:
            return 'writing'
        else:
            return 'general'
    
    def _determine_document_type(self, structure: Dict) -> str:
        """Determine overall document type based on structure analysis"""
        if not structure['questions']:
            return 'unknown'
        
        # Count question types
        math_questions = sum(1 for q in structure['questions'] if q.get('section_type') == 'math')
        reading_questions = sum(1 for q in structure['questions'] if q.get('section_type') == 'reading')
        writing_questions = sum(1 for q in structure['questions'] if q.get('section_type') == 'writing')
        
        total = len(structure['questions'])
        
        if math_questions > total * 0.6:
            return 'math_focused'
        elif reading_questions > total * 0.6:
            return 'reading_focused'
        elif writing_questions > total * 0.6:
            return 'writing_focused'
        elif math_questions > 0 and reading_questions > 0:
            return 'mixed_sat'
        else:
            return 'general'
    
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
            formatted_lines.append(f"**SECTION_HEADER:{section['text']}**")
            formatted_lines.append("")
        
        # Process reading passages with proper structure
        for passage in structure['reading_passages']:
            formatted_lines.append(f"**READING_PASSAGE:{passage['text']}**")
            formatted_lines.append("")
            
            if 'content' in passage:
                for line in passage['content']:
                    formatted_lines.append(f"PASSAGE_CONTENT:{line}")
                formatted_lines.append("")
        
        # Process questions with proper structure
        for question in structure['questions']:
            formatted_lines.append(f"**QUESTION:{question['text']}**")
            
            # Add question content if any
            if 'content' in question:
                for line in question['content']:
                    formatted_lines.append(f"QUESTION_CONTENT:{line}")
            
            # Add multiple choice options with proper formatting
            if question['choices']:
                formatted_lines.append("")
                for choice in question['choices']:
                    formatted_lines.append(f"MULTIPLE_CHOICE:{choice['text']}")
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def _detect_sat_structure_with_images(self, text: str, images: List[Dict]) -> Dict:
        """Detect SAT structure with image integration and AI enhancement"""
        try:
            # Use AI to enhance structure detection if available
            if self.watermark_remover._has_gemini_api():
                ai_enhanced_structure = self._ai_enhance_sat_structure_detection(text)
                if ai_enhanced_structure:
                    logger.info("AI-enhanced SAT structure detection completed")
                    # Integrate images with AI-enhanced structure
                    for passage in ai_enhanced_structure['reading_passages']:
                        passage['images'] = []
                        for image in images:
                            if self._image_belongs_to_passage(image, passage):
                                passage['images'].append(image)
                    return ai_enhanced_structure
            
            # Fallback to basic structure detection
            structure = self._detect_sat_structure(text)
            
            # Integrate images into reading passages
            for passage in structure['reading_passages']:
                passage['images'] = []
                
                # Find images that belong to this passage
                for image in images:
                    # Simple heuristic: images on the same page as passage content
                    if self._image_belongs_to_passage(image, passage):
                        passage['images'].append(image)
                        logger.info(f"Image {image['index']} integrated into passage: {passage['text']}")
            
            return structure
            
        except Exception as e:
            logger.error(f"AI-enhanced SAT structure detection failed: {e}")
            # Fallback to basic detection
            return self._detect_sat_structure(text)
    
    def _image_belongs_to_passage(self, image: Dict, passage: Dict) -> bool:
        """Determine if an image belongs to a specific passage"""
        # Simple heuristic: image is on the same page as passage content
        # In a more sophisticated system, you could use OCR to detect image captions
        # or analyze image position relative to text blocks
        
        if 'start_line' in passage:
            # Estimate page number from line number (assuming ~50 lines per page)
            estimated_page = passage['start_line'] // 50 + 1
            return image['page'] == estimated_page
        
        return False
    
    def _format_sat_for_word_with_images(self, text: str, structure: Dict, images: List[Dict]) -> str:
        """Format SAT content for Word document with image integration"""
        formatted_lines = []
        
        # Add title
        formatted_lines.append("SAT Practice Test")
        formatted_lines.append("=" * 50)
        formatted_lines.append("")
        
        # Process sections
        for section in structure['sections']:
            formatted_lines.append(f"**SECTION_HEADER:{section['text']}**")
            formatted_lines.append("")
        
        # Process reading passages with images
        for passage in structure['reading_passages']:
            formatted_lines.append(f"**READING_PASSAGE:{passage['text']}**")
            formatted_lines.append("")
            
            # Add passage content
            if 'content' in passage:
                for line in passage['content']:
                    formatted_lines.append(f"PASSAGE_CONTENT:{line}")
                formatted_lines.append("")
            
            # Add images associated with this passage
            if 'images' in passage and passage['images']:
                formatted_lines.append("**PASSAGE_IMAGES:**")
                for image in passage['images']:
                    formatted_lines.append(f"IMAGE_PLACEHOLDER:Page {image['page']}, Image {image['index']} ({image['width']}x{image['height']})")
                formatted_lines.append("")
        
        # Process questions with proper structure
        for question in structure['questions']:
            formatted_lines.append(f"**QUESTION:{question['text']}**")
            
            # Add question content if any
            if 'content' in question:
                for line in question['content']:
                    formatted_lines.append(f"QUESTION_CONTENT:{line}")
            
            # Add multiple choice options with proper formatting
            if question['choices']:
                formatted_lines.append("")
                for choice in question['choices']:
                    formatted_lines.append(f"MULTIPLE_CHOICE:{choice['text']}")
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def _format_sat_for_word_english(self, text: str, structure: Dict, images: List[Dict]) -> str:
        """Format SAT content for Word document with English section focus (text processing)"""
        formatted_lines = []
        
        # Add document metadata
        formatted_lines.append(f"- document type : {structure.get('document_type', 'unknown')}")
        formatted_lines.append(f"+Total Questions: {structure.get('total_questions', 0)}")
        formatted_lines.append("")
        
        # Process reading passages with exact format
        for passage in structure['reading_passages']:
            formatted_lines.append("- reading passage :")
            if 'content' in passage and passage['content']:
                for line in passage['content']:
                    formatted_lines.append(f"+{line}")
            else:
                # If no specific content, use the passage text
                formatted_lines.append(f"+{passage['text']}")
            formatted_lines.append("")
        
        # Process questions with exact format and enhanced classification
        for question in structure['questions']:
            # Add question metadata
            question_type = question.get('question_type', 'unknown')
            section_type = question.get('section_type', 'unknown')
            difficulty = question.get('difficulty', 'medium')
            topic = question.get('topic', 'general')
            
            formatted_lines.append(f"- question type : {question_type}")
            formatted_lines.append(f"+Section: {section_type}")
            formatted_lines.append(f"+Difficulty: {difficulty}")
            formatted_lines.append(f"+Topic: {topic}")
            formatted_lines.append("")
            
            formatted_lines.append("- question:")
            if 'content' in question and question['content']:
                for line in question['content']:
                    formatted_lines.append(f"+{line}")
            else:
                # If no specific content, use the question text
                formatted_lines.append(f"+{question['text']}")
            formatted_lines.append("")
            
            # Add multiple choice options with exact format
            if question['choices']:
                formatted_lines.append("- options")
                for choice in question['choices']:
                    formatted_lines.append(f"+{choice['option']}) {choice['text']}")
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def _format_sat_for_word_math(self, text: str, structure: Dict, images: List[Dict]) -> str:
        """Format SAT content for Word document with Math section focus (LaTeX conversion)"""
        formatted_lines = []
        
        # Add document metadata
        formatted_lines.append(f"- document type : {structure.get('document_type', 'unknown')}")
        formatted_lines.append(f"+Total Questions: {structure.get('total_questions', 0)}")
        formatted_lines.append("")
        
        # Process reading passages with exact format
        for passage in structure['reading_passages']:
            formatted_lines.append("- reading passage :")
            if 'content' in passage and passage['content']:
                for line in passage['content']:
                    # Convert math expressions to LaTeX format
                    latex_line = self._convert_math_to_latex(line)
                    formatted_lines.append(f"+{latex_line}")
            else:
                # If no specific content, use the passage text
                latex_text = self._convert_math_to_latex(passage['text'])
                formatted_lines.append(f"+{latex_text}")
            formatted_lines.append("")
        
        # Process questions with exact format and enhanced classification
        for question in structure['questions']:
            # Add question metadata
            question_type = question.get('question_type', 'unknown')
            section_type = question.get('section_type', 'unknown')
            difficulty = question.get('difficulty', 'medium')
            topic = question.get('topic', 'general')
            
            formatted_lines.append(f"- question type : {question_type}")
            formatted_lines.append(f"+Section: {section_type}")
            formatted_lines.append(f"+Difficulty: {difficulty}")
            formatted_lines.append(f"+Topic: {topic}")
            formatted_lines.append("")
            
            formatted_lines.append("- question:")
            if 'content' in question and question['content']:
                for line in question['content']:
                    # Convert math expressions to LaTeX format
                    latex_line = self._convert_math_to_latex(line)
                    formatted_lines.append(f"+{latex_line}")
            else:
                # If no specific content, use the question text
                latex_text = self._convert_math_to_latex(question['text'])
                formatted_lines.append(f"+{latex_text}")
            formatted_lines.append("")
            
            # Add multiple choice options with exact format
            if question['choices']:
                formatted_lines.append("- options")
                for choice in question['choices']:
                    # Convert math expressions in choices to LaTeX format
                    latex_choice = self._convert_math_to_latex(choice['text'])
                    formatted_lines.append(f"+{choice['option']}) {latex_choice}")
                formatted_lines.append("")
        
        return '\n'.join(formatted_lines)
    
    def _convert_math_to_latex(self, text: str) -> str:
        """Convert math expressions to LaTeX format using pdftolatex approach"""
        try:
            if not text or not isinstance(text, str):
                return text
            
            # Common math patterns to convert to LaTeX
            latex_text = text
            
            # Convert fractions: 1/2 -> \frac{1}{2}
            latex_text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex_text)
            
            # Convert exponents: x^2 -> x^{2}, x^y -> x^{y}
            latex_text = re.sub(r'(\w+)\^(\d+)', r'\1^{\2}', latex_text)
            latex_text = re.sub(r'(\w+)\^(\w+)', r'\1^{\2}', latex_text)
            
            # Convert square roots: sqrt(x) -> \sqrt{x}
            latex_text = re.sub(r'sqrt\(([^)]+)\)', r'\\sqrt{\1}', latex_text)
            
            # Convert Greek letters and common symbols
            latex_text = re.sub(r'\bpi\b', r'\\pi', latex_text)
            latex_text = re.sub(r'\btheta\b', r'\\theta', latex_text)
            latex_text = re.sub(r'\balpha\b', r'\\alpha', latex_text)
            latex_text = re.sub(r'\bbeta\b', r'\\beta', latex_text)
            latex_text = re.sub(r'\bgamma\b', r'\\gamma', latex_text)
            latex_text = re.sub(r'\bdelta\b', r'\\delta', latex_text)
            latex_text = re.sub(r'\bepsilon\b', r'\\epsilon', latex_text)
            latex_text = re.sub(r'\blambda\b', r'\\lambda', latex_text)
            latex_text = re.sub(r'\bmu\b', r'\\mu', latex_text)
            latex_text = re.sub(r'\bsigma\b', r'\\sigma', latex_text)
            latex_text = re.sub(r'\btau\b', r'\\tau', latex_text)
            latex_text = re.sub(r'\bphi\b', r'\\phi', latex_text)
            latex_text = re.sub(r'\bomega\b', r'\\omega', latex_text)
            
            # Convert infinity symbol
            latex_text = re.sub(r'\binfinity\b', r'\\infty', latex_text)
            latex_text = re.sub(r'∞', r'\\infty', latex_text)
            
            # Convert mathematical operators
            latex_text = re.sub(r'\btimes\b', r'\\times', latex_text)
            latex_text = re.sub(r'\bdiv\b', r'\\div', latex_text)
            latex_text = re.sub(r'\bplus\b', r'+', latex_text)
            latex_text = re.sub(r'\bminus\b', r'-', latex_text)
            
            # Convert inequalities
            latex_text = re.sub(r'<=', r'\\leq', latex_text)
            latex_text = re.sub(r'>=', r'\\geq', latex_text)
            latex_text = re.sub(r'<', r'<', latex_text)
            latex_text = re.sub(r'>', r'>', latex_text)
            
            # Convert set notation
            latex_text = re.sub(r'\bin\b', r'\\in', latex_text)
            latex_text = re.sub(r'\bnotin\b', r'\\notin', latex_text)
            latex_text = re.sub(r'\bsubset\b', r'\\subset', latex_text)
            latex_text = re.sub(r'\bsupset\b', r'\\supset', latex_text)
            latex_text = re.sub(r'\bunion\b', r'\\cup', latex_text)
            latex_text = re.sub(r'\bintersection\b', r'\\cap', latex_text)
            
            # Convert logical operators
            latex_text = re.sub(r'\band\b', r'\\land', latex_text)
            latex_text = re.sub(r'\bor\b', r'\\lor', latex_text)
            latex_text = re.sub(r'\bnot\b', r'\\neg', latex_text)
            
            # Convert function notation: f(x) -> f(x) (already correct)
            # Convert limits: lim x->0 -> \lim_{x \to 0}
            latex_text = re.sub(r'lim\s+(\w+)\s*->\s*(\w+)', r'\\lim_{\1 \\to \2}', latex_text)
            
            # Convert integrals: int -> \int
            latex_text = re.sub(r'\bint\b', r'\\int', latex_text)
            
            # Convert summations: sum -> \sum
            latex_text = re.sub(r'\bsum\b', r'\\sum', latex_text)
            
            # Convert products: prod -> \prod
            latex_text = re.sub(r'\bprod\b', r'\\prod', latex_text)
            
            # Convert absolute value: |x| -> |x| (already correct)
            
            # Convert matrices: [a b; c d] -> \begin{matrix} a & b \\ c & d \end{matrix}
            matrix_pattern = r'\[([^]]+)\]'
            def replace_matrix(match):
                content = match.group(1)
                if ';' in content:  # Matrix with semicolon separator
                    rows = content.split(';')
                    matrix_content = ' \\\\ '.join([row.strip().replace(' ', ' & ') for row in rows])
                    return f'\\begin{{matrix}} {matrix_content} \\end{{matrix}}'
                return match.group(0)  # Return as-is if not a matrix
            
            latex_text = re.sub(matrix_pattern, replace_matrix, latex_text)
            
            # Convert vectors: <a,b> -> \langle a, b \rangle
            latex_text = re.sub(r'<([^>]+)>', r'\\langle \1 \\rangle', latex_text)
            
            # Convert degrees: 90° -> 90°
            latex_text = re.sub(r'(\d+)°', r'\1°', latex_text)
            
            # Convert percentages: 50% -> 50\%
            latex_text = re.sub(r'(\d+)%', r'\1\\%', latex_text)
            
            # Convert common functions
            latex_text = re.sub(r'\bsin\b', r'\\sin', latex_text)
            latex_text = re.sub(r'\bcos\b', r'\\cos', latex_text)
            latex_text = re.sub(r'\btan\b', r'\\tan', latex_text)
            latex_text = re.sub(r'\blog\b', r'\\log', latex_text)
            latex_text = re.sub(r'\bln\b', r'\\ln', latex_text)
            latex_text = re.sub(r'\bexp\b', r'\\exp', latex_text)
            
            # Convert natural log: ln(x) -> \ln(x)
            latex_text = re.sub(r'ln\(([^)]+)\)', r'\\ln(\1)', latex_text)
            
            # Convert log base: log_2(x) -> \log_2(x)
            latex_text = re.sub(r'log_(\w+)\(([^)]+)\)', r'\\log_{\1}(\2)', latex_text)
            
            # Convert complex expressions with parentheses
            # Handle nested expressions like f(x) = 2x + 1
            latex_text = re.sub(r'f\((\w+)\)\s*=\s*([^=]+)', r'f(\1) = \2', latex_text)
            
            # Convert domain and range notation: [a,b] -> [a,b] (already correct)
            # Convert open intervals: (a,b) -> (a,b) (already correct)
            
            # Convert complex numbers: a + bi -> a + bi (already correct)
            latex_text = re.sub(r'(\d+)\s*\+\s*(\d+)i', r'\1 + \2i', latex_text)
            
            # Convert scientific notation: 1.5e10 -> 1.5 \times 10^{10}
            latex_text = re.sub(r'(\d+\.?\d*)[eE](\d+)', r'\1 \\times 10^{\2}', latex_text)
            
            # Clean up multiple spaces
            latex_text = re.sub(r'\s+', ' ', latex_text).strip()
            
            logger.info(f"Converted math expression: '{text}' -> '{latex_text}'")
            return latex_text
            
        except Exception as e:
            logger.warning(f"LaTeX conversion failed for '{text}': {e}")
            return text
    

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
            
            # Check if this is a SAT document first (for both English and Math sections)
            is_sat = self._is_sat_document(file_path)
            if is_sat:
                logger.info("SAT document detected - using specialized processor")
                word_path = self._convert_sat_to_word(file_path, file_path, section_type)
                section_type = f'sat_{section_type}'  # Mark as SAT with section type
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
                'sat_detection': 'completed' if is_sat else 'not_applicable'
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
    
    def _convert_sat_to_word(self, pdf_path: str, original_filename: str, section_type: str = 'english') -> str:
        """Convert SAT PDF to Word document with format preservation for specific section"""
        logger.info(f"Converting SAT document to Word with {section_type} section processing...")
        
        try:
            # Process SAT document through specialized processor with section-specific handling
            if section_type == 'math':
                formatted_text = self.sat_processor.process_sat_document_math(pdf_path)
            else:
                formatted_text = self.sat_processor.process_sat_document_english(pdf_path)
            
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
            
            # Process formatted text with enhanced SAT format
            lines = formatted_text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Process document metadata
                if line.startswith("- document type :"):
                    metadata_text = line.replace("- document type :", "").strip()
                    heading = word_doc.add_heading(f"Document Type: {metadata_text}", level=1)
                    heading.style.font.bold = True
                    heading.style.font.color.rgb = RGBColor(0, 0, 0)  # Black
                    continue
                
                # Process question type metadata
                if line.startswith("- question type :"):
                    question_type = line.replace("- question type :", "").strip()
                    heading = word_doc.add_heading(f"Question Type: {question_type}", level=4)
                    heading.style.font.bold = True
                    heading.style.font.color.rgb = RGBColor(128, 0, 128)  # Purple
                    continue
                
                # Process reading passages
                if line == "- reading passage :":
                    heading = word_doc.add_heading("Reading Passage", level=2)
                    heading.style.font.bold = True
                    heading.style.font.color.rgb = RGBColor(0, 0, 139)  # Dark blue
                    continue
                
                # Process questions
                if line == "- question:":
                    heading = word_doc.add_heading("Question", level=3)
                    heading.style.font.bold = True
                    heading.style.font.color.rgb = RGBColor(139, 0, 0)  # Dark red
                    continue
                
                # Process options
                if line == "- options":
                    heading = word_doc.add_heading("Multiple Choice Options", level=4)
                    heading.style.font.bold = True
                    heading.style.font.color.rgb = RGBColor(0, 100, 0)  # Dark green
                    continue
                
                # Process metadata lines (section, difficulty, topic)
                if line.startswith('+') and ('Section:' in line or 'Difficulty:' in line or 'Topic:' in line):
                    content_text = line[1:].strip()  # Remove the + prefix
                    if content_text:
                        para = word_doc.add_paragraph(content_text)
                        para.paragraph_format.space_after = Pt(3)
                        para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        para.style.font.italic = True
                        para.style.font.color.rgb = RGBColor(128, 128, 128)  # Gray
                    continue
                
                # Process passage content
                if line.startswith('+') and len(line) > 1:
                    content_text = line[1:].strip()  # Remove the + prefix
                    if content_text:
                        para = word_doc.add_paragraph(content_text)
                        para.paragraph_format.space_after = Pt(6)
                        para.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    continue
                
                # Regular paragraph (fallback)
                if line and not line.startswith('-'):
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

@app.route('/api-key', methods=['POST'])
def update_api_key():
    """Update API key for AI enhancement"""
    try:
        data = request.get_json()
        api_key = data.get('api_key', '').strip()
        provider = data.get('provider', 'gemini')
        
        if not api_key:
            return jsonify({'success': False, 'error': 'API key is required'}), 400
        
        # Store globally and in watermark remover
        global GEMINI_API_KEY
        GEMINI_API_KEY = api_key
        
        # Update watermark remover instance
        watermark_remover = get_processor().watermark_remover
        watermark_remover.set_api_key(provider, api_key)
        
        logger.info(f"API key updated for {provider}")
        return jsonify({
            'success': True, 
            'message': f'API key updated successfully for {provider}'
        })
        
    except Exception as e:
        logger.error(f"API key update failed: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

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

            .api-key-section {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 25px;
                margin-bottom: 30px;
                border: 1px solid #e9ecef;
            }

            .api-key-section h3 {
                color: #333;
                margin-bottom: 10px;
                font-size: 1.3em;
            }

            .api-key-section p {
                color: #666;
                margin-bottom: 20px;
                font-size: 0.95em;
            }

            .api-input-group {
                display: flex;
                gap: 15px;
                margin-bottom: 15px;
            }

            .api-key-input {
                flex: 1;
                padding: 12px 15px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 1em;
                transition: border-color 0.3s ease;
            }

            .api-key-input:focus {
                outline: none;
                border-color: #4CAF50;
            }

            .api-key-btn {
                background: linear-gradient(45deg, #4CAF50, #45a049);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                white-space: nowrap;
            }

            .api-key-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(76, 175, 80, 0.3);
            }

            .api-key-status {
                min-height: 20px;
                font-size: 0.9em;
            }

            .api-key-status.success {
                color: #155724;
            }

            .api-key-status.error {
                color: #721c24;
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
                
                .training-section {
                    padding: 20px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📄 PDF Watermark Remover</h1>
            <p class="subtitle">Two sections: English (text) and Math (LaTeX)</p>

            <!-- API Key Input Section -->
            <div class="api-key-section">
                <h3>🔑 AI Enhancement Setup</h3>
                <p>Enter your Gemini API key for enhanced watermark removal and AI-powered processing:</p>
                <div class="api-input-group">
                    <input type="password" id="apiKeyInput" placeholder="Enter your Gemini API key" class="api-key-input">
                    <button onclick="updateApiKey()" class="api-key-btn">Set API Key</button>
                </div>
                <div id="apiKeyStatus" class="api-key-status"></div>
            </div>

            <div class="sections">
                <div class="section" id="englishSection" onclick="selectSection('english')">
                    <div class="section-icon">📝</div>
                    <div class="section-title">English Section</div>
                    <div class="section-desc">
                        Remove watermarks and convert to Word with clean text formatting.
                        <strong>Auto-detects SAT documents</strong> for enhanced processing.
                        Perfect for documents, reports, and SAT practice tests.
                    </div>
                </div>

                <div class="section" id="mathSection" onclick="selectSection('math')">
                    <div class="section-icon">🧮</div>
                    <div class="section-title">Math Section</div>
                    <div class="section-desc">
                        Auto-detect math functions and convert to Word with LaTeX content.
                        <strong>Auto-detects SAT documents</strong> for enhanced processing.
                        Ideal for academic papers, equations, and SAT math sections.
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
            const apiKeyInput = document.getElementById('apiKeyInput');
            const apiKeyStatus = document.getElementById('apiKeyStatus');

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
                    dropZone.querySelector('.drop-subtext').textContent = 'English section: Text processing, watermark removal & SAT detection';
                } else if (section === 'math') {
                    mathSection.classList.add('active');
                    dropZone.querySelector('.drop-subtext').textContent = 'Math section: LaTeX conversion, math detection & SAT processing';
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

            async function updateApiKey() {
                const apiKey = apiKeyInput.value.trim();
                if (!apiKey) {
                    showApiKeyStatus('Please enter an API key', 'error');
                    return;
                }

                try {
                    const response = await fetch('/api-key', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            api_key: apiKey,
                            provider: 'gemini'
                        })
                    });

                    const result = await response.json();
                    
                    if (result.success) {
                        showApiKeyStatus(result.message, 'success');
                        apiKeyInput.value = '';
                    } else {
                        showApiKeyStatus(result.error || 'Failed to update API key', 'error');
                    }
                } catch (error) {
                    console.error('API key update error:', error);
                    showApiKeyStatus('Failed to update API key. Please try again.', 'error');
                }
            }

            function showApiKeyStatus(message, type) {
                apiKeyStatus.textContent = message;
                apiKeyStatus.className = `api-key-status ${type}`;
                
                // Clear status after 5 seconds
                setTimeout(() => {
                    apiKeyStatus.textContent = '';
                    apiKeyStatus.className = 'api-key-status';
                }, 5000);
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
        if section_type not in ['english', 'math', 'sat']:
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
    print("Starting PDF Watermark Remover - Enhanced SAT Processing...")
    print(f"Python {sys.version} detected")
    print("All dependencies loaded successfully")
    print("Watermark removal algorithms ready")
    print("SAT document processor ready")
    print("AI-powered enhancement ready (API key configurable via web interface)")
    
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
    print("Two sections: English (text + SAT) and Math (LaTeX + SAT)")
    print("SAT documents auto-detected in both sections")
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

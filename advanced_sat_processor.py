#!/usr/bin/env python3
"""
Advanced SAT Document Processor
Inspired by MinerU and other advanced document processing tools
"""

import re
import fitz
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class AdvancedSATProcessor:
    """Advanced SAT document processor with improved detection algorithms"""
    
    def __init__(self):
        self.setup_patterns()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for debugging"""
        logging.basicConfig(level=logging.INFO)
    
    def setup_patterns(self):
        """Setup advanced detection patterns based on digital SAT format research"""
        
        # Digital SAT specific patterns
        self.patterns = {
            # Reading passage indicators (more flexible)
            'passage_start': [
                r'^[A-Z][a-z]+(?:\s+[a-z]+){2,}\s+[a-z]+',  # 4+ word sentences starting with capital
                r'^The\s+[a-z]+\s+[a-z]+',  # "The unique subak..."
                r'^Until\s+\d{4}',  # "Until 1917..."
                r'^[A-Z][a-z]+\s+[a-z]+-[a-z]+',  # "The Egyptian plover-a bird..."
                r'^[A-Z][a-z]+\s+[a-z]+\s+\([^)]+\)',  # "The mihrab (or niche)..."
                r'^[A-Z][a-z]+(?:\s+[a-z]+){3,}',  # 4+ word sentences
                r'^[A-Z][a-z]+(?:\s+[a-z]+){4,}',  # 5+ word sentences
                r'^[A-Z][a-z]+(?:\s+[a-z]+){5,}',  # 6+ word sentences
            ],
            
            # Question number patterns
            'question_number': [
                r'^\d+$',  # Just a number
                r'^\d+\s*$',  # Number with optional spaces
                r'^Question\s+\d+',  # "Question 1"
                r'^\d+\.',  # "1."
                r'^\d+\s*[A-Z]',  # "1 A" (number followed by choice)
            ],
            
            # Multiple choice patterns
            'multiple_choice': [
                r'^[A-D]\.\s+',  # "A. "
                r'^[A-D]\s+',  # "A "
                r'^[A-D]\)\s+',  # "A) "
                r'^[A-D]\s*[-+]\s*',  # "A - " or "A + "
            ],
            
            # Math function patterns
            'math_function': [
                r'f\([^)]+\)\s*=\s*[^,;]+',  # f(x) = 25x + 70
                r'g\([^)]+\)\s*=\s*[^,;]+',  # g(x) = ...
                r'h\([^)]+\)\s*=\s*[^,;]+',  # h(x) = ...
                r'\d+x\s*[+\-]\s*\d+',  # 25x + 70
                r'x\s*[+\-]\s*\d+',  # x + 5
                r'\d+x\s*[+\-]\s*\d+y',  # 25x + 70y
            ],
            
            # Section headers
            'section_header': [
                r'Section\s+\d+.*Math',
                r'Section\s+\d+.*Reading',
                r'Section\s+\d+.*Writing',
                r'Module\s+\d+.*Math',
                r'Module\s+\d+.*Reading',
                r'Module\s+\d+.*Writing',
            ],
            
            # Watermark patterns (to remove)
            'watermark': [
                r'A-\s*A\+\s*\d{2}:\d{2}:\d{2}',  # "A- A+ 21:23:57"
                r'^\s*[-+]+\s*$',  # Lines with just dashes or pluses
                r'^\s*[A-Z]+\s*$',  # Lines with just capital letters
                r'^\s*\d+\s*$',  # Lines with just numbers (if standalone)
            ]
        }
    
    def extract_text_advanced(self, pdf_path: str) -> str:
        """Advanced text extraction with better preprocessing"""
        try:
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    # Clean the text
                    cleaned_text = self.clean_text(text)
                    if cleaned_text:
                        text_parts.append(cleaned_text)
                else:
                    # Try OCR if no text found
                    logger.info(f"No text found on page {page_num + 1}, trying OCR...")
                    # OCR would go here if available
                    
            doc.close()
            return '\n'.join(text_parts)
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Remove watermarks
            if self.is_watermark(line):
                continue
                
            # Clean up the line
            line = re.sub(r'\s+', ' ', line)  # Normalize whitespace
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def is_watermark(self, line: str) -> bool:
        """Check if a line is a watermark"""
        for pattern in self.patterns['watermark']:
            if re.match(pattern, line):
                return True
        return False
    
    def detect_sat_structure(self, text: str) -> Dict:
        """Advanced SAT structure detection"""
        lines = text.split('\n')
        
        structure = {
            'passages': [],
            'questions': [],
            'math_functions': [],
            'sections': [],
            'total_questions': 0
        }
        
        current_passage = None
        current_question = None
        question_counter = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for section headers
            if self.matches_pattern(line, self.patterns['section_header']):
                structure['sections'].append({
                    'text': line,
                    'line_number': i,
                    'type': self.classify_section_type(line)
                })
                continue
            
            # Check for question numbers
            if self.matches_pattern(line, self.patterns['question_number']):
                # Save previous question if exists
                if current_question:
                    structure['questions'].append(current_question)
                    question_counter += 1
                
                # Start new question
                current_question = {
                    'number': self.extract_question_number(line),
                    'text': line,
                    'content': [],
                    'choices': [],
                    'line_number': i,
                    'type': 'unknown'
                }
                continue
            
            # Check for multiple choice options
            if self.matches_pattern(line, self.patterns['multiple_choice']):
                if current_question:
                    choice = self.parse_choice(line)
                    if choice:
                        current_question['choices'].append(choice)
                continue
            
            # Check for math functions
            if self.matches_pattern(line, self.patterns['math_function']):
                structure['math_functions'].append({
                    'text': line,
                    'line_number': i,
                    'latex': self.convert_to_latex(line)
                })
                continue
            
            # Check for passage content
            if self.matches_pattern(line, self.patterns['passage_start']):
                # Save previous passage if exists
                if current_passage:
                    structure['passages'].append(current_passage)
                
                # Start new passage
                current_passage = {
                    'text': line,
                    'content': [line],
                    'line_number': i,
                    'type': 'reading'
                }
                continue
            
            # Add content to current passage or question
            if current_passage and len(line.split()) >= 3:
                current_passage['content'].append(line)
            elif current_question and not self.matches_pattern(line, self.patterns['multiple_choice']):
                current_question['content'].append(line)
        
        # Add final passage and question
        if current_passage:
            structure['passages'].append(current_passage)
        if current_question:
            structure['questions'].append(current_question)
            question_counter += 1
        
        structure['total_questions'] = question_counter
        return structure
    
    def matches_pattern(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the patterns"""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def extract_question_number(self, text: str) -> int:
        """Extract question number from text"""
        match = re.search(r'(\d+)', text)
        return int(match.group(1)) if match else 0
    
    def parse_choice(self, text: str) -> Optional[Dict]:
        """Parse multiple choice option"""
        match = re.match(r'^([A-D])[\.\)]\s*(.+)', text)
        if match:
            return {
                'option': match.group(1),
                'text': match.group(2).strip()
            }
        return None
    
    def classify_section_type(self, text: str) -> str:
        """Classify section type"""
        text_lower = text.lower()
        if 'math' in text_lower:
            return 'math'
        elif 'reading' in text_lower:
            return 'reading'
        elif 'writing' in text_lower:
            return 'writing'
        return 'unknown'
    
    def convert_to_latex(self, text: str) -> str:
        """Convert math expressions to LaTeX"""
        latex_text = text
        
        # Function definitions
        latex_text = re.sub(r'f\(([^)]+)\)\s*=\s*([^,;]+)', r'f(\1) = \2', latex_text)
        latex_text = re.sub(r'g\(([^)]+)\)\s*=\s*([^,;]+)', r'g(\1) = \2', latex_text)
        latex_text = re.sub(r'h\(([^)]+)\)\s*=\s*([^,;]+)', r'h(\1) = \2', latex_text)
        
        # Mathematical expressions
        latex_text = re.sub(r'(\d+)x\s*([+\-])\s*(\d+)', r'\1x \2 \3', latex_text)
        latex_text = re.sub(r'x\s*([+\-])\s*(\d+)', r'x \1 \2', latex_text)
        
        # Fractions
        latex_text = re.sub(r'(\d+)/(\d+)', r'\\frac{\1}{\2}', latex_text)
        
        # Exponents
        latex_text = re.sub(r'(\w+)\^(\d+)', r'\1^{\2}', latex_text)
        
        return latex_text
    
    def format_for_word(self, structure: Dict) -> str:
        """Format structure for Word document"""
        lines = []
        
        # Process questions
        for question in structure['questions']:
            lines.append(f"Question {question['number']}")
            
            # Add question content
            for content_line in question['content']:
                if content_line.strip():
                    lines.append(content_line.strip())
            
            # Add choices
            for choice in question['choices']:
                lines.append(f"{choice['option']}. {choice['text']}")
            
            lines.append("")  # Empty line between questions
        
        return '\n'.join(lines)
    
    def process_pdf(self, pdf_path: str) -> Tuple[str, Dict]:
        """Process PDF and return formatted text and structure"""
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text
        text = self.extract_text_advanced(pdf_path)
        logger.info(f"Extracted {len(text)} characters")
        
        # Detect structure
        structure = self.detect_sat_structure(text)
        logger.info(f"Detected {len(structure['questions'])} questions, {len(structure['passages'])} passages")
        
        # Format for Word
        formatted_text = self.format_for_word(structure)
        
        return formatted_text, structure

def test_advanced_processor():
    """Test the advanced processor"""
    processor = AdvancedSATProcessor()
    
    pdf_files = [
        'real format digital sat exam with image&text watermark (Only Math).pdf',
        'real format digital sat exam with image&text watermark.pdf',
        'real sat many watermark file but still having the sat exam.pdf'
    ]
    
    for pdf_file in pdf_files:
        if Path(pdf_file).exists():
            print(f"\n=== Processing {pdf_file} ===")
            try:
                formatted_text, structure = processor.process_pdf(pdf_file)
                
                print(f"Questions detected: {len(structure['questions'])}")
                print(f"Passages detected: {len(structure['passages'])}")
                print(f"Math functions detected: {len(structure['math_functions'])}")
                print(f"Total questions: {structure['total_questions']}")
                
                # Show sample output
                lines = formatted_text.split('\n')[:20]
                print("\nSample output (first 20 lines):")
                for i, line in enumerate(lines, 1):
                    print(f"{i:2d}. {line}")
                    
            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")
        else:
            print(f"File not found: {pdf_file}")

if __name__ == "__main__":
    test_advanced_processor()


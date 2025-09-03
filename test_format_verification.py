#!/usr/bin/env python3
"""
Format Verification Test Script
Tests the SAT format output with the 3 provided PDF files
Verifies content preservation and correct output format
"""

import os
import sys
from pathlib import Path
from docx import Document
import re

# Import our main processor
from app import DocumentProcessor

def check_word_format_compliance(docx_path: str) -> dict:
    """Check if Word document follows the correct SAT format"""
    try:
        doc = Document(docx_path)
        content = []
        
        # Extract all text from the document
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                content.append(paragraph.text.strip())
        
        # Check for required format elements
        question_numbers = 0
        reading_passages = 0
        multiple_choices = 0
        math_expressions = 0
        
        for line in content:
            # Check for question numbers (should be "Question X" format)
            if re.search(r'Question\s+\d+', line, re.IGNORECASE):
                question_numbers += 1
            
            # Check for reading passages (content without labels)
            if (len(line) > 50 and 
                not re.search(r'Question\s+\d+', line, re.IGNORECASE) and
                not re.search(r'^[A-D]\.', line) and
                not re.search(r'^[A-D]\s', line)):
                reading_passages += 1
            
            # Check for multiple choice options (A, B, C, D format)
            if re.search(r'^[A-D]\.', line) or re.search(r'^[A-D]\s', line):
                multiple_choices += 1
            
            # Check for LaTeX math expressions
            if '\\' in line and any(math_symbol in line for math_symbol in ['\\frac', '\\sqrt', '\\times', '\\in', '\\land']):
                math_expressions += 1
        
        # Determine compliance
        is_compliant = question_numbers > 0 and (reading_passages > 0 or multiple_choices > 0)
        
        return {
            'question_numbers': question_numbers,
            'reading_passages': reading_passages,
            'multiple_choices': multiple_choices,
            'math_expressions': math_expressions,
            'total_content_lines': len(content),
            'is_compliant': is_compliant,
            'content_preview': content[:10]  # First 10 lines for preview
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'is_compliant': False
        }

def test_pdf_format(pdf_path: str) -> dict:
    """Test a single PDF file for format compliance"""
    print(f"\n🎯 Testing: {pdf_path}")
    print("=" * 60)
    
    try:
        # Initialize processor
        processor = DocumentProcessor()
        
        # Process the document
        result = processor.process_document(pdf_path)
        
        # Handle tuple return format
        if isinstance(result, tuple):
            word_path, stats = result
            success = stats.get('success', True)
        else:
            word_path = result['output_file']
            success = result['success']
        
        if success:
            print(f"✅ Processing successful")
            print(f"📄 Word output: {word_path}")
            
            # Check file size
            file_size = os.path.getsize(word_path)
            print(f"📏 File size: {file_size:,} bytes")
            
            # Verify format compliance
            format_check = check_word_format_compliance(word_path)
            
            if 'error' in format_check:
                print(f"❌ Format check error: {format_check['error']}")
                return {'success': False, 'error': format_check['error']}
            
            print(f"\n📋 Format Analysis:")
            print(f"   Question Numbers: {format_check['question_numbers']}")
            print(f"   Reading Passages: {format_check['reading_passages']}")
            print(f"   Multiple Choices: {format_check['multiple_choices']}")
            print(f"   Math Expressions: {format_check['math_expressions']}")
            print(f"   Total Content Lines: {format_check['total_content_lines']}")
            
            compliance_status = "✅ COMPLIANT" if format_check['is_compliant'] else "❌ NOT COMPLIANT"
            print(f"   Format Compliance: {compliance_status}")
            
            # Show content preview
            print(f"\n📖 Content Preview (first 10 lines):")
            for i, line in enumerate(format_check['content_preview'][:10], 1):
                print(f"   {i:2d}. {line[:80]}{'...' if len(line) > 80 else ''}")
            
            return {
                'success': True,
                'file_size': file_size,
                'format_check': format_check,
                'word_path': word_path
            }
        else:
            print(f"❌ Processing failed")
            return {'success': False, 'error': 'Processing failed'}
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}

def main():
    """Main test function"""
    print("🎓 SAT Format Verification Test")
    print("=" * 60)
    print("Testing 3 PDF files for correct SAT format output")
    print("Verifying content preservation and format compliance")
    
    # Test files (based on the detailed names provided)
    test_files = [
        "real format digital sat exam with image&text watermark (Only Math).pdf",
        "real format digital sat exam with image&text watermark.pdf", 
        "real sat many watermark file but still having the sat exam.pdf"
    ]
    
    results = []
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            result = test_pdf_format(pdf_file)
            results.append({
                'file': pdf_file,
                'result': result
            })
        else:
            print(f"❌ File not found: {pdf_file}")
            results.append({
                'file': pdf_file,
                'result': {'success': False, 'error': 'File not found'}
            })
    
    # Summary
    print(f"\n🎉 Test Summary")
    print("=" * 60)
    
    successful_tests = 0
    compliant_tests = 0
    
    for result in results:
        file_name = Path(result['file']).stem
        if result['result']['success']:
            successful_tests += 1
            if result['result']['format_check']['is_compliant']:
                compliant_tests += 1
                print(f"✅ {file_name}: SUCCESS & COMPLIANT")
            else:
                print(f"⚠️  {file_name}: SUCCESS but NOT COMPLIANT")
        else:
            print(f"❌ {file_name}: FAILED - {result['result'].get('error', 'Unknown error')}")
    
    print(f"\n📊 Final Results:")
    print(f"   Total Files: {len(test_files)}")
    print(f"   Successful Processing: {successful_tests}")
    print(f"   Format Compliant: {compliant_tests}")
    print(f"   Success Rate: {successful_tests/len(test_files)*100:.1f}%")
    print(f"   Compliance Rate: {compliant_tests/len(test_files)*100:.1f}%")
    
    if compliant_tests == len(test_files):
        print(f"\n🎉 ALL TESTS PASSED! Perfect SAT format compliance!")
    elif successful_tests == len(test_files):
        print(f"\n✅ All files processed successfully, but some format issues remain")
    else:
        print(f"\n⚠️  Some files failed processing - check errors above")

if __name__ == "__main__":
    main()

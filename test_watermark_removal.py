#!/usr/bin/env python3
"""
Test script for PDF watermark removal
Tests the functionality with the PDF file in the workspace
"""

import os
import sys
import time
from pathlib import Path

def test_watermark_removal():
    """Test watermark removal with the PDF file in workspace"""
    print("🧪 Testing PDF Watermark Removal")
    print("=" * 50)
    
    # Check if PDF file exists
    pdf_file = "Vietaccepted Test 68.pdf"
    if not os.path.exists(pdf_file):
        print(f"❌ PDF file '{pdf_file}' not found in workspace")
        return False
    
    print(f"✅ Found PDF file: {pdf_file}")
    print(f"📁 File size: {os.path.getsize(pdf_file) / (1024*1024):.2f} MB")
    
    try:
        # Import the watermark remover
        from app import SimpleWatermarkRemover
        
        print("✅ Successfully imported SimpleWatermarkRemover")
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize watermark remover
        remover = SimpleWatermarkRemover()
        print("✅ Watermark remover initialized")
        
        # Test watermark removal
        input_path = pdf_file
        output_path = os.path.join(output_dir, "cleaned_test.pdf")
        
        print(f"🚀 Starting watermark removal...")
        print(f"📥 Input: {input_path}")
        print(f"📤 Output: {output_path}")
        
        start_time = time.time()
        
        # Perform watermark removal
        success = remover.remove_watermarks_from_pdf(input_path, output_path)
        
        processing_time = time.time() - start_time
        
        if success:
            print(f"✅ Watermark removal completed successfully!")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"📁 Output file: {output_path}")
            
            # Check output file
            if os.path.exists(output_path):
                output_size = os.path.getsize(output_path) / (1024*1024)
                print(f"📊 Output file size: {output_size:.2f} MB")
                
                # Test Word conversion
                print("\n📝 Testing Word document conversion...")
                try:
                    from app import DocumentProcessor
                    
                    processor = DocumentProcessor()
                    word_path, metadata = processor.process_document(output_path)
                    
                    print(f"✅ Word conversion successful!")
                    print(f"📄 Word file: {word_path}")
                    print(f"📊 Metadata: {metadata}")
                    
                    # Verify Word file can be opened
                    try:
                        from docx import Document
                        test_doc = Document(word_path)
                        # Document objects don't have close() method, just verify it loaded
                        print("✅ Word file verification successful - file can be opened!")
                        
                        return True
                        
                    except Exception as e:
                        print(f"❌ Word file verification failed: {e}")
                        return False
                        
                except Exception as e:
                    print(f"❌ Word conversion failed: {e}")
                    return False
            else:
                print("❌ Output file not found")
                return False
        else:
            print("❌ Watermark removal failed")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 PDF Watermark Remover - Test Suite")
    print("=" * 50)
    
    # Test watermark removal
    success = test_watermark_removal()
    
    if success:
        print("\n🎉 All tests passed! The watermark remover is working correctly.")
        print("✅ Watermark removal: Working")
        print("✅ Word conversion: Working")
        print("✅ File verification: Working")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

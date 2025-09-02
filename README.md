# PDF/Image to Word Converter

A modern web application that converts PDF documents and images to editable text with advanced features including OCR (Optical Character Recognition), mathematical expression recognition, and watermark removal.

## ✨ Features

- **Multi-format Support**: Handles PDF, JPG, PNG, TIFF, and BMP files
- **Drag & Drop Interface**: Intuitive file upload with visual feedback
- **OCR Processing**: Extracts text from images and scanned documents
- **Math Recognition**: Converts mathematical expressions to LaTeX format
- **Watermark Removal**: Automatically detects and removes watermarks
- **Batch Processing**: Process multiple files simultaneously
- **Progress Tracking**: Real-time progress updates during conversion
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Live Demo

The application is designed to be hosted on GitHub Pages. Once deployed, users can:

1. Drag and drop files or click to browse
2. View selected files with size information
3. Monitor real-time processing progress
4. Download converted documents as formatted HTML files

## 🛠️ Technical Implementation

### Frontend Technologies
- **HTML5**: Semantic markup and structure
- **CSS3**: Modern styling with animations and responsive design
- **Vanilla JavaScript**: No framework dependencies for maximum compatibility

### External Libraries
- **PDF-lib**: PDF document processing and manipulation
- **Tesseract.js**: OCR engine for text extraction
- **KaTeX**: Mathematical expression rendering

### Key Functions

#### File Processing Pipeline
1. **File Input**: Supports drag & drop and file browser selection
2. **PDF Conversion**: Converts PDF pages to images for OCR processing
3. **Image Enhancement**: Removes watermarks and improves contrast
4. **Text Extraction**: Performs OCR to extract readable text
5. **Math Conversion**: Transforms mathematical expressions to LaTeX
6. **Output Generation**: Creates formatted HTML documents

#### Watermark Removal Algorithm
- Detects semi-transparent overlays
- Removes light watermark text
- Enhances contrast for better OCR accuracy
- Processes pixel data for optimal results

#### Mathematical Expression Recognition
- Converts fractions (a/b → \\frac{a}{b})
- Handles exponents (x^2 → x^{2})
- Processes square roots (sqrt(x) → \\sqrt{x})
- Supports Greek letters and mathematical symbols
- Generates LaTeX-compatible output

## 📁 File Structure

```
convert_tool/
├── index.html          # Main HTML file
├── styles.css          # CSS styling
├── script.js           # JavaScript functionality
└── README.md           # This documentation
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement include:

- Enhanced PDF processing capabilities
- Advanced watermark removal algorithms
- Extended mathematical expression support
- Performance optimizations
- Additional file format support

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the browser console for error messages
2. Ensure all external libraries are loading correctly
3. Verify file format compatibility
4. Check browser compatibility requirements

---

**Note**: This application is designed for educational and demonstration purposes. For production use, consider implementing server-side processing, user authentication, and enhanced error handling.

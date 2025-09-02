# PDF/Image to Word Converter

A modern web application that converts PDF documents and images to editable text with advanced features including OCR (Optical Character Recognition), mathematical expression recognition, and watermark removal. Now featuring a **two-section interface** for specialized processing.

## ✨ Features

### 🎯 **Two-Section Processing:**
- **🇺🇸 English Section**: Optimized for text documents with advanced watermark removal
- **🧮 Math Section**: Specialized for mathematical formulas and equations with KaTeX support

### 🌍 **Multi-Language Support:**
- **English Section Languages**: English, French, German, Spanish, Italian, Portuguese, Russian, Chinese, Japanese, Korean
- **Math Section Languages**: English, French, German, Spanish, Chinese (with English fallback for math recognition)
- **Intelligent Language Selection**: Automatically uses the best language for optimal OCR accuracy

### 🔧 **Core Functionality:**
- **Multi-format Support**: Handles PDF, JPG, PNG, TIFF, and BMP files
- **Drag & Drop Interface**: Intuitive file upload with visual feedback
- **Advanced OCR Processing**: Multi-language text extraction with confidence scoring
- **Enhanced Watermark Removal**: Sophisticated algorithm for various watermark types
- **Batch Processing**: Process multiple files simultaneously
- **Progress Tracking**: Real-time progress updates during conversion
- **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🚀 Live Demo

The application is designed to be hosted on GitHub Pages. Once deployed, users can:

1. **Choose Processing Section**: Select between English or Math processing
2. **Select Language**: Choose the primary language for OCR
3. **Drag and drop files** or click to browse
4. **View selected files** with size information
5. **Monitor real-time processing** progress
6. **Download converted documents** as Word (.docx) files

## 🛠️ Technical Implementation

### Frontend Technologies
- **HTML5**: Semantic markup with section-based interface
- **CSS3**: Modern styling with responsive design and smooth animations
- **Vanilla JavaScript**: No framework dependencies for maximum compatibility

### External Libraries
- **PDF-lib**: PDF document processing and manipulation
- **Tesseract.js**: Multi-language OCR engine with confidence scoring
- **KaTeX**: Mathematical expression rendering and processing

### Key Functions

#### **Two-Section Processing Pipeline**
1. **Section Selection**: User chooses between English or Math processing
2. **Language Configuration**: Selects primary language for OCR
3. **File Input**: Supports drag & drop and file browser selection
4. **PDF Conversion**: Converts PDF pages to images for OCR processing
5. **Advanced Watermark Removal**: Section-specific watermark detection algorithms
6. **Multi-Language OCR**: Performs OCR with selected language + fallback
7. **Content Processing**: Section-specific text processing and math conversion
8. **Output Generation**: Creates formatted Word documents with proper formatting

#### **English Section Features**
- **Text Preservation**: Maintains document structure and formatting
- **OCR Artifact Cleanup**: Fixes common recognition mistakes
- **Paragraph Structure**: Preserves text layout and organization
- **Advanced Watermark Removal**: Optimized for text documents

#### **Math Section Features**
- **Mathematical Expression Recognition**: Converts math to LaTeX format
- **Enhanced Math Patterns**: Fractions, exponents, roots, Greek letters, operators
- **Matrix Notation**: Supports matrix and vector representations
- **Function Notation**: Proper mathematical function formatting
- **KaTeX Integration**: Generates LaTeX-compatible output

#### **Enhanced Watermark Removal Algorithm**
- **Multi-Type Detection**: Semi-transparent, light, colored watermarks
- **Intelligent Removal**: Preserves content while removing artifacts
- **Contrast Enhancement**: Improves OCR accuracy
- **Section Optimization**: Different algorithms for text vs. math documents

#### **Multi-Language OCR System**
- **Primary Language**: User-selected language for OCR
- **Fallback Languages**: Automatic fallback for better accuracy
- **Confidence Scoring**: Uses highest-confidence results
- **Language-Specific Optimization**: Tailored processing per language

## 📁 File Structure

```
convert_tool/
├── index.html          # Main HTML with two-section interface
├── styles.css          # CSS styling with responsive design
├── script.js           # JavaScript with enhanced functionality
├── README.md           # This documentation
├── DEPLOY.md           # Deployment guide
├── test.html           # Testing page
└── LICENSE             # MIT License
```

## 🎨 Interface Design

### **Section Tabs**
- **Visual Indicators**: Clear icons and labels for each section
- **Active States**: Highlighted active section with smooth transitions
- **Responsive Layout**: Adapts to mobile and desktop screens

### **Language Selection**
- **Dropdown Menus**: Easy language selection for each section
- **Smart Defaults**: Optimized language choices per section
- **Visual Feedback**: Clear indication of selected language

### **Processing Interface**
- **Progress Visualization**: Real-time progress bars and status updates
- **File Management**: Clear file selection and processing status
- **Success/Error Handling**: Comprehensive feedback and error messages

## 🌍 Supported Languages

### **English Section**
- **European Languages**: English, French, German, Spanish, Italian, Portuguese
- **Asian Languages**: Chinese Simplified, Japanese, Korean
- **Other Languages**: Russian

### **Math Section**
- **Primary Languages**: English, French, German, Spanish, Chinese
- **Math Recognition**: Enhanced mathematical expression processing
- **Fallback Support**: English fallback for optimal math recognition

## 📊 Performance Features

- **Confidence Scoring**: Uses highest-confidence OCR results
- **Multi-Language Processing**: Parallel language processing for accuracy
- **Optimized Algorithms**: Section-specific processing for better results
- **Memory Management**: Efficient handling of large files and batches

## 🤝 Contributing

Contributions are welcome! Areas for improvement include:

- **Enhanced PDF Processing**: Real PDF-to-image conversion
- **Advanced Math Recognition**: Machine learning-based math detection
- **Additional Languages**: Support for more OCR languages
- **Performance Optimization**: Faster processing algorithms
- **UI/UX Improvements**: Better user experience and accessibility

## 📄 License

This project is open source and available under the MIT License.

## 🆘 Support

For issues or questions:
1. Check the browser console for error messages
2. Ensure all external libraries are loading correctly
3. Verify file format compatibility
4. Check browser compatibility requirements
5. Test with different languages and document types

## 🚀 Deployment

### **GitHub Pages Setup**
1. Create a new GitHub repository
2. Upload all project files
3. Enable GitHub Pages in repository settings
4. Your app will be live at `https://[username].github.io/[repository-name]`

### **Local Testing**
1. Open `index.html` in your browser
2. Test both English and Math sections
3. Try different languages and file types
4. Verify watermark removal and OCR accuracy

---

**Note**: This application is designed for educational and demonstration purposes. For production use, consider implementing server-side processing, user authentication, and enhanced error handling.

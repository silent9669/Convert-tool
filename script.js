let selectedFiles = [];
let processedContent = '';

// Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const selectedFilesDiv = document.getElementById('selectedFiles');
const fileList = document.getElementById('fileList');
const progressSection = document.getElementById('progressSection');
const progressFill = document.getElementById('progressFill');
const statusText = document.getElementById('statusText');
const successMessage = document.getElementById('successMessage');
const errorMessage = document.getElementById('errorMessage');
const downloadBtn = document.getElementById('downloadBtn');

// Event listeners
dropZone.addEventListener('click', () => fileInput.click());
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

function handleDrop(e) {
    dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    handleFiles(files);
}

function handleFiles(files) {
    selectedFiles = files;
    displaySelectedFiles();
    processFiles();
}

function displaySelectedFiles() {
    fileList.innerHTML = '';
    selectedFiles.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);
        
        fileItem.innerHTML = `
            <span class="file-icon">${getFileIcon(file.type)}</span>
            <span class="file-name">${file.name}</span>
            <span class="file-size">${fileSize} MB</span>
        `;
        
        fileList.appendChild(fileItem);
    });
    
    selectedFilesDiv.style.display = 'block';
}

function getFileIcon(fileType) {
    if (fileType === 'application/pdf') return '📄';
    if (fileType.startsWith('image/')) return '🖼️';
    return '📁';
}

async function processFiles() {
    hideMessages();
    progressSection.style.display = 'block';
    dropZone.classList.add('processing');

    try {
        let allContent = '';
        
        for (let i = 0; i < selectedFiles.length; i++) {
            const file = selectedFiles[i];
            const progress = ((i + 1) / selectedFiles.length) * 100;
            
            updateProgress(progress * 0.3, `Processing ${file.name}...`);
            
            const content = await processFile(file);
            allContent += `\n\n--- ${file.name} ---\n\n${content}`;
        }
        
        processedContent = allContent;
        showSuccess();
        
    } catch (error) {
        console.error('Processing error:', error);
        showError(error.message);
    } finally {
        dropZone.classList.remove('processing');
    }
}

async function processFile(file) {
    let imageData;
    
    // Convert file to image data
    if (file.type === 'application/pdf') {
        updateProgress(null, 'Converting PDF...');
        imageData = await processPDF(file);
    } else {
        imageData = await fileToImageData(file);
    }
    
    // Remove watermarks
    updateProgress(null, 'Removing watermarks...');
    const cleanImage = await removeWatermark(imageData);
    
    // Perform OCR
    updateProgress(null, 'Extracting text...');
    const ocrText = await performOCR(cleanImage);
    
    // Convert math expressions
    updateProgress(null, 'Converting math expressions...');
    const finalText = convertMathToKaTeX(ocrText);
    
    return finalText;
}

async function processPDF(file) {
    try {
        const arrayBuffer = await file.arrayBuffer();
        
        // Load PDF using PDF-lib
        const pdfDoc = await PDFLib.PDFDocument.load(arrayBuffer);
        const pages = pdfDoc.getPages();
        
        if (pages.length === 0) {
            throw new Error('PDF has no pages');
        }
        
        // For demo purposes, we'll create a placeholder image
        // In a production environment, you'd use pdf2pic or similar library
        // to convert PDF pages to images
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 800;
        canvas.height = 1000;
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'black';
        ctx.font = '16px Arial';
        ctx.fillText(`PDF Document - ${pages.length} page(s)`, 50, 50);
        ctx.fillText('Use pdf2pic library for real PDF to image conversion', 50, 80);
        ctx.fillText('This demo shows the interface and workflow', 50, 110);
        
        return canvas.toDataURL();
    } catch (error) {
        console.error('PDF processing error:', error);
        // Fallback to placeholder
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 800;
        canvas.height = 1000;
        
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'black';
        ctx.font = '16px Arial';
        ctx.fillText('PDF Content - Use pdf2pic library for real conversion', 50, 50);
        
        return canvas.toDataURL();
    }
}

function fileToImageData(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = (e) => resolve(e.target.result);
        reader.readAsDataURL(file);
    });
}

async function removeWatermark(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = img.width;
            canvas.height = img.height;
            
            ctx.drawImage(img, 0, 0);
            
            // Enhanced watermark removal algorithm
            const imageDataObj = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageDataObj.data;
            
            for (let i = 0; i < data.length; i += 4) {
                const alpha = data[i + 3];
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                
                // Remove semi-transparent watermarks (common in scanned documents)
                if (alpha < 200 && alpha > 30) {
                    data[i + 3] = 0;
                }
                
                // Remove light watermark text (very light gray/white text)
                if (r > 200 && g > 200 && b > 200 && alpha > 100) {
                    data[i] = 255;
                    data[i + 1] = 255;
                    data[i + 2] = 255;
                    data[i + 3] = 255;
                }
                
                // Remove medium-light watermarks
                if (r > 180 && g > 180 && b > 180 && alpha > 80) {
                    data[i] = 255;
                    data[i + 1] = 255;
                    data[i + 2] = 255;
                    data[i + 3] = 255;
                }
                
                // Enhance contrast for better OCR
                if (r < 150 && g < 150 && b < 150) {
                    data[i] = 0;
                    data[i + 1] = 0;
                    data[i + 2] = 0;
                    data[i + 3] = 255;
                }
                
                // Remove colored watermarks (red, blue, green tints)
                if (Math.abs(r - g) > 50 || Math.abs(r - b) > 50 || Math.abs(g - b) > 50) {
                    if (alpha < 180) {
                        data[i + 3] = 0;
                    }
                }
            }
            
            ctx.putImageData(imageDataObj, 0, 0);
            resolve(canvas.toDataURL());
        };
        img.src = imageData;
    });
}

async function performOCR(imageData) {
    try {
        const result = await Tesseract.recognize(imageData, 'eng', {
            logger: m => {
                if (m.status === 'recognizing text') {
                    const ocrProgress = Math.round(m.progress * 100);
                    updateProgress(50 + (ocrProgress * 0.4), `OCR: ${ocrProgress}%`);
                }
            }
        });
        
        return result.data.text;
    } catch (error) {
        throw new Error('OCR failed: ' + error.message);
    }
}

function convertMathToKaTeX(text) {
    let processedText = text;
    
    // Math conversion patterns for common mathematical expressions
    const conversions = [
        // Fractions
        [/(\w+)\/(\w+)/g, '\\frac{$1}{$2}'],
        [/(\d+)\/(\d+)/g, '\\frac{$1}{$2}'],
        
        // Exponents
        [/(\w+)\^(\d+)/g, '$1^{$2}'],
        [/(\w+)\^(\w+)/g, '$1^{$2}'],
        
        // Square roots
        [/sqrt\(([^)]+)\)/g, '\\sqrt{$1}'],
        [/√\(([^)]+)\)/g, '\\sqrt{$1}'],
        
        // Greek letters
        [/alpha/gi, '\\alpha'],
        [/beta/gi, '\\beta'],
        [/gamma/gi, '\\gamma'],
        [/delta/gi, '\\delta'],
        [/epsilon/gi, '\\epsilon'],
        [/theta/gi, '\\theta'],
        [/lambda/gi, '\\lambda'],
        [/mu/gi, '\\mu'],
        [/pi/gi, '\\pi'],
        [/sigma/gi, '\\sigma'],
        [/phi/gi, '\\phi'],
        [/omega/gi, '\\omega'],
        
        // Mathematical operators
        [/integral/gi, '\\int'],
        [/sum/gi, '\\sum'],
        [/product/gi, '\\prod'],
        [/infinity/gi, '\\infty'],
        [/partial/gi, '\\partial'],
        [/nabla/gi, '\\nabla'],
        
        // Common math symbols
        [/<=/g, '\\leq'],
        [/>=/g, '\\geq'],
        [/!=/g, '\\neq'],
        [/->/g, '\\rightarrow'],
        [/<-/g, '\\leftarrow'],
        [/<->/g, '\\leftrightarrow'],
        
        // Subscripts
        [/(\w+)_(\w+)/g, '$1_{$2}'],
        [/(\w+)_(\d+)/g, '$1_{$2}'],
    ];
    
    conversions.forEach(([pattern, replacement]) => {
        processedText = processedText.replace(pattern, replacement);
    });
    
    // Wrap math expressions in KaTeX delimiters
    processedText = processedText.replace(/\\[a-zA-Z]+(\{[^}]*\})?/g, '$$$&$$');
    
    return processedText;
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
    // Create Word document content using DOCX format
    const content = createWordDocument(processedContent);
    
    // Create and download the file
    const blob = new Blob([content], { 
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'converted_document.docx';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function createWordDocument(content) {
    // Create a simple Word document structure
    // This is a basic implementation - for production use, consider using a library like docx.js
    
    const sections = content.split('---').map((section, index) => {
        if (index === 0) return section;
        const parts = section.split('\n');
        const fileName = parts[0].trim();
        const textContent = parts.slice(1).join('\n').trim();
        
        if (fileName && textContent) {
            return `\n\n${fileName}\n${'='.repeat(fileName.length)}\n\n${textContent}`;
        }
        return '';
    }).join('');
    
    // Convert to Word-compatible format
    // For now, we'll create a rich text format that Word can open
    const wordContent = `{\\rtf1\\ansi\\deff0 {\\fonttbl {\\f0 Times New Roman;}}
\\f0\\fs24 ${sections.replace(/\n/g, '\\par ').replace(/\t/g, '\\tab ')}
}`;
    
    return wordContent;
}

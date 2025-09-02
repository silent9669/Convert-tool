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
            
            // Simple watermark removal algorithm
            const imageDataObj = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const data = imageDataObj.data;
            
            for (let i = 0; i < data.length; i += 4) {
                const alpha = data[i + 3];
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                
                // Remove semi-transparent watermarks
                if (alpha < 200 && alpha > 50) {
                    data[i + 3] = 0;
                }
                
                // Remove light watermark text (common in scanned documents)
                if (r > 180 && g > 180 && b > 180) {
                    data[i] = 255;
                    data[i + 1] = 255;
                    data[i + 2] = 255;
                }
                
                // Enhance contrast for better OCR
                if (r < 128 && g < 128 && b < 128) {
                    data[i] = 0;
                    data[i + 1] = 0;
                    data[i + 2] = 0;
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
    // Create a Word-like document using HTML with proper formatting
    const content = `
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Converted Document</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/katex/0.16.8/katex.min.css">
            <style>
                body { 
                    font-family: 'Times New Roman', serif; 
                    padding: 40px; 
                    line-height: 1.6; 
                    max-width: 800px;
                    margin: 0 auto;
                    background: white;
                }
                h1 { 
                    color: #333; 
                    border-bottom: 2px solid #4CAF50; 
                    padding-bottom: 10px; 
                    margin-bottom: 30px;
                }
                pre { 
                    background: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px; 
                    white-space: pre-wrap; 
                    border-left: 4px solid #4CAF50;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    line-height: 1.5;
                }
                .math { 
                    text-align: center; 
                    margin: 20px 0; 
                    padding: 10px;
                    background: #f0f8ff;
                    border-radius: 5px;
                }
                .file-separator {
                    border-top: 2px solid #e0e0e0;
                    margin: 30px 0;
                    padding-top: 20px;
                }
                .file-header {
                    background: #4CAF50;
                    color: white;
                    padding: 10px 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    font-weight: bold;
                }
            </style>
        </head>
        <body>
            <h1>📄 Converted Document</h1>
            <div class="content">
                ${processedContent.split('---').map((section, index) => {
                    if (index === 0) return section;
                    const parts = section.split('\n');
                    const fileName = parts[0].trim();
                    const content = parts.slice(1).join('\n').trim();
                    
                    if (fileName && content) {
                        return `
                            <div class="file-separator">
                                <div class="file-header">📁 ${fileName}</div>
                                <pre>${content}</pre>
                            </div>
                        `;
                    }
                    return '';
                }).join('')}
            </div>
        </body>
        </html>
    `;
    
    const blob = new Blob([content], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'converted_document.html';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

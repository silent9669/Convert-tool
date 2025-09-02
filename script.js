let selectedFiles = [];
let processedContent = '';
let currentSection = 'english';

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

// Tab elements
const tabBtns = document.querySelectorAll('.tab-btn');
const englishSection = document.getElementById('englishSection');
const mathSection = document.getElementById('mathSection');

// Language selectors
const englishLanguage = document.getElementById('englishLanguage');
const mathLanguage = document.getElementById('mathLanguage');

// Event listeners
dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
downloadBtn.addEventListener('click', downloadResult);

// Tab switching
tabBtns.forEach(btn => {
    btn.addEventListener('click', () => switchSection(btn.dataset.section));
});

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

function switchSection(section) {
    currentSection = section;
    
    // Update tab buttons
    tabBtns.forEach(btn => {
        btn.classList.toggle('active', btn.dataset.section === section);
    });
    
    // Update section content
    englishSection.classList.toggle('active', section === 'english');
    mathSection.classList.toggle('active', section === 'math');
    
    // Reset UI
    hideMessages();
    selectedFilesDiv.style.display = 'none';
    progressSection.style.display = 'none';
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
    
    // Remove watermarks based on section
    updateProgress(null, 'Removing watermarks...');
    const cleanImage = await removeWatermark(imageData);
    
    // Perform OCR with selected language
    updateProgress(null, 'Extracting text...');
    const ocrText = await performOCR(cleanImage);
    
    // Process content based on section
    updateProgress(null, 'Processing content...');
    let finalText;
    
    if (currentSection === 'math') {
        finalText = processMathContent(ocrText);
    } else {
        finalText = processEnglishContent(ocrText);
    }
    
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
        // Get selected language based on current section
        const language = currentSection === 'math' ? mathLanguage.value : englishLanguage.value;
        
        // Use multiple languages for better accuracy
        const languages = [language];
        
        // Add English as fallback for better math recognition
        if (currentSection === 'math' && language !== 'eng') {
            languages.push('eng');
        }
        
        let bestResult = '';
        let bestConfidence = 0;
        
        for (const lang of languages) {
            try {
                const result = await Tesseract.recognize(imageData, lang, {
                    logger: m => {
                        if (m.status === 'recognizing text') {
                            const ocrProgress = Math.round(m.progress * 100);
                            updateProgress(50 + (ocrProgress * 0.4), `OCR (${lang}): ${ocrProgress}%`);
                        }
                    }
                });
                
                // Use result with highest confidence
                if (result.data.confidence > bestConfidence) {
                    bestResult = result.data.text;
                    bestConfidence = result.data.confidence;
                }
            } catch (error) {
                console.warn(`OCR failed for language ${lang}:`, error);
            }
        }
        
        if (!bestResult) {
            throw new Error('OCR failed for all languages');
        }
        
        return bestResult;
    } catch (error) {
        throw new Error('OCR failed: ' + error.message);
    }
}

function processEnglishContent(text) {
    // For English section, focus on text preservation and formatting
    let processedText = text;
    
    // Clean up common OCR artifacts
    processedText = processedText
        .replace(/\|\|/g, 'll')  // Common OCR mistake
        .replace(/\|\//g, 'll')  // Another common mistake
        .replace(/[0O](\d)/g, '0$1')  // Fix number recognition
        .replace(/(\d)[0O]/g, '$10');  // Fix number recognition
    
    // Preserve paragraph structure
    processedText = processedText
        .replace(/\n{3,}/g, '\n\n')  // Normalize multiple newlines
        .trim();
    
    return processedText;
}

function processMathContent(text) {
    // For Math section, focus on mathematical expression recognition
    let processedText = text;
    
    // Enhanced math conversion patterns
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
        
        // Matrix notation
        [/\[([^\]]+)\]/g, '\\begin{bmatrix} $1 \\end{bmatrix}'],
        
        // Function notation
        [/(\w+)\(([^)]+)\)/g, '\\text{$1}($2)'],
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
    // Create Word document content
    const content = createWordDocument(processedContent);
    
    // Create and download the file
    const blob = new Blob([content], { 
        type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document' 
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `converted_document_${currentSection}.docx`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

function createWordDocument(content) {
    // Create a more sophisticated Word document structure
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
    
    // Create RTF content with better formatting
    let rtfContent = '{\\rtf1\\ansi\\deff0 {\\fonttbl {\\f0 Times New Roman;}}';
    
    // Add section-specific formatting
    if (currentSection === 'math') {
        rtfContent += '\\f0\\fs24\\b Math Document\\b0\\par\\par';
    } else {
        rtfContent += '\\f0\\fs24\\b Text Document\\b0\\par\\par';
    }
    
    // Process content with proper formatting
    const formattedContent = sections
        .replace(/\n/g, '\\par ')
        .replace(/\t/g, '\\tab ')
        .replace(/\$\$([^$]+)\$\$/g, '\\i $1\\i0 '); // Format math expressions
    
    rtfContent += formattedContent + '}';
    
    return rtfContent;
}

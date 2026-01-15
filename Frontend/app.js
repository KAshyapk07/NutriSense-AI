// ==================== GLOBALS ====================
const chatMessages = document.getElementById('chat-messages');
const userQuery = document.getElementById('user-query');
const sendBtn = document.querySelector('.send-btn');
const imageFile = document.getElementById('image-file');
const filePreview = document.getElementById('file-preview');

let selectedImage = null;
let isProcessing = false;

// ==================== UTILITY FUNCTIONS ====================
function getTime() {
    return new Date().toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit', 
        hour12: true 
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== IMAGE HANDLING ====================
function handleImageSelect() {
    if (imageFile.files.length > 0) {
        selectedImage = imageFile.files[0];
        filePreview.innerHTML = `
            <span class="file-preview-text">${escapeHtml(selectedImage.name)}</span>
            <span class="file-preview-close" onclick="clearImage()">×</span>
        `;
        filePreview.style.display = 'flex';
        userQuery.focus();
    }
}

function clearImage() {
    selectedImage = null;
    imageFile.value = '';
    filePreview.style.display = 'none';
}

// ==================== MESSAGE RENDERING ====================
function renderMessage(text, isUser = false) {
    const message = document.createElement('div');
    message.className = `message ${isUser ? 'user' : 'ai'}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = text;

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = getTime();

    message.appendChild(bubble);
    message.appendChild(time);
    chatMessages.appendChild(message);

    // Remove welcome area if it exists
    const welcomeArea = document.querySelector('.welcome-area');
    if (welcomeArea) {
        welcomeArea.style.animation = 'fadeOut 0.3s ease-out';
        setTimeout(() => welcomeArea.remove(), 300);
    }

    // Smooth scroll to bottom
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

function renderLoading() {
    const message = document.createElement('div');
    message.className = 'message ai';
    message.id = 'loading-message';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = `
        <div style="display: flex; gap: 12px; align-items: center;">
            <span style="color: var(--text-light); font-weight: 500;">Analyzing</span>
            <div class="loader"><span></span><span></span><span></span></div>
        </div>
    `;

    message.appendChild(bubble);
    chatMessages.appendChild(message);
    
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
    
    return message;
}

// ==================== FORMATTING ====================
function createTable(nutrition) {
    let html = '<table class="nutrition-table"><tbody>';
    for (let key in nutrition) {
        html += `<tr><td>${escapeHtml(key)}</td><td>${escapeHtml(nutrition[key])}</td></tr>`;
    }
    html += '</tbody></table>';
    return html;
}

function formatLLMText(text) {
    if (!text) return '';
    
    // Replace **text** with section headers
    text = text.replace(/\*\*(.+?)\*\*/g, '<div class="section-header">$1</div>');
    
    // Split into lines
    const lines = text.split('\n').map(l => l.trim()).filter(l => l);
    let html = '';
    let inList = false;
    
    for (let line of lines) {
        // Check if it's already a section header
        if (line.includes('class="section-header"')) {
            if (inList) {
                html += '</ul>';
                inList = false;
            }
            html += line;
            continue;
        }
        
        // Check if it's a bullet point
        if (line.match(/^[-•*]\s/)) {
            if (!inList) {
                html += '<ul>';
                inList = true;
            }
            html += `<li>${line.replace(/^[-•*]\s/, '')}</li>`;
        } else {
            if (inList) {
                html += '</ul>';
                inList = false;
            }
            
            // Check for warnings/disclaimers
            if (line.toLowerCase().includes('disclaimer') || 
                line.toLowerCase().includes('these are approximate') ||
                line.toLowerCase().includes('⚠')) {
                html += `<div class="warning-box">${line}</div>`;
            } else if (line.toLowerCase().includes('note:') || line.toLowerCase().includes('ℹ')) {
                html += `<div class="info-box">${line}</div>`;
            } else {
                html += `<p>${line}</p>`;
            }
        }
    }
    
    if (inList) html += '</ul>';
    
    return html;
}

function formatResponse(data) {
    console.log('Formatting response:', data);

    // Error handling
    if (data.error) {
        return `<strong style="color: var(--error);">Error</strong><p>${escapeHtml(data.error)}</p>`;
    }

    // COMPARISON PATHWAY
    if (data.dish_a && data.dish_b && data.nutrition_a && data.nutrition_b) {
        let html = '';
        
        // Show LLM analysis first
        if (data.llm_response) {
            html += formatLLMText(data.llm_response);
        }
        
        // Show comparison table
        html += `
            <div class="comparison-grid">
                <div class="nutrition-card">
                    <h4>${escapeHtml(data.dish_a)}</h4>
                    ${createTable(data.nutrition_a)}
                </div>
                <div class="nutrition-card">
                    <h4>${escapeHtml(data.dish_b)}</h4>
                    ${createTable(data.nutrition_b)}
                </div>
            </div>
        `;
        
        // Badge
        if (data.estimated) {
            html += `<span class="estimated-badge">Estimated Values</span>`;
        } else if (data.accuracy) {
            html += `<span class="confidence-badge">${Math.round(data.accuracy)}% Confidence</span>`;
        }
        
        return html;
    }

    // EXTRACTION/MODIFICATION PATHWAY
    if (data.recipe_name && data.nutrition) {
        let html = `<strong>${escapeHtml(data.recipe_name)}</strong>`;
        
        // Nutrition table
        html += `
            <div class="nutrition-card" style="margin-top: 18px;">
                ${createTable(data.nutrition)}
            </div>
        `;
        
        // Show ingredients if available and meaningful
        if (data.ingredients && 
            !data.ingredients.includes('See modified') && 
            !data.ingredients.includes('Not available') &&
            !data.ingredients.includes('See estimated') &&
            data.ingredients.trim().length > 10) {
            html += `
                <div class="content-section">
                    <div class="content-section-title">Ingredients</div>
                    <div class="content-section-text">${escapeHtml(data.ingredients)}</div>
                </div>
            `;
        }
        
        // Show instructions if available and meaningful
        if (data.instructions && 
            !data.instructions.includes('See modified') && 
            !data.instructions.includes('Not available') &&
            !data.instructions.includes('estimated') &&
            data.instructions.trim().length > 10) {
            html += `
                <div class="content-section">
                    <div class="content-section-title">Instructions</div>
                    <div class="content-section-text">${escapeHtml(data.instructions)}</div>
                </div>
            `;
        }
        
        // Show LLM response if available (modifications, etc.)
        if (data.llm_response) {
            html += formatLLMText(data.llm_response);
        }
        
        // Badge
        if (data.estimated) {
            html += `<span class="estimated-badge">Estimated Values</span>`;
        } else if (data.accuracy) {
            html += `<span class="confidence-badge">${Math.round(data.accuracy)}% Confidence</span>`;
        }
        
        return html;
    }

    // PURE LLM RESPONSE (estimation fallback)
    if (data.llm_response) {
        return formatLLMText(data.llm_response);
    }

    return '<p>Received response from server</p>';
}

// ==================== MESSAGE SENDING ====================
async function sendMessage() {
    const query = userQuery.value.trim();
    
    if (!query && !selectedImage) return;

    // Disable input
    isProcessing = true;
    sendBtn.disabled = true;
    userQuery.disabled = true;

    // Display user message
    let userDisplay = query || 'Uploaded image';
    if (selectedImage && query) {
        userDisplay = `${query} (with image)`;
    } else if (selectedImage) {
        userDisplay = `Analyze: ${selectedImage.name}`;
    }
    
    renderMessage(escapeHtml(userDisplay), true);

    // Save image reference and clear inputs
    const imageToSend = selectedImage;
    userQuery.value = '';
    clearImage();

    // Show loading
    const loadingMsg = renderLoading();

    try {
        // Prepare form data
        const formData = new FormData();
        formData.append('query', query || '');
        if (imageToSend) {
            formData.append('image', imageToSend);
        }

        // Send request
        const response = await fetch('/process', {
            method: 'POST',
            body: formData,
        });

        // Remove loading
        loadingMsg.remove();

        // Check response
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }

        // Parse and display response
        const data = await response.json();
        console.log('Response data:', data);

        renderMessage(formatResponse(data));
        
    } catch (error) {
        console.error('Error:', error);
        loadingMsg.remove();
        renderMessage(
            `<strong style="color: var(--error);">Connection Error</strong>
            <p>Could not reach server. Please check your connection and try again.</p>`
        );
    } finally {
        // Re-enable input
        isProcessing = false;
        sendBtn.disabled = false;
        userQuery.disabled = false;
        userQuery.focus();
    }
}

function sendExample(text) {
    userQuery.value = text;
    sendMessage();
}

// ==================== EVENT LISTENERS ====================
userQuery.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !isProcessing) {
        e.preventDefault();
        sendMessage();
    }
});

// Focus input on load
document.addEventListener('DOMContentLoaded', () => {
    userQuery.focus();
});

// Add fadeOut animation for welcome area
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeOut {
        from {
            opacity: 1;
            transform: scale(1);
        }
        to {
            opacity: 0;
            transform: scale(0.95);
        }
    }
`;
document.head.appendChild(style);
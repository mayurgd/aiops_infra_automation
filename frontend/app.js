const API_BASE = 'http://localhost:8000';
let statusCheckInterval;
let currentStatus = 'idle';
let messageHistory = [];
let isWaitingForUserInput = false;
let isProcessing = false; // NEW: Track if agent is processing

// Initialize the app
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('welcomeTime').textContent = formatTime(new Date());
    checkServerHealth();
});

// Auto-resize textarea
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

// Handle Enter key
function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!isProcessing || isWaitingForUserInput) { // UPDATED: Only allow if not processing or waiting for input
            sendMessage();
        }
    }
}

// Toggle sidebar for mobile
function toggleSidebar() {
    document.getElementById('navSidebar').classList.toggle('open');
}

// Format timestamp
function formatTime(date) {
    return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

// Add message to chat
function addMessage(type, content, avatar = null, isHTML = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatarText = avatar || (type === 'user' ? '👤' : type === 'agent' ? '🤖' : '⚙️');
    const timestamp = formatTime(new Date());
    
    const contentHtml = isHTML ? content : content.replace(/\n/g, '<br>');
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarText}</div>
        <div class="message-content">
            ${contentHtml}
            <div class="message-time">${timestamp}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Store in history
    messageHistory.push({
        type, 
        content: isHTML ? content : contentHtml, 
        timestamp, 
        avatar: avatarText
    });
}

// Show typing indicator
function showTyping() {
    hideTyping(); // UPDATED: Remove any existing typing indicator first
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message agent';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="typing-indicator">
                <span>Agent is thinking</span>
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Hide typing indicator
function hideTyping() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// UPDATED: Update input state based on processing status
function updateInputState() {
    const chatInput = document.getElementById('chatInput');
    const sendBtn = document.getElementById('sendBtn');
    
    if (isProcessing && !isWaitingForUserInput) {
        // Agent is processing, block all input
        chatInput.disabled = true;
        chatInput.placeholder = "Agent is working... Please wait";
        chatInput.classList.add('disabled');
        sendBtn.disabled = true;
    } else if (isWaitingForUserInput) {
        // Agent is waiting for response
        chatInput.disabled = false;
        chatInput.placeholder = "Type your response to the agent's question...";
        chatInput.classList.remove('disabled');
        chatInput.classList.add('waiting-response');
        sendBtn.disabled = false;
        // Ensure input is visible and focusable
        chatInput.style.color = '';
        chatInput.style.backgroundColor = '';
    } else {
        // Ready for new requests
        chatInput.disabled = false;
        chatInput.placeholder = "Describe what you want to automate (e.g., 'Create a GitHub repo for my ML project')...";
        chatInput.classList.remove('disabled', 'waiting-response');
        sendBtn.disabled = false;
        // Ensure input is visible and focusable
        chatInput.style.color = '';
        chatInput.style.backgroundColor = '';
    }
}

// UPDATED: Update input placeholder based on state
function updateInputPlaceholder(isWaitingForResponse = false) {
    isWaitingForUserInput = isWaitingForResponse;
    updateInputState();
}

// Send message or response
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    
    if (!message || (isProcessing && !isWaitingForUserInput)) return;
    
    // Add user message
    addMessage('user', message);
    chatInput.value = '';
    autoResize(chatInput);
    
    if (isWaitingForUserInput) {
        // This is a response to an agent question
        await submitResponse(message);
    } else {
        // This is a new automation request
        await startAutomation(message);
    }
}

// Start new automation
async function startAutomation(message) {
    isProcessing = true; // UPDATED: Set processing flag
    updateInputState(); // UPDATED: Update input state
    showTyping();
    setSendButton(true);
    
    try {
        const response = await fetch(`${API_BASE}/start_crew`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                user_query: message,
                inputs: {}
            })
        });
        
        const result = await response.json();
        hideTyping();
        addMessage('system', `${result.message}`);
        startStatusChecking();
        
    } catch (error) {
        hideTyping();
        addMessage('system', `❌ Error: ${error.message}`, '⚠️');
        isProcessing = false; // UPDATED: Reset processing flag on error
        updateInputState();
        setSendButton(false);
    }
}

// Submit response to agent question
async function submitResponse(response) {
    setSendButton(true);
    isWaitingForUserInput = false; // UPDATED: Reset immediately
    isProcessing = true; // UPDATED: Set processing flag
    updateInputState(); // UPDATED: Update input state
    showTyping(); // UPDATED: Show typing when processing response
    
    try {
        await fetch(`${API_BASE}/submit_input`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                input: response
            })
        });
        
        // Don't show any status messages here, let status checking handle it
        
    } catch (error) {
        addMessage('system', `❌ Error submitting response: ${error.message}`, '⚠️');
        isProcessing = false; // UPDATED: Reset on error
        updateInputState();
        hideTyping();
        setSendButton(false);
    }
}

// Status checking
function startStatusChecking() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    statusCheckInterval = setInterval(checkStatus, 1500); // UPDATED: Faster polling
}

function stopStatusChecking() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
}

let lastPrompt = null; // UPDATED: Track last prompt to avoid duplicates

async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();
        
        updateStatusIndicator(status.status);
        
        // UPDATED: Handle status changes more precisely
        const statusChanged = currentStatus !== status.status;
        
        // Handle agent questions - only show if it's a new prompt
        if (status.status === 'waiting_input' && status.prompt) {
            if (status.prompt !== lastPrompt) {
                hideTyping();
                showAgentQuestion(status.prompt);
                isWaitingForUserInput = true;
                isProcessing = false; // UPDATED: Not processing when waiting for input
                updateInputState();
                setSendButton(false);
                lastPrompt = status.prompt;
            }
        }
        
        // UPDATED: Handle running status
        if (status.status === 'running' && currentStatus !== 'running') {
            isProcessing = true;
            isWaitingForUserInput = false;
            updateInputState();
            if (!document.getElementById('typingIndicator')) {
                showTyping();
            }
        }
        
        // Handle completion - only show result if status changed and we have a result
        if (status.result && statusChanged && (status.status === 'complete' || status.status === 'error')) {
            hideTyping();
            addMessage('agent', status.result);
            isProcessing = false; // UPDATED: Reset processing flag
            isWaitingForUserInput = false;
            updateInputState();
            setSendButton(false);
            stopStatusChecking();
            lastPrompt = null; // UPDATED: Reset prompt tracking
        }
        
        currentStatus = status.status;
        
    } catch (error) {
        console.error('Status check error:', error);
        // UPDATED: Reset flags on error
        isProcessing = false;
        isWaitingForUserInput = false;
        updateInputState();
        hideTyping();
    }
}

// Show agent question in chat
function showAgentQuestion(prompt) {
    // Clean up the prompt format from FastAPI
    let questionContent = prompt;
    
    // Remove the "🤖 Assistant:" prefix and "Your response:" suffix if present
    if (prompt.includes('🤖 Assistant:')) {
        questionContent = prompt
            .replace(/🤖 Assistant:\s*/, '')
            .replace(/\n\nYour response:.*$/, '')
            .trim();
    }
    
    // Add the agent question as a regular chat message
    addMessage('agent', questionContent, '🤖');
    
    // Focus on input for immediate response
    setTimeout(() => {
        if (!isProcessing || isWaitingForUserInput) { // UPDATED: Only focus if appropriate
            document.getElementById('chatInput').focus();
        }
    }, 100);
}

// UI helpers
function updateStatusIndicator(status) {
    const indicator = document.getElementById('statusIndicator');
    indicator.className = `status-indicator status-${status}`;
    
    let statusText = status.charAt(0).toUpperCase() + status.slice(1).replace('_', ' ');    
    indicator.textContent = `Status: ${statusText}`;
}

function setSendButton(isLoading) {
    const sendBtn = document.getElementById('sendBtn');
    const sendIcon = document.getElementById('sendIcon');
    const sendLoader = document.getElementById('sendLoader');
    
    if (isLoading) {
        sendBtn.disabled = true;
        sendIcon.style.display = 'none';
        sendLoader.style.display = 'inline-block';
    } else {
        sendBtn.disabled = false;
        sendIcon.style.display = 'inline';
        sendLoader.style.display = 'none';
    }
}

// Chat management
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="message system">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <strong>Chat cleared!</strong><br>
                Ready to help with your next automation request.
                <div class="message-time">${formatTime(new Date())}</div>
            </div>
        </div>
    `;
    messageHistory = [];
    isWaitingForUserInput = false;
    isProcessing = false; // UPDATED: Reset processing flag
    lastPrompt = null; // UPDATED: Reset prompt tracking
    updateInputState(); // UPDATED: Update input state
    stopStatusChecking();
}

function exportChat() {
    const chatData = {
        timestamp: new Date().toISOString(),
        messages: messageHistory
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aiops-chat-${new Date().toISOString().slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    addMessage('system', 'Chat exported successfully!');
}

// Check server health
async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        addMessage('system', `${data.message}`);
    } catch (error) {
        addMessage('system', `❌ Server connection failed. Make sure the FastAPI server is running on port 8000.`, '⚠️');
    }
}

// UPDATED: Auto-focus chat input only when appropriate
document.addEventListener('DOMContentLoaded', function() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput && !isProcessing) {
        chatInput.focus();
    }
});
const API_BASE = 'http://localhost:8000';
let statusCheckInterval;
let currentStatus = 'idle';
let messageHistory = [];
let isWaitingForUserInput = false;

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
        sendMessage();
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

// Update input placeholder based on state
function updateInputPlaceholder(isWaitingForResponse = false) {
    const chatInput = document.getElementById('chatInput');
    if (isWaitingForResponse) {
        chatInput.placeholder = "Type your response to the agent's question...";
        chatInput.classList.add('waiting-response');
    } else {
        chatInput.placeholder = "Describe what you want to automate (e.g., 'Create a GitHub repo for my ML project')...";
        chatInput.classList.remove('waiting-response');
    }
}

// Send message or response
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    
    if (!message) return;
    
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
        setSendButton(false);
    }
}

// Submit response to agent question
async function submitResponse(response) {
    setSendButton(true);
    
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
        
        // Reset waiting state
        isWaitingForUserInput = false;
        updateInputPlaceholder(false);
        // addMessage('system', 'Response submitted, continuing automation...');
        
    } catch (error) {
        addMessage('system', `❌ Error submitting response: ${error.message}`, '⚠️');
        setSendButton(false);
    }
}

// Status checking
function startStatusChecking() {
    statusCheckInterval = setInterval(checkStatus, 2000);
}

function stopStatusChecking() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
}

async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const status = await response.json();
        
        updateStatusIndicator(status.status);
        
        // Handle agent questions
        if (status.status === 'waiting_input' && status.prompt && !isWaitingForUserInput) {
            showAgentQuestion(status.prompt);
            isWaitingForUserInput = true;
            updateInputPlaceholder(true);
            setSendButton(false);
        }
        
        // Handle completion
        if (status.result && currentStatus !== 'complete' && currentStatus !== 'error') {
            hideTyping();
            addMessage('agent', status.result);
        }
        
        if (status.status === 'complete' || status.status === 'error') {
            isWaitingForUserInput = false;
            updateInputPlaceholder(false);
            setSendButton(false);
            stopStatusChecking();
        }
        
        currentStatus = status.status;
        
    } catch (error) {
        console.error('Status check error:', error);
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
        document.getElementById('chatInput').focus();
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
    updateInputPlaceholder(false);
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

// Auto-focus chat input
document.getElementById('chatInput').focus();
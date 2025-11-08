const API_BASE = 'http://127.0.0.1:8070';
let statusCheckInterval;
let currentStatus = 'idle';
let messageHistory = [];
let currentSessionId = null;
// Configure marked.js options
marked.setOptions({
    breaks: true,
    gfm: true,
    headerIds: false,
    mangle: false,
    highlight: function(code, lang) {
        if (lang && hljs.getLanguage(lang)) {
            try {
                return hljs.highlight(code, { language: lang }).value;
            } catch (err) {}
        }
        return hljs.highlightAuto(code).value;
    }
});

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

// Toggle sidebar for both mobile and desktop
function toggleSidebar() {
    const sidebar = document.getElementById('navSidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const isMobile = window.innerWidth <= 768;
    
    if (isMobile) {
        // Mobile behavior: slide in/out from left
        sidebar.classList.toggle('open');
        overlay.classList.toggle('active');
    } else {
        // Desktop behavior: collapse/expand
        sidebar.classList.toggle('collapsed');
    }
}

// Close sidebar (primarily for mobile overlay)
function closeSidebar() {
    const sidebar = document.getElementById('navSidebar');
    const overlay = document.getElementById('sidebarOverlay');
    
    sidebar.classList.remove('open');
    overlay.classList.remove('active');
}

// Handle window resize to maintain proper sidebar state
window.addEventListener('resize', function() {
    const sidebar = document.getElementById('navSidebar');
    const overlay = document.getElementById('sidebarOverlay');
    const isMobile = window.innerWidth <= 768;
    
    if (!isMobile) {
        // Reset mobile classes when switching to desktop
        sidebar.classList.remove('open');
        overlay.classList.remove('active');
    } else {
        // Reset desktop classes when switching to mobile
        sidebar.classList.remove('collapsed');
    }
});

// Format timestamp
function formatTime(date) {
    return date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
}

// Add message to chat
// Enhanced addMessage function with better code fence detection
function addMessage(type, content, avatar = null, isHTML = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const avatarText = avatar || (type === 'user' ? '👤' : type === 'agent' ? '🤖' : '⚙️');
    const timestamp = formatTime(new Date());
    
    // Render markdown for agent messages, keep HTML/text for others
    let contentHtml;
    if (type === 'agent' && !isHTML) {
        let processedContent = content.trim();
        
        // Remove outer code fences if the ENTIRE content is wrapped in them
        const codeFencePattern = /^```[\w]*\n([\s\S]*?)\n```$/;
        const match = processedContent.match(codeFencePattern);
        
        if (match) {
            // Content was wrapped in code fences, extract the inner content
            processedContent = match[1];
        }
        
        // Parse markdown and wrap in a div with markdown-content class
        contentHtml = `<div class="markdown-content">${marked.parse(processedContent)}</div>`;
    } else if (isHTML) {
        contentHtml = content;
    } else {
        contentHtml = content.replace(/\n/g, '<br>');
    }
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatarText}</div>
        <div class="message-content">
            ${contentHtml}
            <div class="message-time">${timestamp}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    // Highlight code blocks after adding to DOM
    if (type === 'agent') {
        messageDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
    }
    
    // Store in history
    messageHistory.push({
        type, 
        content: content, // Store original content
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
function updateInputPlaceholder(isInConversation = false) {
    const chatInput = document.getElementById('chatInput');
    if (isInConversation) {
        chatInput.placeholder = "Type your message...";
        chatInput.classList.add('waiting-response');
    } else {
        chatInput.placeholder = "Start a conversation with the MLOps Onboarding Assistant...";
        chatInput.classList.remove('waiting-response');
    }
}

// Send message
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    
    if (!message) return;
    
    // Add user message
    addMessage('user', message);
    chatInput.value = '';
    autoResize(chatInput);
    
    // If no session exists, start a new conversation
    if (!currentSessionId) {
        await startConversation(message);
    } else {
        // Continue existing conversation
        await sendMessageToAgent(message);
    }
}

// Start new conversation
// Update startConversation function
async function startConversation(initialMessage = null) {
    showTyping();
    setSendButton(true);
    updateStatusIndicator('starting');
    
    try {
        const response = await fetch(`${API_BASE}/start_conversation`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                initial_message: initialMessage
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        currentSessionId = result.session_id;
        
        hideTyping();
        
        // Display agent's response with streaming animation
        await displayMessageWithStreaming(result.response);
        
        updateStatusIndicator('active');
        updateInputPlaceholder(true);
        setSendButton(false);
        
    } catch (error) {
        hideTyping();
        addMessage('system', `❌ Error starting conversation: ${error.message}`, '⚠️');
        setSendButton(false);
        updateStatusIndicator('error');
    }
}

// Send message to agent in existing conversation
async function sendMessageToAgent(message) {
    if (!currentSessionId) {
        addMessage('system', '❌ No active session. Starting new conversation...', '⚠️');
        await startConversation(message);
        return;
    }
    
    showTyping();
    setSendButton(true);
    updateStatusIndicator('processing');
    
    try {
        const response = await fetch(`${API_BASE}/send_message`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                message: message
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        hideTyping();
        
        // Display agent's response with streaming animation
        await displayMessageWithStreaming(result.response);
        
        // Update status
        if (result.status === 'complete') {
            updateStatusIndicator('complete');
            addMessage('system', '✅ Conversation ended. Start a new conversation anytime!');
            currentSessionId = null;
            updateInputPlaceholder(false);
        } else if (result.status === 'error') {
            updateStatusIndicator('error');
            currentSessionId = null;
            updateInputPlaceholder(false);
        } else {
            updateStatusIndicator('active');
        }
        
        setSendButton(false);
        
    } catch (error) {
        hideTyping();
        addMessage('system', `❌ Error sending message: ${error.message}`, '⚠️');
        setSendButton(false);
        updateStatusIndicator('error');
    }
}


// Check session status
async function checkSessionStatus() {
    if (!currentSessionId) return;
    
    try {
        const response = await fetch(`${API_BASE}/status/${currentSessionId}`);
        
        if (!response.ok) {
            if (response.status === 404) {
                addMessage('system', '⚠️ Session expired. Please start a new conversation.', '⚠️');
                currentSessionId = null;
                updateInputPlaceholder(false);
                updateStatusIndicator('idle');
            }
            return;
        }
        
        const status = await response.json();
        updateStatusIndicator(status.status);
        
    } catch (error) {
        console.error('Status check error:', error);
    }
}

// UI helpers
function updateStatusIndicator(status) {
    const indicator = document.getElementById('statusIndicator');
    indicator.className = `status-indicator status-${status}`;
    
    const statusMap = {
        'idle': 'Idle',
        'starting': 'Starting...',
        'active': 'Active',
        'processing': 'Processing...',
        'complete': 'Complete',
        'error': 'Error'
    };
    
    const statusText = statusMap[status] || status.charAt(0).toUpperCase() + status.slice(1);
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
async function clearChat() {
    // Delete current session if exists
    if (currentSessionId) {
        try {
            await fetch(`${API_BASE}/session/${currentSessionId}`, {
                method: 'DELETE'
            });
        } catch (error) {
            console.error('Error deleting session:', error);
        }
        currentSessionId = null;
    }
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML = `
        <div class="message system">
            <div class="message-avatar">🤖</div>
            <div class="message-content">
                <strong>Chat cleared!</strong><br>
                Ready to help with your MLOps onboarding.
                <div class="message-time">${formatTime(new Date())}</div>
            </div>
        </div>
    `;
    messageHistory = [];
    updateInputPlaceholder(false);
    updateStatusIndicator('idle');
}

function exportChat() {
    const chatData = {
        timestamp: new Date().toISOString(),
        session_id: currentSessionId,
        messages: messageHistory
    };
    
    const blob = new Blob([JSON.stringify(chatData, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mlops-chat-${new Date().toISOString().slice(0, 19)}.json`;
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
        
        if (data.agent_initialized) {
            addMessage('system', `✅ ${data.message} | Active Sessions: ${data.active_sessions}`);
        } else {
            addMessage('system', `⚠️ Server is running but agent not initialized`, '⚠️');
        }
    } catch (error) {
        addMessage('system', `❌ Server connection failed. Make sure the FastAPI server is running on port 8070.`, '⚠️');
    }
}

// Add this new function to simulate streaming display
async function displayMessageWithStreaming(content, delayPerChar = 10) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message agent';
    messageDiv.id = 'streamingMessage';
    
    const timestamp = formatTime(new Date());
    
    messageDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-content">
            <div class="markdown-content streaming-content"></div>
            <div class="message-time">${timestamp}</div>
        </div>
    `;
    
    chatMessages.appendChild(messageDiv);
    
    const contentDiv = messageDiv.querySelector('.streaming-content');
    let displayedContent = '';
    
    // Stream character by character
    for (let i = 0; i < content.length; i++) {
        displayedContent += content[i];
        
        // Process content to remove outer code fences if present
        let processedContent = displayedContent.trim();
        const codeFencePattern = /^```[\w]*\n([\s\S]*?)\n```$/;
        const match = processedContent.match(codeFencePattern);
        
        if (match) {
            processedContent = match[1];
        }
        
        // Update with markdown rendering
        contentDiv.innerHTML = marked.parse(processedContent);
        
        // Highlight code blocks
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        // Add delay
        await new Promise(resolve => setTimeout(resolve, delayPerChar));
    }
    
    // Finalize
    messageDiv.removeAttribute('id');
    
    // Store in history
    messageHistory.push({
        type: 'agent', 
        content: content,
        timestamp, 
        avatar: '🤖'
    });
}

// Start a new conversation on page load (optional - can be removed if you want user to initiate)
// Uncomment the line below if you want automatic greeting on page load
// setTimeout(() => startConversation(), 1000);

// Auto-focus chat input
document.getElementById('chatInput').focus();

// Periodic session status check (optional)
setInterval(checkSessionStatus, 30000); // Check every 30 seconds
// ── CHATBOT ──────────────────────────────────────────────────────────────────

function toggleChat() {
    const chatbox = document.getElementById('chatbox');
    const icon = document.getElementById('chatbot-icon');
    
    if (chatbox.style.display === 'flex') {
        chatbox.style.display = 'none';
        icon.style.transform = 'scale(1)';
    } else {
        chatbox.style.display = 'flex';
        icon.style.transform = 'scale(0.9)';
        
        // Add welcome message if chat is empty
        const messages = document.getElementById('messages');
        if (messages.children.length === 0) {
            addBotMessage("👋 Hello! I'm your AI assistant powered by Google Gemini. I can answer questions about:<br><br>🏥 Medical topics & breast cancer<br>💊 Treatments & medications<br>🥗 Diet & nutrition<br>🌍 General knowledge<br>💬 Anything else you'd like to know!<br><br>What would you like to ask?");
        }
    }
}

function addUserMessage(text) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg-user';
    div.textContent = text;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function addBotMessage(text) {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg-bot';
    div.innerHTML = text; // Use innerHTML to support formatting
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function addTypingIndicator() {
    const messages = document.getElementById('messages');
    const div = document.createElement('div');
    div.className = 'msg-bot typing-indicator';
    div.id = 'typing';
    div.innerHTML = '<span></span><span></span><span></span>';
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
}

function removeTypingIndicator() {
    const typing = document.getElementById('typing');
    if (typing) {
        typing.remove();
    }
}

async function sendMessage() {
    const input = document.getElementById('userInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    addUserMessage(message);
    input.value = '';
    
    // Show typing indicator
    addTypingIndicator();
    
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        // Add bot response
        if (data.response) {
            addBotMessage(data.response);
        } else {
            addBotMessage("I'm sorry, I couldn't process that. Please try again.");
        }
        
    } catch (error) {
        console.error('Chat error:', error);
        removeTypingIndicator();
        addBotMessage("⚠️ Connection error. Please check your internet and try again.");
    }
}

// Allow Enter key to send message
document.addEventListener('DOMContentLoaded', function() {
    const userInput = document.getElementById('userInput');
    if (userInput) {
        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }
});


// Quick action buttons (optional enhancement)
function addQuickActions() {
    const messages = document.getElementById('messages');
    const quickActions = document.createElement('div');
    quickActions.className = 'quick-actions';
    quickActions.innerHTML = `
        <button onclick="askQuestion('What are the symptoms of breast cancer?')">Symptoms</button>
        <button onclick="askQuestion('What treatment options are available?')">Treatment</button>
        <button onclick="askQuestion('What foods should I eat?')">Diet</button>
        <button onclick="askQuestion('How can I manage side effects?')">Side Effects</button>
    `;
    messages.appendChild(quickActions);
}

function askQuestion(question) {
    document.getElementById('userInput').value = question;
    sendMessage();
}


class ChatbotUI {
    constructor() {
        this.sessionId = this.generateSessionId();
        this.apiBaseUrl = 'http://localhost:8000';
        this.isStreaming = false;
        this.currentMessageElement = null;
        this.currentStartTime = null;

        this.initElements();
        this.initEventListeners();
        this.checkServerHealth();
    }

    generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substring(2, 11);
    }

    initElements() {
        this.messagesContainer = document.getElementById('messagesContainer');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.statusText = document.getElementById('statusText');
    }

    initEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Auto-resize textarea
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        });

        // Initial scroll to bottom
        setTimeout(() => this.scrollToBottom(), 100);
    }

    async checkServerHealth() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (response.ok) {
                this.updateStatus('Connected', true);
            } else {
                this.updateStatus('Server error', false);
            }
        } catch (error) {
            this.updateStatus('Disconnected', false);
            console.error('Health check failed:', error);
        }
    }

    updateStatus(text, isConnected) {
        this.statusText.textContent = text;
        const statusDot = document.querySelector('.status-dot');
        statusDot.style.background = isConnected ? '#10B981' : '#EF4444';
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message || this.isStreaming) return;

        // Add user message
        this.addMessage('user', message);
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';

        // Disable input
        this.setInputState(false);
        this.showTypingIndicator(true);

        // Track timing
        this.currentStartTime = Date.now();
        let firstTokenReceived = false;
        let ttft = null;

        try {
            const response = await fetch(`${this.apiBaseUrl}/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    session_id: this.sessionId,
                    message: message,
                    debug: false // Set to true to see debug info
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let currentContent = '';

            // Create assistant message element
            this.currentMessageElement = this.addMessage('assistant', '', true);

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);

                        if (data === '[DONE]') {
                            this.finishStreaming();
                            return;
                        }

                        try {
                            const parsed = JSON.parse(data);

                            // Track TTFT
                            if (!firstTokenReceived && parsed.choices?.[0]?.delta?.content) {
                                firstTokenReceived = true;
                                ttft = Date.now() - this.currentStartTime;
                                this.updateTTFT(this.currentMessageElement, ttft, message);
                            }

                            // Handle content
                            if (parsed.choices?.[0]?.delta?.content) {
                                currentContent += parsed.choices[0].delta.content;
                                this.updateMessageContent(this.currentMessageElement, currentContent);
                            }

                            // Handle tool usage
                            if (parsed.tools_used && parsed.tools_used.length > 0) {
                                this.addToolInfo(this.currentMessageElement, parsed.tools_used);
                            }

                            // Handle debug info
                            if (parsed.debug_info) {
                                this.addDebugInfo(this.currentMessageElement, parsed.debug_info);
                            }

                        } catch (e) {
                            console.warn('Failed to parse chunk:', data, e);
                        }
                    }
                }
            }

        } catch (error) {
            console.error('Chat request failed:', error);
            this.showError(`Failed to send message: ${error.message}`);
        } finally {
            this.showTypingIndicator(false);
            this.setInputState(true);
            this.currentStartTime = null;
        }
    }

    addMessage(role, content, streaming = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        // Add avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.textContent = role === 'user' ? 'U' : 'C';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = content;

        if (role === 'user') {
            messageDiv.appendChild(contentDiv);
            messageDiv.appendChild(avatarDiv);
        } else {
            messageDiv.appendChild(avatarDiv);
            messageDiv.appendChild(contentDiv);
        }

        if (!streaming) {
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = new Date().toLocaleTimeString();
            contentDiv.parentElement.appendChild(metaDiv);
        }

        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();

        return contentDiv;
    }

    updateMessageContent(element, content) {
        element.textContent = content;
        // Ensure scrolling happens during streaming
        this.scrollToBottom();
    }

    updateTTFT(element, ttft, originalMessage) {
        const existingTTFT = element.parentElement.querySelector('.ttft-info');
        if (existingTTFT) {
            existingTTFT.remove();
        }

        // Create shortened message display: first two words + ... + last word
        const words = originalMessage.trim().split(/\s+/);
        let shortMessage;
        if (words.length <= 3) {
            shortMessage = originalMessage;
        } else {
            const firstTwo = words.slice(0, 2).join(' ');
            const lastWord = words[words.length - 1];
            shortMessage = `${firstTwo}...${lastWord}`;
        }

        const ttftDiv = document.createElement('div');
        ttftDiv.className = 'ttft-info';
        ttftDiv.innerHTML = `⚡ TTFT: ${ttft}ms <span style="opacity: 0.7; font-size: 0.9em;">"${shortMessage}"</span>`;

        element.parentElement.appendChild(ttftDiv);
        this.scrollToBottom();
    }

    addToolInfo(element, tools) {
        const existingToolInfo = element.parentElement.querySelector('.tool-info');
        if (existingToolInfo) {
            existingToolInfo.remove();
        }

        const toolDiv = document.createElement('div');
        toolDiv.className = 'debug-info';
        toolDiv.innerHTML = `🔧 Tools used: ${tools.join(', ')}`;
        element.parentElement.appendChild(toolDiv);
    }

    addDebugInfo(element, debugInfo) {
        const debugDiv = document.createElement('div');
        debugDiv.className = 'debug-info';
        debugDiv.innerHTML = `🐛 Debug: ${JSON.stringify(debugInfo, null, 2)}`;
        element.parentElement.appendChild(debugDiv);
    }

    finishStreaming() {
        if (this.currentMessageElement) {
            // Add timestamp
            const metaDiv = document.createElement('div');
            metaDiv.className = 'message-meta';
            metaDiv.textContent = new Date().toLocaleTimeString();
            this.currentMessageElement.parentElement.appendChild(metaDiv);
        }
        this.currentMessageElement = null;
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        this.messagesContainer.appendChild(errorDiv);
        this.scrollToBottom();

        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    showTypingIndicator(show) {
        this.typingIndicator.classList.toggle('show', show);
        if (show) {
            this.scrollToBottom();
        }
    }

    setInputState(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
        this.sendButton.textContent = enabled ? 'Send' : 'Sending...';
    }

    scrollToBottom() {
        // Use requestAnimationFrame for smoother scrolling
        requestAnimationFrame(() => {
            this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
        });
    }
}

// Initialize chatbot when page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatbotUI();
});
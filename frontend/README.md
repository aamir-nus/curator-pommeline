# Curator Pommeline Chatbot UI

A web interface for the Curator Pommeline AI shopping assistant.

## Quick Start

1. **Start the API Server**:
   ```bash
   cd /path/to/curator-pommeline
   source .venv/bin/activate
   python3 -m api.main
   ```

2. **Open the UI**:
   ```bash
   open frontend/index.html
   ```
   Or open `frontend/index.html` in your browser.

## Features

- Real-time streaming responses
- TTFT (Time to First Token) tracking
- ML-based content filtering
- Product search and knowledge retrieval
- Multi-turn conversations
- Responsive design

## File Structure

```
frontend/
├── index.html          # Main HTML file
├── styles.css          # Styling
├── chatbot.js          # JavaScript logic
└── README.md           # This file
```

## Usage

The chatbot connects to the FastAPI backend and provides an interface for:
- Shopping assistance
- Product information
- Policy questions
- General inquiries

Type your message in the input field and press Enter or click Send to chat.
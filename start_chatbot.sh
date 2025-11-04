#!/bin/bash

# Chatbot Startup Script
# This script starts the chatbot server with proper environment setup and resource warm-up

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "Starting Chatbot Server Setup..."
echo "========================================"

# 1. Stop current server/kill process on port 8000
print_status "Step 1: Stopping any existing server on port 8000..."

# Kill any process using port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Found process running on port 8000, stopping it..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null || true
    sleep 2
    print_success "Port 8000 cleared"
else
    print_success "No process found on port 8000"
fi

# Kill any remaining python3 -m api.main processes
pkill -f "python3 -m api.main" 2>/dev/null || true
sleep 1

# 2. Activate virtual environment
print_status "Step 2: Activating virtual environment..."

if [ -d ".venv" ]; then
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        print_success "Virtual environment activated"
    else
        print_error "Virtual environment found but activation script missing"
        exit 1
    fi
else
    print_error "Virtual environment not found. Please create it first with:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1)
print_status "Using Python: $PYTHON_VERSION"

# 3. Start the server in background
print_status "Step 3: Starting server with python3 -m api.main..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Start server in background with logging
python3 -m api.main > logs/server.log 2>&1 &
SERVER_PID=$!

print_status "Server started with PID: $SERVER_PID"
print_status "Server logs will be written to: logs/server.log"

# Wait for server to start
print_status "Waiting for server to initialize..."
sleep 5

# Check if server is running
if ps -p $SERVER_PID > /dev/null 2>&1; then
    print_success "Server process is running"
else
    print_error "Server failed to start. Check logs/server.log for details."
    exit 1
fi

# Check if server is responding
print_status "Checking server health..."
for i in {1..10}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Server is responding on port 8000"
        break
    else
        if [ $i -eq 10 ]; then
            print_error "Server not responding after 10 attempts"
            print_error "Check logs/server.log for error details"
            exit 1
        fi
        print_status "Attempt $i/10: Server not ready yet, waiting..."
        sleep 2
    fi
done

# 4. Warmup calls to load embedding models and other resources
print_status "Step 4: Warming up endpoints and loading resources..."

# Function to make warmup call
make_warmup_call() {
    local endpoint=$1
    local method=$2
    local data=$3
    local description=$4

    print_status "Warming up: $description"

    if [ "$method" = "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" "http://localhost:8000$endpoint" || echo "000")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST \
            -H "Content-Type: application/json" \
            -d "$data" \
            "http://localhost:8000$endpoint" || echo "000")
    fi

    http_code=$(echo "$response" | tail -n1)
    response_body=$(echo "$response" | head -n -1)

    if [ "$http_code" = "200" ] || [ "$http_code" = "201" ]; then
        print_success "✓ $description - Ready"
    else
        print_warning "⚠ $description - HTTP $http_code (may be expected)"
    fi
}

# Health check
make_warmup_call "/health" "GET" "" "Health check endpoint"

# Retrieval endpoint warmup
make_warmup_call "/retrieve" "POST" '{"query": "test query", "top_k": 3}' "Retrieval service (loading embedding models)"

# Guardrails endpoint warmup
make_warmup_call "/guardrails" "POST" '{"user_input": "test input"}' "Guardrails service"

# Chat endpoint warmup (this will load LLM and other resources)
make_warmup_call "/chat" "POST" '{"session_id": "warmup_session", "message": "Hello, this is a warmup test"}' "Chat service (loading LLM and tools)"

# Search tool warmup
make_warmup_call "/chat" "POST" '{"session_id": "warmup_search", "message": "Search for iPhone products"}' "Product search tool"

# 5. Open browser
print_status "Step 5: Opening browser for chat interface..."

# Determine the HTML file path
HTML_FILE="index.html"
if [ ! -f "$HTML_FILE" ]; then
    # Look for HTML files in common locations
    for possible_file in "chat.html" "ui/index.html" "web/index.html" "public/index.html"; do
        if [ -f "$possible_file" ]; then
            HTML_FILE="$possible_file"
            break
        fi
    done
fi

if [ -f "$HTML_FILE" ]; then
    # Convert to absolute path for macOS open command
    ABS_HTML_PATH="$(pwd)/$HTML_FILE"
    print_status "Opening chat interface: $ABS_HTML_PATH"

    # Try to open with default browser (macOS)
    if command -v open >/dev/null 2>&1; then
        open "$ABS_HTML_PATH"
        print_success "Browser opened with chat interface"
    else
        # Fallback for other systems
        if command -v xdg-open >/dev/null 2>&1; then
            xdg-open "$ABS_HTML_PATH"
            print_success "Browser opened with chat interface"
        else
            print_warning "Could not auto-open browser. Please manually open: $ABS_HTML_PATH"
        fi
    fi
else
    print_error "No HTML interface found. Expected files: index.html, chat.html, ui/index.html"
    print_status "You can access the API directly at: http://localhost:8000"
    print_status "API documentation: http://localhost:8000/docs"
fi

# 6. Final status
echo "========================================"
print_success "🚀 Chatbot server is now running!"
echo ""
echo "Server Details:"
echo "  - PID: $SERVER_PID"
echo "  - Port: 8000"
echo "  - Logs: logs/server.log"
echo "  - Health: http://localhost:8000/health"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "To stop the server:"
echo "  kill $SERVER_PID"
echo "  or run: ./stop_chatbot.sh"
echo ""
print_status "Chat session is ready! 🎉"

# Optional: Create stop script
if [ ! -f "stop_chatbot.sh" ]; then
    cat > stop_chatbot.sh << 'EOF'
#!/bin/bash

# Stop Chatbot Server Script

echo "Stopping chatbot server..."

# Kill process on port 8000
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    lsof -ti:8000 | xargs kill -9
    echo "Server on port 8000 stopped"
else
    echo "No server found on port 8000"
fi

# Kill any remaining api.main processes
pkill -f "python3 -m api.main" 2>/dev/null || true

echo "Chatbot server stopped successfully"
EOF
    chmod +x stop_chatbot.sh
    print_status "Created stop_chatbot.sh for easy server shutdown"
fi
# GenAI-Labs

This project implements a chatbot assistant for e-commerce that handles product information and order data queries. It's built using a microservices architecture.

## Architecture

The system consists of three main microservices:
- **Chat Service**: Main entry point for user queries
- **Product Service**: Handles product information retrieval
- **Order Service**: Interfaces with order data API

## Features

- RAG-based product information retrieval
- Order data lookup via API integration
- Natural language query processing

## Running the Application

### Prerequisites
- Python3.10
- OpenAI API key (set as environment variable)

### Local Development
1. Clone the repository
2. Set your OpenAI API key: `export OPENAI_API_KEY=your-api-key`
3. Make a python virtual environment and run all the services (chat_api, mock_api, order_api, product_api)
4. Access the chat service at http://localhost:8000/docs

### API Testing

You can test the API endpoints using curl:

```bash
# Test the chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the top-rated guitar products?", "session_id": "id"}'

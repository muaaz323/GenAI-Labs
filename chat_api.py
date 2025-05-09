# chat_service/api.py
from fastapi import FastAPI, HTTPException
import httpx
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import logging
from dotenv import load_dotenv
import uuid
from datetime import datetime
import json

load_dotenv()

app = FastAPI(title="Chat Service", description="Main entry point for the chatbot")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("chat_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("chat_service")

# Configuration
PRODUCT_SERVICE_URL = "http://127.0.0.1:8002"  # Adjust in deployment
ORDER_SERVICE_URL = "http://127.0.0.1:8003"  # Adjust in deployment
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# print(OPENAI_API_KEY)

# Define request model
# Store conversation state
conversations = {}

# Define request models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str

# Initialize LLM
llm = OpenAI()

# Define prompt templates
product_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""
    You are an e-commerce assistant specializing in musical instruments and other products.
    Answer the user's query based on the retrieved product information.
    
    User Query: {query}
    
    Retrieved Product Information:
    {context}
    
    Provide a helpful, informative response that directly addresses the user's query.
    Format product information in a clean, readable way. Include prices and ratings where available.
    """
)

order_prompt = PromptTemplate(
    input_variables=["query", "context", "customer_id"],
    template="""
    You are an e-commerce assistant helping customers with their order information.
    Answer the user's query based on the retrieved order data.
    
    User Query: {query}
    Customer ID: {customer_id}
    
    Retrieved Order Data:
    {context}
    
    Provide a helpful, informative response that directly addresses the user's query.
    Format order information in a clean, readable way. Include order dates, products, and prices where available.
    """
)

product_chain = LLMChain(llm=llm, prompt=product_prompt)
order_chain = LLMChain(llm=llm, prompt=order_prompt)

def get_conversation(session_id: str = None):
    """Create or retrieve a conversation session"""
    if not session_id or session_id not in conversations:
        # Create new session
        new_session_id = session_id or str(uuid.uuid4())
        logger.info(f"Creating new conversation session: {new_session_id}")
        conversations[new_session_id] = {
            "customer_id": None,
            "last_query_type": None,
            "waiting_for_customer_id": False,
            "pending_query": None,
            "history": [],
            "created_at": datetime.now().isoformat()
        }
        return new_session_id, conversations[new_session_id]
    
    logger.info(f"Retrieving existing conversation session: {session_id}")
    return session_id, conversations[session_id]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process user messages and return appropriate responses."""
    try:
        session_id, conversation = get_conversation(request.session_id)
        message = request.message.lower()
        
        logger.info(f"Session {session_id}: Received message: '{message}'")
        logger.debug(f"Current conversation state: {json.dumps(conversation, indent=2)}")
        
        # Add message to history
        conversation["history"].append({"role": "user", "content": message, "timestamp": datetime.now().isoformat()})
        
        # Check if we're waiting for customer ID
        # Check if we're waiting for customer ID
        if conversation["waiting_for_customer_id"]:
            logger.info(f"Session {session_id}: Waiting for customer ID, received: '{message}'")
            
            # Extract customer ID from message
            customer_id = None
            
            # Case 1: Message is just a number
            if message.strip().isdigit():
                customer_id = message.strip()
            # Case 2: Natural language pattern (e.g., "My Customer ID is 37077.")
            else:
                # Look for common patterns where numbers appear after ID-related words
                import re
                patterns = [
                    r"customer\s*id\s*(?:is|:|=)?\s*(\d+)",  # "customer id is 37077" or "customer id: 37077"
                    r"id\s*(?:is|:|=)?\s*(\d+)",            # "id is 37077" or "id: 37077"
                    r"number\s*(?:is|:|=)?\s*(\d+)",        # "number is 37077"
                    r"(\d+)\s*(?:is my|as my)?\s*(?:customer)?\s*id"  # "37077 is my customer id"
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, message, re.IGNORECASE)
                    if match:
                        customer_id = match.group(1)
                        break
                        
                # As a fallback, just find any number in the message
                if not customer_id:
                    numbers = re.findall(r'\d+', message)
                    if numbers:
                        # Use the first number found as customer ID
                        customer_id = numbers[0]
            
            if customer_id:
                logger.info(f"Session {session_id}: Customer ID extracted: {customer_id}")
                conversation["customer_id"] = customer_id
                conversation["waiting_for_customer_id"] = False
                
                # Process the pending query with the customer ID
                if conversation["last_query_type"]:
                    query_type = conversation["last_query_type"]
                    logger.info(f"Session {session_id}: Processing pending query of type: {query_type}")
                    
                    if query_type == "recent_order":
                        return await process_recent_order(customer_id, session_id, conversation)
                    elif query_type == "order_by_category":
                        category = conversation.get("pending_query", {}).get("category")
                        logger.info(f"Session {session_id}: Processing category order with category: {category}")
                        return await process_category_order(customer_id, category, session_id, conversation)
                    else:
                        # Default to general order for any other query type
                        return await process_general_order(customer_id, session_id, conversation)
                else:
                    # Fallback if no query type is set
                    logger.info(f"Session {session_id}: No pending query type found, defaulting to general order lookup")
                    return await process_general_order(customer_id, session_id, conversation)
            else:
                # The user didn't provide a valid customer ID
                logger.warning(f"Session {session_id}: No customer ID found in message: '{message}'")
                response = "I couldn't find a valid Customer ID in your message. Please provide just a number, or say something like 'My customer ID is 12345'."
                conversation["history"].append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})
                return ChatResponse(response=response, session_id=session_id)
        
        # Determine if this is a product or order query
        order_keywords = ["order", "shipped", "delivery", "purchase", "bought", "status"]
        personal_keywords = ["my", "i ", "mine", "me"]
        
        is_order_query = any(keyword in message for keyword in order_keywords)
        is_personal_query = any(keyword in message for keyword in personal_keywords)
        
        logger.info(f"Session {session_id}: Query classification - order query: {is_order_query}, personal query: {is_personal_query}")
        
        if is_order_query:
    # This is likely an order query
            if is_personal_query:
                # Personal order query requires customer ID
                if not conversation["customer_id"]:
                    logger.info(f"Session {session_id}: Personal order query detected but no customer ID available")
                    conversation["waiting_for_customer_id"] = True
                    
                    # FIXED: Always set a default query type even if we don't detect specifics
                    conversation["last_query_type"] = "general_order"
                    conversation["pending_query"] = {"original_message": message}
                    
                    # Check for more specific query types
                    if "recent" in message or "last" in message:
                        logger.info(f"Session {session_id}: Identified as recent order query")
                        conversation["last_query_type"] = "recent_order"
                    elif any(cat in message for cat in ["car", "cell", "phone", "mobile", "guitar"]):
                        # Extract the category
                        category = None
                        if "car" in message:
                            category = "car"
                        elif any(word in message for word in ["cell", "phone", "mobile"]):
                            category = "mobile"
                        elif "guitar" in message:
                            category = "guitar"
                            
                        logger.info(f"Session {session_id}: Identified as category-specific order query for category: {category}")
                        conversation["last_query_type"] = "order_by_category"
                        conversation["pending_query"] = {"category": category, "original_message": message}
                    
                    response = "Could you please provide your Customer ID?"
                    logger.info(f"Session {session_id}: Requesting customer ID")
                    conversation["history"].append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})
                    return ChatResponse(response=response, session_id=session_id)
                
                # We have the customer ID, proceed with the query
                customer_id = conversation["customer_id"]
                logger.info(f"Session {session_id}: Processing personal order query with customer ID: {customer_id}")
                
                # Handle different types of order queries
                if "recent" in message or "last" in message:
                    logger.info(f"Session {session_id}: Processing recent order query")
                    return await process_recent_order(customer_id, session_id, conversation)
                
                # Check for category-specific orders
                category = None
                if "car" in message:
                    category = "car"
                elif any(word in message for word in ["cell", "phone", "mobile"]):
                    category = "mobile"
                elif "guitar" in message:
                    category = "guitar"
                    
                if category:
                    logger.info(f"Session {session_id}: Processing category order query for: {category}")
                    return await process_category_order(customer_id, category, session_id, conversation)
                    
                # Default: general order query
                logger.info(f"Session {session_id}: Processing general order query")
                return await process_general_order(customer_id, session_id, conversation)
                
            elif "priority" in message:
                # Handle priority-based queries (these don't require customer ID)
                priority = "High"  # Default
                if "critical" in message:
                    priority = "Critical"
                elif "medium" in message:
                    priority = "Medium"
                elif "low" in message:
                    priority = "Low"
                    
                logger.info(f"Session {session_id}: Processing priority-based order query for priority: {priority}")
                return await process_priority_orders(priority, session_id, conversation)
            else:
                # Generic order query without personal reference
                logger.info(f"Session {session_id}: Generic order query without customer context")
                response = "I need more information about which orders you're interested in. Are you looking for information about your own order? If so, please provide your Customer ID."
                conversation["history"].append({"role": "assistant", "content": response, "timestamp": datetime.now().isoformat()})
                return ChatResponse(response=response, session_id=session_id)
        else:
            # This is likely a product query
            logger.info(f"Session {session_id}: Processing product query")
            return await process_product_query(message, session_id, conversation)
            
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def process_recent_order(customer_id, session_id, conversation):
    """Process a request for the most recent order."""
    try:
        logger.info(f"Session {session_id}: Fetching recent order for customer: {customer_id}")
        
        async with httpx.AsyncClient() as client:
            logger.debug(f"Session {session_id}: Making request to {ORDER_SERVICE_URL}/order/recent/{customer_id}")
            response = await client.get(f"{ORDER_SERVICE_URL}/order/recent/{customer_id}")
            order_data = response.json()
            
            logger.debug(f"Session {session_id}: Received order data: {json.dumps(order_data, indent=2)}")
            
        context = str(order_data)
        logger.info(f"Session {session_id}: Running LLM chain for recent order response")
        response_text = order_chain.run(
            query="What are the details of my most recent order?", 
            context=context, 
            customer_id=customer_id
        )
        
        logger.info(f"Session {session_id}: LLM generated response: {response_text}")
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        logger.error(f"Session {session_id}: Error processing recent order: {str(e)}", exc_info=True)
        response_text = f"I'm sorry, I encountered an error while fetching your recent order: {str(e)}"
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)


async def process_category_order(customer_id, category, session_id, conversation):
    """Process a request for orders in a specific category."""
    try:
        logger.info(f"Session {session_id}: Fetching {category} orders for customer: {customer_id}")
        
        async with httpx.AsyncClient() as client:
            logger.debug(f"Session {session_id}: Making request to {ORDER_SERVICE_URL}/order/category/{customer_id}/{category}")
            response = await client.get(f"{ORDER_SERVICE_URL}/order/category/{customer_id}/{category}")
            order_data = response.json()
            
            logger.debug(f"Session {session_id}: Received order data: {json.dumps(order_data, indent=2)}")
            
        context = str(order_data)
        logger.info(f"Session {session_id}: Running LLM chain for category order response")
        response_text = order_chain.run(
            query=f"What are my orders for {category} products?", 
            context=context, 
            customer_id=customer_id
        )
        
        logger.info(f"Session {session_id}: LLM generated response: {response_text}")
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        logger.error(f"Session {session_id}: Error processing category order: {str(e)}", exc_info=True)
        response_text = f"I'm sorry, I encountered an error while fetching your {category} orders: {str(e)}"
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)


async def process_general_order(customer_id, session_id, conversation):
    """Process a general order lookup request."""
    try:
        logger.info(f"Session {session_id}: Fetching all orders for customer: {customer_id}")
        
        async with httpx.AsyncClient() as client:
            logger.debug(f"Session {session_id}: Making request to {ORDER_SERVICE_URL}/order/customer/{customer_id}")
            response = await client.get(f"{ORDER_SERVICE_URL}/order/customer/{customer_id}")
            order_data = response.json()
            
            logger.debug(f"Session {session_id}: Received order data: {json.dumps(order_data, indent=2)}")
            
        context = str(order_data)
        logger.info(f"Session {session_id}: Running LLM chain for general order response")
        response_text = order_chain.run(
            query="What are my order details?", 
            context=context, 
            customer_id=customer_id
        )
        
        logger.info(f"Session {session_id}: LLM generated response: {response_text}")
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        logger.error(f"Session {session_id}: Error processing general order: {str(e)}", exc_info=True)
        response_text = f"I'm sorry, I encountered an error while fetching your order information: {str(e)}"
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)


async def process_priority_orders(priority, session_id, conversation):
    """Process a request for orders with a specific priority."""
    try:
        logger.info(f"Session {session_id}: Fetching {priority}-priority orders")
        
        async with httpx.AsyncClient() as client:
            logger.debug(f"Session {session_id}: Making request to {ORDER_SERVICE_URL}/order/priority/{priority}/recent")
            response = await client.get(f"{ORDER_SERVICE_URL}/order/priority/{priority}/recent")
            order_data = response.json()
            
            logger.debug(f"Session {session_id}: Received order data: {json.dumps(order_data, indent=2)}")
            
        context = str(order_data)
        logger.info(f"Session {session_id}: Running LLM chain for priority orders response")
        response_text = order_chain.run(
            query=f"What are the recent {priority}-priority orders?", 
            context=context, 
            customer_id=None
        )
        
        logger.info(f"Session {session_id}: LLM generated response: {response_text}")
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)
    except Exception as e:
        logger.error(f"Session {session_id}: Error processing priority orders: {str(e)}", exc_info=True)
        response_text = f"I'm sorry, I encountered an error while fetching {priority}-priority orders: {str(e)}"
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)


async def process_product_query(message, session_id, conversation):
    """Process product-related queries."""
    try:
        if "top" in message and "rated" in message:
            # Handle top-rated products query
            category = None
            if "guitar" in message:
                category = "guitar"
                
            logger.info(f"Session {session_id}: Processing top-rated products query for category: {category}")
            
            async with httpx.AsyncClient() as client:
                logger.debug(f"Session {session_id}: Making request to {PRODUCT_SERVICE_URL}/product/top-rated with category={category}")
                response = await client.get(
                    f"{PRODUCT_SERVICE_URL}/product/top-rated",
                    params={"category": category, "limit": 5}
                )
                product_data = response.json()
                
                logger.debug(f"Session {session_id}: Received product data: {json.dumps(product_data, indent=2)}")
                
            context = str(product_data)
            logger.info(f"Session {session_id}: Running LLM chain for top-rated products response")
            response_text = product_chain.run(query=message, context=context)
            
            logger.info(f"Session {session_id}: LLM generated response: {response_text}")
            conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
            return ChatResponse(response=response_text, session_id=session_id)
            
        # Handle feature-specific queries
        features = {
            "thin": "thin strings",
            "microphone": "microphone",
            "stand": "stand",
            "wireless": "wireless",
            "beginner": "beginner"
        }
        
        feature = None
        for key, value in features.items():
            if key in message:
                feature = value
                break
                
        if feature:
            logger.info(f"Session {session_id}: Processing feature-specific product query for feature: {feature}")
            
            async with httpx.AsyncClient() as client:
                logger.debug(f"Session {session_id}: Making request to {PRODUCT_SERVICE_URL}/product/by-feature with feature={feature}")
                response = await client.get(
                    f"{PRODUCT_SERVICE_URL}/product/by-feature",
                    params={"feature": feature, "limit": 5}
                )
                product_data = response.json()
                
                logger.debug(f"Session {session_id}: Received product data: {json.dumps(product_data, indent=2)}")
                
            context = str(product_data)
            logger.info(f"Session {session_id}: Running LLM chain for feature-specific product response")
            response_text = product_chain.run(query=message, context=context)
            
            logger.info(f"Session {session_id}: LLM generated response: {response_text}")
            conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
            return ChatResponse(response=response_text, session_id=session_id)
            
        # Default: general search
        logger.info(f"Session {session_id}: Processing general product search query")
        
        async with httpx.AsyncClient() as client:
            logger.debug(f"Session {session_id}: Making request to {PRODUCT_SERVICE_URL}/product/search with query={message}")
            response = await client.get(
                f"{PRODUCT_SERVICE_URL}/product/search",
                params={"query": message, "limit": 5}
            )
            product_data = response.json()
            
            logger.debug(f"Session {session_id}: Received product data: {json.dumps(product_data, indent=2)}")
            
        context = str(product_data)
        logger.info(f"Session {session_id}: Running LLM chain for general product search response")
        response_text = product_chain.run(query=message, context=context)
        
        logger.info(f"Session {session_id}: LLM generated response: {response_text}")
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)
        
    except Exception as e:
        logger.error(f"Session {session_id}: Error processing product query: {str(e)}", exc_info=True)
        response_text = f"I'm sorry, I encountered an error while processing your product query: {str(e)}"
        conversation["history"].append({"role": "assistant", "content": response_text, "timestamp": datetime.now().isoformat()})
        return ChatResponse(response=response_text, session_id=session_id)

# Log application startup
@app.on_event("startup")
async def startup_event():
    logger.info("Chat service starting up")
    logger.info(f"Product Service URL: {PRODUCT_SERVICE_URL}")
    logger.info(f"Order Service URL: {ORDER_SERVICE_URL}")
    
# Log application shutdown
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Chat service shutting down")
    logger.info(f"Total sessions handled: {len(conversations)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("chat_api:app", host="127.0.0.1", port=8000, reload=True)


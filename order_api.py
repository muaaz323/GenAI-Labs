# order_service/api.py
from fastapi import FastAPI, HTTPException
import httpx
import asyncio
from typing import List, Dict, Any, Optional

app = FastAPI(title="Order Service", description="Service for querying order data")

MOCK_API_BASE_URL = "http://127.0.0.1:8001"  # Adjust as needed in deployment

@app.get("/order/customer/{customer_id}")
async def get_customer_orders(customer_id: int):
    """Get all orders for a specific customer."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MOCK_API_BASE_URL}/data/customer/{customer_id}")
            data = response.json()
            
            if "error" in data:
                raise HTTPException(status_code=404, detail=data["error"])
                
            # Sort by date (most recent first)
            sorted_data = sorted(data, key=lambda x: x["Order_Date"], reverse=True)
            return {"orders": sorted_data}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order/recent/{customer_id}")
async def get_most_recent_order(customer_id: int):
    """Get the most recent order for a customer."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MOCK_API_BASE_URL}/data/customer/{customer_id}")
            data = response.json()
            
            if "error" in data:
                raise HTTPException(status_code=404, detail=data["error"])
                
            # Sort by date (most recent first) and return the first one
            sorted_data = sorted(data, key=lambda x: x["Order_Date"], reverse=True)
            if sorted_data:
                return {"order": sorted_data[0]}
            else:
                return {"order": None}
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order/priority/{priority}/recent")
async def get_recent_priority_orders(priority: str, limit: int = 5):
    """Get the most recent orders with the specified priority."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MOCK_API_BASE_URL}/data/order-priority/{priority}")
            data = response.json()
            
            if "error" in data:
                raise HTTPException(status_code=404, detail=data["error"])
                
            # Sort by date (most recent first) and limit
            sorted_data = sorted(data, key=lambda x: x["Order_Date"], reverse=True)[:limit]
            return {"orders": sorted_data}
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/order/category/{customer_id}/{category}")
async def get_customer_category_orders(customer_id: int, category: str):
    """Get orders for a specific customer and product category."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MOCK_API_BASE_URL}/data/customer/{customer_id}")
            data = response.json()
            
            if "error" in data:
                raise HTTPException(status_code=404, detail=data["error"])
                
            # Filter by category
            filtered_data = [order for order in data if category.lower() in order["Product_Category"].lower()]
            
            if not filtered_data:
                return {"message": f"No orders found for customer {customer_id} in category '{category}'"}
                
            # Sort by date (most recent first)
            sorted_data = sorted(filtered_data, key=lambda x: x["Order_Date"], reverse=True)
            return {"orders": sorted_data}
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("order_api:app", host="127.0.0.1", port=8003, reload=True)


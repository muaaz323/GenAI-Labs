# product_service/api.py
from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Optional
from data_loader import ProductDataLoader

app = FastAPI(title="Product Service", description="Service for querying product data")

# Initialize data loader
product_data = ProductDataLoader("Product_Information_Dataset.csv")

@app.get("/product/search")
async def search_products(query: str, limit: int = 5):
    """Search products based on a query string."""
    try:
        results = product_data.search(query, k=limit)
        return {"products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/product/top-rated")
async def get_top_rated_products(category: Optional[str] = None, limit: int = 5):
    """Get top-rated products, optionally filtered by category."""
    try:
        df = product_data.df
        
        if category:
            filtered_df = df[df['categories'].str.contains(category, case=False, na=False)]
        else:
            filtered_df = df
            
        top_rated = filtered_df.sort_values(by='average_rating', ascending=False).head(limit)
        
        results = []
        for _, row in top_rated.iterrows():
            results.append({
                'title': row['title'],
                'description': row['description'],
                'features': row['features'],
                'average_rating': row['average_rating'],
                'price': row['price'],
                'categories': row['categories']
            })
            
        return {"products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/product/by-feature")
async def get_products_by_feature(feature: str, limit: int = 5):
    """Get products containing a specific feature."""
    try:
        df = product_data.df
        filtered_df = df[df['features'].str.contains(feature, case=False, na=False)]
        
        results = []
        for _, row in filtered_df.head(limit).iterrows():
            results.append({
                'title': row['title'],
                'description': row['description'],
                'features': row['features'],
                'average_rating': row['average_rating'],
                'price': row['price'],
                'categories': row['categories']
            })
            
        return {"products": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("product_api:app", host="127.0.0.1", port=8002, reload=True)


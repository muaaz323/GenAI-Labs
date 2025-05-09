# product_service/data_loader.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

class ProductDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.df = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_data()
        self.create_embeddings()
        
    def load_data(self):
        self.df = pd.read_csv(self.dataset_path)
        self.df.fillna('', inplace=True)
        
    def create_embeddings(self):
        # Combine relevant columns for embedding
        self.df['combined_text'] = (
            self.df['title'] + ' ' + 
            self.df['description'] + ' ' + 
            self.df['features'] + ' ' + 
            self.df['categories']
        )
        
        # Create embeddings
        embeddings = self.model.encode(self.df['combined_text'].tolist())
        embeddings = np.array([embedding for embedding in embeddings]).astype('float32')
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.embeddings = embeddings
        
    def search(self, query, k=5):
        query_vector = self.model.encode([query])
        query_vector = np.array(query_vector).astype('float32')
        
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df):
                results.append({
                    'title': self.df.iloc[idx]['title'],
                    'description': self.df.iloc[idx]['description'],
                    'features': self.df.iloc[idx]['features'],
                    'average_rating': self.df.iloc[idx]['average_rating'],
                    'price': self.df.iloc[idx]['price'],
                    'categories': self.df.iloc[idx]['categories'],
                    'relevance_score': 1 / (1 + distances[0][i])
                })
        
        return results
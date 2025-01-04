from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Configure the Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI
app = FastAPI()

def find_relevant_chunk(query: str):
    """Finds the most relevant chunk using embeddings from Pinecone."""
    # Generate the embedding using OpenAI's model
    try:
        response = openai.Embedding.create(
            input=query,
            model="text-embedding-ada-002",
            api_key=openai_api_key  
        )
        query_embedding = response['data'][0]['embedding']
        print(f"Generated Embeddings: {query_embedding}")
        query_results = index.query(vector=query_embedding, top_k=1)
        print(f"Pinecone Query Results: {query_results}")
        match = query_results['matches'][0]
        return match.get('metadata', {}).get('text', 'No text available')
    except Exception as e:
        logger.error(f"Error generating embeddings or querying Pinecone: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve relevant data.")

def generate_response(query: str):
    """Generates a chatbot response using Google's Gemini API."""
    if query.lower() in ["exit", "bye"]:
        return "Thank you for using the service. Goodbye!"

    selected_chunk = find_relevant_chunk(query)
    print(selected_chunk)
    prompt = f"Context: {selected_chunk}\n\nUser: {query}\nChatbot:"

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    
    except Exception as e:
        logger.error(f"Error generating response from Gemini API: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {e}")

@app.post("/process-text/")
async def process_text(request: Request):
    """
    Accepts JSON input with `text` and generates a chatbot response.
    """
    try:
        body = await request.json()
        text = body.get("text")
        if not text:
            raise HTTPException(status_code=400, detail="No text provided.")

        chatbot_response = generate_response(text)
        return JSONResponse(content={"response": chatbot_response})

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 

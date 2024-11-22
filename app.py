from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import os
import nltk
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

# Download NLTK data
nltk.download('punkt')

# Initialize FastAPI
app = FastAPI()

# Load SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Constants
TOKEN_LIMIT = 200
PDF_FILE_PATH = "countries.pdf"  # Replace this with your PDF file path

def extract_text_from_pdf(file_path):
    """Extracts text from a PDF file."""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {e}")

def preprocess_text(text, token_limit=TOKEN_LIMIT, overlap=2):
    """Tokenizes and chunks text."""
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0

    for sentence in sentences:
        sentence_length = len(nltk.word_tokenize(sentence))
        if current_chunk_size + sentence_length > token_limit:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] + [sentence]
            current_chunk_size = sum(len(nltk.word_tokenize(sent)) for sent in current_chunk)
        else:
            current_chunk.append(sentence)
            current_chunk_size += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def find_relevant_chunk(query, chunks):
    """Finds the most relevant chunk using cosine similarity."""
    query_embedding = embedding_model.encode([query])
    chunk_embeddings = embedding_model.encode(chunks)

    similarities = cosine_similarity(query_embedding, chunk_embeddings).flatten()
    most_relevant_chunk_index = np.argmax(similarities)
    return chunks[most_relevant_chunk_index]

def generate_response(query, chunks):
    """Generates a chatbot response using OpenAI."""
    if query.lower() in ["exit", "bye"]:
        return "Thank you for using the service. Goodbye!"
    # print(query)
    selected_chunk = find_relevant_chunk(query, chunks)
    # print(selected_chunk)
    prompt = f"Context: {selected_chunk}\n\nUser: {query}\nChatbot:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
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

        # Load and preprocess the PDF context only once (consider caching for optimization)
        if not os.path.exists(PDF_FILE_PATH):
            raise HTTPException(status_code=500, detail=f"PDF file '{PDF_FILE_PATH}' not found.")
        
        pdf_text = extract_text_from_pdf(PDF_FILE_PATH)
        chunks = preprocess_text(pdf_text)

        # Generate a chatbot response based on the input text
        chatbot_response = generate_response(text, chunks)

        # Return the chatbot response
        return JSONResponse(content={"response": chatbot_response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

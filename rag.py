import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from groq import Groq

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to generate embeddings
def generate_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = text.split('. ')  # Split text into sentences
    embeddings = model.encode(sentences)
    return sentences, embeddings

# Function to find the most relevant sentence using FAISS
def find_most_relevant_sentence(query, sentences, embeddings):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    # Create a FAISS index for the embeddings
    dimension = embeddings.shape[1]  # Embedding dimensions
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean distance)

    # Add embeddings to the FAISS index
    index.add(embeddings)

    # Search for the most similar sentence to the query
    _, indices = index.search(np.array(query_embedding), k=1)  # k=1 for most similar sentence
    most_relevant_index = indices[0][0]

    return sentences[most_relevant_index]

# Function to generate an answer using Groq
def generate_answer_with_groq(query, context):
    client = Groq(api_key="your_api_key")  # Replace with your Groq API key
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful learning assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"}
        ],
        model="llama-3.3-70b-versatile",  # Use the appropriate Groq model
    )
    return response.choices[0].message.content

# Main execution with loop and exit condition
def main():
    pdf_path = "sample.pdf"  
    text = extract_text_from_pdf(pdf_path)
    sentences, embeddings = generate_embeddings(text)
    embeddings = np.array(embeddings).astype('float32')  # Ensure embeddings are float32 for FAISS
    
    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the program. Goodbye!")
            break
        
        relevant_sentence = find_most_relevant_sentence(query, sentences, embeddings)
        answer = generate_answer_with_groq(query, relevant_sentence)
        
        print("\nMost Relevant Context:", relevant_sentence)
        print("\nGenerated Answer:", answer)

if __name__ == "__main__":
    main()
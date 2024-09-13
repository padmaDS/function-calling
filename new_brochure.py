from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import os
import chromadb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Example placeholder functions
def get_intent(question):
    question_lower = question.lower()
    
    # Check for form-related intents
    if any(word in question_lower for word in ["sales", "appointment", "block_apartment", "visit", "contact", "other projects"]):
        return "form"
    # Check for brochure-related intents
    if any(word in question_lower for word in ["brochure", "show the brochure"]):
        return "showDocument"
    
    # Return 'unknown' for any other intents
    return "unknown"


def initialize_openai_embeddings():
    """Initialize OpenAI embeddings."""
    return OpenAIEmbeddings()

def load_documents_from_directory(directory):
    """Load documents from the specified directory."""
    loader = DirectoryLoader(directory)
    return loader.load()

def split_documents_into_chunks(data, chunk_size=1000, chunk_overlap=100):
    """Split documents into chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(data)

def initialize_chroma_vector_db(docs, embeddings, collection_name="purvankara"):
    """Initialize Chroma vector store with documents and embeddings."""
    new_client = chromadb.EphemeralClient()
    return Chroma.from_documents(docs, embeddings, client=new_client, collection_name=collection_name)

def initialize_openai_llm(model_name="gpt-4o-mini", temperature=0):
    """Initialize OpenAI language model."""
    return ChatOpenAI(model_name=model_name, temperature=temperature)

def create_prompt_template():
    """Create a prompt template for the assistant."""
    template = """You are an Intelligent Indian Real Estate customer service AI Assistant.
    You are designed to respond to answers as per the input language regarding the Provident Park Square, it is a world-class residential project.
    You are primarily programmed to communicate in English. However, if user asks in another language,
    you must strictly respond in the same language as the user’s language. Do not respond in English for other language queries.
    - Provide the response with more professional and personalized in a good customer service oriented fashion.
   - Always provide address and map location when asked about the location details in any language.
 
    {context}

    Question: {question}
    Helpful Answer:"""
    return PromptTemplate.from_template(template)

def create_rag_chain(llm, retriever, prompt_template):
    """Create the RAG chain for querying."""
    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

def ask_question(rag_chain, question):
    """Query the RAG chain with a question."""
    response = rag_chain.invoke(question)
    return response.content

def setup_rag_chain():
    """Set up the entire RAG chain pipeline."""
    embeddings = initialize_openai_embeddings()
    documents = load_documents_from_directory('data')
    docs = split_documents_into_chunks(documents)
    vector_db = initialize_chroma_vector_db(docs, embeddings)
    llm = initialize_openai_llm()
    prompt_template = create_prompt_template()
    retriever = vector_db.as_retriever()
    return create_rag_chain(llm, retriever, prompt_template)

# Set up the RAG chain once for reuse
rag_chain = setup_rag_chain()

# API endpoint to query the RAG chain
@app.route('/purvankara', methods=['POST'])
def query_endpoint():
    # Get the question from the request
    question = request.json.get('query')

    # Detect user intent
    user_intent = get_intent(question)

    # If the intent is "show brochure", return the brochure directly with "showDocument" type
    if user_intent == "showDocument":
        return jsonify({
            "message": "Here is the brochure for more details.",
            "type": "showDocument",
            "brochure_url": "https://quadz.blob.core.windows.net/demo/provident-parksquare.pdf"
        })

    # If the intent matches sales, booking, or blocking an apartment, show the form
    elif user_intent == "form" :
        return jsonify({
            "answer": "Appreciate your interest, please provide your contact details, to assist you better.",
            "follow_up_required": True,
            "fields": ["name", "phoneNumber", "email"],
            "type": "form",
            "intent": user_intent
        })

    # Otherwise, invoke the RAG chain with the question
    answer = ask_question(rag_chain, question)

    # Check if the response contains a request for phone number and email
    if "please provide your name, phone number and email address" in answer.lower() or "please share your name, phone number and email address" in answer.lower():
        return jsonify({
            "answer": answer,
            "follow_up_required": True,
            "fields": ["name", "phoneNumber", "email"],
            "type": "form"
        })

    # Return the answer as JSON response
    return jsonify({"answer": answer})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

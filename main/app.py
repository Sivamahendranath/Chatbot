import asyncio
import torch
import streamlit as st
import os
import tempfile
import time
import fitz
import speech_recognition as sr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import requests
import base64
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Ensure an event loop is running
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Prevent Streamlit from misinterpreting torch.classes
torch.classes.__path__ = []

# Create directories if they don't exist
os.makedirs("documents", exist_ok=True)
os.makedirs("index", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Custom CSS to enhance UI
def local_css():
    st.markdown("""
    <style>
        .main {
            background-color: #f5f7f9;
        }
        .stButton>button {
            background-color: #4e8cff;
            color: white;
            border-radius: 5px;
            border: none;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton>button:hover {
            background-color: #3a7aff;
        }
        .user-bubble {
            background-color: #e6f3ff;
            padding: 15px;
            border-radius: 15px 15px 15px 0;
            margin-bottom: 15px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .bot-bubble {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 15px 15px 0 15px;
            margin-bottom: 15px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        .sidebar .stButton>button {
            width: 100%;
        }
        .chat-header {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .metadata {
            font-size: 0.8rem;
            color: #888;
            text-align: right;
        }
        .stExpander {
            border: none !important;
            box-shadow: none !important;
        }
        .upload-section {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .chat-section {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .sidebar-section {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }
        .source-box {
            background-color: #f8f9fa;
            border-left: 3px solid #4e8cff;
            padding: 10px;
            font-size: 0.9rem;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-active {
            background-color: #28a745;
        }
        .status-inactive {
            background-color: #dc3545;
        }
        .custom-file-upload {
            border: 1px dashed #ccc;
            display: inline-block;
            padding: 80px 0;
            cursor: pointer;
            width: 100%;
            text-align: center;
            border-radius: 5px;
        }
        .info-box {
            background-color: #e7f3fe;
            border-left: 4px solid #2196F3;
            padding: 12px;
            margin-bottom: 15px;
        }
        .error-box {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            padding: 12px;
            margin-bottom: 15px;
        }
        .success-box {
            background-color: #e8f5e9;
            border-left: 4px solid #4CAF50;
            padding: 12px;
            margin-bottom: 15px;
        }
        .truncate {
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            max-width: 100%;
        }
    </style>
    """, unsafe_allow_html=True)

def log_activity(action, details=None):
    """Log user activities for analytics and debugging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"{timestamp} - {action}"
    if details:
        log_entry += f" - {details}"
    
    log_file = os.path.join("logs", "app_activity.log")
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF with metadata"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name
    
    doc = fitz.open(tmp_path)
    text = ""
    metadata = {}
    
    # Extract document metadata
    metadata["title"] = doc.metadata.get("title", "Unknown")
    metadata["author"] = doc.metadata.get("author", "Unknown")
    metadata["creation_date"] = doc.metadata.get("creationDate", "Unknown")
    metadata["page_count"] = len(doc)
    
    # Extract text with page numbers
    for i, page in enumerate(doc):
        page_text = page.get_text()
        text += f"\n\n--- Page {i+1} ---\n\n" + page_text
    
    doc.close()
    os.unlink(tmp_path)
    return text, metadata

def process_text(text):
    """Process text into chunks with improved context preservation"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Add page number metadata to chunks where available
    enhanced_chunks = []
    for chunk in chunks:
        page_marker = None
        for line in chunk.split("\n"):
            if line.startswith("--- Page ") and line.endswith(" ---"):
                page_marker = line.replace("--- Page ", "").replace(" ---", "")
                break
        
        # Remove page markers from the actual text
        clean_chunk = chunk.replace("\n--- Page ", " [Page ").replace(" ---\n", "] ")
        enhanced_chunks.append(clean_chunk)
    
    return enhanced_chunks

def get_embeddings_model():
    """Load and return the embeddings model"""
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    
    # Cache the embeddings model in session state
    if 'embeddings_model' not in st.session_state:
        with st.spinner("Loading embeddings model..."):
            st.session_state.embeddings_model = HuggingFaceEmbeddings(
                model_name=model_name, 
                model_kwargs=model_kwargs
            )
    
    return st.session_state.embeddings_model

def create_vector_store(chunks, embeddings):
    """Create vector store from chunks"""
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def get_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = [model["name"] for model in response.json()["models"]]
            if models:
                return models
    except:
        pass
    
    # Default models if we can't get them from the API
    return ["mistral", "llama2", "gemma", "phi", "mixtral", "codellama"]

def check_ollama_status():
    """Check if Ollama is running and return status"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

def get_llm(temperature=0.3, ollama_model="mistral"):
    """Initialize and return the LLM"""
    try:
        from langchain.llms import Ollama
        return Ollama(model=ollama_model, temperature=temperature, num_ctx=4096)
    except Exception as e:
        st.error(f"Failed to load Ollama: {e}")
        log_activity("LLM loading error", str(e))
        return None

def setup_qa_chain(vectorstore, llm, use_memory=True):
    """Setup QA chain with improved prompting"""
    if llm is None:
        return None
    
    # Enhanced prompt with focus on context-specific answers
    prompt_template = """You are an expert AI assistant that provides accurate, helpful, and comprehensive answers based on the documents provided.

    Instructions:
    1. Use ONLY the following context to answer the question. Don't use prior knowledge.
    2. If the answer isn't fully contained in the context, say so clearly and explain what's missing.
    3. Include specific page numbers if available in the context.
    4. If you're uncertain about a point, express your level of confidence based on the context.
    5. Format your answer for clarity with bullet points, paragraphs, or other helpful structure.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Configure memory if enabled
    memory = None
    if use_memory:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="result"
        )
    
    # Configure retriever with semantic search
    retriever = vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance
        search_kwargs={
            "k": 5,  # Number of documents to return
            "fetch_k": 10,  # Fetch more, then rerank
            "lambda_mult": 0.7  # Diversity parameter
        }
    )
    
    # Setup the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True,
    )
    
    return qa_chain

def listen_to_speech():
    """Enhanced voice input function"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Please speak now", icon="🔊")
        r.adjust_for_ambient_noise(source, duration=0.5)
        audio = r.listen(source, timeout=7, phrase_time_limit=20)
        st.info("Processing your speech...", icon="⏳")
    
    try:
        text = r.recognize_google(audio)
        return True, text
    except sr.UnknownValueError:
        return False, "Sorry, I couldn't understand what you said. Please try again."
    except sr.RequestError:
        return False, "Could not request results. Please check your network connection."
    except Exception as e:
        return False, f"Error: {str(e)}"

def generate_text_visualization(text, title="Word Frequency"):
    """Generate word frequency visualization"""
    from collections import Counter
    import re
    
    # Clean and tokenize text
    words = re.findall(r'\b[a-zA-Z]{3,15}\b', text.lower())
    
    # Remove common stopwords
    stopwords = {'the', 'and', 'to', 'of', 'a', 'in', 'for', 'on', 'is', 'that', 'this', 'with', 'be', 'are', 'as', 'not'}
    filtered_words = [word for word in words if word not in stopwords]
    
    # Count word frequencies
    word_counts = Counter(filtered_words).most_common(15)
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    words, counts = zip(*word_counts) if word_counts else ([], [])
    ax.barh(words, counts, color='steelblue')
    ax.set_title(title)
    ax.set_xlabel('Frequency')
    
    # Save to buffer
    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    return buf

def get_document_statistics(text, metadata=None):
    """Generate document statistics"""
    stats = {}
    stats["Total Characters"] = len(text)
    stats["Total Words"] = len(text.split())
    stats["Total Paragraphs"] = len([p for p in text.split("\n\n") if p.strip()])
    stats["Average Word Length"] = round(sum(len(word) for word in text.split()) / max(1, len(text.split())), 2)
    
    if metadata:
        stats.update(metadata)
    
    return stats

def export_chat_history(chat_history):
    """Export chat history to markdown"""
    markdown = "# PDF Chat Session\n\n"
    markdown += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for i, exchange in enumerate(chat_history):
        markdown += f"## Exchange {i+1}\n\n"
        markdown += f"**Question:** {exchange['user']}\n\n"
        markdown += f"**Answer:** {exchange['bot']}\n\n"
        if "sources" in exchange and exchange["sources"]:
            markdown += "### Sources\n\n"
            markdown += exchange["sources"].replace("**Sources:**", "") + "\n\n"
        if "time" in exchange and exchange["time"]:
            markdown += f"*{exchange['time']}*\n\n"
        markdown += "---\n\n"
    
    return markdown

# Helper function to handle voice input
def handle_voice_input():
    try:
        success, spoken_text = listen_to_speech()
        if success:
            st.session_state.input_text = spoken_text
            return True
        else:
            st.warning(spoken_text)
            return False
    except Exception as e:
        st.error(f"Speech recognition error: {e}")
        st.info("Make sure your microphone is connected and permissions are granted")
        return False

# Initialize session state variables at the start
def init_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    if "pdf_metadata" not in st.session_state:
        st.session_state.pdf_metadata = None
    if "pdf_text" not in st.session_state:
        st.session_state.pdf_text = None
    if "use_memory" not in st.session_state:
        st.session_state.use_memory = True
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""

def main():
    st.set_page_config(
        page_title="Smart PDF Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state first
    init_session_state()
    
    local_css()
    
    # Header with logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/color/96/000000/pdf.png", width=80)
    with col2:
        st.title("Smart PDF Assistant")
        st.markdown("Your intelligent document analysis companion powered by Ollama")
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Configuration")
    
    with st.sidebar.container():
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        
        # Ollama status indicator
        ollama_status = check_ollama_status()
        status_color = "status-active" if ollama_status else "status-inactive"
        status_text = "Active" if ollama_status else "Inactive"
        st.markdown(f'<div><span class="status-indicator {status_color}"></span> Ollama Status: {status_text}</div>', unsafe_allow_html=True)
        
        if not ollama_status:
            st.warning("⚠️ Ollama is not running. Start it with: `ollama serve`")
        
        # Model selection
        ollama_models = get_ollama_models()
        ollama_model = st.selectbox(
            "Select Ollama model", 
            ollama_models,
            help="Select a model you've pulled with 'ollama pull <model-name>'"
        )
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            temperature = st.slider(
                "Response Creativity", 
                0.0, 1.0, 0.3, 
                help="Lower = more focused, Higher = more creative"
            )
            
            st.session_state.use_memory = st.checkbox(
                "Enable Memory", 
                value=st.session_state.use_memory,
                help="Remember previous questions in the same session"
            )
            
            st.session_state.dark_mode = st.checkbox(
                "Dark Mode", 
                value=st.session_state.dark_mode,
                help="Toggle dark mode appearance"
            )
        
        # Update model settings
        if st.session_state.pdf_processed and st.button("Update Model Settings", key="update_model"):
            with st.spinner("Updating model settings..."):
                llm = get_llm(temperature, ollama_model)
                if llm:
                    st.session_state.qa_chain = setup_qa_chain(
                        st.session_state.vectorstore, 
                        llm,
                        use_memory=st.session_state.use_memory
                    )
                    st.success("✅ Model settings updated!")
                    log_activity("Model settings updated", f"Model: {ollama_model}, Temp: {temperature}")
                else:
                    st.error(f"❌ Failed to initialize model '{ollama_model}'")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # PDF Stats and Export section
    if st.session_state.pdf_processed:
        st.sidebar.markdown("### 📊 Document Information")
        with st.sidebar.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            if st.session_state.pdf_metadata:
                st.markdown(f"**File:** {st.session_state.pdf_name}")
                st.markdown(f"**Pages:** {st.session_state.pdf_metadata.get('page_count', 'Unknown')}")
                st.markdown(f"**Title:** {st.session_state.pdf_metadata.get('title', 'Unknown')}")
                
                # Word count and other stats
                if st.session_state.pdf_text:
                    word_count = len(st.session_state.pdf_text.split())
                    st.markdown(f"**Words:** {word_count:,}")
                    
                    # Generate and show visualization
                    if st.button("Generate Insights", key="generate_insights"):
                        viz_buffer = generate_text_visualization(st.session_state.pdf_text)
                        st.image(viz_buffer)
            
            # Export functionality
            if st.session_state.chat_history:
                if st.button("Export Chat History"):
                    markdown_export = export_chat_history(st.session_state.chat_history)
                    b64 = base64.b64encode(markdown_export.encode()).decode()
                    now = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pdf_chat_export_{now}.md"
                    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">📥 Download Markdown</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    log_activity("Exported chat history")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    main_col1, main_col2 = st.columns([2, 3])
    
    # PDF Upload Section
    with main_col1:
        st.markdown("### 📄 Upload Document")
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("Choose a PDF file", type="pdf", help="Select a PDF document to analyze")
        
        if pdf_file:
            # Display file info
            file_details = {
                "Filename": pdf_file.name,
                "File size": f"{pdf_file.size / 1024:.2f} KB"
            }
            st.json(file_details)
            
            # Process PDF button
            process_col1, process_col2 = st.columns(2)
            process_button = process_col1.button("Process PDF", key="process_pdf")
            reset_button = process_col2.button("Reset", key="reset_pdf")
            
            if reset_button:
                st.session_state.pdf_processed = False
                st.session_state.pdf_name = None
                st.session_state.vectorstore = None
                st.session_state.qa_chain = None
                st.session_state.chat_history = []
                st.session_state.pdf_metadata = None
                st.session_state.pdf_text = None
                st.session_state.input_text = ""
                st.rerun()
            
            if process_button or (pdf_file and (not st.session_state.pdf_processed or st.session_state.pdf_name != pdf_file.name)):
                with st.spinner("Processing PDF... This may take a moment."):
                    # Extract text and metadata
                    text, metadata = extract_text_from_pdf(pdf_file)
                    st.session_state.pdf_text = text
                    st.session_state.pdf_metadata = metadata
                    
                    # Process text into chunks
                    chunks = process_text(text)
                    
                    # Get embeddings model and create vector store
                    embeddings = get_embeddings_model()
                    st.session_state.vectorstore = create_vector_store(chunks, embeddings)
                    
                    # Initialize LLM and QA chain
                    llm = get_llm(temperature, ollama_model)
                    if llm:
                        st.session_state.qa_chain = setup_qa_chain(
                            st.session_state.vectorstore, 
                            llm,
                            use_memory=st.session_state.use_memory
                        )
                        st.session_state.pdf_processed = True
                        st.session_state.pdf_name = pdf_file.name
                        st.session_state.chat_history = []  # Clear chat history for new PDF
                        
                        st.success(f"✅ Successfully processed: {pdf_file.name}")
                        log_activity("PDF processed", f"File: {pdf_file.name}, Size: {pdf_file.size / 1024:.2f} KB")
                    else:
                        st.error(f"❌ Failed to initialize Ollama with model '{ollama_model}'")
                        st.info(f"Try running: ollama pull {ollama_model}")
        else:
            st.info("Please upload a PDF document to get started", icon="ℹ️")
        
        # Document preview 
        if st.session_state.pdf_processed and st.session_state.pdf_text:
            with st.expander("Document Preview"):
                preview_text = st.session_state.pdf_text[:1500] + "..." if len(st.session_state.pdf_text) > 1500 else st.session_state.pdf_text
                st.text_area("Document Content (Preview)", preview_text, height=300, disabled=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat Interface
    with main_col2:
        st.markdown("### 💬 Chat with Your Document")
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        
        if not st.session_state.pdf_processed:
            st.info("Upload and process a PDF to start asking questions", icon="👈")
        else:
            # Voice input button - place before text input
            voice_col1, voice_col2, voice_col3 = st.columns([1, 1, 1])
            
            with voice_col1:
                if st.button("🎙️ Voice Input", key="voice_button"):
                    handle_voice_input()
                    st.rerun()
            
            with voice_col3:
                if st.button("Clear Chat", key="clear_chat"):
                    st.session_state.chat_history = []
                    st.session_state.last_query = None
                    st.session_state.input_text = ""
                    log_activity("Chat history cleared")
                    st.rerun()
            
            # Input area with pre-populated value from session state
            query = st.text_input(
                "Ask a question about your document:", 
                key="query_input",
                value=st.session_state.input_text
            )
            
            # Clear the input_text after it's used
            if st.session_state.input_text:
                st.session_state.input_text = ""
            
            with voice_col2:
                submit_button = st.button("Submit Question", key="submit_button")
            
            # Process query
            if submit_button or (query and query.strip() and st.session_state.pdf_processed and 
                             (st.session_state.last_query is None or st.session_state.last_query != query)):
                if not query:
                    st.warning("Please enter a question")
                else:
                    current_query = query  # Store the current query to use in this iteration
                    st.session_state.last_query = current_query
                    with st.spinner("Generating answer..."):
                        try:
                            start_time = time.time()
                            # Get response and properly extract answer and sources
                            response = st.session_state.qa_chain({"query": current_query})
                            end_time = time.time()
                            
                            # Extract the answer from the response
                            if "result" in response:
                                answer = response["result"]
                            else:
                                # Fallback for different response formats
                                answer = response.get("answer", response.get("output", "No answer found"))
                            
                            # Get source documents
                            sources = response.get("source_documents", [])
                            
                            # Process and format source documents
                            source_texts = []
                            for i, doc in enumerate(sources):
                                source_text = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                # Extract page numbers if available
                                page_info = ""
                                if "[Page " in source_text:
                                    import re
                                    page_match = re.search(r'\[Page (\d+)\]', source_text)
                                    if page_match:
                                        page_info = f" (Page {page_match.group(1)})"
                                
                                source_texts.append(f"Source {i+1}{page_info}: {source_text}")
                            
                            source_info = "\n\n".join(source_texts)
                            
                            # Record the exchange
                            st.session_state.chat_history.append({
                                "user": current_query, 
                                "bot": answer,
                                "sources": source_info,
                                "time": f"Response time: {end_time - start_time:.2f} seconds",
                                "timestamp": datetime.now().strftime("%H:%M:%S")
                            })
                            
                            # Log the activity
                            log_activity("Question answered", f"Q: {current_query[:50]}...")
                            
                            # Clear input by forcing a rerun
                            st.session_state.input_text = ""
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")
                            log_activity("Error generating answer", str(e))
            
            # Display chat history
            if st.session_state.chat_history:
                st.markdown("#### Conversation History")
                for i, message in enumerate(reversed(st.session_state.chat_history)):
                    # User message
                    st.markdown(f'<div class="user-bubble"><div class="chat-header">You:</div>{message["user"]}</div>', unsafe_allow_html=True)
                    
                    # Bot message
                    st.markdown(f'<div class="bot-bubble"><div class="chat-header">AI Assistant:</div>{message["bot"]}</div>', unsafe_allow_html=True)
                    
                    # Show sources in an expander
                    with st.expander("Show Sources"):
                        st.markdown('<div class="source-box">', unsafe_allow_html=True)
                        st.markdown(f"**Sources:**\n{message['sources']}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show timing info
                    st.markdown(f'<div class="metadata">{message["time"]} | {message["timestamp"]}</div>', unsafe_allow_html=True)
                    
                    # Add separator between messages
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")
            else:
                if st.session_state.pdf_processed:
                    st.info("Start asking questions about your document!", icon="💡")
                    
                    # Suggest some questions based on the document
                    st.markdown("#### Suggested Questions:")
                    example_questions = [
                        "What is the main topic of this document?",
                        "Can you summarize this document for me?",
                        "What are the key points in this document?",
                        f"What information does this document contain about {st.session_state.pdf_name.split('.')[0]}?",
                        "Are there any tables or figures in this document? What do they show?"
                    ]
                    
                    for question in example_questions:
                        if st.button(question, key=f"suggest_{hash(question)}"):
                            st.session_state.input_text = question
                            st.rerun()
                        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888;">
        Smart PDF Assistant | Powered by Ollama, LangChain & Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

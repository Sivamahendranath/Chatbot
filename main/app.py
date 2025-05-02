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
from bs4 import BeautifulSoup
from lxml.html.clean import Cleaner
from newspaper import Article
import html2text
import trafilatura
from urllib.parse import urlparse
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
        .tab-section {
            margin-top: 15px;
            margin-bottom: 15px;
        }
        .url-input {
            margin-bottom: 15px;
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

def normalize_url(url):
    """Ensure URL has proper scheme"""
    if not url.startswith(('http://', 'https://')):
        return 'https://' + url
    return url

def get_domain(url):
    """Extract domain from URL"""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    if domain.startswith('www.'):
        domain = domain[4:]
    return domain

def fetch_web_content(url):
    """Fetch web content from a URL with enhanced error handling and user-agent"""
    try:
        # Normalize URL if needed
        url = normalize_url(url)
        
        # Use a realistic user agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        
        # Make request with timeout and headers
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        
        # Check if response is valid HTML
        content_type = resp.headers.get('Content-Type', '')
        if 'text/html' not in content_type and 'application/xhtml+xml' not in content_type:
            return None, f"URL returned non-HTML content: {content_type}"
        
        return resp.text, None
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') else "unknown"
        return None, f"HTTP Error: {status_code}"
    except requests.exceptions.ConnectionError:
        return None, "Connection Error: Failed to connect to the server"
    except requests.exceptions.Timeout:
        return None, "Timeout Error: The request timed out"
    except requests.exceptions.RequestException as e:
        return None, f"Request Error: {str(e)}"
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"

def extract_text_from_url(url):
    """Enhanced function to extract text from a URL using multiple methods"""
    html, error = fetch_web_content(url)
    
    if error:
        return None, None, f"Error fetching URL: {error}"
    
    if not html:
        return None, None, "No content received from the URL"
    
    metadata = {
        "source": url,
        "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "domain": get_domain(url)
    }
    
    # Method 1: Trafilatura - specialized for article extraction
    trafilatura_text = ""
    try:
        downloaded = trafilatura.extract(html, include_comments=False, include_tables=True, output_format="text")
        if downloaded:
            trafilatura_text = downloaded
            metadata["extraction_method"] = "trafilatura"
    except Exception as e:
        metadata["trafilatura_error"] = str(e)
    
    # Method 2: Newspaper3k - good for news articles
    newspaper_text = ""
    try:
        article = Article(url)
        article.set_html(html)
        article.parse()
        
        # Extract metadata
        if article.title:
            metadata["title"] = article.title
        if article.authors:
            metadata["authors"] = ", ".join(article.authors)
        if article.publish_date:
            metadata["article_date"] = article.publish_date.strftime("%Y-%m-%d") if hasattr(article.publish_date, "strftime") else str(article.publish_date)
            
        newspaper_text = article.text
    except Exception as e:
        metadata["newspaper_error"] = str(e)
    
    # Method 3: BeautifulSoup - general HTML parsing
    bs4_text = ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        
        # Get title from soup if not already found
        if "title" not in metadata and soup.title:
            metadata["title"] = soup.title.text.strip()
        
        # Extract metadata
        meta_tags = soup.find_all("meta")
        for tag in meta_tags:
            if tag.get("name") in ["description", "author", "keywords"]:
                metadata[tag.get("name")] = tag.get("content")
            elif tag.get("property") in ["og:title", "og:description", "og:site_name"]:
                prop = tag.get("property")[3:]  # Remove 'og:' prefix
                metadata[f"og_{prop}"] = tag.get("content")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
            element.decompose()
            
        # Get article content if available
        article_content = None
        for elem in ["article", "main", "#content", ".content", "#main", ".main", ".post", ".article"]:
            if elem.startswith(("#", ".")):
                article_content = soup.select_one(elem)
            else:
                article_content = soup.find(elem)
                
            if article_content:
                break
        
        # If article content is found, use it
        if article_content:
            paragraphs = article_content.find_all("p")
            bs4_text = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        else:
            # Fallback to all paragraphs
            paragraphs = soup.find_all("p")
            bs4_text = "\n\n".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
        # Get all headings for structure
        headings = []
        for i in range(1, 7):
            for h in soup.find_all(f'h{i}'):
                text = h.get_text().strip()
                if text:  # Only add non-empty headings
                    headings.append(f"Heading {i}: {text}")
        
        # Combine headings and paragraphs if headings found
        if headings:
            bs4_text = "\n\n".join(headings) + "\n\n" + bs4_text
    
    except Exception as e:
        metadata["bs4_error"] = str(e)
    
    # Method 4: HTML2Text as a fallback
    html2text_result = ""
    try:
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_tables = False
        h.single_line_break = True
        html2text_result = h.handle(html)
    except Exception as e:
        metadata["html2text_error"] = str(e)
    
    # Select the best extracted text based on quality and length
    candidates = [
        (trafilatura_text, "trafilatura", len(trafilatura_text.split())),
        (newspaper_text, "newspaper3k", len(newspaper_text.split())),
        (bs4_text, "beautifulsoup", len(bs4_text.split())),
        (html2text_result, "html2text", len(html2text_result.split()))
    ]
    
    # Sort by word count (descending)
    candidates.sort(key=lambda x: x[2], reverse=True)
    
    # Choose the best candidate with at least 50 words
    final_text = None
    for text, method, word_count in candidates:
        if word_count >= 50:
            final_text = text
            metadata["extraction_method"] = method
            metadata["word_count"] = word_count
            break
    
    # If no good candidate found
    if not final_text:
        # Try combining all methods as a last resort
        combined_text = "\n\n".join([t for t, _, _ in candidates if t])
        if len(combined_text.split()) >= 30:
            final_text = combined_text
            metadata["extraction_method"] = "combined"
            metadata["word_count"] = len(combined_text.split())
        else:
            return None, None, "Failed to extract meaningful text from the URL"
        
    # Clean up the final text
    final_text = "\n".join([line.strip() for line in final_text.splitlines() if line.strip()])
    
    # Add a few additional metadata points
    metadata["text_length"] = len(final_text)
    metadata["extraction_success"] = True
    
    return final_text, metadata, None

def process_text(text, source_type="pdf"):
    """Process text into chunks with improved context preservation"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Add source type and page number metadata to chunks where available
    enhanced_chunks = []
    for chunk in chunks:
        # For PDFs with page markers
        if source_type == "pdf":
            page_marker = None
            for line in chunk.split("\n"):
                if line.startswith("--- Page ") and line.endswith(" ---"):
                    page_marker = line.replace("--- Page ", "").replace(" ---", "")
                    break
            
            # Remove page markers from the actual text
            clean_chunk = chunk.replace("\n--- Page ", " [Page ").replace(" ---\n", "] ")
        else:
            # For web content
            clean_chunk = chunk
            
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
    markdown = "# Document Chat Session\n\n"
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
    if "document_processed" not in st.session_state:
        st.session_state.document_processed = False
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    if "document_metadata" not in st.session_state:
        st.session_state.document_metadata = None
    if "document_text" not in st.session_state:
        st.session_state.document_text = None
    if "use_memory" not in st.session_state:
        st.session_state.use_memory = True
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "document_source" not in st.session_state:
        st.session_state.document_source = None  # "pdf" or "url"
    if "url" not in st.session_state:
        st.session_state.url = None

def main():
    st.set_page_config(
        page_title="Smart Document Assistant",
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
        st.title("Smart Document Assistant")
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
        if st.session_state.document_processed and st.button("Update Model Settings", key="update_model"):
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
    
    # Stats and Export section
    if st.session_state.document_processed:
        st.sidebar.markdown("### 📊 Document Information")
        with st.sidebar.container():
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            
            if st.session_state.document_metadata:
                st.markdown(f"**Source:** {st.session_state.document_source.upper()}")
                st.markdown(f"**Name:** {st.session_state.document_name}")
                
                # Display source-specific metadata
                if st.session_state.document_source == "pdf" and "page_count" in st.session_state.document_metadata:
                    st.markdown(f"**Pages:** {st.session_state.document_metadata.get('page_count', 'Unknown')}")
                    st.markdown(f"**Title:** {st.session_state.document_metadata.get('title', 'Unknown')}")
                    st.markdown(f"**Author:** {st.session_state.document_metadata.get('author', 'Unknown')}")
                elif st.session_state.document_source == "url" and "domain" in st.session_state.document_metadata:
                    st.markdown(f"**Domain:** {st.session_state.document_metadata.get('domain', 'Unknown')}")
                    st.markdown(f"**Title:** {st.session_state.document_metadata.get('title', st.session_state.document_metadata.get('og_title', 'Unknown'))}")
                    if "word_count" in st.session_state.document_metadata:
                        st.markdown(f"**Word Count:** {st.session_state.document_metadata.get('word_count', 'Unknown')}")
            
            # Document statistics
            if st.session_state.document_text:
                with st.expander("📈 View Statistics"):
                    total_words = len(st.session_state.document_text.split())
                    total_chars = len(st.session_state.document_text)
                    st.markdown(f"**Total Words:** {total_words}")
                    st.markdown(f"**Total Characters:** {total_chars}")
                    
                    # Generate and display visualization
                    if total_words > 50:  # Only if enough words
                        viz_buffer = generate_text_visualization(st.session_state.document_text)
                        st.image(viz_buffer, caption="Word Frequency", use_container_width=True)
            
            # Export functionality
            if st.session_state.chat_history:
                if st.button("📥 Export Chat History"):
                    markdown_text = export_chat_history(st.session_state.chat_history)
                    
                    # Create download button for the markdown file
                    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"chat_history_{time_str}.md"
                    
                    # Convert to bytes for download
                    b64 = base64.b64encode(markdown_text.encode()).decode()
                    href = f'<a href="data:file/markdown;base64,{b64}" download="{filename}">Click here to download</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    log_activity("Export chat history")
            
            # Reset button
            if st.button("🔄 Reset Assistant"):
                for key in list(st.session_state.keys()):
                    if key not in ["dark_mode"]:
                        del st.session_state[key]
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    # Document upload section
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("### 📄 Upload Your Document")
        
        # Tabs for different input methods
        tab1, tab2 = st.tabs(["Upload PDF", "Enter URL"])
        
        with tab1:
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
                if st.button("Process PDF", key="process_pdf"):
                    with st.spinner("Processing PDF..."):
                        # Extract text from PDF
                        text, metadata = extract_text_from_pdf(uploaded_file)
                        
                        if text:
                            # Process text into chunks
                            chunks = process_text(text, source_type="pdf")
                            
                            # Get embeddings model
                            embeddings = get_embeddings_model()
                            
                            # Create vector store
                            vectorstore = create_vector_store(chunks, embeddings)
                            
                            # Get LLM
                            llm = get_llm(temperature, ollama_model)
                            
                            if llm:
                                # Setup QA chain
                                qa_chain = setup_qa_chain(
                                    vectorstore, 
                                    llm,
                                    use_memory=st.session_state.use_memory
                                )
                                
                                # Store in session state
                                st.session_state.document_processed = True
                                st.session_state.vectorstore = vectorstore
                                st.session_state.qa_chain = qa_chain
                                st.session_state.document_name = uploaded_file.name
                                st.session_state.document_metadata = metadata
                                st.session_state.document_text = text
                                st.session_state.document_source = "pdf"
                                st.session_state.chat_history = []
                                
                                st.success(f"✅ PDF processed successfully: {uploaded_file.name}")
                                log_activity("PDF processed", uploaded_file.name)
                            else:
                                st.error("❌ Failed to initialize LLM. Is Ollama running?")
                        else:
                            st.error("❌ Failed to extract text from PDF")
        
        with tab2:
            url = st.text_input("Enter webpage URL", key="url_input")
            if url:
                if st.button("Process URL", key="process_url"):
                    with st.spinner("Processing webpage..."):
                        # Extract text from URL
                        text, metadata, error = extract_text_from_url(url)
                        
                        if text and not error:
                            # Process text into chunks
                            chunks = process_text(text, source_type="url")
                            
                            # Get embeddings model
                            embeddings = get_embeddings_model()
                            
                            # Create vector store
                            vectorstore = create_vector_store(chunks, embeddings)
                            
                            # Get LLM
                            llm = get_llm(temperature, ollama_model)
                            
                            if llm:
                                # Setup QA chain
                                qa_chain = setup_qa_chain(
                                    vectorstore, 
                                    llm,
                                    use_memory=st.session_state.use_memory
                                )
                                
                                # Store in session state
                                st.session_state.document_processed = True
                                st.session_state.vectorstore = vectorstore
                                st.session_state.qa_chain = qa_chain
                                st.session_state.document_name = metadata.get("title", url)
                                st.session_state.document_metadata = metadata
                                st.session_state.document_text = text
                                st.session_state.document_source = "url"
                                st.session_state.url = url
                                st.session_state.chat_history = []
                                
                                st.success(f"✅ URL processed successfully")
                                log_activity("URL processed", url)
                            else:
                                st.error("❌ Failed to initialize LLM. Is Ollama running?")
                        else:
                            st.error(f"❌ {error}")
        
        # Preview document if processed
        if st.session_state.document_processed and st.session_state.document_text:
            with st.expander("📝 Document Preview"):
                preview_text = st.session_state.document_text[:2000] + "..." if len(st.session_state.document_text) > 2000 else st.session_state.document_text
                st.text_area("Document content", preview_text, height=300)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat interface
    with col2:
        st.markdown('<div class="chat-section">', unsafe_allow_html=True)
        st.markdown("### 💬 Chat with Your Document")
        
        # Display chat history
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.markdown(f'<div class="user-bubble"><div class="chat-header">You:</div>{chat["user"]}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="bot-bubble"><div class="chat-header">Assistant:</div>{chat["bot"]}</div>', unsafe_allow_html=True)
                
                if "sources" in chat and chat["sources"]:
                    with st.expander("View Sources"):
                        st.markdown(f'<div class="source-box">{chat["sources"]}</div>', unsafe_allow_html=True)
        
        # Input area
        if st.session_state.document_processed:
            input_col1, input_col2 = st.columns([6, 1])
            
            with input_col1:
                user_input = st.text_area("Your question", value=st.session_state.input_text, height=80, key="question_input")
            
            with input_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                voice_button = st.button("🎤", help="Voice Input")
                if voice_button:
                    handle_voice_input()
                    st.rerun()
            
            # Submit button
            submit_col1, submit_col2 = st.columns([1, 6])
            with submit_col1:
                submit_button = st.button("Send")
            with submit_col2:
                # Voice input button
                if st.session_state.qa_chain is None:
                    st.warning("⚠️ LLM not initialized. Is Ollama running?")
            
            # Process user input when submit button is clicked
            if submit_button and user_input.strip():
                # Store the query
                query = user_input.strip()
                st.session_state.last_query = query
                st.session_state.input_text = ""  # Clear input
                
                # Create a placeholder for the assistant's response
                response_placeholder = st.empty()
                response_placeholder.markdown('<div class="bot-bubble" style="background-color: black; color: white;"><div class="chat-header" style="color: white;">Assistant:</div>Thinking...</div>', unsafe_allow_html=True)                
                try:
                    # Get response from QA chain
                    start_time = time.time()
                    result = st.session_state.qa_chain({"query": query})
                    end_time = time.time()
                    
                    answer = result.get("result", "I couldn't find an answer in the document.")
                    source_docs = result.get("source_documents", [])
                    
                    # Format sources
                    sources_text = ""
                    if source_docs:
                        sources_text = "**Sources:**\n\n"
                        for i, doc in enumerate(source_docs[:3]):  # Show top 3 sources
                            content = doc.page_content
                            # Truncate if too long
                            if len(content) > 300:
                                content = content[:300] + "..."
                            sources_text += f"Source {i+1}:\n{content}\n\n"
                    
                    # Calculate time taken
                    time_taken = round(end_time - start_time, 2)
                    time_text = f"Response time: {time_taken} seconds"
                    
                    # Update chat history
                    chat_entry = {
                        "user": query,
                        "bot": answer,
                        "time": time_text
                    }
                    
                    if sources_text:
                        chat_entry["sources"] = sources_text
                    
                    st.session_state.chat_history.append(chat_entry)
                    
                    # Remove the placeholder and show the complete chat history
                    response_placeholder.empty()
                    
                    # Log activity
                    log_activity("Question answered", f"Q: {query[:50]}...")
                    
                    # Rerun to refresh chat display
                    st.rerun()
                except Exception as e:
                    response_placeholder.error(f"Error: {str(e)}")
                    log_activity("Error in QA", str(e))
        else:
            st.info("📤 Please upload and process a document first")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

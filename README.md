📚 Smart PDF Assistant (Offline Chatbot)
An intelligent, voice-enabled, offline chatbot that lets you chat with any PDF file in natural language. It uses local Large Language Models (LLMs) through Ollama, LangChain for document retrieval, and Streamlit for a friendly interface — all running entirely on your local machine with no internet required once setup is complete.

🔧 Key Features

Feature	Description
📄 PDF Upload	Upload and process any PDF document
🧠 Local LLM	Powered by Ollama to run LLMs like Mistral, LLaMA2 locally
🔍 Intelligent Retrieval	Uses LangChain + FAISS for semantic search and context-aware question answering
💬 Chat Interface	Ask natural language questions and receive detailed answers
🧠 Memory Option	Optionally remember previous questions in a session
🎙️ Voice Input	Speak your questions using your mic (Google Speech or Whisper)
📊 Document Insights	Get visual analysis like word frequency and document stats
📥 Export Chats	Export your conversation in Markdown format
🌗 Theming	Optional dark mode UI
📁 Folder Structure
perl
Copy
Edit
📦 project/
 ┣ 📁 documents/          # Uploaded PDFs
 ┣ 📁 index/              # FAISS vector index
 ┣ 📁 logs/               # User activity logs
 ┣ 📄 app.txt         # Main app file (rename to app.py)
 ┣ 📄 requirements.txt    # Python dependencies
 ┗ 📄 README.md           # You're here!
🖥️ Requirements
Make sure Python 3.8+ is installed. Then install the required packages:

requirements.txt
txt
Copy
Edit
streamlit
torch
PyMuPDF
speechrecognition
langchain
langchain-community
faiss-cpu
sentence-transformers
requests
matplotlib
pandas
numpy
scipy
pydub
transformers
accelerate
python-dotenv
✅ For offline voice input, you can optionally install OpenAI's Whisper:

pip install git+https://github.com/openai/whisper.git
⚙️ Setup Instructions
1. Clone the Project

git clone https://github.com/yourusername/pdf-assistant.git
cd pdf-assistant
2. Install Dependencies

pip install -r requirements.txt
3. Install Ollama
Visit https://ollama.com and install Ollama.

Start the server:

ollama serve
Pull a local LLM:

ollama pull mistral
🚀 Run the App
Rename the main script from app.txt to app.py for easier execution:

mv app.txt app.py
streamlit run app.py
This launches the app on your local browser (usually at http://localhost:8501).

🧠 How It Works
You upload a PDF.

The app extracts the text and metadata.

Text is split into meaningful chunks using LangChain.

Chunks are converted into embeddings using sentence-transformers.

A FAISS vector index is created for fast similarity search.

You ask a question — via text or voice.

The app retrieves the most relevant chunks and generates an answer using a local LLM.

Your chat history, answers, sources, and insights are all visualized and exportable.

🗣️ Voice Input
🎤 Google Speech Recognition is the default (requires internet).

🛠️ For offline mode, install Whisper:

pip install git+https://github.com/openai/whisper.git
Then update the voice input section to use Whisper instead of recognize_google.

💡 Suggested Use Cases
Academic research assistants

Legal document analysis

Offline reading companion

Personal knowledge management

Enterprise document Q&A bots

📥 Exporting Conversations
Click "Export Chat History" to download your Q&A session as a clean Markdown file for future reference or reporting.

⚠️ Known Limitations
PDF parsing may skip complex layouts (e.g., tables, images)

Google Speech API needs internet — use Whisper for offline STT

Ollama must be installed and running

Large PDFs may take time to process (optimize chunking for performance)

📝 License
This project is licensed under the MIT License. You're free to use, modify, and distribute it as long as you include the original copyright.

🤝 Credits
LangChain

Streamlit

Ollama

OpenAI Whisper

FAISS by Meta

Sentence Transformers

Would you like me to package this into a downloadable ZIP structure or generate a Dockerfile for easy deployment?



You said:
app.txt
Document
for these give me an clear readme information for understanding the functionality of the chatbot


ChatGPT said:
Here is a clear and complete README.md that explains the functionality of your Smart Document Assistant (offline chatbot) project using Streamlit, Ollama, LangChain, FAISS, and more:

📚 Smart Document Assistant (Offline PDF/URL Chatbot)
Smart Document Assistant is an offline chatbot application that lets you upload a PDF document or enter a web URL, and then interact with the content using natural language—via text or voice. It uses local LLMs (via Ollama) and retrieval-based QA to give accurate, context-aware answers based on the document's content.

💡 What Does This App Do?
Accepts a PDF or Web URL

Extracts and splits content into chunks

Creates semantic vector embeddings

Stores them in a FAISS index for similarity search

Lets users ask natural language questions

Answers are generated by a local LLM (e.g., Mistral) via Ollama

All responses are grounded in the document — no hallucination!

Users can speak their questions using a microphone

Includes document statistics, word frequency charts, and export options

📁 Supported Inputs

Input Type	Description
📄 PDF File	Upload a local PDF for processing and Q&A
🌐 Web URL	Enter a URL; the app will fetch and clean the content
🧠 Core Functionalities Explained

Feature	Description
📂 Upload/URL Fetching	Choose between uploading a PDF or fetching content from a web URL
📄 Text Extraction	PDF parsing using fitz, web parsing via BeautifulSoup and newspaper3k
🧩 Text Chunking	Splits text into 1000-character chunks with overlap for context
🔍 Embedding + Indexing	Each chunk is embedded using sentence-transformers and indexed with FAISS
🤖 LLM Integration	Local models like mistral, llama2, etc., are used via Ollama for response generation
💬 QA Pipeline	Questions are passed to a LangChain QA chain that pulls relevant chunks and generates a contextual response
🧠 Memory	Optionally keeps track of previous questions and answers
🎙️ Voice Input	Use your microphone to ask questions (defaults to Google Speech, can be extended to Whisper)
📈 Visualization	Word frequency charts, document statistics, and metadata display
📤 Export	Download your full chat history as a Markdown file
🖥️ How to Use the App
1. Clone and Set Up

git clone https://github.com/yourusername/document-assistant.git
cd document-assistant
pip install -r requirements.txt
2. Install and Run Ollama

# Install from https://ollama.com
ollama pull mistral
ollama serve
3. Start the App
Rename your main file if needed:

mv app.txt app.py
streamlit run app.py
Visit: http://localhost:8501

⚙️ Requirements
requirements.txt
txt
Copy
Edit
streamlit
torch
PyMuPDF
speechrecognition
langchain
langchain-community
faiss-cpu
sentence-transformers
requests
matplotlib
pandas
numpy
scipy
pydub
transformers
accelerate
beautifulsoup4
lxml
newspaper3k
✅ Optional: Use Whisper for offline voice input:

pip install git+https://github.com/openai/whisper.git
🧾 Example Use Cases
Research assistant for academic PDFs

Summarizing news articles from a URL

Legal or policy document Q&A

Offline chatbot for restricted environments

Internal document analysis in enterprises

📤 Export & Insights
Generate and download word frequency charts

View and copy document metadata

Export your Q&A history as a Markdown file

🔒 Offline Mode
Once dependencies and models are installed: ✅ No internet is required
✅ All processing is local
✅ Whisper can replace Google STT for voice input

🧰 Technologies Used
Streamlit for UI

Ollama for local LLMs

LangChain for RAG

FAISS for vector search

BeautifulSoup + Newspaper3k for URL extraction

📝 License
MIT License. Use it freely for personal and commercial purposes.


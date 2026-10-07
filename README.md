# PolicyMind-AI

![PolicyMind AI banner](assets/policymind-banner.png)

A company policy and employee handbook assistant built with Python, Streamlit, and local AI.
PolicyMind AI helps users find information in policy PDFs by asking questions in plain language. It uses Retrieval-Augmented Generation (RAG) to retrieve relevant document sections and provide them to a language model as context for an answer. Source filenames and page references help users check the retrieved material.
Why this project?
Employee handbooks often spread information about leave, remote work, expenses, and travel across several documents. PolicyMind AI provides a single interface for exploring those documents without manually searching each PDF.
This is a learning and portfolio project demonstrating document processing, semantic retrieval, local model integration, and a Streamlit interface.
Features
- Upload and process multiple policy PDFs.
- Split extracted text into searchable sections.
- Retrieve relevant sections for each question; the UI requests up to four results.
- Generate answers using the retrieved context.
- Display source filenames and page references in expandable panels.
- View uploaded documents and searchable section counts.
- Keep the current conversation visible during the session.
- Start a new conversation while retaining the indexed documents.
- Clear the workspace, including documents and conversation history.
- Use starter questions about leave, remote work, and expenses.
Technology
Component	Role
Python	Application and document-processing logic
Streamlit	Web interface and session state
Ollama	Local model runtime
llama3.2:3b	Intended answer-generation model
nomic-embed-text	Intended text-embedding model
Vector store	Indexing and retrieval through src/vector_store.py


The model names above match the original project's displayed configuration. Check src/embeddings.py and src/rag_pipeline.py for the actual model names and connection settings used by your copy. Python dependencies belong in requirements.txt.
How it works
1. Read PDFs: Extract text from the uploaded documents, retaining source metadata.
2. Prepare the index: Split the text into smaller sections, generate embeddings, and build a vector index.
3. Retrieve context: Search the index for sections relevant to the user's question.
4. Generate a response: Send the question and retrieved sections to the answer-generation pipeline.
5. Show supporting sources: Display the answer alongside the retrieved documents' filenames and page references.
The source list identifies retrieved material. It does not guarantee that every claim in an answer is supported.
Project files
Path	Purpose
app.py	Streamlit interface, upload flow, and chat controls
src/document_processor.py	PDF text extraction and splitting
src/embeddings.py	Embedding configuration
src/vector_store.py	Vector index creation and policy search
src/rag_pipeline.py	Answer generation from retrieved context
requirements.txt	Python dependency list
assets/policymind-banner.png	Project cover artwork
assets/app-screenshot.png	Optional screenshot of the running application


Run locally
1. Get the project
Download this repository using Code → Download ZIP, extract it, and open the extracted folder in VS Code. Open a terminal in the folder containing app.py and requirements.txt.
Install Python and Ollama if they are not already installed. Use a Python version compatible with the dependencies in requirements.txt.
2. Create a virtual environment
python -m venv .venv
Activate it using the command for your terminal.
Windows PowerShell:
.\.venv\Scripts\Activate.ps1
Windows Command Prompt:
.venv\Scripts\activate.bat
macOS or Linux:
source .venv/bin/activate
3. Install Python dependencies
python -m pip install -r requirements.txt
4. Prepare Ollama
Start the Ollama application or service. If it is not already running, start it in a separate terminal:
ollama serve
Download the models in another terminal:
ollama pull llama3.2:3b
ollama pull nomic-embed-text
Confirm the installed models:
ollama ls
If your backend uses different model names, download those models instead. An internet connection is needed for initial package and model downloads. Runtime data handling depends on the endpoints configured in the backend; keep them local for local inference.
5. Start the application
With the virtual environment active, run:
python -m streamlit run app.py
Open the local URL printed in the terminal. Keep Ollama running while using the application.
Usage
1. Upload one or more text-based policy PDFs in the sidebar.
2. Select Process policies and wait for indexing to finish.
3. Choose a starter question or enter your own question.
4. Expand View sources to check document and page references.
5. Select New conversation to reset the chat, or Clear workspace to reset the document workspace too.
Processing a new selection replaces the current document selection and clears the conversation in the redesigned interface.
Example questions
- What is the annual leave policy?
- What are the work-from-home eligibility rules?
- How do I claim travel expenses?
- What documents are required for reimbursement?
- What does the handbook say about the probation period?
Answers depend on what is present in the uploaded PDFs.
Application screenshot
The banner at the top is project artwork. A screenshot of the actual running app can be added here after saving it as assets/app-screenshot.png.
<!-- Uncomment after adding the screenshot:
![PolicyMind AI application](assets/app-screenshot.png)
-->

Limitations
- AI-generated answers can be incomplete or incorrect. Verify important answers against the original policy or with HR.
- Scanned PDFs may need OCR before their text can be extracted.
- Retrieval quality depends on PDF extraction, text splitting, embeddings, and document content.
- The displayed chat history is not passed into ask_policy_question() by the current interface. Ask self-contained questions rather than relying on conversational follow-ups.
- Chat and workspace references use Streamlit session state; the UI does not provide saved conversations across sessions.
- There is no user login or document-level access control in this interface.
- Response time depends on local hardware, model size, and document volume.
- Publishing the repository does not deploy the app or provide a hosted Ollama server.
Repository hygiene
Keep virtual environments, credentials, private policy documents, downloaded model weights, and indexes containing private document content out of the public repository. Use synthetic or explicitly shareable PDFs for demonstrations.
Possible improvements
- OCR support for scanned documents.
- Conversation-aware follow-up questions.
- Persistent document indexes and saved conversations.
- Search-quality evaluation with a small policy question-and-answer dataset.
- User authentication and document access controls.
These are proposed improvements, not current features.
References
- Ollama CLI documentation
- Llama 3.2 3B model
- Nomic embedding model
Author
Vantakula Manikanta
B.Tech in Artificial Intelligence and Data Science.

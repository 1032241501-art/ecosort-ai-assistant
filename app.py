"""
EcoSort AI - Municipal Waste Segregation Assistant
A RAG-powered application using IBM WatsonX AI and LangChain
"""

import streamlit as st
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
import tempfile
import os

# Page configuration
st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextInput > label {
        font-weight: 600;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #E3F2FD;
    }
    .assistant-message {
        background-color: #F1F8E9;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None

# Sidebar for configuration
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/recycle-sign.png", width=80)
    st.title("⚙️ Configuration")
    
    st.markdown("---")
    
    # IBM Cloud credentials
    st.subheader("🔐 IBM WatsonX Credentials")
    api_key = st.text_input(
        "IBM Cloud API Key",
        type="password",
        help="Enter your IBM Cloud API Key"
    )
    
    project_id = st.text_input(
        "WatsonX Project ID",
        type="password",
        help="Enter your WatsonX Project ID"
    )
    
    st.markdown("---")
    
    # PDF upload
    st.subheader("📄 Upload Policy Document")
    uploaded_file = st.file_uploader(
        "Municipal Waste Policy PDF",
        type=['pdf'],
        help="Upload your local municipal waste segregation policy document"
    )
    
    st.markdown("---")
    
    # Model selection (CHANGED TO SUPPORTED TIER MODEL)
    model_choice = st.selectbox(
        "🤖 Select Model",
        ["ibm/granite-4-h-small"],
        help="Choose the IBM Granite model for responses"
    )
    
    # Process button
    process_button = st.button("🚀 Initialize RAG System", type="primary", use_container_width=True)

# Main content area
st.markdown('<div class="main-header">♻️ EcoSort AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your Intelligent Municipal Waste Segregation Assistant</div>', unsafe_allow_html=True)

# Function to initialize WatsonX LLM
def initialize_watsonx_llm(api_key, project_id, model_id):
    """
    Initialize IBM WatsonX LLM with specified credentials and model.
    """
    try:
        # Configure WatsonX credentials
        credentials = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": api_key
        }
        
        # Set model parameters for optimal RAG performance
        parameters = {
            GenParams.DECODING_METHOD: "greedy",
            GenParams.MAX_NEW_TOKENS: 500,
            GenParams.MIN_NEW_TOKENS: 1,
            GenParams.TEMPERATURE: 0.1,  
            GenParams.TOP_K: 50,
            GenParams.TOP_P: 1
        }
        
        # Initialize WatsonX LLM through LangChain integration
        watsonx_llm = WatsonxLLM(
            model_id=model_id,
            url=credentials["url"],
            apikey=credentials["apikey"],
            project_id=project_id,
            params=parameters
        )
        
        return watsonx_llm
    
    except Exception as e:
        st.error(f"Error initializing WatsonX LLM: {str(e)}")
        return None

# Function to process PDF and create vector store
def process_pdf_and_create_vectorstore(uploaded_file):
    """
    Process uploaded PDF, chunk it, and create FAISS vector store.
    """
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load PDF using LangChain's PyPDFLoader
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Initialize embeddings model (using HuggingFace sentence-transformers)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Create FAISS vector store from document chunks
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return vector_store
    
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Process PDF and initialize RAG system
if process_button:
    if not api_key or not project_id:
        st.error("⚠️ Please provide both IBM Cloud API Key and WatsonX Project ID")
    elif not uploaded_file:
        st.error("⚠️ Please upload a Municipal Waste Policy PDF")
    else:
        with st.spinner("🔄 Initializing RAG system... (This takes a few seconds)"):
            # Process PDF and create vector store
            vector_store = process_pdf_and_create_vectorstore(uploaded_file)
            
            if vector_store:
                st.session_state.vector_store = vector_store
                
                # Initialize WatsonX LLM
                watsonx_llm = initialize_watsonx_llm(api_key, project_id, model_choice)
                
                if watsonx_llm:
                    # Create RetrievalQA chain
                    st.session_state.qa_chain = RetrievalQA.from_chain_type(
                        llm=watsonx_llm,
                        chain_type="stuff",
                        retriever=vector_store.as_retriever(
                            search_kwargs={"k": 3}  # Retrieve top 3 relevant chunks
                        ),
                        return_source_documents=True
                    )
                    
                    st.success("✅ RAG system initialized successfully! You can now start asking questions.")
                    st.balloons()

# Chat interface
if st.session_state.qa_chain:
    st.markdown("### 💬 Ask Your Waste Segregation Questions")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("e.g., I am doing home renovations and have broken bricks. Can I throw them in the regular trash?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking... Reading the Municipal PDF..."):
                try:
                    # Create context-aware prompt
                    enhanced_prompt = f"""You are a municipal waste management assistant. Answer the following question based ONLY on the provided policy document. If the answer is not in the document, say "I don't have information about this in the policy document."

Question: {prompt}

Provide a clear, actionable answer that promotes sustainable recycling practices."""
                    
                    # Query the RAG chain
                    response = st.session_state.qa_chain.invoke({"query": enhanced_prompt})
                    answer = response['result']
                    
                    # Display response
                    st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

else:
    # Instructions when system is not initialized
    st.info("👈 Please configure your credentials and upload a policy document in the sidebar, then click 'Initialize RAG System' to start.")
    
    st.markdown("### 📋 Example Questions you can ask:")
    st.markdown("""
    - What is the fine if a restaurant mixes wet and dry waste?
    - How do I dispose of old earphones?
    - I am doing home renovations and have broken bricks. Can I throw them in the regular trash?
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>EcoSort AI | Powered by IBM WatsonX & LangChain | Built for Sustainable Communities</div>",
    unsafe_allow_html=True
)

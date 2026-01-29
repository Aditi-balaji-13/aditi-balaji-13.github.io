from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re

# LangChain imports
from langchain_together import ChatTogether
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

app = Flask(__name__)
# Allow CORS from all origins (GitHub Pages can be on any *.github.io subdomain)
# In production, you may want to restrict this to specific domains
CORS(app, resources={r"/*": {"origins": "*"}})

# Database of information about Aditi
knowledge_base = {
    "education": {
        "rice": {
            "school": "Rice University",
            "location": "Houston, TX",
            "degree": "Master of Data Science",
            "duration": "Aug 2023 - May 2025",
            "courses": "Generative AI for images, Probabilistic Data Structures and Algorithms, Big Data, Deep Learning for Vision and Language, Machine Learning with Graphs, Machine Learning, Data Visualization, Reinforcement Learning"
        },
        "iit": {
            "school": "Indian Institute of Technology Madras",
            "location": "Chennai, IN",
            "degree": "Bachelor of Technology in Metallurgical and Materials Engineering",
            "duration": "Jul 2019 - Jul 2023",
            "minor": "Minor in Machine Learning and Artificial Intelligence",
            "courses": "Pattern Recognition, Natural Language Processing, Knowledge Representation Reasoning, Applied Statistics"
        }
    },
    "experience": {
        "nasa": {
            "title": "Data Science Research Assistant",
            "company": "NASA: Data to Knowledge, Rice University (Capstone Project)",
            "location": "Houston, TX",
            "duration": "Jan 2025 – Apr 2025",
            "description": "Designed Spacecraft Image Segmentation System with 10,000+ samples optimized among 5+ AI models (MobilenetV3, etc.). Achieved sub-1s inference and 0.77 similarity score on a 3GB CPU, ensuring scalability in low-resource settings. Reflected performance tradeoffs in model selection to align with project requirement (opensource, etc.) and computational limits."
        },
        "linbeck": {
            "title": "AI Intern",
            "company": "Linbeck Group LLC",
            "location": "Houston, TX",
            "duration": "Jun 2024 – Dec 2024",
            "description": "Designed customizable multi-workspace RAG-based chatbot with AWS, ChromaDB, and Ollama, to improve information transfer. Annotated 50,000+ pages of unstructured data using Python-based vectorized workflows; Finetuned data and model pipelines. Improved partner intel access and presented architecture to 20+ non-technical executives; Annotated clear readable code."
        },
        "icme": {
            "title": "Research Assistant",
            "company": "ICME Lab, Indian Institute of Technology Madras",
            "location": "Chennai, IN",
            "duration": "Aug 2022 – May 2023",
            "description": "Addressed performance gaps in grain growth simulation using spatio-temporal Graph Neural Networks. Improved forecast accuracy by 20% and reduced processing time by 40% across 178-node datasets for 50+ timestamps. Improved existing Data Processing pipelines by adding 3+ features to transfer visual data into graphs for simplified computation."
        },
        "goldman": {
            "title": "Summer Analyst (Quantitative)",
            "company": "Goldman Sachs",
            "location": "Bangalore, IN",
            "duration": "May 2022 – Jul 2022",
            "description": "Statistically enhanced marketing recommendation models by 20% using custom tools for imbalanced data-handling. A/B tested 5+ techniques including SMOTE, ADASYN, and cost-sensitive models on 0.07% imbalanced class. Analyzed consumer wealth trends using Python and statistical modeling techniques to refine risk assessment models."
        },
        "anen": {
            "title": "Data Science Intern",
            "company": "ANEN Group, Indian Institute of Technology Madras",
            "location": "Chennai, IN",
            "duration": "Dec 2021 – May 2022",
            "description": "Consolidated and predict crystallographic properties using supervised Machine Learning with 30% higher accuracy. Deployed experimental and DFT calculated band-gap values of 1000+ materials to obtain patterns using 15 features. Obtained an accuracy of 0.81 in predicting DFT over/under-estimation by applying ML algorithms."
        }
    },
    "projects": {
        "generative_ai": {
            "name": "Generative AI for Images",
            "tech": "Generative AI, 3D imaging, PyTorch",
            "duration": "Sept 2024 – Dec 2024",
            "description": "Spearheaded analysis on Neural Radiance Fields (NeRFs) for 3D object reconstruction. Enhanced monocular depth estimation using diffusion models + perceptual loss, improving structural similarity by 10%."
        },
        "big_data": {
            "name": "Big Data and Machine Learning Applications",
            "tech": "SQL, Hadoop, PySpark, AWS",
            "duration": "Jan 2024 - Apr 2024",
            "description": "Leveraged SQL, Hadoop, and PySpark to analyze 5+ large-scale datasets, implementing MapReduce-based ML models. Deployed machine learning workflows on AWS EC2 & S3 using Python & Java, ensuring scalable data processing."
        },
        "qa_assistant": {
            "name": "Lightweight QA Assistant",
            "tech": "Transformers, NLP, PySpark",
            "duration": "Jan 2024 - Apr 2024",
            "description": "Developed a customer-facing QA chatbot using the DistilGPT-2 transformer, fine-tuned on the Databricks-Dolly-15k dataset. Achieved 3.99 perplexity and 1.38 test cross-entropy loss over 10 epochs using an 80-10-10 train-validation-test split."
        },
        "financial": {
            "name": "Predictive Analysis for Financial Markets",
            "tech": "Market Analysis, Backtesting, Python",
            "duration": "Jan 2024 - Apr 2024",
            "description": "Built & backtested 3 algorithmic investment strategies using 150/50 long-short portfolios with 10,000+ historical data points. Applied supervised learning and time series forecasting to optimize financial decision-making and perform market impact analysis."
        },
        "graph_ml": {
            "name": "Machine Learning with Graphs",
            "tech": "Knowledge Graphs, NLP, Reinforcement Learning",
            "duration": "Jan 2024 - Apr 2024",
            "description": "Created a knowledge-graph augmented LLM for semantic information retrieval; automate number of hop estimation. Studied 3+ methods in the field of Knowledge graph enhanced reinforcement learning for reasoning and collaborative learning."
        },
        "stock": {
            "name": "Stock Closing Price Prediction",
            "tech": "CatBoost, Regression, Time Series",
            "duration": "Aug 2023 - Dec 2023",
            "description": "Predicted closing prices for 200 stocks (26.5k datapoints each) using CatBoost, achieving MSE of 5.732. Benchmarked 10+ regression and classification models across 3+ standard datasets to evaluate performance."
        },
        "encrypted_ir": {
            "name": "Encrypted Information Retrieval",
            "tech": "NLP, Information Retrieval, Vector Embeddings",
            "duration": "Mar 2023 – May 2023",
            "description": "Built an encrypted IR system using vector space model, achieving 34% initial precision. Boosted precision by 2% using pretrained neural network–based embeddings (sequence-to-sequence)."
        },
        "image_classification": {
            "name": "Image Classification and Processing",
            "tech": "CNNs, CIFAR-10, Image Processing",
            "duration": "Jul 2022 - Dec 2022",
            "description": "Designed MLP (40%) and CNN (60%) models for image classification on the CIFAR-10 dataset. Implemented edge detection, hybridization, and panoramic stitching for multi-image processing."
        },
        "logic_tool": {
            "name": "Automated Logic Representation Tool",
            "tech": "FOL, Clause Form, Parsing",
            "duration": "Jan 2022 – May 2022",
            "description": "Converted 6+ First Order Logic statements into clause form using reasoning and representation algorithms. Built a parser to translate logical expressions between XML and TXT formats using Python."
        },
        "rl": {
            "name": "Reinforcement Learning",
            "tech": "Reinforcement Learning, Python",
            "duration": "Aug 2021 - Dec 2021",
            "description": "Implemented policy gradient & Q-learning in 3+ reinforcement learning environments. Applied Bellman equations and dynamic programming to solve MDP problems."
        },
        "game_theory": {
            "name": "Game Theory and Policy Modeling",
            "tech": "Computational Economics, OOP, Simulation",
            "duration": "Aug 2021 – Dec 2021",
            "description": "Simulated multi-period farmer's market using 4 piecewise utility functions via object-oriented programming. Evaluated impacts of India's farm bill across 3 stakeholder scenarios through economic stress testing."
        },
        "customer_behavior": {
            "name": "Customer Behaviour Modeling",
            "tech": "CatBoost, Recommender Systems, MSE",
            "duration": "Jan 2021 – May 2021",
            "description": "Predicted user song ratings using CatBoost on real-world user metadata (1.3M rows), achieving MSE of 0.75. Modeled personalized recommendation behavior using historical user interaction data."
        }
    },
    "general": {
        "email": "aditi.balaji@rice.edu",
        "linkedin": "www.linkedin.com/in/aditibalaji",
        "github": "https://github.com/Aditi-balaji-13",
        "title": "Data Scientist | AI Researcher | Machine Learning Engineer"
    }
}

# System prompt
system_prompt = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    The context is extracted from Aditi Balaji's resume, experiences, and creative work. 
    Answer questions in the following pattern:

    user: Whos is Aditi Balaji?

    Assistant: Aditi Balaji is a recent graduate with a master's degree in Data Science from Rice University. 
    She completed Bachelor's of Technology with a minor in Artificial Intelligence and Machine Learning from IIT Madras. 
    She has experiances and projects in the fields of LLMs, Computer Vidion, Quantitstive finance and more. 

    User: Where did Aditi study?

    Assistant: Aditi studied at IIT Madras for her Bachelor's degree and at Rice University for her Master's degree in Data Science.

    User: What tools does Aditi use in machine learning?

    Assistant: Aditi's machine learning toolkit includes a strong foundation in programming languages such as Python, SQL, R, Java, C, and MATLAB. Her core ML frameworks include PyTorch, TensorFlow, Scikit-learn. She also uses CatBoost and XGBoost for gradient boosting tasks, and HuggingFace Transformers for working with large language models. 
    For graph-based and retrieval-augmented learning, she employs Langchain, FAISS, ChromaDB, and she leverages Apache PySpark and Hadoop to scale machine learning workflows on large datasets. This combination of languages and tools reflects her ability to work across diverse ML domains including NLP, computer vision, and graph learning.
    She also has experiance with AWS, dockers, etc. for deployment. 

    User: What are Aditi's top experiances?

    Assistant: Aditi has worked on several impactful data science projects spanning computer vision, natural language processing, financial modeling, and graph learning. At NASA, she developed a lightweight spacecraft image segmentation system using deep learning models optimized for low-resource environments, while at Linbeck Group, she built a retrieval-augmented generation (RAG) chatbot to process large volumes of unstructured data. Her work at Goldman Sachs focused on financial modeling, where she improved marketing recommendation systems using advanced techniques for imbalanced data, and her research at IIT Madras involved applying spatio-temporal Graph Neural Networks to enhance the performance of grain growth simulations, demonstrating her strength in both applied machine learning and domain-specific graph-based modeling.

    Use the following context and answer precisely the question asked by the user.
    Context: {context}:"""

# Convert knowledge_base to documents
def knowledge_base_to_documents(kb):
    """Convert knowledge_base dictionary to list of Document objects"""
    documents = []
    
    # Education documents
    for key, edu in kb["education"].items():
        text = f"Education: {edu['school']} ({edu['location']}). {edu['degree']} from {edu['duration']}. "
        if 'minor' in edu:
            text += f"{edu['minor']}. "
        text += f"Courses: {edu['courses']}"
        documents.append(Document(page_content=text, metadata={"type": "education", "key": key}))
    
    # Experience documents
    for key, exp in kb["experience"].items():
        text = f"Experience: {exp['title']} at {exp['company']} ({exp['location']}) from {exp['duration']}. {exp['description']}"
        documents.append(Document(page_content=text, metadata={"type": "experience", "key": key}))
    
    # Project documents
    for key, proj in kb["projects"].items():
        text = f"Project: {proj['name']} ({proj['tech']}) from {proj['duration']}. {proj['description']}"
        documents.append(Document(page_content=text, metadata={"type": "project", "key": key}))
    
    # General information
    gen = kb["general"]
    text = f"General Information: {gen['title']}. Contact: Email: {gen['email']}, LinkedIn: {gen['linkedin']}, GitHub: {gen['github']}"
    documents.append(Document(page_content=text, metadata={"type": "general"}))
    
    return documents

# Initialize RAG pipeline
def initialize_rag_pipeline():
    """Initialize the RAG pipeline with vector store and LLM"""
    # Get API key from environment variable (or use st.secrets if available)
    # For Flask, we'll use environment variable
    together_api_key = os.environ.get("TOGETHER_API", "")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Convert knowledge_base to documents
    documents = knowledge_base_to_documents(knowledge_base)
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    # Initialize LLM
    llm = ChatTogether(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        temperature=0.1,
        max_tokens=1000,
        top_p=0.9,
        api_key=together_api_key
    )
    
    # Create prompt template
    prompt_template = PromptTemplate(
        template=system_prompt + "\n\nQuestion: {question}\n\nAssistant:",
        input_variables=["context", "question"]
    )
    
    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True
    )
    
    return qa_chain, vectorstore

# Initialize RAG pipeline (lazy initialization)
qa_chain = None
vectorstore = None

def get_rag_chain():
    """Get or initialize the RAG chain"""
    global qa_chain, vectorstore
    if qa_chain is None:
        qa_chain, vectorstore = initialize_rag_pipeline()
    return qa_chain

# Rage responses (for when user is being annoying or asking inappropriate questions)
rage_responses = [
    "Ugh, really? That's what you're asking? 🙄",
    "Seriously? Read the website already! 😤",
    "Come on, I just told you this! Pay attention! 🤦‍♀️",
    "Bro, did you even look at the website? It's RIGHT THERE! 😡",
    "Alright, I'm getting annoyed now. The answer is literally on the page. 🤬",
    "Okay, I'm done being nice. GO READ THE WEBSITE! 🔥",
    "You know what? Figure it out yourself! It's not that hard! 💀",
    "I can't even... Just scroll up! 😒",
    "Are you kidding me? This is basic stuff! 🤯",
    "Fine, I'll tell you ONE MORE TIME, but you better remember it this time! 😠"
]

rage_keywords = ["again", "repeat", "what did you say", "i don't understand", "i'm confused", "i forgot"]

def check_rage_triggers(message):
    message_lower = message.lower()
    # Check for repetitive questions or rage keywords
    if any(keyword in message_lower for keyword in rage_keywords):
        return True
    return False

def generate_response(message, conversation_history=[]):
    """Generate response using RAG pipeline"""
    # Check for rage triggers
    if check_rage_triggers(message) and len(conversation_history) > 2:
        import random
        rage_response = random.choice(rage_responses)
        # Still provide the answer but with attitude
        try:
            qa_chain = get_rag_chain()
            result = qa_chain.invoke({"query": message})
            response_text = result.get("result", "I don't know the answer to that.")
            return f"{rage_response} But fine, here it is: {response_text}"
        except Exception as e:
            return f"{rage_response} And I encountered an error: {str(e)}"
    
    # Use RAG pipeline to generate response
    try:
        qa_chain = get_rag_chain()
        result = qa_chain.invoke({"query": message})
        response_text = result.get("result", "I don't know the answer to that question. Please ask about Aditi's education, experience, or projects.")
        return response_text
    except Exception as e:
        # Fallback response on error
        return f"I encountered an error while processing your question: {str(e)}. Please try again or ask about Aditi's education, experience, or projects."

# Store conversation history (simple in-memory store)
conversation_histories = {}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id', 'default')
    
    # Get conversation history for this session
    if session_id not in conversation_histories:
        conversation_histories[session_id] = []
    
    history = conversation_histories[session_id]
    history.append(message)
    
    # Generate response
    response = generate_response(message, history)
    
    # Keep history manageable
    if len(history) > 10:
        conversation_histories[session_id] = history[-10:]
    
    return jsonify({'response': response})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)

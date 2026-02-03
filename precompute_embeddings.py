"""
Pre-compute embeddings for all documents and save as JSON
Run this script locally to generate embeddings.json for GitHub Pages

Usage:
    pip install langchain langchain-community langchain-text-splitters sentence-transformers torch
    python precompute_embeddings.py
"""
import json
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Knowledge base (same structure as in script.js)
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
        "email": "aditi.balaji.ds@gmail.com",
        "linkedin": "www.linkedin.com/in/aditibalaji",
        "github": "https://github.com/Aditi-balaji-13",
        "title": "Data Scientist | AI Researcher | Machine Learning Engineer"
    }
}

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

def precompute_embeddings():
    """Pre-compute embeddings for all documents"""
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    print("Converting knowledge base to documents...")
    documents = knowledge_base_to_documents(knowledge_base)
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(documents)
    
    print(f"Generating embeddings for {len(splits)} chunks...")
    texts = [doc.page_content for doc in splits]
    metadata_list = [doc.metadata for doc in splits]
    
    # Generate embeddings
    embedding_vectors = embeddings.embed_documents(texts)
    
    print("Creating embeddings data structure...")
    embeddings_data = {
        "documents": [
            {
                "text": text,
                "metadata": metadata
            }
            for text, metadata in zip(texts, metadata_list)
        ],
        "embeddings": embedding_vectors,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": len(embedding_vectors[0]) if embedding_vectors else 384
    }
    
    print(f"Saving to embeddings.json ({len(embedding_vectors)} embeddings)...")
    with open("embeddings.json", "w") as f:
        json.dump(embeddings_data, f)
    
    file_size_kb = len(json.dumps(embeddings_data)) / 1024
    print(f"✅ Done! Created embeddings.json with {len(embedding_vectors)} embeddings")
    print(f"   File size: {file_size_kb:.2f} KB")
    print(f"   Dimension: {embeddings_data['dimension']}")
    
    return embeddings_data

if __name__ == "__main__":
    precompute_embeddings()

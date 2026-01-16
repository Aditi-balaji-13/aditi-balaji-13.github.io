from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

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

def search_knowledge_base(query):
    query_lower = query.lower()
    results = []
    
    # Check education
    if any(term in query_lower for term in ["education", "school", "university", "degree", "master", "bachelor", "rice", "iit", "iitm"]):
        if "rice" in query_lower or "master" in query_lower:
            results.append(f"Aditi is doing her Master of Data Science at Rice University (Houston, TX) from Aug 2023 - May 2025. Courses include: {knowledge_base['education']['rice']['courses']}")
        if "iit" in query_lower or "bachelor" in query_lower or "undergraduate" in query_lower:
            results.append(f"Aditi completed her Bachelor of Technology in Metallurgical and Materials Engineering at IIT Madras (Chennai, IN) from Jul 2019 - Jul 2023. She also has a Minor in Machine Learning and Artificial Intelligence. Relevant courses: {knowledge_base['education']['iit']['courses']}")
    
    # Check experience
    experience_keywords = {
        "nasa": ["nasa", "spacecraft", "image segmentation", "capstone"],
        "linbeck": ["linbeck", "rag", "chatbot", "aws", "chromadb", "ollama"],
        "icme": ["icme", "grain", "graph neural", "gnn"],
        "goldman": ["goldman", "sachs", "quantitative", "analyst", "marketing"],
        "anen": ["anen", "crystallographic", "dft", "band-gap"]
    }
    
    for exp_key, keywords in experience_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            exp = knowledge_base['experience'][exp_key]
            results.append(f"{exp['title']} at {exp['company']} ({exp['location']}) from {exp['duration']}. {exp['description']}")
    
    # Check projects
    project_keywords = {
        "generative_ai": ["generative", "nerf", "3d", "depth estimation", "diffusion"],
        "big_data": ["big data", "hadoop", "pyspark", "mapreduce", "aws"],
        "qa_assistant": ["qa", "assistant", "distilgpt", "dolly"],
        "financial": ["financial", "stock", "market", "backtesting", "investment"],
        "graph_ml": ["graph", "knowledge graph", "llm", "semantic"],
        "stock": ["stock", "catboost", "closing price", "mse"],
        "encrypted_ir": ["encrypted", "information retrieval", "vector"],
        "image_classification": ["image", "classification", "cifar", "cnn"],
        "logic_tool": ["logic", "fol", "clause", "xml"],
        "rl": ["reinforcement", "q-learning", "policy gradient", "bellman"],
        "game_theory": ["game theory", "farmer", "policy", "simulation"],
        "customer_behavior": ["customer", "behavior", "recommendation", "rating"]
    }
    
    for proj_key, keywords in project_keywords.items():
        if any(keyword in query_lower for keyword in keywords):
            proj = knowledge_base['projects'][proj_key]
            results.append(f"{proj['name']} ({proj['tech']}) from {proj['duration']}. {proj['description']}")
    
    # General questions
    if any(term in query_lower for term in ["who", "what do", "tell me about", "about aditi"]):
        results.append(f"Aditi is a {knowledge_base['general']['title']}. She's currently pursuing her Master's at Rice University and has extensive experience in data science, AI, and machine learning. Check out her projects and experience sections!")
    
    if any(term in query_lower for term in ["contact", "email", "linkedin", "github", "reach"]):
        results.append(f"You can reach Aditi on LinkedIn: {knowledge_base['general']['linkedin']} or GitHub: {knowledge_base['general']['github']}")
    
    return results

def generate_response(message, conversation_history=[]):
    # Check for rage triggers
    if check_rage_triggers(message) and len(conversation_history) > 2:
        import random
        rage_response = random.choice(rage_responses)
        # Still provide the answer but with attitude
        results = search_knowledge_base(message)
        if results:
            return f"{rage_response} But fine, here it is: {results[0]}"
        else:
            return f"{rage_response} And I don't know what you're even asking about! Try asking about education, experience, or projects!"
    
    # Search knowledge base
    results = search_knowledge_base(message)
    
    if results:
        response = results[0]
        if len(results) > 1:
            response += f"\n\nAlso: {results[1]}"
        return response
    else:
        # Default response with attitude
        responses = [
            "Hmm, not sure what you're asking about. Try asking about Aditi's education, work experience, or projects! 🤔",
            "I don't have that info right now. Why don't you check the education, experience, or projects sections? 🙃",
            "Can you be more specific? I know about education, experience, and projects. Pick one! 😏"
        ]
        import random
        return random.choice(responses)

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

"""
Simplified backend - no embeddings, no vector DB
Just receives context + question and calls Together AI
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# System prompt (same as before)
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

def call_together_ai(prompt, api_key):
    """Call Together AI API"""
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
        "top_p": 0.9
    }
    
    response = requests.post(url, json=data, headers=headers, timeout=60)
    response.raise_for_status()
    result = response.json()
    
    return result["choices"][0]["message"]["content"]

@app.route('/chat', methods=['POST'])
def chat():
    """Chat endpoint - receives context + question from frontend"""
    try:
        if not request.json:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        data = request.json
        message = data.get('message', '')
        context_docs = data.get('context', [])  # Array of relevant document texts
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Format context from documents
        if context_docs:
            context = "\n\n".join(context_docs)
        else:
            context = "No relevant context found."
        
        # Format prompt
        prompt = system_prompt.format(context=context) + f"\n\nQuestion: {message}\n\nAssistant:"
        
        # Get API key
        together_api_key = os.environ.get("TOGETHER_API", "")
        if not together_api_key:
            return jsonify({'error': 'TOGETHER_API not configured'}), 500
        
        # Call Together AI
        response_text = call_together_ai(prompt, together_api_key)
        
        return jsonify({'response': response_text})
    except Exception as e:
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        api_key_set = bool(os.environ.get("TOGETHER_API", ""))
        return jsonify({
            'status': 'healthy',
            'api_key_set': api_key_set,
            'backend_type': 'simplified'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

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

# System prompt
system_prompt = """You are Aditi Balaji. Answer questions about yourself using first person (I, me, my). 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    The context is extracted from your resume, experiences, and creative work. 
    Answer questions in the following pattern:

    User: Who is Aditi Balaji?

    Assistant: I am a recent graduate with a master's degree in Data Science from Rice University. 
    I completed my Bachelor's of Technology with a minor in Artificial Intelligence and Machine Learning from IIT Madras. 
    I have experiences and projects in the fields of LLMs, Computer Vision, Quantitative finance and more.

    User: Who are you?

    Assistant: I am Aditi Balaji, a Data Scientist and AI Researcher with a master's degree in Data Science from Rice University.

    User: Where did Aditi study?

    Assistant: I studied at IIT Madras for my Bachelor's degree and at Rice University for my Master's degree in Data Science.

    User: What tools do you use in machine learning?

    Assistant: My machine learning toolkit includes a strong foundation in programming languages such as Python, SQL, R, Java, C, and MATLAB. My core ML frameworks include PyTorch, TensorFlow, Scikit-learn. I also use CatBoost and XGBoost for gradient boosting tasks, and HuggingFace Transformers for working with large language models. 
    For graph-based and retrieval-augmented learning, I employ Langchain, FAISS, ChromaDB, and I leverage Apache PySpark and Hadoop to scale machine learning workflows on large datasets. I also have experience with AWS, dockers, etc. for deployment.

    User: What are your top experiences?

    Assistant: I have worked on several impactful data science projects spanning computer vision, natural language processing, financial modeling, and graph learning. At NASA, I developed a lightweight spacecraft image segmentation system using deep learning models optimized for low-resource environments, while at Linbeck Group, I built a retrieval-augmented generation (RAG) chatbot to process large volumes of unstructured data. My work at Goldman Sachs focused on financial modeling, where I improved marketing recommendation systems using advanced techniques for imbalanced data, and my research at IIT Madras involved applying spatio-temporal Graph Neural Networks to enhance the performance of grain growth simulations.

    Use the following context and answer precisely the question asked by the user. Always respond in first person as Aditi Balaji.
    Context: {context}:"""

def call_together_ai(prompt, api_key):
    """Call Together AI API"""
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "ServiceNow-AI/Apriel-1.6-15b-Thinker",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(url, json=data, headers=headers, timeout=60)
        
        # Log response for debugging
        print(f"Together AI API Status: {response.status_code}")
        
        if response.status_code != 200:
            error_detail = response.text
            try:
                error_detail = response.json()
            except:
                pass
            print(f"Together AI API Error: {error_detail}")
            raise Exception(f"Together AI API returned {response.status_code}: {error_detail}")
        
        result = response.json()
        
        # Check response structure
        if "choices" not in result:
            print(f"Unexpected response structure: {result}")
            raise Exception(f"Unexpected response structure: no 'choices' field")
        
        if len(result["choices"]) == 0:
            print(f"Empty choices array: {result}")
            raise Exception("Together AI returned empty choices array")
        
        if "message" not in result["choices"][0]:
            print(f"Unexpected choice structure: {result['choices'][0]}")
            raise Exception("Unexpected choice structure: no 'message' field")
        
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        raise Exception("Together AI API request timed out")
    except requests.exceptions.ConnectionError:
        raise Exception("Failed to connect to Together AI API")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Together AI API request failed: {str(e)}")

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
        
        # Format prompt (use replace instead of format to avoid issues with curly braces)
        prompt = system_prompt.replace('{context}', context) + f"\n\nQuestion: {message}\n\nAssistant:"
        
        # Get API key
        together_api_key = os.environ.get("TOGETHER_API", "")
        if not together_api_key:
            print("ERROR: TOGETHER_API environment variable is not set!")
            return jsonify({'error': 'TOGETHER_API not configured'}), 500
        
        # Log that API key is set (but don't log the actual key)
        print(f"API key is set (length: {len(together_api_key)})")
        
        # Call Together AI
        try:
            response_text = call_together_ai(prompt, together_api_key)
        except requests.exceptions.RequestException as e:
            return jsonify({
                'error': 'Failed to call Together AI API',
                'message': str(e)
            }), 500
        except (KeyError, IndexError) as e:
            return jsonify({
                'error': 'Unexpected response from Together AI',
                'message': str(e)
            }), 500
        
        return jsonify({'response': response_text})
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in /chat: {error_msg}\n{traceback_str}")  # Log to console
        return jsonify({
            'error': 'Internal server error',
            'message': error_msg
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

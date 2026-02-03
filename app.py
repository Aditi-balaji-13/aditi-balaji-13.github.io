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
system_prompt = """You are Aditi Balaji's AI assistant. Answer the question directly. Use the context provided. Give only the answer, no reasoning or explanations. Use third person to answer the questions and mention infromation from context.

Context: {context}

Question: {question}

Answer:"""

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
        "max_tokens": 300,  # Reduced to prevent verbose reasoning
        "top_p": 0.9,
        "stop": ["\n\nThe user", "\n\nUser:", "\n\nQuestion:", "\n\nContext:", "\n\nAnswer:", "The user asks", "The context", "The pattern"]
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
        
        response_text = result["choices"][0]["message"]["content"]
        
        # Post-process to remove reasoning from "Thinker" models
        # Extract only the direct answer (before any reasoning starts)
        lines = response_text.split('\n')
        direct_answer = []
        reasoning_keywords = ["The user", "The context", "The pattern", "The question", "The answer", "We need", "Thus", "So", "However", "The examples"]
        
        for line in lines:
            line_stripped = line.strip()
            # Stop if we hit reasoning keywords
            if any(line_stripped.startswith(keyword) for keyword in reasoning_keywords):
                break
            # Skip empty lines at start
            if not direct_answer and not line_stripped:
                continue
            # Add non-reasoning lines
            if line_stripped and not any(line_stripped.startswith(keyword) for keyword in reasoning_keywords):
                direct_answer.append(line)
            # Stop if we see reasoning patterns
            if "The user asks" in line or "The context:" in line or "The pattern:" in line:
                break
        
        # If we found a direct answer, use it; otherwise use original
        if direct_answer:
            cleaned_response = '\n'.join(direct_answer).strip()
            # Remove any trailing reasoning markers
            for marker in ["Answer:", "Answer directly:", "Thus answer:", "Final answer:"]:
                if cleaned_response.startswith(marker):
                    cleaned_response = cleaned_response[len(marker):].strip()
            return cleaned_response
        
        return response_text
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
        prompt = system_prompt.replace('{context}', context).replace('{question}', message)
        
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

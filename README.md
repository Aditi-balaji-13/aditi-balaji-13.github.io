# Aditi Balaji - Portfolio Website

A starry-themed portfolio website with an interactive RAG-powered chatbot using LangChain and Together AI.

## Features

- ✨ Beautiful starry night theme with animated stars
- 📚 Education timeline
- 💼 Work experience timeline
- 🚀 Projects displayed in a 3-column grid
- 📧 Contact form with email functionality
- 💬 **RAG-powered chatbot** using LangChain, ChromaDB, and Together AI's Llama 3.3 70B

## Architecture

This project uses a **two-part architecture**:

- **Frontend (GitHub Pages)**: Static HTML/CSS/JS files served on GitHub Pages
- **Backend (Hosting Service)**: Flask API with LangChain RAG pipeline hosted separately

## Quick Start

### 1. Deploy Backend

The backend needs to be hosted separately. See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

**Quick option using Render:**
1. Sign up at [render.com](https://render.com)
2. Create new Web Service from this GitHub repo
3. Set environment variable: `TOGETHER_API=your-api-key`
4. Deploy (takes ~5-10 minutes)

### 2. Update Frontend

1. Open `script.js`
2. Update `BACKEND_URL` with your deployed backend URL:
   ```javascript
   const BACKEND_URL = 'https://your-backend-url.onrender.com';
   ```

### 3. Deploy Frontend to GitHub Pages

1. Push to GitHub
2. Go to Settings → Pages
3. Select branch and root folder
4. Your site will be live!

See [DEPLOYMENT.md](DEPLOYMENT.md) for complete deployment guide.

## Local Development

### Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key
export TOGETHER_API="your-api-key"

# Run Flask app
python app.py
```

Backend will run on `http://localhost:5000`

### Frontend

1. Update `BACKEND_URL` in `script.js` to `http://localhost:5000`
2. Open `index.html` in browser or use:
   ```bash
   python -m http.server 8000
   ```

## Files

- `index.html` - Main website HTML
- `styles.css` - Styling and animations
- `script.js` - Frontend JavaScript (connects to Flask backend)
- `app.py` - Flask backend with LangChain RAG pipeline
- `requirements.txt` - Python dependencies
- `Procfile` - For Heroku/Railway deployment
- `render.yaml` - Render deployment config
- `DEPLOYMENT.md` - Complete deployment guide

## Chatbot Features

The chatbot uses a **LangChain RAG pipeline** with:

- **Vector Store**: ChromaDB with HuggingFace embeddings
- **LLM**: Together AI's Llama 3.3 70B Instruct Turbo
- **Knowledge Base**: Aditi's education, experience, and projects
- **Smart Retrieval**: Semantic search to find relevant context

It can answer questions about:
- Education (Rice University, IIT Madras)
- Work experience (NASA, Linbeck, ICME Lab, Goldman Sachs, ANEN Group)
- Projects (all 12 projects)
- General information about Aditi

## Tech Stack

**Frontend:**
- HTML5, CSS3, JavaScript
- GitHub Pages (hosting)

**Backend:**
- Flask (Python web framework)
- LangChain (RAG framework)
- ChromaDB (vector database)
- HuggingFace Embeddings
- Together AI (LLM provider)

## Environment Variables

- `TOGETHER_API`: Your Together AI API key (required for backend)

## Cost

- **GitHub Pages**: Free
- **Render Free Tier**: Free (with limitations - spins down after inactivity)
- **Together AI**: Free tier available for Llama 3.3 70B

## Troubleshooting

See [DEPLOYMENT.md](DEPLOYMENT.md) for troubleshooting guide.

## License

This project is open source and available for personal use.

Enjoy the starry experience! 🌟

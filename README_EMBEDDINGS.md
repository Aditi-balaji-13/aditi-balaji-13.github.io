# Client-Side Embeddings Setup Guide

This setup moves the vector database to the frontend, dramatically reducing backend memory usage.

## Architecture

- **Frontend (GitHub Pages)**: 
  - Loads pre-computed embeddings from `embeddings.json`
  - Performs similarity search in JavaScript
  - Sends relevant context + question to backend

- **Backend (Render)**:
  - Receives question + pre-retrieved context
  - Calls Together AI API
  - Returns response
  - **No embeddings, no vector DB, no PyTorch!**

## Setup Steps

### Step 1: Generate Embeddings (One-time, Local)

1. **Install dependencies locally:**
   ```bash
   pip install langchain langchain-community langchain-text-splitters sentence-transformers torch
   ```

2. **Run pre-computation script:**
   ```bash
   python precompute_embeddings.py
   ```

3. **This creates `embeddings.json`** with:
   - All document texts
   - Pre-computed embeddings
   - Metadata

4. **Commit `embeddings.json` to GitHub:**
   ```bash
   git add embeddings.json
   git commit -m "Add pre-computed embeddings"
   git push
   ```

### Step 2: Update Backend

1. **Replace `app.py` with `app_simple.py`:**
   ```bash
   mv app.py app_old.py
   mv app_simple.py app.py
   ```

2. **Update `requirements.txt`:**
   ```bash
   cp requirements_simple.txt requirements.txt
   ```

3. **Update `Procfile` and `render.yaml`** (no changes needed, they're already correct)

4. **Deploy to Render:**
   - Push to GitHub
   - Render will auto-deploy
   - Backend will be much lighter!

### Step 3: Update Frontend

The frontend (`script.js`) is already updated to:
- Load `embeddings.json` on page load
- Perform similarity search
- Send context to backend

Just make sure `embeddings.json` is in the root directory and accessible.

## File Sizes

- `embeddings.json`: ~500KB - 2MB (depending on number of documents)
- Frontend JavaScript: +~5KB (similarity search code)
- Backend: **~15MB total** (vs ~400MB before!)

## Memory Usage Comparison

### Before (Full RAG Backend):
- PyTorch: ~150-200MB
- sentence-transformers: ~80-100MB  
- ChromaDB: ~50-100MB
- **Total: ~300-400MB** ❌ (exceeds 512MB limit)

### After (Simplified Backend):
- Flask: ~10MB
- requests: ~5MB
- **Total: ~15MB** ✅ (fits easily in 512MB!)

## How It Works

1. **User asks question** → Frontend
2. **Frontend embeds query** → Simple hash-based embedding
3. **Frontend finds top 3 docs** → Cosine similarity with pre-computed embeddings
4. **Frontend sends to backend** → `{message: "...", context: ["doc1", "doc2", "doc3"]}`
5. **Backend formats prompt** → Adds context to system prompt
6. **Backend calls Together AI** → Gets response
7. **Backend returns response** → Frontend displays

## Updating Knowledge Base

If you update the knowledge base:

1. Update `knowledge_base` in `app.py`
2. Run `precompute_embeddings.py` again
3. Commit new `embeddings.json`
4. No backend redeployment needed!

## Limitations

- **Query embedding quality**: The JavaScript embedding function is simpler than the Python model. For better results, consider:
  - Using TensorFlow.js with the same model
  - Calling a lightweight embedding API
  - Using a more sophisticated hash-based approach

- **Initial page load**: Users download `embeddings.json` (~1-2MB), but this is cached by the browser.

## Troubleshooting

### Embeddings not loading
- Check browser console for errors
- Verify `embeddings.json` is in root directory
- Check file is committed to GitHub

### Similarity search not working
- Check browser console for errors
- Verify embeddings are loaded: `console.log(embeddingsData)`
- Try simpler queries first

### Backend still using memory
- Make sure you're using `app_simple.py` (not `app.py`)
- Check `requirements.txt` doesn't include PyTorch/sentence-transformers
- Verify Render logs show simplified backend

## Next Steps

1. Generate embeddings: `python precompute_embeddings.py`
2. Commit `embeddings.json` to GitHub
3. Replace backend with simplified version
4. Deploy and test!

Your backend should now fit easily in Render's free tier! 🎉

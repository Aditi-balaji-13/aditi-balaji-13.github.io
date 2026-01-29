# Deployment Guide: Running RAG Pipeline on GitHub Pages

Since GitHub Pages only serves static files, you need to host the Flask backend separately. This guide shows you how to deploy both parts.

## Architecture

- **Frontend (GitHub Pages)**: Static HTML/CSS/JS files
- **Backend (Hosting Service)**: Flask API with LangChain RAG pipeline

## Step 1: Deploy Backend to Render (Free Tier Available)

### Option A: Using Render (Recommended - Easiest)

1. **Create a Render Account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

3. **Configure Service**
   - **Name**: `aditi-portfolio-backend` (or any name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Plan**: Free (or paid if you need more resources)

4. **Add Environment Variable**
   - Go to "Environment" tab
   - Add: `TOGETHER_API` = `your-together-api-key`
   - Click "Save Changes"

5. **Deploy**
   - Render will automatically deploy
   - Wait for build to complete (~5-10 minutes first time)
   - Your backend URL will be: `https://your-service-name.onrender.com`

### Option B: Using Railway

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Add environment variable: `TOGETHER_API`
6. Railway auto-detects Python and deploys
7. Get your backend URL from the service settings

### Option C: Using Fly.io

1. Install Fly CLI: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Create app: `fly launch`
4. Set secret: `fly secrets set TOGETHER_API=your-key`
5. Deploy: `fly deploy`

## Step 2: Update Frontend to Use Backend URL

1. Open `script.js`
2. Find this line:
   ```javascript
   const BACKEND_URL = 'https://your-backend-url.onrender.com';
   ```
3. Replace with your actual backend URL from Step 1
4. Commit and push to GitHub

## Step 3: Deploy Frontend to GitHub Pages

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Update frontend to use RAG backend"
   git push origin main
   ```

2. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Settings → Pages
   - Source: Deploy from branch
   - Branch: `main` (or `master`)
   - Folder: `/ (root)`
   - Click Save

3. **Your site will be live at:**
   - `https://yourusername.github.io/repository-name`

## Step 4: Enable CORS (Already Done)

The Flask app already has CORS enabled, so your GitHub Pages frontend can call the backend API.

## Testing

1. **Test Backend Locally:**
   ```bash
   export TOGETHER_API="your-api-key"
   python app.py
   ```
   Visit: `http://localhost:5000/health`

2. **Test Frontend Locally:**
   - Update `BACKEND_URL` in `script.js` to `http://localhost:5000`
   - Open `index.html` in browser or use a local server

3. **Test Production:**
   - Visit your GitHub Pages URL
   - Open the chatbot
   - Ask a question about Aditi

## Troubleshooting

### Backend Issues

- **Build fails**: Check that all dependencies in `requirements.txt` are correct
- **API key error**: Make sure `TOGETHER_API` environment variable is set
- **Slow first response**: First request initializes the vector store (takes ~30 seconds)

### Frontend Issues

- **CORS errors**: Backend already has CORS enabled, but check browser console
- **Connection refused**: Verify `BACKEND_URL` in `script.js` is correct
- **Timeout**: Free tier services may spin down after inactivity (Render free tier)

### Render Free Tier Limitations

- Services spin down after 15 minutes of inactivity
- First request after spin-down takes ~30-60 seconds
- Consider upgrading to paid tier for always-on service

## Alternative: Use Serverless Functions

If you want to avoid hosting a full Flask app, you could:
- Use Vercel Serverless Functions
- Use Netlify Functions
- Use AWS Lambda

However, these require more setup and may have limitations with LangChain dependencies.

## Cost

- **GitHub Pages**: Free
- **Render Free Tier**: Free (with limitations)
- **Railway Free Tier**: $5 credit/month
- **Fly.io**: Free tier available

## Security Notes

- Never commit your `TOGETHER_API` key to GitHub
- Use environment variables only
- The `.env.example` file shows what variables are needed (don't commit actual `.env`)

## Next Steps

1. Deploy backend to Render/Railway/Fly.io
2. Update `BACKEND_URL` in `script.js`
3. Deploy frontend to GitHub Pages
4. Test the chatbot!

Your RAG-powered chatbot will now work on GitHub Pages! 🚀

# Aditi Balaji - Portfolio Website

A starry-themed portfolio website with an interactive rage-based chatbot. Fully compatible with GitHub Pages!

## Features

- ✨ Beautiful starry night theme with animated stars
- 📚 Education timeline
- 💼 Work experience timeline
- 🚀 Projects displayed in a 3-column grid
- 📧 Contact form with email functionality
- 💬 Rage-based chatbot (client-side, works on GitHub Pages!)

## Deployment to GitHub Pages

### Quick Setup:

1. Push all files to a GitHub repository
2. Go to repository Settings → Pages
3. Select the branch (usually `main` or `master`) and `/ (root)` folder
4. Click Save
5. Your site will be live at `https://yourusername.github.io/repository-name`

**Note:** Make sure `index.html` is in the root directory of your repository.

### Files Structure:
```
repository-root/
├── index.html
├── styles.css
├── script.js
├── Profile.jpeg
└── README.md
```

## Local Development

Simply open `index.html` in your web browser, or use a local server:

```bash
# Using Python's built-in server
python -m http.server 8000
```

Then navigate to `http://localhost:8000` in your browser.

## Files

- `index.html` - Main website HTML
- `styles.css` - Styling and animations
- `script.js` - Frontend JavaScript with client-side chatbot (no backend needed!)
- `Profile.jpeg` - Profile image

## Chatbot

The chatbot is **rage-based** and gets progressively more annoyed if you ask repetitive questions! It's fully client-side (no backend required) and contains all the information from the website in its knowledge base. It can answer questions about:

- Education (Rice University, IIT Madras)
- Work experience (NASA, Linbeck, ICME Lab, Goldman Sachs, ANEN Group)
- Projects (all 12 projects)
- General information about Aditi

**Note:** The `app.py` file is included for reference but is **not needed** for GitHub Pages deployment. The chatbot runs entirely in the browser using JavaScript.

Enjoy the starry experience! 🌟

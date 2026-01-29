// Generate stars dynamically
function createStars() {
    const starsContainer1 = document.getElementById('stars');
    const starsContainer2 = document.getElementById('stars2');
    const starsContainer3 = document.getElementById('stars3');
    
    // Create small stars (layer 1) - increased to 300
    for (let i = 0; i < 300; i++) {
        const star = document.createElement('div');
        star.style.position = 'absolute';
        star.style.width = '1px';
        star.style.height = '1px';
        star.style.backgroundColor = 'white';
        star.style.borderRadius = '50%';
        star.style.opacity = Math.random() * 0.8 + 0.2;
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animation = `twinkle ${Math.random() * 3 + 2}s ease-in-out infinite`;
        starsContainer1.appendChild(star);
    }
    
    // Create medium stars (layer 2) - increased to 150
    for (let i = 0; i < 150; i++) {
        const star = document.createElement('div');
        star.style.position = 'absolute';
        star.style.width = '2px';
        star.style.height = '2px';
        star.style.backgroundColor = 'white';
        star.style.borderRadius = '50%';
        star.style.opacity = Math.random() * 0.9 + 0.1;
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animation = `twinkle ${Math.random() * 4 + 3}s ease-in-out infinite`;
        starsContainer2.appendChild(star);
    }
    
    // Create large stars (layer 3) - increased to 75
    for (let i = 0; i < 75; i++) {
        const star = document.createElement('div');
        star.style.position = 'absolute';
        star.style.width = '3px';
        star.style.height = '3px';
        star.style.backgroundColor = '#ffd700';
        star.style.borderRadius = '50%';
        star.style.opacity = Math.random() * 0.8 + 0.2;
        star.style.boxShadow = '0 0 6px #ffd700';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 100 + '%';
        star.style.animation = `twinkle ${Math.random() * 5 + 4}s ease-in-out infinite`;
        starsContainer3.appendChild(star);
    }
}

// Add twinkle animation
const style = document.createElement('style');
style.textContent = `
    @keyframes twinkle {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }
`;
document.head.appendChild(style);

// Initialize stars on page load
document.addEventListener('DOMContentLoaded', createStars);

// Smooth scrolling for navigation links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Contact Form Email Handler
document.getElementById('contact-form').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const subject = document.getElementById('subject').value;
    const message = document.getElementById('message').value;
    
    // Create mailto link
    const mailtoLink = `mailto:aditi.balaji@rice.edu?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(`From: ${name} (${email})\n\n${message}`)}`;
    
    // Open email client
    window.location.href = mailtoLink;
    
    // Show success message
    alert('Your email client should open now. If it doesn\'t, please email aditi.balaji@rice.edu directly.');
    
    // Reset form
    this.reset();
});

// Chatbot functionality - Connects to Flask backend
const chatbotToggle = document.getElementById('chatbot-toggle');
const chatbotWindow = document.getElementById('chatbot-window');
const chatbotClose = document.getElementById('chatbot-close');
const chatbotSend = document.getElementById('chatbot-send');
const chatbotInput = document.getElementById('chatbot-input');
const chatbotMessages = document.getElementById('chatbot-messages');

// Backend API URL - Update this with your deployed backend URL
// For local development: 'http://localhost:5000'
// For production: 'https://your-backend-url.onrender.com' (or your hosting service)
const BACKEND_URL = 'https://aditi-balaji-13-github-io.onrender.com'; // UPDATE THIS!

// Knowledge base (client-side)
const knowledgeBase = {
    education: {
        rice: {
            school: "Rice University",
            location: "Houston, TX",
            degree: "Master of Data Science",
            duration: "Aug 2023 - May 2025",
            courses: "Generative AI for images, Probabilistic Data Structures and Algorithms, Big Data, Deep Learning for Vision and Language, Machine Learning with Graphs, Machine Learning, Data Visualization, Reinforcement Learning"
        },
        iit: {
            school: "Indian Institute of Technology Madras",
            location: "Chennai, IN",
            degree: "Bachelor of Technology in Metallurgical and Materials Engineering",
            duration: "Jul 2019 - Jul 2023",
            minor: "Minor in Machine Learning and Artificial Intelligence",
            courses: "Pattern Recognition, Natural Language Processing, Knowledge Representation Reasoning, Applied Statistics"
        }
    },
    experience: {
        nasa: {
            title: "Data Science Research Assistant",
            company: "NASA: Data to Knowledge, Rice University (Capstone Project)",
            location: "Houston, TX",
            duration: "Jan 2025 – Apr 2025",
            description: "Designed Spacecraft Image Segmentation System with 10,000+ samples optimized among 5+ AI models (MobilenetV3, etc.). Achieved sub-1s inference and 0.77 similarity score on a 3GB CPU, ensuring scalability in low-resource settings. Reflected performance tradeoffs in model selection to align with project requirement (opensource, etc.) and computational limits."
        },
        linbeck: {
            title: "AI Intern",
            company: "Linbeck Group LLC",
            location: "Houston, TX",
            duration: "Jun 2024 – Dec 2024",
            description: "Designed customizable multi-workspace RAG-based chatbot with AWS, ChromaDB, and Ollama, to improve information transfer. Annotated 50,000+ pages of unstructured data using Python-based vectorized workflows; Finetuned data and model pipelines. Improved partner intel access and presented architecture to 20+ non-technical executives; Annotated clear readable code."
        },
        icme: {
            title: "Research Assistant",
            company: "ICME Lab, Indian Institute of Technology Madras",
            location: "Chennai, IN",
            duration: "Aug 2022 – May 2023",
            description: "Addressed performance gaps in grain growth simulation using spatio-temporal Graph Neural Networks. Improved forecast accuracy by 20% and reduced processing time by 40% across 178-node datasets for 50+ timestamps. Improved existing Data Processing pipelines by adding 3+ features to transfer visual data into graphs for simplified computation."
        },
        goldman: {
            title: "Summer Analyst (Quantitative)",
            company: "Goldman Sachs",
            location: "Bangalore, IN",
            duration: "May 2022 – Jul 2022",
            description: "Statistically enhanced marketing recommendation models by 20% using custom tools for imbalanced data-handling. A/B tested 5+ techniques including SMOTE, ADASYN, and cost-sensitive models on 0.07% imbalanced class. Analyzed consumer wealth trends using Python and statistical modeling techniques to refine risk assessment models."
        },
        anen: {
            title: "Data Science Intern",
            company: "ANEN Group, Indian Institute of Technology Madras",
            location: "Chennai, IN",
            duration: "Dec 2021 – May 2022",
            description: "Consolidated and predict crystallographic properties using supervised Machine Learning with 30% higher accuracy. Deployed experimental and DFT calculated band-gap values of 1000+ materials to obtain patterns using 15 features. Obtained an accuracy of 0.81 in predicting DFT over/under-estimation by applying ML algorithms."
        }
    },
    projects: {
        generative_ai: {
            name: "Generative AI for Images",
            tech: "Generative AI, 3D imaging, PyTorch",
            duration: "Sept 2024 – Dec 2024",
            description: "Spearheaded analysis on Neural Radiance Fields (NeRFs) for 3D object reconstruction. Enhanced monocular depth estimation using diffusion models + perceptual loss, improving structural similarity by 10%."
        },
        big_data: {
            name: "Big Data and Machine Learning Applications",
            tech: "SQL, Hadoop, PySpark, AWS",
            duration: "Jan 2024 - Apr 2024",
            description: "Leveraged SQL, Hadoop, and PySpark to analyze 5+ large-scale datasets, implementing MapReduce-based ML models. Deployed machine learning workflows on AWS EC2 & S3 using Python & Java, ensuring scalable data processing."
        },
        qa_assistant: {
            name: "Lightweight QA Assistant",
            tech: "Transformers, NLP, PySpark",
            duration: "Jan 2024 - Apr 2024",
            description: "Developed a customer-facing QA chatbot using the DistilGPT-2 transformer, fine-tuned on the Databricks-Dolly-15k dataset. Achieved 3.99 perplexity and 1.38 test cross-entropy loss over 10 epochs using an 80-10-10 train-validation-test split."
        },
        financial: {
            name: "Predictive Analysis for Financial Markets",
            tech: "Market Analysis, Backtesting, Python",
            duration: "Jan 2024 - Apr 2024",
            description: "Built & backtested 3 algorithmic investment strategies using 150/50 long-short portfolios with 10,000+ historical data points. Applied supervised learning and time series forecasting to optimize financial decision-making and perform market impact analysis."
        },
        graph_ml: {
            name: "Machine Learning with Graphs",
            tech: "Knowledge Graphs, NLP, Reinforcement Learning",
            duration: "Jan 2024 - Apr 2024",
            description: "Created a knowledge-graph augmented LLM for semantic information retrieval; automate number of hop estimation. Studied 3+ methods in the field of Knowledge graph enhanced reinforcement learning for reasoning and collaborative learning."
        },
        stock: {
            name: "Stock Closing Price Prediction",
            tech: "CatBoost, Regression, Time Series",
            duration: "Aug 2023 - Dec 2023",
            description: "Predicted closing prices for 200 stocks (26.5k datapoints each) using CatBoost, achieving MSE of 5.732. Benchmarked 10+ regression and classification models across 3+ standard datasets to evaluate performance."
        },
        encrypted_ir: {
            name: "Encrypted Information Retrieval",
            tech: "NLP, Information Retrieval, Vector Embeddings",
            duration: "Mar 2023 – May 2023",
            description: "Built an encrypted IR system using vector space model, achieving 34% initial precision. Boosted precision by 2% using pretrained neural network–based embeddings (sequence-to-sequence)."
        },
        image_classification: {
            name: "Image Classification and Processing",
            tech: "CNNs, CIFAR-10, Image Processing",
            duration: "Jul 2022 - Dec 2022",
            description: "Designed MLP (40%) and CNN (60%) models for image classification on the CIFAR-10 dataset. Implemented edge detection, hybridization, and panoramic stitching for multi-image processing."
        },
        logic_tool: {
            name: "Automated Logic Representation Tool",
            tech: "FOL, Clause Form, Parsing",
            duration: "Jan 2022 – May 2022",
            description: "Converted 6+ First Order Logic statements into clause form using reasoning and representation algorithms. Built a parser to translate logical expressions between XML and TXT formats using Python."
        },
        rl: {
            name: "Reinforcement Learning",
            tech: "Reinforcement Learning, Python",
            duration: "Aug 2021 - Dec 2021",
            description: "Implemented policy gradient & Q-learning in 3+ reinforcement learning environments. Applied Bellman equations and dynamic programming to solve MDP problems."
        },
        game_theory: {
            name: "Game Theory and Policy Modeling",
            tech: "Computational Economics, OOP, Simulation",
            duration: "Aug 2021 – Dec 2021",
            description: "Simulated multi-period farmer's market using 4 piecewise utility functions via object-oriented programming. Evaluated impacts of India's farm bill across 3 stakeholder scenarios through economic stress testing."
        },
        customer_behavior: {
            name: "Customer Behaviour Modeling",
            tech: "CatBoost, Recommender Systems, MSE",
            duration: "Jan 2021 – May 2021",
            description: "Predicted user song ratings using CatBoost on real-world user metadata (1.3M rows), achieving MSE of 0.75. Modeled personalized recommendation behavior using historical user interaction data."
        }
    },
    general: {
        email: "aditi.balaji@rice.edu",
        linkedin: "www.linkedin.com/in/aditibalaji",
        github: "https://github.com/Aditi-balaji-13",
        title: "Data Scientist | AI Researcher | Machine Learning Engineer"
    }
};

// Rage responses
const rageResponses = [
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
];

const rageKeywords = ["again", "repeat", "what did you say", "i don't understand", "i'm confused", "i forgot"];

// Conversation history
let conversationHistory = [];

// Chatbot functions
function checkRageTriggers(message) {
    const messageLower = message.toLowerCase();
    return rageKeywords.some(keyword => messageLower.includes(keyword));
}

function searchKnowledgeBase(query) {
    const queryLower = query.toLowerCase();
    const results = [];
    
    // Check education
    if (["education", "school", "university", "degree", "master", "bachelor", "rice", "iit", "iitm"].some(term => queryLower.includes(term))) {
        if (queryLower.includes("rice") || queryLower.includes("master")) {
            results.push(`Aditi is doing her Master of Data Science at Rice University (Houston, TX) from Aug 2023 - May 2025. Courses include: ${knowledgeBase.education.rice.courses}`);
        }
        if (queryLower.includes("iit") || queryLower.includes("bachelor") || queryLower.includes("undergraduate")) {
            results.push(`Aditi completed her Bachelor of Technology in Metallurgical and Materials Engineering at IIT Madras (Chennai, IN) from Jul 2019 - Jul 2023. She also has a Minor in Machine Learning and Artificial Intelligence. Relevant courses: ${knowledgeBase.education.iit.courses}`);
        }
    }
    
    // Check experience
    const experienceKeywords = {
        nasa: ["nasa", "spacecraft", "image segmentation", "capstone"],
        linbeck: ["linbeck", "rag", "chatbot", "aws", "chromadb", "ollama"],
        icme: ["icme", "grain", "graph neural", "gnn"],
        goldman: ["goldman", "sachs", "quantitative", "analyst", "marketing"],
        anen: ["anen", "crystallographic", "dft", "band-gap"]
    };
    
    for (const [expKey, keywords] of Object.entries(experienceKeywords)) {
        if (keywords.some(keyword => queryLower.includes(keyword))) {
            const exp = knowledgeBase.experience[expKey];
            results.push(`${exp.title} at ${exp.company} (${exp.location}) from ${exp.duration}. ${exp.description}`);
        }
    }
    
    // Check projects
    const projectKeywords = {
        generative_ai: ["generative", "nerf", "3d", "depth estimation", "diffusion"],
        big_data: ["big data", "hadoop", "pyspark", "mapreduce", "aws"],
        qa_assistant: ["qa", "assistant", "distilgpt", "dolly"],
        financial: ["financial", "stock", "market", "backtesting", "investment"],
        graph_ml: ["graph", "knowledge graph", "llm", "semantic"],
        stock: ["stock", "catboost", "closing price", "mse"],
        encrypted_ir: ["encrypted", "information retrieval", "vector"],
        image_classification: ["image", "classification", "cifar", "cnn"],
        logic_tool: ["logic", "fol", "clause", "xml"],
        rl: ["reinforcement", "q-learning", "policy gradient", "bellman"],
        game_theory: ["game theory", "farmer", "policy", "simulation"],
        customer_behavior: ["customer", "behavior", "recommendation", "rating"]
    };
    
    for (const [projKey, keywords] of Object.entries(projectKeywords)) {
        if (keywords.some(keyword => queryLower.includes(keyword))) {
            const proj = knowledgeBase.projects[projKey];
            results.push(`${proj.name} (${proj.tech}) from ${proj.duration}. ${proj.description}`);
        }
    }
    
    // General questions
    if (["who", "what do", "tell me about", "about aditi"].some(term => queryLower.includes(term))) {
        results.push(`Aditi is a ${knowledgeBase.general.title}. She's currently pursuing her Master's at Rice University and has extensive experience in data science, AI, and machine learning. Check out her projects and experience sections!`);
    }
    
    if (["contact", "email", "linkedin", "github", "reach"].some(term => queryLower.includes(term))) {
        results.push(`You can reach Aditi on LinkedIn: ${knowledgeBase.general.linkedin} or GitHub: ${knowledgeBase.general.github}`);
    }
    
    return results;
}

function generateResponse(message) {
    // Check for rage triggers
    if (checkRageTriggers(message) && conversationHistory.length > 2) {
        const rageResponse = rageResponses[Math.floor(Math.random() * rageResponses.length)];
        const results = searchKnowledgeBase(message);
        if (results.length > 0) {
            return `${rageResponse} But fine, here it is: ${results[0]}`;
        } else {
            return `${rageResponse} And I don't know what you're even asking about! Try asking about education, experience, or projects!`;
        }
    }
    
    // Search knowledge base
    const results = searchKnowledgeBase(message);
    
    if (results.length > 0) {
        let response = results[0];
        if (results.length > 1) {
            response += `\n\nAlso: ${results[1]}`;
        }
        return response;
    } else {
        const defaultResponses = [
            "Hmm, not sure what you're asking about. Try asking about Aditi's education, work experience, or projects! 🤔",
            "I don't have that info right now. Why don't you check the education, experience, or projects sections? 🙃",
            "Can you be more specific? I know about education, experience, and projects. Pick one! 😏"
        ];
        return defaultResponses[Math.floor(Math.random() * defaultResponses.length)];
    }
}

chatbotToggle.addEventListener('click', () => {
    chatbotWindow.classList.toggle('hidden');
});

chatbotClose.addEventListener('click', () => {
    chatbotWindow.classList.add('hidden');
});

function addMessage(text, isUser) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
    // Handle newlines in text
    messageDiv.innerHTML = text.split('\n').join('<br>');
    chatbotMessages.appendChild(messageDiv);
    chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    return messageDiv; // Return element so it can be removed if needed
}

async function sendMessage() {
    const message = chatbotInput.value.trim();
    if (!message) return;
    
    addMessage(message, true);
    chatbotInput.value = '';
    chatbotSend.disabled = true;
    chatbotInput.disabled = true;
    
    // Show loading indicator
    const loadingMessage = addMessage("Thinking...", false);
    
    try {
        // Call Flask backend API
        const response = await fetch(`${BACKEND_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: 'github-pages-session'
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove loading message and add actual response
        loadingMessage.remove();
        addMessage(data.response || "I'm sorry, I couldn't generate a response.", false);
        
    } catch (error) {
        console.error('Error calling backend:', error);
        loadingMessage.remove();
        addMessage("Sorry, I'm having trouble connecting to the server. Please try again later.", false);
    } finally {
        chatbotSend.disabled = false;
        chatbotInput.disabled = false;
        chatbotInput.focus();
    }
}

chatbotSend.addEventListener('click', sendMessage);
chatbotInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendMessage();
    }
});

// Add welcome message when chatbot opens
let welcomeShown = false;
chatbotToggle.addEventListener('click', () => {
    if (!welcomeShown && !chatbotWindow.classList.contains('hidden')) {
        setTimeout(() => {
            addMessage('Hey! I\'m Aditi\'s rage-based chatbot. Ask me anything about her education, experience, or projects! 🔥', false);
            welcomeShown = true;
        }, 300);
    }
});

// Navbar scroll effect
let lastScroll = 0;
let shootingStarTimeout = null;

window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.background = 'rgba(10, 10, 26, 0.95)';
    } else {
        navbar.style.background = 'rgba(10, 10, 26, 0.9)';
    }
    
    // Trigger shooting star randomly during scroll
    if (Math.random() < 0.02 && !shootingStarTimeout) { // 2% chance on each scroll event
        createShootingStar();
    }
    
    lastScroll = currentScroll;
});

// Shooting star function
function createShootingStar() {
    const shootingStar = document.getElementById('shooting-star');
    
    // Random starting position from top-left area
    const startX = Math.random() * 30; // 0-30% from left
    const startY = Math.random() * 30; // 0-30% from top
    const angle = 45 + Math.random() * 20; // 45-65 degrees
    const distance = 50 + Math.random() * 30; // Distance to travel
    
    // Calculate end position
    const endX = startX + Math.cos(angle * Math.PI / 180) * distance;
    const endY = startY + Math.sin(angle * Math.PI / 180) * distance;
    
    // Set starting position
    shootingStar.style.left = startX + '%';
    shootingStar.style.top = startY + '%';
    shootingStar.style.opacity = '1';
    
    // Animate to end position
    shootingStar.style.transform = `translate(${endX - startX}%, ${endY - startY}%)`;
    
    // Reset after animation
    shootingStarTimeout = setTimeout(() => {
        shootingStar.style.opacity = '0';
        shootingStar.style.transform = 'translate(0, 0)';
        shootingStarTimeout = null;
    }, 1000);
}

// Also trigger shooting stars on page load and randomly
setInterval(() => {
    if (Math.random() < 0.1 && !shootingStarTimeout) { // 10% chance every interval
        createShootingStar();
    }
}, 5000); // Check every 5 seconds

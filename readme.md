🚀 AI Resume Evaluator

An intelligent web application that analyzes resumes and provides actionable feedback using AI. It helps users improve their resumes by evaluating structure, skills, keywords, and overall quality.

📌 Features
📄 Upload Resume (PDF/DOCX)
🤖 AI-powered Resume Analysis
📊 Section-wise Feedback (Skills, Experience, Projects, etc.)
🎯 ATS Optimization Suggestions
🏢 Suggested Companies based on Resume (optional feature)
⚡ Clean UI with loading states and result cards
📥 Download Evaluation Report
🛠️ Tech Stack
Frontend
React.js
Tailwind CSS
Axios
Backend
Node.js
Express.js
AI / Processing
OpenAI API / Claude API
NLP techniques (TF-IDF, keyword extraction)
Other Tools
Multer (file uploads)
PDF / DOCX parser
📂 Project Structure
AI-Resume-Evaluator/
│
├── frontend/
│   ├── components/
│   ├── pages/
│   ├── services/
│   └── App.jsx
│
├── backend/
│   ├── controllers/
│   ├── routes/
│   ├── utils/
│   └── server.js
│
├── uploads/
├── README.md
└── package.json
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-resume-evaluator.git
cd ai-resume-evaluator
2️⃣ Setup Backend
cd backend
npm install

Create a .env file:

PORT=5000
OPENAI_API_KEY=your_api_key_here

Run backend:

npm start
3️⃣ Setup Frontend
cd frontend
npm install
npm run dev
🔄 How It Works
User uploads resume
Backend extracts text from file
AI analyzes:
Skills
Experience
Keywords
Formatting
Structured feedback is generated
Frontend displays results in card format
📊 Example Output
✅ Strong Skills Section
⚠️ Missing Keywords for ATS
❌ Weak Project Descriptions
💡 Suggestions for improvement
🌟 Future Improvements
🔐 User Authentication
📈 Resume Score System
🧠 Advanced NLP (BERT / embeddings)
📊 Resume Comparison
🌍 Multi-language Support
🏢 Live Job Matching API
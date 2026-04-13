🚀 AI Resume Evaluator

An intelligent web application that analyzes resumes and provides actionable feedback using AI.
It evaluates resume quality based on ATS standards, keyword optimization, and role-specific expectations.

📌 Features
📄 Upload Resume (PDF / DOCX / TXT)
🤖 AI-powered Resume Analysis
📊 Section-wise Scoring:
Basic Details
Summary
Education
Experience
Projects
Certifications
Skills
🎯 ATS Optimization Suggestions
🧠 Keyword Extraction using NLP (TF-IDF)
📈 Role-based Evaluation (Data Analyst, SDE roles)
⚡ Clean UI with instant feedback
📥 Downloadable evaluation report (optional / extendable)
🛠️ Tech Stack
Frontend
HTML5
CSS3
JavaScript
Axios
Backend
Python
Flask
AI / Processing
Claude API
NLP (TF-IDF, keyword extraction, rule-based scoring)
File Handling
Multer (for uploads)
PDF / DOCX parsing libraries
⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-resume-evaluator.git
cd ai-resume-evaluator
2️⃣ Backend Setup (Flask)
cd backend
python -m venv venv
Activate Virtual Environment
Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate
Install Dependencies
pip install -r requirements.txt
Set Environment Variables

Create a .env file:

CLAUDE_API_KEY=your_api_key_here
Run Backend Server
python app.py

Backend runs on:

http://127.0.0.1:5000
3️⃣ Frontend Setup
cd frontend

Simply open:

index.html

Or use Live Server (recommended in VS Code)

📂 Project Structure
ai-resume-evaluator/
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│
├── backend/
│   ├── app.py
│   ├── utils/
│   │   ├── parser.py
│   │   ├── scorer.py
│   │   ├── nlp.py
│   ├── requirements.txt
│
├── uploads/
├── README.md
🧠 How It Works
User uploads resume
Backend parses document (PDF/DOCX)
Text is processed using NLP techniques
Claude API + heuristic scoring:
Extracts sections
Identifies missing components
Scores each section
Final structured report is generated with:
Score
Weak areas
Suggestions
📊 Example Output
Overall Score: 70.12 / 100
Skills: ✅ 20/20
Experience: ✅ 20/20
Projects: ❌ 0/20
Certifications: ⚠️ Low score
Suggestions:
Add GitHub & LinkedIn links
Include 2–3 projects with measurable impact
Improve certification relevance
🚧 Limitations
Heuristic scoring (not fully semantic understanding)
Limited contextual reasoning across sections
No deep fine-tuned model (yet)
Depends on keyword matching + structure detection
🔮 Future Improvements
Fine-tuned LLM for deeper resume understanding
Semantic similarity scoring (job description vs resume)
Resume rewriting suggestions
Real-time ATS simulation
Resume vs job match percentage
Dashboard for tracking improvements
🧪 Example Use Cases
Students improving resumes before placements
Professionals optimizing for ATS systems
Recruiters screening resumes quickly
Career coaching platforms
🤝 Contributing

Contributions are welcome.

fork → clone → create branch → commit → push → PR
📜 License

This project is licensed under the MIT License.

👤 Author

Vardhan Boi

GitHub: https://github.com/VardhanBoi
LinkedIn: (add your link)
⭐ If you found this useful

Give this repo a star ⭐ — it helps visibility and credibility.

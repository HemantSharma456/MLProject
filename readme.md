# ResumeIQ 🧠📄
### AI-Powered Resume Evaluator

> Upload your resume. Pick a role. Get instant, structured feedback with scores and actionable suggestions.

![ResumeIQ Banner](https://img.shields.io/badge/ResumeIQ-AI%20Resume%20Evaluator-orange?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square&logo=flask)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 What is ResumeIQ?

ResumeIQ is an intelligent resume evaluation web application that analyzes your resume against role-specific expectations and gives you a **score out of 100** — broken down section by section — along with **AI-generated improvement suggestions** that tell you exactly what to fix and how many points you'd gain.

No more guessing why your resume isn't getting callbacks. ResumeIQ shows you.

---

## ✨ Features

- 📄 **Upload Resume** — Supports PDF and DOCX formats (up to 10 MB)
- 🎯 **Role-Specific Evaluation** — Choose from Data Analyst, SDE Frontend, SDE Backend, or SDE Fullstack
- 📊 **Section-wise Scoring** — 8 sections evaluated with a weighted rubric totaling 100 points
- 🤖 **AI-Powered Suggestions** — Adzuna AI generates actionable feedback with point-impact indicators (+1, +2, +3)
- 🏷️ **ATS Optimization Tips** — Know if your resume will pass Applicant Tracking Systems
- 👁️ **Resume Overview Dashboard** — Extracted metadata (name, email, skills, experience count) displayed visually
- 📥 **Download Evaluation Report** — Save your results for offline reference
- ⚡ **Instant Results** — Analysis completes in seconds

---

## 🎯 Scoring Rubric

| Section | Max Score | What's Evaluated |
|---|---|---|
| Basic Details | 10 | Name, email, phone, LinkedIn, GitHub |
| Professional Summary | 2.5 | Presence, role alignment, quantified impact |
| Education | 10 | Degree, institution, graduation dates |
| Experience | 20 | Roles, duration, achievements, relevant tools |
| Projects | 20 | Count, tech stack, outcomes, links |
| Certifications | 7.5 | Relevance, issuing org, recency |
| Skills | 20 | Coverage, breadth, role-relevant grouping |
| Section Order | 10 | ATS-standard ordering |
| **Total** | **100** | |

---

## 🛠️ Tech Stack

**Frontend**
- HTML5, CSS3, JavaScript
- Axios (async API calls)

**Backend**
- Python 3.10+
- Flask (REST API & file handling)

**AI / NLP**
- Adzuna API
- scikit-learn — TF-IDF keyword extraction

**Document Parsing**
- PyMuPDF (`fitz`) — PDF text extraction
- python-docx — DOCX text extraction

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+

### 1. Clone the repository

```bash
git clone https://github.com/HemantSharma456/MLProject.git
cd MLProject
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Open your browser and go to `http://127.0.0.1:5000`

---



---

## 🖥️ How It Works

```
User uploads resume (PDF/DOCX)
        ↓
Flask receives file → DocumentParser extracts raw text
        ↓
NLPProcessor: NLTK tokenization + TF-IDF keyword extraction
        ↓
Claude API: structured prompt → JSON evaluation response
        ↓
ScoringEngine: applies weighted rubric → section scores + overall score
        ↓
Frontend renders dashboard: score, overview, section bars, AI suggestions
```

---

## 📸 Screenshots

| Landing Page | Upload & Role Selection |
|---|---|
| *Know exactly where your resume stands* | *Select role → Drop resume → Instant analysis* |

| Score Dashboard | Section Feedback |
|---|---|
| *Overall score out of 100 with section breakdown* | *AI suggestions with point-impact badges* |

---

## 🔮 Roadmap

- [ ] Support for more job roles (Data Scientist, DevOps, Product Manager, UI/UX)
- [ ] User accounts with resume score history and progress tracking
- [ ] OCR support for scanned PDF resumes
- [ ] Job description import for job-specific (not just role-generic) evaluation
- [ ] Suggested companies based on resume skills and experience profile
- [ ] Cloud deployment (AWS / Render / Railway)

---

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Anthropic](https://www.anthropic.com) for the Claude API
- [Bennett University SCSET](https://www.bennett.edu.in) for project support
- The open-source community behind Flask, PyMuPDF, python-docx, NLTK, and scikit-learn
EOF

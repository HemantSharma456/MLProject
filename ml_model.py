import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class ResumeEvaluator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.classifier = MultinomialNB(alpha=1.0)
        self.role_mapping = {
            'Data Analyst': ['Data Science', 'Business Analyst', 'Database', 'SQL Developer', 'ETL Developer'],
            'SDE(Frontend)': ['Web Designing', 'React Developer', 'UI-UX', 'Frontend Developer'],
            'SDE(Backend)': ['Java Developer', 'Python Developer', 'DotNet Developer', 'SAP Developer', 'Backend Developer'],
            'SDE(Fullstack)': ['React Developer', 'Web Designing', 'Java Developer', 'Python Developer', 'Full Stack Developer', 'Web Developer']
        }
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess resume text"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def load_and_prepare_data(self, csv_path='Dataset.csv'):
        """Load dataset and map to target roles"""
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # Create role mapping
        role_data = []
        for target_role, categories in self.role_mapping.items():
            for category in categories:
                category_data = df[df['Category'].str.contains(category, case=False, na=False)]
                for _, row in category_data.iterrows():
                    role_data.append({
                        'role': target_role,
                        'text': self.preprocess_text(row['Text'])
                    })
        
        # Also add some generic software/data roles to balance the dataset
        software_roles = df[df['Category'].str.contains('Developer|Engineer', case=False, na=False)]
        for _, row in software_roles.head(300).iterrows():
            category = str(row['Category']).lower()
            if 'react' in category or 'web' in category:
                role_data.append({
                    'role': 'SDE(Frontend)',
                    'text': self.preprocess_text(row['Text'])
                })
            elif 'java' in category or 'python' in category or 'dotnet' in category:
                role_data.append({
                    'role': 'SDE(Backend)',
                    'text': self.preprocess_text(row['Text'])
                })
            else:
                role_data.append({
                    'role': 'SDE(Fullstack)',
                    'text': self.preprocess_text(row['Text'])
                })
        
        role_df = pd.DataFrame(role_data)
        print(f"Prepared {len(role_df)} samples")
        return role_df
    
    def train(self, csv_path='Dataset.csv'):
        """Train the Multinomial Naive Bayes classifier"""
        print("Training ML model...")
        df = self.load_and_prepare_data(csv_path)
        
        if len(df) == 0:
            raise ValueError("No data found for training")
        
        X = df['text'].values
        y = df['role'].values
        
        # Split data (use shuffle instead of stratify if not enough samples per class)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # If stratification fails, use shuffle instead
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Vectorize
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        return accuracy
    
    def predict_role(self, resume_text):
        """Predict the role for a resume"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_text = self.preprocess_text(resume_text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(text_tfidf)[0]
        probabilities = self.classifier.predict_proba(text_tfidf)[0]
        
        role_probs = dict(zip(self.classifier.classes_, probabilities))
        return prediction, role_probs
    
    def calculate_section_scores(self, resume_data, target_role):
        """Calculate scores for each section based on ML analysis"""
        scores = {
            'order_of_sections': 0,
            'basic_details': 0,
            'professional_summary': 0,
            'education': 0,
            'experience': 0,
            'projects': 0,
            'certification': 0,
            'skills': 0
        }
        
        max_scores = {
            'order_of_sections': 10,
            'basic_details': 10,
            'professional_summary': 2.5,
            'education': 10,
            'experience': 20,
            'projects': 20,
            'certification': 7.5,
            'skills': 20
        }
        
        # Order of Sections (check if standard order is followed)
        sections_order = ['basic_details', 'professional_summary', 'education', 'experience', 'projects', 'certification', 'skills']
        found_sections = []
        for section in sections_order:
            if section in resume_data and resume_data[section]:
                found_sections.append(section)
        
        if len(found_sections) >= 5:
            scores['order_of_sections'] = max_scores['order_of_sections']
        elif len(found_sections) >= 3:
            scores['order_of_sections'] = 7
        else:
            scores['order_of_sections'] = 5
        
        # Basic Details Score
        basic_fields = ['name', 'email', 'phone', 'github', 'linkedin']
        found_fields = sum(1 for field in basic_fields if resume_data.get('basic_details', {}).get(field))
        scores['basic_details'] = (found_fields / len(basic_fields)) * max_scores['basic_details']
        
        # Professional Summary Score
        summary = resume_data.get('professional_summary', '')
        if summary:
            # Handle both list and string formats
            if isinstance(summary, list):
                summary_text = ' '.join(summary)
                bullet_count = len(summary)
            else:
                summary_text = str(summary)
                bullet_count = 1
            
            word_count = len(summary_text.split())
            # Score based on bullet points (3-5 is ideal) and word count
            if 3 <= bullet_count <= 5 and 30 <= word_count <= 200:
                scores['professional_summary'] = max_scores['professional_summary']
            elif 2 <= bullet_count <= 6 and 20 <= word_count <= 250:
                scores['professional_summary'] = max_scores['professional_summary'] * 0.75
            elif bullet_count >= 1 and 15 <= word_count <= 300:
                scores['professional_summary'] = max_scores['professional_summary'] * 0.5
            else:
                scores['professional_summary'] = max_scores['professional_summary'] * 0.3
        else:
            scores['professional_summary'] = 0
        
        # Education Score
        education_count = len(resume_data.get('education', []))
        if education_count >= 2:
            scores['education'] = max_scores['education']
        elif education_count == 1:
            scores['education'] = max_scores['education'] * 0.75
        else:
            scores['education'] = 0
        
        # Experience Score
        experience_count = len(resume_data.get('experience', []))
        if experience_count >= 3:
            scores['experience'] = max_scores['experience']
        elif experience_count == 2:
            scores['experience'] = max_scores['experience'] * 0.8
        elif experience_count == 1:
            scores['experience'] = max_scores['experience'] * 0.5
        else:
            scores['experience'] = 0
        
        # Projects Score
        projects_count = len(resume_data.get('projects', []))
        if projects_count >= 3:
            scores['projects'] = max_scores['projects']
        elif projects_count == 2:
            scores['projects'] = max_scores['projects'] * 0.75
        elif projects_count == 1:
            scores['projects'] = max_scores['projects'] * 0.5
        else:
            scores['projects'] = 0
        
        # Certification Score
        cert_count = len(resume_data.get('certification', []))
        if cert_count >= 3:
            scores['certification'] = max_scores['certification']
        elif cert_count == 2:
            scores['certification'] = max_scores['certification'] * 0.6
        elif cert_count == 1:
            scores['certification'] = max_scores['certification'] * 0.3
        else:
            scores['certification'] = 0
        
        # Skills Score (role-specific)
        skills = resume_data.get('skills', [])
        role_keywords = {
            'Data Analyst': ['python', 'sql', 'excel', 'tableau', 'power bi', 'pandas', 'numpy', 'statistics'],
            'SDE(Frontend)': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'typescript'],
            'SDE(Backend)': ['java', 'python', 'node.js', 'spring', 'django', 'database', 'api'],
            'SDE(Fullstack)': ['react', 'node.js', 'javascript', 'python', 'java', 'mongodb', 'sql']
        }
        
        required_skills = role_keywords.get(target_role, [])
        found_skills = sum(1 for skill in skills if any(keyword in skill.lower() for keyword in required_skills))
        
        if found_skills >= len(required_skills) * 0.6:
            scores['skills'] = max_scores['skills']
        elif found_skills >= len(required_skills) * 0.4:
            scores['skills'] = max_scores['skills'] * 0.75
        elif found_skills >= len(required_skills) * 0.2:
            scores['skills'] = max_scores['skills'] * 0.5
        else:
            scores['skills'] = max_scores['skills'] * 0.25
        
        return scores, max_scores
    
    def generate_suggestions(self, resume_data, target_role, scores):
        """Generate reliable, role-specific AI suggestions for improvement"""
        suggestions = []
        import re
        
        # Basic Details Suggestions
        basic = resume_data.get('basic_details', {})
        if not basic.get('github') and target_role.startswith('SDE'):
            suggestions.append({
                'section': 'basic_details',
                'points': 2,
                'suggestion': 'Add your GitHub profile to showcase your code repositories and projects'
            })
        elif not basic.get('github') and target_role == 'Data Analyst':
            suggestions.append({
                'section': 'basic_details',
                'points': 1,
                'suggestion': 'Consider adding GitHub profile to showcase data analysis projects and code samples'
            })
        
        if not basic.get('linkedin'):
            suggestions.append({
                'section': 'basic_details',
                'points': 2,
                'suggestion': 'Include your LinkedIn profile for professional networking and credibility'
            })
        
        if not basic.get('email'):
            suggestions.append({
                'section': 'basic_details',
                'points': 3,
                'suggestion': 'Add a professional email address for contact'
            })
        
        if not basic.get('phone'):
            suggestions.append({
                'section': 'basic_details',
                'points': 2,
                'suggestion': 'Include a contact phone number'
            })
        
        # Professional Summary Suggestions
        summary = resume_data.get('professional_summary', '')
        summary_text = ' '.join(summary) if isinstance(summary, list) else str(summary)
        
        # Check for typos
        if re.search(r'\bul\b', summary_text, re.IGNORECASE) and 'ui' not in summary_text.lower():
            suggestions.append({
                'section': 'professional_summary',
                'points': 1,
                'suggestion': "Fix typo: 'Ul' should be 'UI' (User Interface)"
            })
        
        # Check summary length and quality
        word_count = len(summary_text.split())
        if word_count < 30:
            suggestions.append({
                'section': 'professional_summary',
                'points': 2,
                'suggestion': 'Expand your professional summary to 3-5 bullet points highlighting key achievements and skills'
            })
        elif word_count > 200:
            suggestions.append({
                'section': 'professional_summary',
                'points': 1,
                'suggestion': 'Condense your summary to 3-5 concise bullet points for better readability'
            })
        
        # Role-specific summary suggestions
        role_keywords_in_summary = {
            'Data Analyst': ['data', 'analysis', 'analytics', 'sql', 'python', 'statistics'],
            'SDE(Frontend)': ['frontend', 'ui', 'ux', 'react', 'javascript', 'web'],
            'SDE(Backend)': ['backend', 'api', 'server', 'database', 'java', 'python'],
            'SDE(Fullstack)': ['fullstack', 'full-stack', 'web', 'application', 'development']
        }
        
        required_keywords = role_keywords_in_summary.get(target_role, [])
        found_keywords = sum(1 for kw in required_keywords if kw in summary_text.lower())
        if found_keywords < 2:
            suggestions.append({
                'section': 'professional_summary',
                'points': 1,
                'suggestion': f'Include role-specific keywords related to {target_role} in your summary'
            })
        
        # Education Suggestions
        education = resume_data.get('education', [])
        if len(education) == 0:
            suggestions.append({
                'section': 'education',
                'points': 3,
                'suggestion': 'Add your educational background including degree, institution, and graduation year'
            })
        elif len(education) == 1:
            suggestions.append({
                'section': 'education',
                'points': 1,
                'suggestion': 'Consider adding relevant coursework, GPA (if >3.5), or additional certifications'
            })
        
        # Experience Suggestions
        experience = resume_data.get('experience', [])
        if len(experience) == 0:
            suggestions.append({
                'section': 'experience',
                'points': 3,
                'suggestion': 'Add work experience with job titles, company names, dates, and key achievements using bullet points'
            })
        elif len(experience) == 1:
            suggestions.append({
                'section': 'experience',
                'points': 2,
                'suggestion': 'Add more work experience entries. Include internships, freelance work, or relevant projects as experience'
            })
        else:
            # Check for quantified achievements
            exp_text = ' '.join(experience).lower()
            if not re.search(r'\d+%|\d+\s*(years?|months?)|increased|decreased|improved|reduced', exp_text):
                suggestions.append({
                    'section': 'experience',
                    'points': 1,
                    'suggestion': 'Add quantified achievements (numbers, percentages, metrics) to make your experience more impactful'
                })
        
        # Projects Suggestions
        projects = resume_data.get('projects', [])
        if len(projects) == 0:
            suggestions.append({
                'section': 'projects',
                'points': 3,
                'suggestion': f'Add 2-3 projects relevant to {target_role} with technologies used, problem solved, and outcomes'
            })
        elif len(projects) < 2:
            suggestions.append({
                'section': 'projects',
                'points': 2,
                'suggestion': 'Include at least 2-3 projects. For each project, mention: technologies, your role, and impact/results'
            })
        else:
            # Check if projects mention technologies
            proj_text = ' '.join(projects).lower()
            tech_mentioned = re.search(r'\b(react|python|java|javascript|sql|node|django|flask|spring|aws|docker)\b', proj_text)
            if not tech_mentioned:
                suggestions.append({
                    'section': 'projects',
                    'points': 1,
                    'suggestion': 'Mention specific technologies and tools used in each project'
                })
        
        # Certification Suggestions
        certifications = resume_data.get('certification', [])
        if len(certifications) == 0:
            role_cert_suggestions = {
                'Data Analyst': 'Consider certifications like Google Data Analytics, Microsoft Power BI, or Tableau',
                'SDE(Frontend)': 'Consider certifications like AWS Certified Developer, React Certification, or Frontend Development courses',
                'SDE(Backend)': 'Consider certifications like AWS Certified Developer, Oracle Java Certification, or Backend Development courses',
                'SDE(Fullstack)': 'Consider certifications like AWS Certified Developer, Full Stack Development courses, or MERN Stack certifications'
            }
            suggestions.append({
                'section': 'certification',
                'points': 2,
                'suggestion': role_cert_suggestions.get(target_role, 'Add relevant professional certifications to strengthen your profile')
            })
        
        # Skills Suggestions - More comprehensive
        role_required_skills = {
            'Data Analyst': {
                'essential': ['python', 'sql', 'excel'],
                'recommended': ['tableau', 'power bi', 'pandas', 'numpy', 'statistics', 'r', 'machine learning'],
                'tools': ['jupyter', 'git', 'github']
            },
            'SDE(Frontend)': {
                'essential': ['html', 'css', 'javascript'],
                'recommended': ['react', 'angular', 'vue', 'typescript', 'responsive design'],
                'tools': ['git', 'github', 'npm', 'webpack']
            },
            'SDE(Backend)': {
                'essential': ['java', 'python', 'sql'],
                'recommended': ['node.js', 'spring', 'django', 'rest api', 'database', 'mongodb', 'postgresql'],
                'tools': ['git', 'github', 'docker', 'aws']
            },
            'SDE(Fullstack)': {
                'essential': ['javascript', 'html', 'css', 'sql'],
                'recommended': ['react', 'node.js', 'mongodb', 'express', 'rest api', 'git'],
                'tools': ['git', 'github', 'docker', 'aws', 'npm']
            }
        }
        
        skills = [s.lower() for s in resume_data.get('skills', [])]
        skill_reqs = role_required_skills.get(target_role, {})
        essential = skill_reqs.get('essential', [])
        recommended = skill_reqs.get('recommended', [])
        
        missing_essential = [s for s in essential if not any(s in skill for skill in skills)]
        missing_recommended = [s for s in recommended[:5] if not any(s in skill for skill in skills)]
        
        if missing_essential:
            suggestions.append({
                'section': 'skills',
                'points': 3,
                'suggestion': f'Add essential skills for {target_role}: {", ".join(missing_essential[:3])}'
            })
        
        if missing_recommended and len(missing_essential) == 0:
            suggestions.append({
                'section': 'skills',
                'points': 1,
                'suggestion': f'Consider adding recommended skills: {", ".join(missing_recommended[:3])}'
            })
        
        # Order of Sections suggestion
        if scores.get('order_of_sections', 0) < 8:
            suggestions.append({
                'section': 'order_of_sections',
                'points': 1,
                'suggestion': 'Follow standard resume order: Contact Info → Summary → Education → Experience → Projects → Certifications → Skills'
            })
        
        return suggestions
    
    def save_model(self, model_path='resume_model.joblib', vectorizer_path='vectorizer.joblib'):
        """Save trained model"""
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path}")
        print(f"Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_path='resume_model.joblib', vectorizer_path='vectorizer.joblib'):
        """Load trained model"""
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.classifier = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_trained = True
            print("Model loaded successfully")
            return True
        return False


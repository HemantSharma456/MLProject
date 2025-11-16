import pandas as pd
import re
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
        if pd.isna(text):
            return ""
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text).lower())
        return re.sub(r'\s+', ' ', text).strip()
    
    def load_and_prepare_data(self, csv_path='Dataset.csv'):
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
        
        software_roles = df[df['Category'].str.contains('Developer|Engineer', case=False, na=False)]
        for _, row in software_roles.head(300).iterrows():
            cat = str(row['Category']).lower()
            role = 'SDE(Frontend)' if 'react' in cat or 'web' in cat else \
                   'SDE(Backend)' if any(x in cat for x in ['java', 'python', 'dotnet']) else \
                   'SDE(Fullstack)'
            role_data.append({'role': role, 'text': self.preprocess_text(row['Text'])})
        
        role_df = pd.DataFrame(role_data)
        print(f"Prepared {len(role_df)} samples")
        return role_df
    
    def train(self, csv_path='Dataset.csv'):
        print("Training ML model...")
        df = self.load_and_prepare_data(csv_path)
        
        if len(df) == 0:
            raise ValueError("No data found for training")
        
        X = df['text'].values
        y = df['role'].values
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
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
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        processed_text = self.preprocess_text(resume_text)
        text_tfidf = self.vectorizer.transform([processed_text])
        prediction = self.classifier.predict(text_tfidf)[0]
        probabilities = self.classifier.predict_proba(text_tfidf)[0]
        
        role_probs = dict(zip(self.classifier.classes_, probabilities))
        return prediction, role_probs
    
    def calculate_section_scores(self, resume_data, target_role):
        max_scores = {
            'order_of_sections': 10, 'basic_details': 10, 'professional_summary': 2.5,
            'education': 10, 'experience': 20, 'projects': 20, 'certification': 7.5, 'skills': 20
        }
        scores = {k: 0 for k in max_scores.keys()}
        
        sections_order = ['basic_details', 'professional_summary', 'education', 'experience', 
                         'projects', 'certification', 'skills']
        found_sections = [s for s in sections_order if resume_data.get(s)]
        scores['order_of_sections'] = 10 if len(found_sections) >= 5 else 7 if len(found_sections) >= 3 else 5
        
        basic_fields = ['name', 'email', 'phone', 'github', 'linkedin']
        found = sum(1 for f in basic_fields if resume_data.get('basic_details', {}).get(f))
        scores['basic_details'] = (found / len(basic_fields)) * max_scores['basic_details']
        
        summary = resume_data.get('professional_summary', '')
        if summary:
            summary_text = ' '.join(summary) if isinstance(summary, list) else str(summary)
            bullet_count = len(summary) if isinstance(summary, list) else 1
            word_count = len(summary_text.split())
            if 3 <= bullet_count <= 5 and 30 <= word_count <= 200:
                scores['professional_summary'] = max_scores['professional_summary']
            elif 2 <= bullet_count <= 6 and 20 <= word_count <= 250:
                scores['professional_summary'] = max_scores['professional_summary'] * 0.75
            elif bullet_count >= 1 and 15 <= word_count <= 300:
                scores['professional_summary'] = max_scores['professional_summary'] * 0.5
            else:
                scores['professional_summary'] = max_scores['professional_summary'] * 0.3
        
        count_scores = {
            'education': [(2, 1.0), (1, 0.75)],
            'experience': [(3, 1.0), (2, 0.8), (1, 0.5)],
            'projects': [(3, 1.0), (2, 0.75), (1, 0.5)],
            'certification': [(3, 1.0), (2, 0.6), (1, 0.3)]
        }
        for section, thresholds in count_scores.items():
            count = len(resume_data.get(section, []))
            for min_count, multiplier in thresholds:
                if count >= min_count:
                    scores[section] = max_scores[section] * multiplier
                    break
        
        role_keywords = {
            'Data Analyst': ['python', 'sql', 'excel', 'tableau', 'power bi', 'pandas', 'numpy', 'statistics'],
            'SDE(Frontend)': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'typescript'],
            'SDE(Backend)': ['java', 'python', 'node.js', 'spring', 'django', 'database', 'api'],
            'SDE(Fullstack)': ['react', 'node.js', 'javascript', 'python', 'java', 'mongodb', 'sql']
        }
        required = role_keywords.get(target_role, [])
        found = sum(1 for s in resume_data.get('skills', []) 
                   if any(kw in s.lower() for kw in required))
        ratio = found / len(required) if required else 0
        scores['skills'] = max_scores['skills'] * (1.0 if ratio >= 0.6 else 0.75 if ratio >= 0.4 
                                                   else 0.5 if ratio >= 0.2 else 0.25)
        
        return scores, max_scores
    
    def generate_suggestions(self, resume_data, target_role, scores):
        suggestions = []
        basic = resume_data.get('basic_details', {})
        
        basic_suggestions = [
            ('email', 3, 'Add a professional email address for contact'),
            ('phone', 2, 'Include a contact phone number'),
            ('linkedin', 2, 'Include your LinkedIn profile for professional networking and credibility'),
        ]
        for field, points, msg in basic_suggestions:
            if not basic.get(field):
                suggestions.append({'section': 'basic_details', 'points': points, 'suggestion': msg})
        
        if not basic.get('github'):
            msg = 'Add your GitHub profile to showcase your code repositories and projects' if target_role.startswith('SDE') else 'Consider adding GitHub profile to showcase data analysis projects'
            suggestions.append({'section': 'basic_details', 'points': 2 if target_role.startswith('SDE') else 1, 'suggestion': msg})
        
        summary = resume_data.get('professional_summary', '')
        summary_text = ' '.join(summary) if isinstance(summary, list) else str(summary)
        word_count = len(summary_text.split())
        
        if word_count < 30:
            suggestions.append({'section': 'professional_summary', 'points': 2, 
                              'suggestion': 'Expand your professional summary to 3-5 bullet points highlighting key achievements and skills'})
        elif word_count > 200:
            suggestions.append({'section': 'professional_summary', 'points': 1, 
                              'suggestion': 'Condense your summary to 3-5 concise bullet points for better readability'})
        
        role_keywords = {
            'Data Analyst': ['data', 'analysis', 'analytics', 'sql', 'python', 'statistics'],
            'SDE(Frontend)': ['frontend', 'ui', 'ux', 'react', 'javascript', 'web'],
            'SDE(Backend)': ['backend', 'api', 'server', 'database', 'java', 'python'],
            'SDE(Fullstack)': ['fullstack', 'full-stack', 'web', 'application', 'development']
        }
        if sum(1 for kw in role_keywords.get(target_role, []) if kw in summary_text.lower()) < 2:
            suggestions.append({'section': 'professional_summary', 'points': 1, 
                              'suggestion': f'Include role-specific keywords related to {target_role} in your summary'})
        
        section_suggestions = {
            'education': [(0, 3, 'Add your educational background including degree, institution, and graduation year'),
                         (1, 1, 'Consider adding relevant coursework, GPA (if >3.5), or additional certifications')],
            'experience': [(0, 3, 'Add work experience with job titles, company names, dates, and key achievements using bullet points'),
                          (1, 2, 'Add more work experience entries. Include internships, freelance work, or relevant projects as experience')],
            'projects': [(0, 3, f'Add 2-3 projects relevant to {target_role} with technologies used, problem solved, and outcomes'),
                        (1, 2, 'Include at least 2-3 projects. For each project, mention: technologies, your role, and impact/results')]
        }
        
        for section, rules in section_suggestions.items():
            count = len(resume_data.get(section, []))
            for min_count, points, msg in rules:
                if count == min_count:
                    suggestions.append({'section': section, 'points': points, 'suggestion': msg})
                    break
        
        experience = resume_data.get('experience', [])
        if len(experience) > 1 and not re.search(r'\d+%|\d+\s*(years?|months?)|increased|decreased|improved|reduced', ' '.join(experience).lower()):
            suggestions.append({'section': 'experience', 'points': 1, 
                              'suggestion': 'Add quantified achievements (numbers, percentages, metrics) to make your experience more impactful'})
        
        projects = resume_data.get('projects', [])
        if len(projects) >= 2 and not re.search(r'\b(react|python|java|javascript|sql|node|django|flask|spring|aws|docker)\b', ' '.join(projects).lower()):
            suggestions.append({'section': 'projects', 'points': 1, 
                              'suggestion': 'Mention specific technologies and tools used in each project'})
        
        if not resume_data.get('certification'):
            cert_msgs = {
                'Data Analyst': 'Consider certifications like Google Data Analytics, Microsoft Power BI, or Tableau',
                'SDE(Frontend)': 'Consider certifications like AWS Certified Developer, React Certification, or Frontend Development courses',
                'SDE(Backend)': 'Consider certifications like AWS Certified Developer, Oracle Java Certification, or Backend Development courses',
                'SDE(Fullstack)': 'Consider certifications like AWS Certified Developer, Full Stack Development courses, or MERN Stack certifications'
            }
            suggestions.append({'section': 'certification', 'points': 2, 
                              'suggestion': cert_msgs.get(target_role, 'Add relevant professional certifications to strengthen your profile')})
        
        role_skills = {
            'Data Analyst': {'essential': ['python', 'sql', 'excel'], 'recommended': ['tableau', 'power bi', 'pandas', 'numpy']},
            'SDE(Frontend)': {'essential': ['html', 'css', 'javascript'], 'recommended': ['react', 'angular', 'vue', 'typescript']},
            'SDE(Backend)': {'essential': ['java', 'python', 'sql'], 'recommended': ['node.js', 'spring', 'django', 'rest api']},
            'SDE(Fullstack)': {'essential': ['javascript', 'html', 'css', 'sql'], 'recommended': ['react', 'node.js', 'mongodb', 'express']}
        }
        skills = [s.lower() for s in resume_data.get('skills', [])]
        skill_reqs = role_skills.get(target_role, {})
        missing_essential = [s for s in skill_reqs.get('essential', []) if not any(s in sk for sk in skills)]
        missing_recommended = [s for s in skill_reqs.get('recommended', [])[:5] if not any(s in sk for sk in skills)]
        
        if missing_essential:
            suggestions.append({'section': 'skills', 'points': 3, 
                              'suggestion': f'Add essential skills for {target_role}: {", ".join(missing_essential[:3])}'})
        elif missing_recommended:
            suggestions.append({'section': 'skills', 'points': 1, 
                              'suggestion': f'Consider adding recommended skills: {", ".join(missing_recommended[:3])}'})
        
        if scores.get('order_of_sections', 0) < 8:
            suggestions.append({'section': 'order_of_sections', 'points': 1, 
                              'suggestion': 'Follow standard resume order: Contact Info → Summary → Education → Experience → Projects → Certifications → Skills'})
        
        return suggestions
    
    def save_model(self, model_path='resume_model.joblib', vectorizer_path='vectorizer.joblib'):
        joblib.dump(self.classifier, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        print(f"Model saved to {model_path} and {vectorizer_path}")
    
    def load_model(self, model_path='resume_model.joblib', vectorizer_path='vectorizer.joblib'):
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            self.classifier = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            self.is_trained = True
            print("Model loaded successfully")
            return True
        return False


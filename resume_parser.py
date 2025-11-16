import re
import pdfplumber
from docx import Document
import PyPDF2

class ResumeParser:
    def __init__(self):
        self.section_keywords = {
            'name': [r'^[A-Z][a-z]+\s+[A-Z][a-z]+', r'name\s*:', r'full\s*name'],
            'email': [r'[\w\.-]+@[\w\.-]+\.\w+'],
            'phone': [r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', r'\+\d{1,3}[-.\s]?\d{10}'],
            'github': [r'github\.com/[\w-]+', r'github\s*:'],
            'linkedin': [r'linkedin\.com/in/[\w-]+', r'linkedin\s*:'],
            'professional_summary': ['summary', 'objective', 'profile', 'about'],
            'education': ['education', 'academic', 'qualification', 'degree'],
            'experience': ['experience', 'employment', 'work history', 'career'],
            'projects': ['project', 'portfolio'],
            'certification': ['certification', 'certificate', 'certified', 'license'],
            'skills': ['skill', 'technical skill', 'competence', 'expertise']
        }
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        except:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error reading PDF: {e}")
        return text
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def extract_text(self, file_path):
        """Extract text from resume file"""
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx') or file_path.endswith('.doc'):
            return self.extract_text_from_docx(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def extract_basic_details(self, text):
        """Extract basic contact information"""
        details = {
            'name': '',
            'email': '',
            'phone': '',
            'github': '',
            'linkedin': ''
        }
        
        # Extract email
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            details['email'] = email_match.group()
        
        # Extract phone
        phone_patterns = [
            r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{10}'
        ]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                details['phone'] = phone_match.group()
                break
        
        # Extract GitHub
        github_match = re.search(r'github\.com/([\w-]+)', text, re.IGNORECASE)
        if github_match:
            details['github'] = f"github.com/{github_match.group(1)}"
        
        # Extract LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/([\w-]+)', text, re.IGNORECASE)
        if linkedin_match:
            details['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        # Extract name (usually at the top)
        lines = text.split('\n')[:10]
        for line in lines:
            line = line.strip()
            if len(line) > 0 and len(line) < 50:
                # Check if it looks like a name
                name_pattern = r'^[A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?$'
                if re.match(name_pattern, line):
                    details['name'] = line
                    break
        
        return details
    
    def extract_section(self, text, section_name):
        """Extract a specific section from resume"""
        keywords = self.section_keywords.get(section_name, [])
        text_lower = text.lower()
        
        # Find section start
        section_start = -1
        for keyword in keywords:
            pattern = rf'\b{keyword}\b'
            match = re.search(pattern, text_lower)
            if match:
                section_start = match.start()
                break
        
        if section_start == -1:
            return ""
        
        # Find section end (next section or end of text)
        lines = text.split('\n')
        section_lines = []
        in_section = False
        
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this line starts a section
            if any(re.search(rf'\b{k}\b', line_lower) for k in keywords):
                in_section = True
                continue
            
            if in_section:
                # Check if we hit another major section
                other_sections = ['education', 'experience', 'project', 'certification', 'skill', 'summary', 'objective']
                if any(re.search(rf'\b{s}\b', line_lower) for s in other_sections if s not in keywords):
                    if line_lower and not line_lower.startswith(('•', '-', '*', '·')):
                        break
                
                if line.strip():
                    section_lines.append(line.strip())
        
        return '\n'.join(section_lines)
    
    def extract_professional_summary(self, text):
        """Extract professional summary/objective and convert to bullet points"""
        summary = self.extract_section(text, 'professional_summary')
        if not summary:
            # Try to get first few paragraphs
            lines = text.split('\n')
            summary_lines = []
            for line in lines[:15]:
                line = line.strip()
                if line and len(line) > 20:
                    summary_lines.append(line)
                    if len(summary_lines) >= 3:
                        break
            summary = ' '.join(summary_lines)
        
        # Convert to bullet points
        if summary:
            # Split by sentences
            sentences = re.split(r'[.!?]+', summary)
            bullet_points = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15:  # Only meaningful sentences
                    # Remove common prefixes
                    sentence = re.sub(r'^(i am|i\'m|i have|i\'ve|i can|i will|i would|i\'d)\s+', '', sentence, flags=re.IGNORECASE)
                    sentence = sentence.strip().capitalize()
                    if sentence:
                        bullet_points.append(sentence)
            
            # If we have bullet points, return them, otherwise return original
            if bullet_points:
                return bullet_points[:5]  # Max 5 bullet points
            else:
                # Split by commas or semicolons as fallback
                parts = re.split(r'[,;]', summary)
                bullet_points = [p.strip() for p in parts if len(p.strip()) > 15]
                return bullet_points[:5] if bullet_points else [summary[:200]]
        
        return []
    
    def extract_education(self, text):
        """Extract education entries with improved parsing"""
        education_text = self.extract_section(text, 'education')
        if not education_text:
            # Try alternative section names
            alt_keywords = ['academic', 'qualification', 'qualifications', 'educational background']
            for keyword in alt_keywords:
                alt_text = self.extract_section(text, keyword)
                if alt_text:
                    education_text = alt_text
                    break
        
        if not education_text:
            return []
        
        # Split by common patterns
        entries = []
        lines = education_text.split('\n')
        current_entry = []
        
        for line in lines:
            line = line.strip()
            # Remove bullet points
            line = re.sub(r'^[•\-\*·–]\s*', '', line)
            
            if not line:
                if current_entry:
                    entries.append(' '.join(current_entry))
                    current_entry = []
                continue
            
            # Check if this looks like a new entry (degree name, university, year)
            degree_pattern = r'\b(bachelor|master|phd|ph\.?d|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|b\.?tech|m\.?tech|degree|diploma|bachelor\'?s|master\'?s|doctorate)\b'
            year_pattern = r'\d{4}[-–]\d{4}|\d{4}\s*[-–]\s*(present|current)|\d{4}'
            
            if re.search(degree_pattern, line, re.IGNORECASE) or (re.search(year_pattern, line) and len(current_entry) > 0):
                if current_entry:
                    entries.append(' '.join(current_entry))
                current_entry = [line]
            elif re.search(r'\b(university|college|institute|school)\b', line, re.IGNORECASE):
                if current_entry:
                    current_entry.append(line)
                else:
                    current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append(' '.join(current_entry))
        
        # Clean entries
        cleaned_entries = []
        for entry in entries:
            entry = ' '.join(entry.split())  # Normalize whitespace
            if len(entry) > 10:  # Meaningful entry
                cleaned_entries.append(entry)
        
        return cleaned_entries[:5]  # Limit to 5 entries
    
    def extract_experience(self, text):
        """Extract work experience entries with improved parsing"""
        experience_text = self.extract_section(text, 'experience')
        if not experience_text:
            # Try alternative section names
            alt_keywords = ['employment', 'work history', 'career', 'professional experience', 'work experience']
            for keyword in alt_keywords:
                alt_text = self.extract_section(text, keyword)
                if alt_text:
                    experience_text = alt_text
                    break
        
        if not experience_text:
            return []
        
        entries = []
        lines = experience_text.split('\n')
        current_entry = []
        
        for line in lines:
            line = line.strip()
            # Remove bullet points
            line = re.sub(r'^[•\-\*·–]\s*', '', line)
            
            if not line:
                if current_entry:
                    entries.append(' '.join(current_entry))
                    current_entry = []
                continue
            
            # Check if this looks like a new entry (date pattern, company, or job title)
            date_pattern = r'\d{4}[-–]\d{4}|\d{4}\s*[-–]\s*(present|current)|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}'
            
            if re.search(date_pattern, line, re.IGNORECASE):
                if current_entry:
                    entries.append(' '.join(current_entry))
                current_entry = [line]
            elif re.search(r'\b(software|developer|engineer|analyst|manager|intern|assistant|specialist|consultant|lead|senior|junior)\b', line, re.IGNORECASE) and len(line) < 100:
                # Likely a job title
                if current_entry and len(current_entry) > 2:
                    entries.append(' '.join(current_entry))
                    current_entry = [line]
                else:
                    current_entry.append(line)
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append(' '.join(current_entry))
        
        # Clean entries
        cleaned_entries = []
        for entry in entries:
            entry = ' '.join(entry.split())  # Normalize whitespace
            if len(entry) > 15:  # Meaningful entry
                cleaned_entries.append(entry)
        
        return cleaned_entries[:5]  # Limit to 5 entries
    
    def extract_projects(self, text):
        """Extract project entries with improved parsing"""
        projects_text = self.extract_section(text, 'projects')
        if not projects_text:
            # Try alternative section names
            alt_keywords = ['project', 'portfolio', 'personal projects', 'academic projects']
            for keyword in alt_keywords:
                if keyword != 'project':  # Already tried
                    alt_text = self.extract_section(text, keyword)
                    if alt_text:
                        projects_text = alt_text
                        break
        
        if not projects_text:
            return []
        
        entries = []
        lines = projects_text.split('\n')
        current_entry = []
        
        for line in lines:
            line = line.strip()
            # Remove bullet points but keep track
            original_line = line
            line = re.sub(r'^[•\-\*·–]\s*', '', line)
            
            if not line:
                if current_entry:
                    entries.append(' '.join(current_entry))
                    current_entry = []
                continue
            
            # Check if this looks like a project title (short line, no bullet, might have tech stack)
            is_likely_title = (len(line) < 100 and 
                             not original_line.startswith(('•', '-', '*', '·')) and
                             (re.search(r'\b(project|application|system|website|app)\b', line, re.IGNORECASE) or
                              re.search(r'[A-Z][a-z]+\s+[A-Z]', line)))  # Title case
            
            if is_likely_title and current_entry:
                entries.append(' '.join(current_entry))
                current_entry = [line]
            else:
                current_entry.append(line)
        
        if current_entry:
            entries.append(' '.join(current_entry))
        
        # Clean entries
        cleaned_entries = []
        for entry in entries:
            entry = ' '.join(entry.split())  # Normalize whitespace
            if len(entry) > 10:  # Meaningful entry
                cleaned_entries.append(entry)
        
        return cleaned_entries[:5]  # Limit to 5 entries
    
    def extract_certifications(self, text):
        """Extract certification entries with improved detection"""
        # Try multiple section names
        cert_keywords = ['certification', 'certificate', 'certified', 'license', 'licenses', 
                        'certifications', 'credentials', 'professional certification', 
                        'training & certification', 'training and certification']
        
        cert_text = ""
        text_lower = text.lower()
        
        # Find certification section
        for keyword in cert_keywords:
            pattern = rf'\b{re.escape(keyword)}\b'
            match = re.search(pattern, text_lower)
            if match:
                # Extract section content
                lines = text.split('\n')
                in_section = False
                section_lines = []
                
                for line in lines:
                    line_lower = line.lower().strip()
                    if re.search(pattern, line_lower):
                        in_section = True
                        continue
                    
                    if in_section:
                        # Stop at next major section
                        major_sections = ['education', 'experience', 'project', 'skill', 'summary', 
                                         'objective', 'achievement', 'award', 'publication']
                        if any(re.search(rf'\b{s}\b', line_lower) for s in major_sections):
                            if line.strip() and not line.strip().startswith(('•', '-', '*', '·', '–')):
                                break
                        
                        if line.strip():
                            section_lines.append(line.strip())
                
                cert_text = '\n'.join(section_lines)
                break
        
        # Also search entire text for certification patterns
        if not cert_text:
            # Look for common certification patterns
            cert_patterns = [
                r'(?:certified|certificate|license)\s+[A-Z][^.!?]*(?:\d{4})?',
                r'[A-Z][^.!?]*(?:certified|certificate|license)[^.!?]*(?:\d{4})?',
                r'(?:AWS|Azure|Google Cloud|Oracle|Microsoft|Cisco|CompTIA|PMP|CPA|CFA)[^.!?]*(?:certified|certificate)[^.!?]*',
            ]
            
            found_certs = []
            for pattern in cert_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    cert = match.group().strip()
                    if len(cert) > 10 and len(cert) < 200:
                        found_certs.append(cert)
            
            if found_certs:
                return list(set(found_certs))[:10]
        
        # Parse certification text
        entries = []
        if cert_text:
            lines = cert_text.split('\n')
            current_entry = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_entry:
                        entries.append(' '.join(current_entry))
                        current_entry = []
                    continue
                
                # Remove bullet points
                line = re.sub(r'^[•\-\*·–]\s*', '', line)
                
                # Check if it looks like a certification
                if re.search(r'\b(certified|certificate|license|certification|credential|aws|azure|google|oracle|microsoft|cisco|comptia|pmp|cpa|cfa)\b', line, re.IGNORECASE):
                    if current_entry:
                        entries.append(' '.join(current_entry))
                    current_entry = [line]
                elif line and len(line) > 5:
                    # Could be continuation or date
                    if re.search(r'\d{4}', line) or len(line) < 50:
                        if current_entry:
                            current_entry.append(line)
                        else:
                            # Might be a standalone cert
                            entries.append(line)
                    else:
                        current_entry.append(line)
            
            if current_entry:
                entries.append(' '.join(current_entry))
        
        # Clean and deduplicate
        cleaned_entries = []
        seen = set()
        for entry in entries:
            entry_clean = ' '.join(entry.split())  # Normalize whitespace
            entry_lower = entry_clean.lower()
            if entry_lower not in seen and len(entry_clean) > 5:
                seen.add(entry_lower)
                cleaned_entries.append(entry_clean)
        
        return cleaned_entries[:10]  # Limit to 10 entries
    
    def extract_skills(self, text):
        """Extract skills with improved detection"""
        skills_text = self.extract_section(text, 'skills')
        if not skills_text:
            # Try alternative section names
            alt_keywords = ['technical skill', 'technical skills', 'competence', 'expertise', 'technologies', 'tools']
            for keyword in alt_keywords:
                alt_text = self.extract_section(text, keyword)
                if alt_text:
                    skills_text = alt_text
                    break
        
        skills = []
        
        if skills_text:
            # Extract skills from skills section
            lines = skills_text.split('\n')
            
            for line in lines:
                line = line.strip()
                # Remove bullet points
                line = re.sub(r'^[•\-\*·–]\s*', '', line)
                
                # Split by common delimiters
                items = re.split(r'[,;|•\-\*–]', line)
                for item in items:
                    item = item.strip()
                    if item and len(item) > 1 and len(item) < 50:
                        # Remove common prefixes/suffixes
                        item = re.sub(r'^(proficient in|experienced with|knowledge of|familiar with)\s+', '', item, flags=re.IGNORECASE)
                        item = item.strip()
                        if item:
                            skills.append(item)
        
        # Also search entire text for common technical skills
        tech_skills_pattern = r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js|express|django|flask|spring|sql|mysql|postgresql|mongodb|redis|html|css|sass|less|bootstrap|tailwind|aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab|agile|scrum|jira|tableau|power.?bi|excel|pandas|numpy|tensorflow|pytorch|machine.?learning|data.?science|rest.?api|graphql|microservices|ci.?cd)\b'
        found_skills = re.findall(tech_skills_pattern, text, re.IGNORECASE)
        skills.extend([s.lower() for s in found_skills])
        
        # Clean and deduplicate
        cleaned_skills = []
        seen = set()
        for skill in skills:
            skill_lower = skill.lower().strip()
            if skill_lower and skill_lower not in seen and len(skill_lower) > 1:
                seen.add(skill_lower)
                cleaned_skills.append(skill.strip())
        
        return cleaned_skills[:25]  # Limit to 25 skills
    
    def parse_resume(self, file_path):
        """Parse resume and extract all sections"""
        text = self.extract_text(file_path)
        
        resume_data = {
            'basic_details': self.extract_basic_details(text),
            'professional_summary': self.extract_professional_summary(text),
            'education': self.extract_education(text),
            'experience': self.extract_experience(text),
            'projects': self.extract_projects(text),
            'certification': self.extract_certifications(text),
            'skills': self.extract_skills(text),
            'raw_text': text
        }
        
        return resume_data


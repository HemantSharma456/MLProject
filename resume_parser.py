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
        try:
            with pdfplumber.open(file_path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except:
            try:
                with open(file_path, 'rb') as file:
                    return "\n".join(page.extract_text() or "" for page in PyPDF2.PdfReader(file).pages)
            except:
                return ""
    
    def extract_text_from_docx(self, file_path):
        return "\n".join(p.text for p in Document(file_path).paragraphs)
    
    def extract_text(self, file_path):
        if file_path.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith(('.docx', '.doc')):
            return self.extract_text_from_docx(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def extract_basic_details(self, text):
        details = {'name': '', 'email': '', 'phone': '', 'github': '', 'linkedin': ''}
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if email_match:
            details['email'] = email_match.group()
        
        for pattern in [r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', r'\+\d{1,3}[-.\s]?\d{10}']:
            match = re.search(pattern, text)
            if match:
                details['phone'] = match.group()
                break
        
        github_match = re.search(r'github\.com/([\w-]+)', text, re.IGNORECASE)
        if github_match:
            details['github'] = f"github.com/{github_match.group(1)}"
        
        linkedin_match = re.search(r'linkedin\.com/in/([\w-]+)', text, re.IGNORECASE)
        if linkedin_match:
            details['linkedin'] = f"linkedin.com/in/{linkedin_match.group(1)}"
        
        for line in text.split('\n')[:10]:
            line = line.strip()
            if 0 < len(line) < 50 and re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)?$', line):
                details['name'] = line
                break
        
        return details
    
    def extract_section(self, text, section_name):
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
        summary = self.extract_section(text, 'professional_summary')
        if not summary:
            lines = [l.strip() for l in text.split('\n')[:15] if l.strip() and len(l.strip()) > 20]
            summary = ' '.join(lines[:3])
        
        if summary:
            sentences = [s.strip().capitalize() for s in re.split(r'[.!?]+', summary) 
                        if len(s.strip()) > 15]
            sentences = [re.sub(r'^(i am|i\'m|i have|i\'ve|i can|i will|i would|i\'d)\s+', '', s, flags=re.IGNORECASE).strip() 
                        for s in sentences if s]
            if sentences:
                return sentences[:5]
            parts = [p.strip() for p in re.split(r'[,;]', summary) if len(p.strip()) > 15]
            return parts[:5] if parts else [summary[:200]]
        return []
    
    def _extract_list_items(self, text, section_name, alt_keywords, new_entry_pattern, min_length=10, max_items=5):
        section_text = self.extract_section(text, section_name)
        if not section_text:
            for kw in alt_keywords:
                alt_text = self.extract_section(text, kw)
                if alt_text:
                    section_text = alt_text
                    break
        if not section_text:
            return []
        
        entries, current = [], []
        for line in section_text.split('\n'):
            line = re.sub(r'^[•\-\*·–]\s*', '', line.strip())
            if not line:
                if current:
                    entries.append(' '.join(current))
                    current = []
                continue
            
            if re.search(new_entry_pattern, line, re.IGNORECASE) and current:
                entries.append(' '.join(current))
                current = [line]
            else:
                current.append(line)
        
        if current:
            entries.append(' '.join(current))
        
        return [' '.join(e.split()) for e in entries if len(' '.join(e.split())) > min_length][:max_items]
    
    def extract_education(self, text):
        pattern = r'\b(bachelor|master|phd|ph\.?d|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?|b\.?tech|m\.?tech|degree|diploma|university|college|institute|school|\d{4})\b'
        return self._extract_list_items(text, 'education', ['academic', 'qualification', 'qualifications'], pattern, 10, 5)
    
    def extract_experience(self, text):
        pattern = r'\d{4}[-–]\d{4}|\d{4}\s*[-–]\s*(present|current)|\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|\b(software|developer|engineer|analyst|manager|intern|assistant|specialist|consultant|lead|senior|junior)\b'
        return self._extract_list_items(text, 'experience', ['employment', 'work history', 'career', 'professional experience'], pattern, 15, 5)
    
    def extract_projects(self, text):
        projects_text = self.extract_section(text, 'projects')
        if not projects_text:
            for kw in ['portfolio', 'personal projects', 'academic projects']:
                alt_text = self.extract_section(text, kw)
                if alt_text:
                    projects_text = alt_text
                    break
        if not projects_text:
            return []
        
        entries, current = [], []
        for line in projects_text.split('\n'):
            original = line
            line = re.sub(r'^[•\-\*·–]\s*', '', line.strip())
            if not line:
                if current:
                    entries.append(' '.join(current))
                    current = []
                continue
            
            is_title = (len(line) < 100 and not original.strip().startswith(('•', '-', '*', '·')) and
                       (re.search(r'\b(project|application|system|website|app)\b', line, re.IGNORECASE) or
                        re.search(r'[A-Z][a-z]+\s+[A-Z]', line)))
            
            if is_title and current:
                entries.append(' '.join(current))
                current = [line]
            else:
                current.append(line)
        
        if current:
            entries.append(' '.join(current))
        
        return [' '.join(e.split()) for e in entries if len(' '.join(e.split())) > 10][:5]
    
    def extract_certifications(self, text):
        cert_text = self.extract_section(text, 'certification')
        if not cert_text:
            for kw in ['certificate', 'certified', 'license', 'licenses', 'credentials']:
                cert_text = self.extract_section(text, kw)
                if cert_text:
                    break
        
        if not cert_text:
            patterns = [
                r'(?:certified|certificate|license)\s+[A-Z][^.!?]*(?:\d{4})?',
                r'[A-Z][^.!?]*(?:certified|certificate|license)[^.!?]*(?:\d{4})?',
                r'(?:AWS|Azure|Google Cloud|Oracle|Microsoft|Cisco|CompTIA|PMP|CPA|CFA)[^.!?]*(?:certified|certificate)[^.!?]*',
            ]
            found = [m.group().strip() for p in patterns for m in re.finditer(p, text, re.IGNORECASE)
                    if 10 < len(m.group().strip()) < 200]
            return list(set(found))[:10] if found else []
        
        entries, current = [], []
        for line in cert_text.split('\n'):
            line = re.sub(r'^[•\-\*·–]\s*', '', line.strip())
            if not line:
                if current:
                    entries.append(' '.join(current))
                    current = []
                continue
            
            if re.search(r'\b(certified|certificate|license|certification|credential|aws|azure|google|oracle|microsoft|cisco|comptia|pmp|cpa|cfa)\b', line, re.IGNORECASE):
                if current:
                    entries.append(' '.join(current))
                current = [line]
            elif line and len(line) > 5:
                if re.search(r'\d{4}', line) or len(line) < 50:
                    if current:
                        current.append(line)
                    else:
                        entries.append(line)
                else:
                    current.append(line)
        
        if current:
            entries.append(' '.join(current))
        
        seen = set()
        return [e for e in [' '.join(entry.split()) for entry in entries] 
                if e.lower() not in seen and not seen.add(e.lower()) and len(e) > 5][:10]
    
    def extract_skills(self, text):
        skills_text = self.extract_section(text, 'skills')
        if not skills_text:
            for kw in ['technical skill', 'technical skills', 'competence', 'expertise', 'technologies', 'tools']:
                skills_text = self.extract_section(text, kw)
                if skills_text:
                    break
        
        skills = []
        if skills_text:
            for line in skills_text.split('\n'):
                line = re.sub(r'^[•\-\*·–]\s*', '', line.strip())
                for item in re.split(r'[,;|•\-\*–]', line):
                    item = re.sub(r'^(proficient in|experienced with|knowledge of|familiar with)\s+', '', item.strip(), flags=re.IGNORECASE)
                    if 1 < len(item) < 50:
                        skills.append(item)
        
        tech_pattern = r'\b(python|java|javascript|typescript|react|angular|vue|node\.?js|express|django|flask|spring|sql|mysql|postgresql|mongodb|redis|html|css|sass|less|bootstrap|tailwind|aws|azure|gcp|docker|kubernetes|jenkins|git|github|gitlab|agile|scrum|jira|tableau|power.?bi|excel|pandas|numpy|tensorflow|pytorch|machine.?learning|data.?science|rest.?api|graphql|microservices|ci.?cd)\b'
        skills.extend(re.findall(tech_pattern, text, re.IGNORECASE))
        
        seen = set()
        return [s.strip() for s in skills if s.lower().strip() and s.lower().strip() not in seen 
                and not seen.add(s.lower().strip()) and len(s.strip()) > 1][:25]
    
    def parse_resume(self, file_path):
        text = self.extract_text(file_path)
        return {
            'basic_details': self.extract_basic_details(text),
            'professional_summary': self.extract_professional_summary(text),
            'education': self.extract_education(text),
            'experience': self.extract_experience(text),
            'projects': self.extract_projects(text),
            'certification': self.extract_certifications(text),
            'skills': self.extract_skills(text),
            'raw_text': text
        }


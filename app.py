from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from ml_model import ResumeEvaluator
from resume_parser import ResumeParser
import time

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize models
evaluator = ResumeEvaluator()
parser = ResumeParser()

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Try to load existing model, otherwise train new one
if not evaluator.load_model():
    print("Training new model...")
    evaluator.train('Dataset.csv')
    evaluator.save_model()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/roles', methods=['GET'])
def get_roles():
    """Get available roles"""
    roles = ['Data Analyst', 'SDE(Frontend)', 'SDE(Backend)', 'SDE(Fullstack)']
    return jsonify({'roles': roles})

@app.route('/api/upload', methods=['POST'])
def upload_resume():
    """Upload and analyze resume"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        role = request.form.get('role', 'Data Analyst')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PDF, DOCX, DOC, TXT'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Parse resume
        resume_data = parser.parse_resume(file_path)
        
        # Predict role match
        predicted_role, role_probs = evaluator.predict_role(resume_data['raw_text'])
        
        # Calculate scores
        scores, max_scores = evaluator.calculate_section_scores(resume_data, role)
        
        # Calculate total score
        total_score = sum(scores.values())
        max_total = sum(max_scores.values())
        score_percentage = (total_score / max_total) * 100 if max_total > 0 else 0
        
        # Generate suggestions
        suggestions = evaluator.generate_suggestions(resume_data, role, scores)
        
        # Group suggestions by section
        suggestions_by_section = {}
        for suggestion in suggestions:
            section = suggestion['section']
            if section not in suggestions_by_section:
                suggestions_by_section[section] = []
            suggestions_by_section[section].append(suggestion)
        
        # Prepare response
        response = {
            'filename': filename,
            'role': role,
            'predicted_role': predicted_role,
            'role_match_probability': role_probs.get(role, 0),
            'total_score': round(total_score, 2),
            'max_score': max_total,
            'score_percentage': round(score_percentage, 2),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'max_scores': max_scores,
            'resume_data': {
                'basic_details': resume_data['basic_details'],
                'professional_summary': resume_data['professional_summary'],
                'education': resume_data['education'],
                'experience': resume_data['experience'],
                'projects': resume_data['projects'],
                'certification': resume_data['certification'],
                'skills': resume_data['skills']
            },
            'suggestions': suggestions_by_section
        }
        
        # Clean up file
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_trained': evaluator.is_trained})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


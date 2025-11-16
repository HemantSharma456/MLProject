from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from werkzeug.utils import secure_filename
from ml_model import ResumeEvaluator
from resume_parser import ResumeParser

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'doc', 'txt'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

evaluator = ResumeEvaluator()
parser = ResumeParser()
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
    return jsonify({'roles': ['Data Analyst', 'SDE(Frontend)', 'SDE(Backend)', 'SDE(Fullstack)']})

@app.route('/api/upload', methods=['POST'])
def upload_resume():
    try:
        if 'file' not in request.files or not request.files['file'].filename:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        role = request.form.get('role', 'Data Analyst')
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PDF, DOCX, DOC, TXT'}), 400
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
        file.save(file_path)
        
        resume_data = parser.parse_resume(file_path)
        predicted_role, role_probs = evaluator.predict_role(resume_data['raw_text'])
        scores, max_scores = evaluator.calculate_section_scores(resume_data, role)
        
        total_score = sum(scores.values())
        max_total = sum(max_scores.values())
        
        suggestions = evaluator.generate_suggestions(resume_data, role, scores)
        suggestions_by_section = {}
        for s in suggestions:
            section = s['section']
            suggestions_by_section.setdefault(section, []).append(s)
        
        try:
            os.remove(file_path)
        except:
            pass
        
        return jsonify({
            'filename': filename,
            'role': role,
            'predicted_role': predicted_role,
            'role_match_probability': role_probs.get(role, 0),
            'total_score': round(total_score, 2),
            'max_score': max_total,
            'score_percentage': round((total_score / max_total * 100) if max_total > 0 else 0, 2),
            'scores': {k: round(v, 2) for k, v in scores.items()},
            'max_scores': max_scores,
            'resume_data': {k: resume_data[k] for k in ['basic_details', 'professional_summary', 
                                                         'education', 'experience', 'projects', 
                                                         'certification', 'skills']},
            'suggestions': suggestions_by_section
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_trained': evaluator.is_trained})

if __name__ == '__main__':
    app.run(debug=True, port=5000)


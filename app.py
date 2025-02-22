from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# Ensure the 'uploads' directory exists to store audio files
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html', pages=[['Home', 'index'], ['Scan', 'scan']])

@app.route('/scan')
def scan():
    return render_template('scan.html', pages=[['Home', 'index'], ['Scan', 'scan']])

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file found"}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
    audio_file.save(file_path)

    # TODO: Process audio for lung cancer classification
    return jsonify({"message": "Audio received successfully!", "file_path": file_path})

if __name__ == '__main__':
    app.run(debug=True)

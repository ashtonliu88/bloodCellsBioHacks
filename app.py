import os
import openai
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv


load_dotenv()
app = Flask(__name__)

# Ensure the uploads directory exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set OpenAI API Key (Use an environment variable in production)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

@app.route('/')
def index():
    return render_template('index.html', pages=[['Home', 'index'], ['Scan', 'scan']])

@app.route('/scan')
def scan():
    return render_template('scan.html', pages=[['Home', 'index'], ['Scan', 'scan']])

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file found"}), 400
        
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], "recorded_audio.mp3")
        audio_file.save(file_path)
        print(f"✅ Audio file saved at: {file_path}")  # Debug print

        # Check if OpenAI API Key is set
        if not OPENAI_API_KEY:
            print("🚨 OpenAI API Key is missing!")
            return jsonify({"error": "OpenAI API Key is missing!"}), 500

        # ✅ Send the file to OpenAI Whisper using the new API method
        whisper_response = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=open(file_path, "rb")
        )
        transcribed_text = whisper_response.text
        print(f"🎙️ Transcribed text: {transcribed_text}")  # Debug print

        # ✅ Send transcribed text to GPT-4 using the NEW API
        gpt_response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a medical assistant specializing in lung cancer detection. You should try to respond to the patient in relatively short responses, rich with detail. (Maximum of 3 sentences)"},
                {"role": "user", "content": transcribed_text}
            ]
        )
        ai_response = gpt_response.choices[0].message.content
        print(f"🤖 AI Response: {ai_response}")  # Debug print

        return jsonify({
            "message": "Audio received successfully!",
            "ai_response": ai_response
        })

    except Exception as e:
        print(f"🚨 Error: {str(e)}")  # Debug print
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)

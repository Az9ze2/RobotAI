"""
Simple Flask API for Coqui TTS (KhanomTan Thai model)
"""
from flask import Flask, request, jsonify, send_file
from TTS.api import TTS
import time
import os

app = Flask(__name__)

# Initialize TTS with actual KhanomTan Thai model
print("Loading KhanomTan Thai TTS model from HuggingFace...")
try:
    # Use the actual KhanomTan model from wannaphong
    tts = TTS(model_name="tts_models/th/cv/vits", progress_bar=False)
    print("✅ KhanomTan model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load KhanomTan: {e}")
    print("Trying to download from HuggingFace directly...")
    try:
        # Try loading from HuggingFace hub
        from TTS.utils.manage import ModelManager
        manager = ModelManager()
        # Download and use the model
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)
        print("✅ Loaded multilingual model as fallback")
    except Exception as e2:
        print(f"❌ All attempts failed: {e2}")
        raise

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "model": "KhanomTan (Coqui TTS)"})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """
    Synthesize Thai text to speech
    
    Request JSON:
    {
        "text": "สวัสดีครับ",
        "output_path": "output.wav"  # optional
    }
    """
    try:
        data = request.json
        text = data.get('text', '')
        output_path = data.get('output_path', f'/tmp/tts_{int(time.time())}.wav')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Measure synthesis time
        start_time = time.time()
        # Use language parameter only
        tts.tts_to_file(
            text=text,
            file_path=output_path,
            language='th-th'  # Specify Thai language
        )
        synthesis_time = time.time() - start_time
        
        # Get file size
        file_size = os.path.getsize(output_path)
        
        return jsonify({
            "success": True,
            "output_path": output_path,
            "synthesis_time": synthesis_time,
            "file_size": file_size,
            "text": text
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/download/<path:filename>', methods=['GET'])
def download(filename):
    """Download generated audio file"""
    return send_file(filename, mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

# app.py
import ffmpeg
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import whisper
import tempfile
import os
import ollama

app = Flask(__name__)
socketio = SocketIO(app)

# Load a smaller Whisper model to improve speed
model = whisper.load_model("tiny")  # Try "tiny" or "small" for faster transcription

# Preload the Ollama model to avoid download delays
model_name = 'llama3.1'
try:
    ollama.pull(model_name)
except Exception as e:
    print(f"Error preloading model '{model_name}': {e}")

# Global variable to store the transcript
transcript = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speaker')
def speaker():
    return render_template('speaker.html')

@app.route('/listener')
def listener():
    return render_template('listener.html')

@socketio.on('connect')
def handle_connect():
    emit('load_transcript', {'transcript': transcript})

@socketio.on('speech_data')
def handle_speech(data):
    audio_data = data.get('audio', None)
    if audio_data:
        try:
            # Write WebM audio data to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_audio_file:
                temp_audio_file.write(audio_data)
                temp_audio_path = temp_audio_file.name

            # Convert WebM to WAV for Whisper transcription
            temp_wav_path = tempfile.mktemp(suffix=".wav")
            ffmpeg.input(temp_audio_path).output(temp_wav_path).run(overwrite_output=True)

            # Transcribe the audio with Whisper (smaller model for faster processing)
            result = model.transcribe(temp_wav_path)
            text = result["text"]

            # Append new text to transcript
            transcript.append(text)
            emit('new_speech', {'text': text}, broadcast=True)

            # Clean up temporary files
            os.remove(temp_audio_path)
            os.remove(temp_wav_path)
        except Exception as e:
            emit('new_speech', {'text': f"Error: {str(e)}"}, broadcast=True)

@socketio.on('question')
def handle_question(data):
    question = data.get('text', '')
    # Summarize context to the last 1000 characters to speed up response generation
    context = ' '.join(transcript)[-1000:]

    # Construct the prompt to constrain answers within context and limit response length
    prompt = (
        f"You are a helpful assistant answering questions based strictly on the following transcript. "
        f"if the user greets you then politely greet them back. if the user says bye then respond appropriately as well"
        f"If the question is outside this content, reply with 'I'm sorry, but I can only answer questions related to the provided transcript.' "
        f"Answer concisely, limiting your response to 200 characters. Here is the transcript: {context}"
    )

    try:
        # Generate response using Ollama, if possible limit max_tokens (approx. 200 characters)
        response = ollama.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': question}
            ],
        )

        # Extract the generated response
        answer = response['message']['content']
        emit('answer', {'text': answer})
    except Exception as e:
        emit('answer', {'text': f'Error generating response: {str(e)}'})


if __name__ == '__main__':
    socketio.run(app, debug=True)

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import numpy as np
import librosa
import tempfile
import joblib
import os
from tensorflow.keras.models import load_model

from llm.utils.audio_processing import preprocess_audio
from llm.utils.whisper_transcriber import transcribe_audio_whisper
from llm.utils.stutter_detection import segment_audio_by_words, detect_stuttering_per_segment
from llm.utils.phoneme_analysis import find_most_stuttered_phoneme
from llm.utils.ollama_generator import generate_practice_passage

@csrf_exempt
def stutter_analysis_view(request):
    if request.method == "POST":
        audio_file = request.FILES.get('file')
        if not audio_file:
            return JsonResponse({'error': 'No file uploaded'}, status=400)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
            for chunk in audio_file.chunks():
                temp.write(chunk)
            temp_path = temp.name

        try:
            y, sr = librosa.load(temp_path, sr=None)
            y_proc, sr_proc = preprocess_audio(y, sr)

            text, word_timestamps = transcribe_audio_whisper(temp_path)
            segments = segment_audio_by_words(y_proc, sr_proc, word_timestamps)

            scaler = joblib.load('C:\college\capstone\StutterGo\StutterGo\llm\models\standard_scaler.save')
            model = load_model('C:\college\capstone\StutterGo\StutterGo\llm\models\Stutter_Detection_Model.h5', compile=False)
            stuttered = detect_stuttering_per_segment(segments, sr_proc, scaler, model)

            phoneme, count = find_most_stuttered_phoneme(stuttered)
            passage = generate_practice_passage(phoneme) if phoneme else "No phoneme to generate practice passage."

            return JsonResponse({
                "transcript": text,
                "num_stutters": len(stuttered),
                "most_stuttered_phoneme": phoneme,
                "practice_passage": passage
            })
        finally:
            os.remove(temp_path)

    return JsonResponse({'message': 'Send a POST request with audio file'}, status=405)

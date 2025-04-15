import whisper

def transcribe_audio_whisper(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, word_timestamps=True)
    word_timestamps = []
    for segment in result["segments"]:
        word_timestamps.extend(segment.get("words", []))
    return result["text"], word_timestamps

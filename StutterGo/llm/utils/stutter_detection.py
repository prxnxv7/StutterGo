from llm.utils.feature_extraction import extract_features_from_segment

def segment_audio_by_words(y, sr, word_timestamps):
    segments = []
    for word_info in word_timestamps:
        start_sample = int(word_info["start"] * sr)
        end_sample = int(word_info["end"] * sr)
        segment = y[start_sample:end_sample]
        segments.append((segment, word_info["start"], word_info["end"], word_info["word"]))
    return segments

def detect_stuttering_per_segment(segments, sr, scaler, model, threshold=0.98):
    stuttered = []
    for segment, start, end, word in segments:
        features = extract_features_from_segment(segment, sr)
        if features is None or features.shape[0] != 43:
            continue
        scaled = scaler.transform(features.reshape(1, -1))
        prediction = model.predict(scaled, verbose=0)[0][0]
        if prediction >= threshold:
            stuttered.append((start, end, word, prediction))
    return stuttered

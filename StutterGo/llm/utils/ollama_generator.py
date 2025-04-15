import ollama

def generate_practice_passage(phoneme):
    try:
        phoneme_sound = phoneme.lower()
        replacements = {
            'CH': 'ch', 'DH': 'th (as in this)', 'JH': 'j', 'NG': 'ng',
            'SH': 'sh', 'TH': 'th (as in think)', 'ZH': 'zh (as in vision)'
        }
        phoneme_sound = replacements.get(phoneme, phoneme_sound)
        
        prompt = (
            f"Create a short, creative story (3-5 sentences) for speech therapy practice, "
            f"emphasizing words starting with or containing the consonant sound /{phoneme_sound}/. "
            f"Use at least 5 distinct words with /{phoneme_sound}/, highlight these with asterisks (*word*), "
            f"and keep it engaging for practice."
        )

        client = ollama.Client(host='http://127.0.0.1:11434')
        response = client.generate(model='llama3', prompt=prompt)
        passage = response['response'].strip()
        return passage
    except Exception as e:
        return f"Error generating passage: {str(e)}"

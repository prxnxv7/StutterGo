import re
from g2p_en import G2p
from collections import Counter

def get_phonemes(word, g2p):
    clean = re.sub(r'[.,!?]', '', word).strip().lower()
    if not clean:
        return []
    if clean == "20th": clean = "twentieth"
    elif clean == "20": clean = "twenty"
    try:
        phonemes = g2p(clean)
        phonemes = [re.sub(r'[0-2]', '', p) for p in phonemes]
        consonants = {'B','CH','D','DH','F','G','HH','JH','K','L','M','N','NG','P','R','S','SH','T','TH','V','W','Y','Z','ZH'}
        return [p for p in phonemes if p and p in consonants]
    except:
        return []

def find_most_stuttered_phoneme(stuttered_segments):
    g2p = G2p()
    word_counts = Counter([word for _, _, word, _ in stuttered_segments])
    all_phonemes = []
    for word, count in word_counts.items():
        phonemes = get_phonemes(word, g2p)
        all_phonemes.extend(p for p in phonemes for _ in range(count))
    phoneme_counts = Counter(all_phonemes)
    return phoneme_counts.most_common(1)[0] if phoneme_counts else (None, 0)

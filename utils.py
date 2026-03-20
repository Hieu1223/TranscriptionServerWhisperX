from sudachipy import tokenizer
from sudachipy import dictionary
import re
import yt_dlp

tokenizer_obj = dictionary.Dictionary(dict='full').create()
mode = tokenizer.Tokenizer.SplitMode.C

def split_japanese_sentences(text):
    sentences = re.split(r'(?<=[。！？?])', text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize(text):
    
    tokens = tokenizer_obj.tokenize(text, mode)
    words = []
    noun_buffer = ""

    i = 0
    while i < len(tokens):

        m = tokens[i]
        pos = m.part_of_speech()[0]
        word = m.surface()

        # 1️⃣ merge consecutive nouns
        if pos == "名詞":
            noun_buffer += word
            i += 1
            continue

        if noun_buffer:
            words.append(noun_buffer)
            noun_buffer = ""

        # 2️⃣ generic grammar merge after verbs
        if pos == "動詞":
            merged = word
            j = i + 1

            while j < len(tokens):
                next_m = tokens[j]
                next_pos = next_m.part_of_speech()[0]
                next_word = next_m.surface()

                # attach grammar pieces
                if next_pos in ["助詞", "助動詞", "形容詞"]:
                    merged += next_word
                    j += 1
                    continue

                break

            words.append(merged)
            i = j
            continue

        # normal token
        words.append(word)
        i += 1


    if noun_buffer:
        words.append(noun_buffer)
    return words    

def gather_words(whisper_result):
    words = []

    for seg in whisper_result["segments"]:
        for w in seg["words"]:
            if "start" in w:
                words.append(w)

    return words

def remove_punctuation(tokens):
    return [t for t in tokens if not re.fullmatch(r"[。、！？?,!]", t)]

def align_tokens(tokens, words):

    durations = []
    word_idx = 0

    for token in tokens:

        start = None
        end = None
        collected = ""

        while word_idx < len(words):

            w = words[word_idx]

            word_text = w["word"].replace(" ", "").strip()

            if start is None:
                start = w["start"]

            collected += word_text
            end = w["end"]

            word_idx += 1

            if collected == token:
                durations.append({
                    "word": token,
                    "start": start,
                    "end": end
                })
                break

            if not token.startswith(collected):
                break

    return durations

def gather_tokens(whisper_result):
    all_tokens = []

    for seg in whisper_result["segments"]:
        sentences = split_japanese_sentences(seg["text"])

        for sentence in sentences:
            all_tokens += tokenize(sentence)

    return all_tokens

def split_sentences(whisper_result):

    tokens = gather_tokens(whisper_result)
    tokens = remove_punctuation(tokens)
    words = gather_words(whisper_result)

    return align_tokens(tokens, words)


def download_from_url(path, url):
    ydl_opts = {
        'format': 'bestaudio/best',
        # --- ADD THIS LINE ---
        'noplaylist': True, 
        # ---------------------
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        # Using path directly as the template
        'outtmpl': f'{path}.%(ext)s'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def get_video_id(url):
    ydl_opts = {
        'quiet': True,  # Suppress console output
        'no_warnings': True,
        'force_generic_extractor': True, # Ensures extraction works for various URLs
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        # Extract info without downloading
        info_dict = ydl.extract_info(url, download=False)

        # The 'id' field contains the video ID
        video_id = info_dict.get("id", None)
        return video_id
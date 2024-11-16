import difflib
import json

def flatten(xss):
    return [x for xs in xss for x in xs]

def tokenize(text):
    # Simple tokenizer that splits on whitespace
    return text.strip().split()

def align_texts(original_tokens, corrected_tokens):
    original_words = [token['word'] for token in original_tokens]
    corrected_words = corrected_tokens

    # SequenceMatcher will find the best alignment between the two sequences
    matcher = difflib.SequenceMatcher(None, original_words, corrected_words)
    aligned_tokens = []

    opcodes = matcher.get_opcodes()
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            # Words are the same, copy timestamps
            for idx in range(i1, i2):
                aligned_tokens.append(original_tokens[idx])
        elif tag == 'replace':
            # Words are replaced, assign timestamps from original
            for idx in range(j1, j2):
                # Map to the original token's timestamp if possible
                orig_idx = min(i1 + idx - j1, len(original_tokens) - 1)
                aligned_tokens.append({
                    'word': corrected_words[idx],
                    'start': original_tokens[orig_idx]['start'],
                    'end': original_tokens[orig_idx]['end']
                })
        elif tag == 'insert':
            # New words inserted, estimate timestamps
            prev_end = original_tokens[i1 - 1]['end'] if i1 > 0 else original_tokens[0]['start']
            next_start = original_tokens[i1]['start'] if i1 < len(original_tokens) else original_tokens[-1]['end']
            duration = (next_start - prev_end) / max(j2 - j1, 1)
            for idx in range(j1, j2):
                aligned_tokens.append({
                    'word': corrected_words[idx],
                    'start': prev_end + (idx - j1) * duration,
                    'end': prev_end + (idx - j1 + 1) * duration
                })
        elif tag == 'delete':
            # Words are deleted in corrected text, skip them
            continue
    return aligned_tokens

def align_corrected_transcription(segments, corrected_text):
    # with open('./dst/وام_بانکی_و_اموال_ربوی/audio_2024-11-12_06-41-32.json', 'r') as f:
    #     corrected_text = json.loads(f.read())['correction']['corrected_transcription']
    # with open('./dst/وام_بانکی_و_اموال_ربوی/audio_2024-11-12_06-41-32.jsonl', 'r') as f:
    #     segments = [json.loads(line) for line in f]

    # Tokenize corrected text
    corrected_tokens = tokenize(corrected_text)

    # Get time-stamped tokens from original segments
    original_tokens = flatten([s['words'] for s in segments])
    for t in original_tokens:
        t['word'] = t['word'].strip()

    # Align texts and get corrected tokens with timestamps
    aligned_tokens = align_texts(original_tokens, corrected_tokens)


    # with open('align-1.txt', 'w', encoding='utf-8') as f:
    #     for token in original_tokens:
    #         f.write(f"{token['word']} ({token['start']:.2f}s - {token['end']:.2f}s)\n")

    # with open('align-2.txt', 'w', encoding='utf-8') as f:
    #     for token in aligned_tokens:
    #         f.write(f"{token['word']} ({token['start']:.2f}s - {token['end']:.2f}s)\n")

    return aligned_tokens

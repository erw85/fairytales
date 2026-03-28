#!/usr/bin/env python3
"""
Preprocess three fairy tale collections and generate JSON data for visualization.
"""

import re
import json
import os
from collections import Counter, defaultdict

# Download NLTK data
import nltk
for pkg in ['punkt', 'stopwords', 'punkt_tab']:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))
# Add common narrative/Gutenberg stopwords
EXTRA_STOPWORDS = {
    'said', 'would', 'could', 'upon', 'one', 'little', 'great', 'came',
    'went', 'come', 'go', 'know', 'told', 'like', 'man', 'time', 'old',
    'good', 'long', 'made', 'went', 'take', 'got', 'saw', 'away', 'back',
    'day', 'night', 'let', 'put', 'well', 'never', 'also', 'away',
    'project', 'gutenberg', 'ebook', 'electronic', 'works', 'www',
    'http', 'org', 'copyright', 'license', 'terms', 'use', 'page',
    'illustration', 'chapter', 'vol', 'i', 'ii', 'iii', 'iv', 'v',
    'us', 'even', 'still', 'though', 'much', 'many', 'every', 'way',
    'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'first', 'last', 'next', 'soon', 'yet', 'must', 'shall', 'may',
    'without', 'nothing', 'something', 'everything', 'another',
    'set', 'new', 'see', 'say', 'yes', 'no', 'ever', 'quite', 'got',
    'took', 'looked', 'asked', 'replied', 'cried', 'began',
}
STOPWORDS |= EXTRA_STOPWORDS

CORPUS_CONFIGS = [
    {
        'key': 'hca',
        'label': 'Hans Christian Andersen',
        'short': 'HCA',
        'color': '#4e79a7',
        'path': '/Users/erinwebreck/Desktop/HCA Fairy Tales.txt',
    },
    {
        'key': 'grimms',
        'label': "Grimm's Fairy Tales",
        'short': 'Grimms',
        'color': '#f28e2b',
        'path': '/Users/erinwebreck/Desktop/Grimms Fairy Tales.txt',
    },
    {
        'key': 'russian',
        'label': 'Russian Fairy Tales',
        'short': 'Russian',
        'color': '#e15759',
        'path': '/Users/erinwebreck/Desktop/Russian Fairy Tales.txt',
    },
]

# Thematic word groups for the Themes tab
THEME_GROUPS = [
    {
        'name': 'Royalty & Power',
        'icon': '♔',
        'words': ['king', 'queen', 'prince', 'princess', 'castle', 'throne',
                  'crown', 'kingdom', 'royal', 'palace', 'emperor', 'majesty'],
    },
    {
        'name': 'Family & Kinship',
        'icon': '👨‍👩‍👧',
        'words': ['father', 'mother', 'son', 'daughter', 'brother', 'sister',
                  'child', 'wife', 'husband', 'children', 'widow', 'stepmother'],
    },
    {
        'name': 'Magic & Supernatural',
        'icon': '✨',
        'words': ['witch', 'magic', 'spell', 'fairy', 'enchanted', 'dragon',
                  'ghost', 'spirit', 'wizard', 'wicked', 'curse', 'charm',
                  'sorcerer', 'mysterious', 'supernatural'],
    },
    {
        'name': 'Animals',
        'icon': '🦊',
        'words': ['horse', 'bird', 'wolf', 'bear', 'fox', 'dog', 'cat',
                  'fish', 'snake', 'eagle', 'deer', 'lion', 'duck', 'swan',
                  'rabbit', 'raven', 'crow'],
    },
    {
        'name': 'Nature & Elements',
        'icon': '🌿',
        'words': ['water', 'sea', 'river', 'mountain', 'forest', 'tree',
                  'flower', 'wind', 'sun', 'moon', 'fire', 'earth', 'snow',
                  'ice', 'rain', 'sky', 'wood', 'stone', 'lake', 'field'],
    },
    {
        'name': 'Death & Danger',
        'icon': '⚔️',
        'words': ['death', 'dead', 'kill', 'blood', 'evil', 'darkness',
                  'afraid', 'danger', 'sword', 'war', 'enemy', 'wicked',
                  'cruel', 'poison', 'die', 'wound'],
    },
    {
        'name': 'Love & Beauty',
        'icon': '❤️',
        'words': ['love', 'beautiful', 'heart', 'kiss', 'marry', 'bride',
                  'wedding', 'lovely', 'fair', 'handsome', 'beloved',
                  'marriage', 'joy', 'happy'],
    },
    {
        'name': 'Journey & Quest',
        'icon': '🗺️',
        'words': ['journey', 'road', 'travel', 'seek', 'find', 'search',
                  'quest', 'path', 'walk', 'return', 'arrive', 'set',
                  'home', 'far', 'land', 'world'],
    },
]


def strip_gutenberg(text):
    """Remove Project Gutenberg header and footer."""
    start_markers = [
        r'\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
    ]
    end_markers = [
        r'\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'\*\*\*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*',
        r'End of the Project Gutenberg',
        r'End of Project Gutenberg',
    ]
    for marker in start_markers:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            text = text[m.end():]
            break
    for marker in end_markers:
        m = re.search(marker, text, re.IGNORECASE)
        if m:
            text = text[:m.start()]
            break
    return text.strip()


def clean_text(text):
    """Strip formatting artifacts."""
    text = re.sub(r'\[Illustration[^\]]*\]', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\n[A-Z][A-Z\s\-\',\.]+\n', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def tokenize(text):
    """Lowercase, tokenize, keep only alphabetic tokens."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and len(t) > 2]


def filter_stopwords(tokens):
    return [t for t in tokens if t not in STOPWORDS]


def extract_sentences(text):
    """Split text into clean sentences for concordance search."""
    # Split on sentence-ending punctuation
    raw = re.split(r'(?<=[.!?])\s+', text)
    sentences = []
    for s in raw:
        s = s.strip()
        # Clean up the sentence
        s = re.sub(r'\s+', ' ', s)
        # Only keep sentences of reasonable length
        if 25 <= len(s) <= 400:
            sentences.append(s)
    return sentences


def sentence_concordance(sentences, keyword, max_results=25):
    """Find sentences containing keyword with highlight markup."""
    kw = keyword.lower()
    results = []
    for sent in sentences:
        if kw in sent.lower():
            highlighted = re.sub(
                r'\b' + re.escape(keyword) + r'\b',
                f'<mark>{keyword}</mark>',
                sent,
                flags=re.IGNORECASE
            )
            results.append({'text': sent, 'highlighted': highlighted})
            if len(results) >= max_results:
                break
    return results


def process_corpus(config):
    with open(config['path'], 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    text = strip_gutenberg(raw)
    sentences = extract_sentences(text)
    text = clean_text(text)
    tokens_all = tokenize(text)
    tokens_filtered = filter_stopwords(tokens_all)

    freq = Counter(tokens_filtered)
    total = sum(freq.values())

    top_words = [
        {'word': w, 'count': c, 'freq': round(c / total * 1000, 4)}
        for w, c in freq.most_common(500)
    ]

    return {
        'key': config['key'],
        'label': config['label'],
        'short': config['short'],
        'color': config['color'],
        'total_tokens': len(tokens_all),
        'unique_tokens': len(set(tokens_filtered)),
        'top_words': top_words,
        'tokens': tokens_filtered,
        'tokens_all': tokens_all,
        'sentences': sentences,
        'freq': freq,
    }


def compute_shared_terms(corpora, top_n=100):
    """Find terms significant across all three corpora."""
    all_freqs = [c['freq'] for c in corpora]
    all_words = set.intersection(*[set(f.keys()) for f in all_freqs])

    scored = []
    for w in all_words:
        counts = [f[w] for f in all_freqs]
        totals = [c['total_tokens'] for c in corpora]
        normalized = [c / t * 10000 for c, t in zip(counts, totals)]
        min_norm = min(normalized)
        avg_norm = sum(normalized) / len(normalized)
        scored.append({
            'word': w,
            'counts': {c['key']: all_freqs[i][w] for i, c in enumerate(corpora)},
            'normalized': {c['key']: round(normalized[i], 4) for i, c in enumerate(corpora)},
            'min_norm': round(min_norm, 4),
            'avg_norm': round(avg_norm, 4),
        })

    scored.sort(key=lambda x: x['avg_norm'], reverse=True)
    return scored[:top_n]


def compute_themes(corpora):
    """Compute per-theme normalized frequency scores for each corpus."""
    results = []
    for group in THEME_GROUPS:
        group_data = {
            'name': group['name'],
            'icon': group['icon'],
            'words': group['words'],
            'scores': {},
            'word_scores': {},  # individual word breakdown
        }
        for corpus in corpora:
            total = corpus['total_tokens']
            freq = corpus['freq']
            # Sum normalized frequencies for all theme words
            word_scores = {}
            for w in group['words']:
                count = freq.get(w, 0)
                word_scores[w] = round(count / total * 10000, 4)
            group_data['scores'][corpus['key']] = round(
                sum(word_scores.values()), 4
            )
            group_data['word_scores'][corpus['key']] = word_scores
        results.append(group_data)
    return results


def build_concordance_data(corpora, keywords):
    """Build sentence-level concordance entries for a set of keywords."""
    result = {}
    for kw in keywords:
        result[kw] = {}
        for corpus in corpora:
            entries = sentence_concordance(corpus['sentences'], kw, max_results=25)
            result[kw][corpus['key']] = entries
    return result


def main():
    os.makedirs('data', exist_ok=True)

    print("Processing corpora...")
    corpora = []
    for config in CORPUS_CONFIGS:
        print(f"  Processing {config['label']}...")
        data = process_corpus(config)
        corpora.append(data)
        print(f"    {data['total_tokens']:,} tokens, {data['unique_tokens']:,} unique, "
              f"{len(data['sentences']):,} sentences")

    print("Computing shared terms...")
    shared = compute_shared_terms(corpora)
    print(f"  Found {len(shared)} shared terms")

    print("Computing themes...")
    themes = compute_themes(corpora)
    print(f"  Computed {len(themes)} thematic groups")

    # ── corpus stats JSON (top 500 words for client-side theme lookup) ──
    corpus_stats = []
    for c in corpora:
        corpus_stats.append({
            'key': c['key'],
            'label': c['label'],
            'short': c['short'],
            'color': c['color'],
            'total_tokens': c['total_tokens'],
            'unique_tokens': c['unique_tokens'],
            'top_words': c['top_words'],  # now 500 words
        })

    with open('data/corpora.json', 'w') as f:
        json.dump(corpus_stats, f, separators=(',', ':'))
    print("  Wrote data/corpora.json")

    with open('data/shared.json', 'w') as f:
        json.dump(shared, f, separators=(',', ':'))
    print("  Wrote data/shared.json")

    with open('data/themes.json', 'w') as f:
        json.dump(themes, f, separators=(',', ':'))
    print("  Wrote data/themes.json")

    # ── sentences for full client-side concordance ──
    print("Exporting sentences for full concordance...")
    sentences_data = {}
    for c in corpora:
        # Limit to 3500 sentences per corpus to keep file size reasonable
        sentences_data[c['key']] = c['sentences'][:3500]
        print(f"  {c['key']}: {len(sentences_data[c['key']]):,} sentences")
    with open('data/sentences.json', 'w') as f:
        json.dump(sentences_data, f, separators=(',', ':'))
    print("  Wrote data/sentences.json")

    # ── pre-computed concordance for top 60 shared terms ──
    print("Building pre-computed concordances (top 60 shared terms)...")
    top_shared_words = [s['word'] for s in shared[:60]]
    concordance = build_concordance_data(corpora, top_shared_words)
    with open('data/concordance.json', 'w') as f:
        json.dump(concordance, f, separators=(',', ':'))
    print("  Wrote data/concordance.json")

    # ── frequency comparison table (top 60 per corpus, union) ──
    all_top = set()
    for c in corpora:
        all_top.update(w['word'] for w in c['top_words'][:60])

    freq_table = []
    for word in sorted(all_top):
        row = {'word': word}
        for c in corpora:
            # Look up count from top_words list
            entry = next((w for w in c['top_words'] if w['word'] == word), None)
            row[c['key']] = entry['count'] if entry else 0
            total = c['total_tokens']
            row[c['key'] + '_norm'] = round(
                (entry['count'] if entry else 0) / total * 10000, 4
            )
        freq_table.append(row)
    freq_table.sort(
        key=lambda x: sum(x.get(c['key'], 0) for c in corpora), reverse=True
    )

    with open('data/freq_table.json', 'w') as f:
        json.dump(freq_table, f, separators=(',', ':'))
    print("  Wrote data/freq_table.json")

    print("\nDone! Generated files in data/")
    for fname in ['corpora.json', 'shared.json', 'themes.json',
                  'sentences.json', 'concordance.json', 'freq_table.json']:
        size = os.path.getsize(f'data/{fname}')
        print(f"  data/{fname}: {size / 1024:.1f} KB")


if __name__ == '__main__':
    main()

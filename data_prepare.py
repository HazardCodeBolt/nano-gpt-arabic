"""
Data preparation for Arabic Text Generation.
Downloads ArabicText-Large from HuggingFace, preprocesses, tokenizes,
and saves binary files for nanoGPT-style training.
"""

import os
import re
import pickle
import random
import numpy as np
from collections import Counter

# ── Arabic preprocessing ──────────────────────────────────────────────────

DIACRITICS  = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]')
NON_ARABIC  = re.compile(r'[^؀-ۿ\s\n\.\،\؟\!\:\-]')
URL_PAT     = re.compile(r'http\S+|www\.\S+')
HTML_PAT    = re.compile(r'<[^>]+>')
MULTI_SPACE = re.compile(r'[ \t]+')

ARABIC_NORM = str.maketrans({
    'آ': 'ا', 'أ': 'ا', 'إ': 'ا', 'ٱ': 'ا',
    'ة': 'ه',
    'ى': 'ي',
})

NOISE_DIACRITICS = ['َ', 'ُ', 'ِ', 'ْ', 'ّ']


def remove_urls(text):        return URL_PAT.sub(' ', text)
def remove_html(text):        return HTML_PAT.sub(' ', text)
def remove_diacritics(text):  return DIACRITICS.sub('', text)
def normalize_arabic(text):   return text.translate(ARABIC_NORM)
def remove_non_arabic(text):  return NON_ARABIC.sub(' ', text)
def normalize_whitespace(text):
    text = MULTI_SPACE.sub(' ', text)
    # collapse more than 2 consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def preprocess_text(text: str) -> str:
    """Full 6-step preprocessing pipeline."""
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_diacritics(text)
    text = normalize_arabic(text)
    text = remove_non_arabic(text)
    text = normalize_whitespace(text)
    return text


def inject_noise(text: str, noise_ratio: float = 0.02) -> str:
    """Insert random Arabic diacritics to simulate a noisy corpus."""
    chars = list(text)
    n_noise = max(1, int(len(chars) * noise_ratio))
    for _ in range(n_noise):
        pos = random.randint(0, len(chars) - 1)
        if chars[pos] not in (' ', '\n'):
            chars.insert(pos + 1, random.choice(NOISE_DIACRITICS))
    return ''.join(chars)


# ── Character-level tokenizer ─────────────────────────────────────────────

class ArabicCharTokenizer:
    SPECIAL = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build_vocab(self, texts, max_vocab=512):
        counter = Counter()
        for text in texts:
            counter.update(text)
        common = [ch for ch, _ in counter.most_common(max_vocab - len(self.SPECIAL))]
        vocab = self.SPECIAL + common
        self.char2idx = {ch: i for i, ch in enumerate(vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(vocab)}
        self.vocab_size = len(vocab)
        print(f"Vocabulary: {self.vocab_size} characters")
        return self

    def encode(self, text, max_len=None):
        unk = self.char2idx['<UNK>']
        ids = [self.char2idx.get(ch, unk) for ch in text]
        if max_len is not None:
            ids = ids[:max_len]
            ids += [self.char2idx['<PAD>']] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        specials = set(self.SPECIAL)
        return ''.join(
            self.idx2char.get(i, '')
            for i in ids
            if self.idx2char.get(i, '') not in specials
        )

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({
                'char2idx': self.char2idx,
                'idx2char': self.idx2char,
                'vocab_size': self.vocab_size,
            }, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path, 'rb') as f:
            d = pickle.load(f)
        tok.char2idx   = d['char2idx']
        tok.idx2char   = d['idx2char']
        tok.vocab_size = d['vocab_size']
        return tok


# ── BPE tokenizer (second feature representation) ─────────────────────────

class SimpleBPETokenizer:
    """
    Minimal byte-pair encoding tokenizer built from scratch.
    Used as the second feature representation for comparison.
    """

    def __init__(self, vocab_size=1000):
        self.target_vocab_size = vocab_size
        self.merges = {}       # (pair) -> merged_token
        self.vocab  = {}       # token -> id
        self.id2tok = {}

    def _get_pairs(self, word_freqs):
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_pair(self, pair, word_freqs):
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        return new_word_freqs

    def fit(self, texts, num_merges=200):
        # Initialize: every character is a token, words split into chars
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word_freqs[' '.join(list(word)) + ' </w>'] += 1

        # Build initial vocab from characters
        all_tokens = set()
        for word in word_freqs:
            all_tokens.update(word.split())
        self.vocab = {tok: i for i, tok in enumerate(sorted(all_tokens))}

        # BPE merges
        for _ in range(num_merges):
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
            best = pairs.most_common(1)[0][0]
            word_freqs = self._merge_pair(best, word_freqs)
            self.merges[best] = ''.join(best)
            merged = ''.join(best)
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)

        self.id2tok = {v: k for k, v in self.vocab.items()}
        print(f"BPE vocab size: {len(self.vocab)}")
        return self

    def encode(self, text, max_len=None):
        ids = []
        unk_id = 0
        for word in text.split():
            word_tok = ' '.join(list(word)) + ' </w>'
            for pair, merged in self.merges.items():
                word_tok = word_tok.replace(' '.join(pair), merged)
            for tok in word_tok.split():
                ids.append(self.vocab.get(tok, unk_id))
        if max_len is not None:
            ids = ids[:max_len]
            ids += [0] * (max_len - len(ids))
        return ids

    def decode(self, ids):
        tokens = [self.id2tok.get(i, '') for i in ids]
        text = ''.join(tokens).replace('</w>', ' ')
        return text.strip()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'merges': self.merges, 'vocab': self.vocab,
                         'id2tok': self.id2tok,
                         'target_vocab_size': self.target_vocab_size}, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        tok = cls(d['target_vocab_size'])
        tok.merges = d['merges']
        tok.vocab  = d['vocab']
        tok.id2tok = d['id2tok']
        return tok


# ── Main preparation ──────────────────────────────────────────────────────

def prepare_data(
    sample_size: int = 50000,
    val_ratio:   float = 0.1,
    out_dir:     str   = 'data',
    seed:        int   = 42,
):
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading ArabicText-Large from HuggingFace (streaming)...")
    from datasets import load_dataset
    dataset = load_dataset("Jr23xd23/ArabicText-Large", split='train', streaming=True)

    raw_texts = []
    for article in dataset:
        text = article.get('text', '') or ''
        if len(text) > 200:
            raw_texts.append(text)
        if len(raw_texts) >= sample_size:
            break
    print(f"Collected {len(raw_texts)} articles")

    # Save representative noisy/clean examples for the notebook
    import json
    examples = []
    for raw in raw_texts[:20]:
        noisy = inject_noise(raw[:400])
        clean = preprocess_text(raw[:400])
        examples.append({'raw': raw[:400], 'noisy': noisy, 'clean': clean})
    with open(os.path.join(out_dir, 'preprocessing_examples.json'), 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)

    # Preprocess all texts
    print("Preprocessing...")
    clean_texts = [preprocess_text(t) for t in raw_texts]
    clean_texts = [t for t in clean_texts if len(t) > 100]
    print(f"Articles after filtering: {len(clean_texts)}")

    # Save article stats for EDA
    stats = [{'length': len(t), 'word_count': len(t.split())} for t in clean_texts]
    with open(os.path.join(out_dir, 'article_stats.json'), 'w') as f:
        json.dump(stats, f)

    # Concatenate into one big corpus with article separators
    corpus = '\n\n'.join(clean_texts)
    print(f"Total corpus characters: {len(corpus):,}")

    # ── Character tokenizer ───────────────────────────────────────────────
    print("Building character tokenizer...")
    char_tok = ArabicCharTokenizer()
    char_tok.build_vocab(clean_texts[:5000])  # build vocab from subset
    char_tok.save(os.path.join(out_dir, 'char_tokenizer.pkl'))

    # ── BPE tokenizer ─────────────────────────────────────────────────────
    print("Building BPE tokenizer (this takes a few minutes)...")
    bpe_tok = SimpleBPETokenizer(vocab_size=1000)
    bpe_tok.fit(clean_texts[:2000], num_merges=500)
    bpe_tok.save(os.path.join(out_dir, 'bpe_tokenizer.pkl'))

    # ── Encode with char tokenizer & split ────────────────────────────────
    print("Encoding corpus with character tokenizer...")
    char_ids = np.array(char_tok.encode(corpus), dtype=np.uint16)

    split_idx = int(len(char_ids) * (1 - val_ratio))
    train_ids = char_ids[:split_idx]
    val_ids   = char_ids[split_idx:]

    train_ids.tofile(os.path.join(out_dir, 'train.bin'))
    val_ids.tofile(os.path.join(out_dir,   'val.bin'))
    print(f"Char tokens — train: {len(train_ids):,}, val: {len(val_ids):,}")

    # ── Encode with BPE tokenizer & split ─────────────────────────────────
    print("Encoding corpus with BPE tokenizer...")
    bpe_ids = np.array(bpe_tok.encode(corpus), dtype=np.uint16)
    bpe_split = int(len(bpe_ids) * (1 - val_ratio))
    bpe_ids[:bpe_split].tofile(os.path.join(out_dir, 'bpe_train.bin'))
    bpe_ids[bpe_split:].tofile(os.path.join(out_dir, 'bpe_val.bin'))
    print(f"BPE tokens  — train: {bpe_split:,}, val: {len(bpe_ids)-bpe_split:,}")

    # ── Save meta ─────────────────────────────────────────────────────────
    meta = {
        'vocab_size':     char_tok.vocab_size,
        'bpe_vocab_size': len(bpe_tok.vocab),
        'char2idx':       char_tok.char2idx,
        'idx2char':       char_tok.idx2char,
        'corpus_len':     len(corpus),
        'train_tokens':   len(train_ids),
        'val_tokens':     len(val_ids),
        'n_articles':     len(clean_texts),
    }
    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print(f"\nData ready in '{out_dir}/'")
    return meta, char_tok, bpe_tok, clean_texts


if __name__ == '__main__':
    prepare_data(sample_size=50000)

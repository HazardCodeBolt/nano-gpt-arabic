"""
Streamlit Web App – Arabic Text Generation
CSDS4102 NLP Mini Project | Semester 2 2025-2026

Run: streamlit run app.py
"""

import os
import pickle
import math
import re

import numpy as np
import streamlit as st
import torch
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Arabic Text Generator",
    page_icon="🖊️",
    layout="wide",
)

DATA_DIR  = 'data'
MODEL_DIR = '.'

# ── Preprocessing (mirrors data_prepare.py) ───────────────────────────────
DIACRITICS  = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۤۧۨ-ۭ]')
NON_ARABIC  = re.compile(r'[^؀-ۿ\s\n\.\،\؟\!\:\-]')
URL_PAT     = re.compile(r'http\S+|www\.\S+')
HTML_PAT    = re.compile(r'<[^>]+>')
MULTI_SPACE = re.compile(r'[ \t]+')
ARABIC_NORM = str.maketrans({'آ':'ا','أ':'ا','إ':'ا','ٱ':'ا','ة':'ه','ى':'ي'})


def preprocess(text):
    text = URL_PAT.sub(' ', text)
    text = HTML_PAT.sub(' ', text)
    text = DIACRITICS.sub('', text)
    text = text.translate(ARABIC_NORM)
    text = NON_ARABIC.sub(' ', text)
    text = MULTI_SPACE.sub(' ', text).strip()
    return text


# ── Model loading ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_models():
    models = {}

    # Character tokenizer
    tok_path = os.path.join(DATA_DIR, 'char_tokenizer.pkl')
    if os.path.exists(tok_path):
        from data_prepare import ArabicCharTokenizer
        models['tokenizer'] = ArabicCharTokenizer.load(tok_path)

    # NanoGPT
    gpt_path = os.path.join(MODEL_DIR, 'best_gpt_model.pt')
    if os.path.exists(gpt_path) and 'tokenizer' in models:
        from model import GPT, GPTConfig
        ckpt = torch.load(gpt_path, map_location='cpu')
        cfg  = ckpt['config']
        gpt  = GPT(cfg)
        gpt.load_state_dict(ckpt['model'])
        gpt.eval()
        models['gpt'] = gpt
        models['gpt_val_loss'] = ckpt.get('val_loss', None)

    # N-gram models
    for name, path in [('bigram', 'ngram_bigram.pkl'), ('trigram', 'ngram_trigram.pkl')]:
        full = os.path.join(MODEL_DIR, path)
        if os.path.exists(full):
            with open(full, 'rb') as f:
                models[name] = pickle.load(f)

    return models


# ── Generation helpers ────────────────────────────────────────────────────
def generate_gpt(model, tokenizer, seed_text, max_new, temperature, top_k, top_p):
    clean = preprocess(seed_text)
    ids   = tokenizer.encode(clean)
    if not ids:
        return seed_text
    x = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            x, max_new_tokens=max_new,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p if top_p < 1.0 else None,
        )
    return tokenizer.decode(out[0].cpu().tolist())


def generate_ngram(lm, tokenizer, seed_text, max_new, temperature):
    clean    = preprocess(seed_text)
    seed_ids = tokenizer.encode(clean)
    if not seed_ids:
        return seed_text
    gen_ids = lm.generate(seed_ids, max_new=max_new, temperature=temperature)
    return tokenizer.decode(gen_ids)


# ── UI ────────────────────────────────────────────────────────────────────
def main():
    st.markdown(
        "<h1 style='text-align:center;'>🖊️ Arabic Text Generator</h1>"
        "<p style='text-align:center; color:gray;'>NLP Mini Project – CSDS4102 | "
        "NanoGPT × ArabicText-Large</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    models = load_models()
    tok = models.get('tokenizer')

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Generation Settings")

        model_choice = st.selectbox(
            "Model",
            options=["NanoGPT (DL)", "Trigram LM (ML)", "Bigram LM (ML)"],
        )

        max_new = st.slider("Max new characters / tokens", 50, 500, 200, 25)

        temperature = st.slider(
            "Temperature", 0.1, 2.0, 0.85, 0.05,
            help="Lower = more focused; higher = more creative"
        )

        top_k = st.slider(
            "Top-k (NanoGPT only)", 0, 200, 40,
            help="0 = disabled"
        )

        top_p = st.slider(
            "Top-p / nucleus (NanoGPT only)", 0.5, 1.0, 1.0, 0.05,
            help="1.0 = disabled"
        )

        st.markdown("---")
        if models.get('gpt_val_loss') is not None:
            vl = models['gpt_val_loss']
            st.metric("NanoGPT val loss",  f"{vl:.4f}")
            st.metric("NanoGPT val PPL",   f"{math.exp(vl):.1f}")
        st.markdown("---")
        st.markdown("**About**")
        st.markdown(
            "NanoGPT is trained **from scratch** on 50K Arabic articles "
            "from the ArabicText-Large corpus (HuggingFace). "
            "No pretrained weights are used."
        )

    # ── Main panel ────────────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📝 Seed Text")
        seed = st.text_area(
            label="",
            value="العلوم والتكنولوجيا",
            height=120,
            help="Type any Arabic seed phrase. The model will continue from here.",
        )
        generate_btn = st.button("✨ Generate", type="primary", use_container_width=True)

    with col2:
        st.subheader("💡 Example Seeds")
        examples = {
            "Science":   "العلوم والتكنولوجيا تشهد",
            "History":   "في القرن العاشر الميلادي",
            "Sports":    "فاز المنتخب بعد مباراة",
            "Religion":  "قال الله تعالى في القرآن",
            "Geography": "تقع المدينة على ضفاف",
        }
        for label, ex in examples.items():
            if st.button(f"▶ {label}", key=label, use_container_width=True):
                seed = ex
                generate_btn = True

    # ── Generation ────────────────────────────────────────────────────────
    if generate_btn:
        st.markdown("---")

        model_key_map = {
            "NanoGPT (DL)":    'gpt',
            "Trigram LM (ML)": 'trigram',
            "Bigram LM (ML)":  'bigram',
        }
        key = model_key_map[model_choice]

        if key not in models or tok is None:
            st.error(
                "⚠️ Model not found. Run the notebook first to train and save models, "
                "then restart the app."
            )
            return

        with st.spinner(f"Generating with {model_choice}…"):
            try:
                if key == 'gpt':
                    output = generate_gpt(
                        models['gpt'], tok, seed,
                        max_new, temperature, top_k, top_p
                    )
                else:
                    output = generate_ngram(
                        models[key], tok, seed, max_new, temperature
                    )
            except Exception as e:
                st.error(f"Generation error: {e}")
                return

        # Display
        st.subheader("📄 Generated Text")
        # Highlight the seed portion
        if seed in output:
            split_at = output.index(seed) + len(seed)
            seed_part = output[:split_at]
            gen_part  = output[split_at:]
        else:
            seed_part = seed
            gen_part  = output

        st.markdown(
            f"<div style='background:#f8f9fa; padding:16px; border-radius:8px; "
            f"font-size:18px; line-height:1.8; direction:rtl; text-align:right;'>"
            f"<span style='background:#fff3cd; padding:2px 4px;'>{seed_part}</span>"
            f"{gen_part}"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.caption(f"Model: {model_choice} | Temperature: {temperature} | "
                   f"Max new: {max_new} | Top-k: {top_k} | Top-p: {top_p}")

        # Token probability bar (approximate — show top-token distribution for last step)
        if key == 'gpt':
            st.markdown("---")
            st.subheader("🔢 Next-Token Probability (after seed)")
            clean_seed = preprocess(seed)
            ids = tok.encode(clean_seed)
            x   = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits, _ = models['gpt'](x)
            probs = torch.softmax(logits[0, -1, :] / temperature, dim=-1).numpy()
            top_n = 15
            top_ids = np.argsort(probs)[::-1][:top_n]
            top_chars  = [tok.idx2char.get(int(i), f'id{i}') for i in top_ids]
            top_probs  = [float(probs[i]) for i in top_ids]

            fig, ax = plt.subplots(figsize=(8, 3))
            ax.barh(top_chars[::-1], top_probs[::-1], color='steelblue')
            ax.set_xlabel('Probability')
            ax.set_title('Top-15 Next Character Probabilities')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:gray; font-size:12px;'>"
        "CSDS4102 NLP Mini Project | NanoGPT trained from scratch on ArabicText-Large"
        "</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

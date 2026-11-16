#!/usr/bin/env python3
"""AI Model to Identify Wrong / Fake News trained on a 10,000 Sample Dataset.

Architecture:
- Subword & Token Embedding Layer + Stylometric Feature Fusion
- Deep Residual LayerNorm MLP Classifier (`ox.nn.Module`)
- Hugging Face `.safetensors` weight persistence via `ox.save_file`
- Interactive and programmatic evaluation (`eval_input`) for arbitrary news headlines
"""

from collections import Counter
import argparse
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import oxideon as ox
import oxideon.nn as nn


# ---------------------------------------------------------------------------
# 1. 10,000 Sample Balanced News Dataset Generator
# ---------------------------------------------------------------------------

def generate_10k_news_dataset(seed: int = 42):
    """Generates a rich, balanced 10,000 dataset (5,000 Real News, 5,000 Wrong/Fake News).

    Covers multiple domains: Science, Politics, Economy, Medicine, Technology,
    and Climate, with authentic linguistic signatures for both classes.
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- Real News Combinatorial Elements ---
    real_entities = [
        "NASA astronomers", "Researchers at Stanford University", "The Federal Reserve Board",
        "The World Health Organization", "European Space Agency scientists", "Reuters journalists",
        "The Department of Energy", "MIT robotics engineers", "Clinical researchers at Mayo Clinic",
        "The International Monetary Fund", "Geologists at USGS", "Oxford University epidemiologists",
        "The Bureau of Labor Statistics", "CERN particle physicists", "The Bank of England",
        "The National Science Foundation", "The Environmental Protection Agency", "Johns Hopkins medicine",
    ]
    real_actions = [
        "published peer-reviewed findings confirming", "announced a 25 basis point adjustment in",
        "released official quarterly telemetry measuring", "concluded a five-year double-blind trial on",
        "reported verified economic data showing", "demonstrated a 14 percent efficiency increase in",
        "confirmed successful laboratory tests on", "approved revised regulatory standards for",
        "completed deep-sea sonar mapping across", "documented atmospheric carbon reduction from",
        "deployed seismic sensor networks along", "identified potential biomarkers associated with",
    ]
    real_subjects = [
        "quantum dot semiconductor manufacturing", "lithium-iron-phosphate battery longevity",
        "subsurface ocean salinity on Jupiter's moon Europa", "pediatric immunotherapy tolerance rates",
        "quarterly core inflation rate stabilization", "silicon-perovskite photovoltaic solar cells",
        "high-speed magnetic levitation track tests", "deep ocean hydrothermal vent biodiversity",
        "interbank liquidity reserve requirements", "drought-resistant CRISPR maize crop yield",
        "early-stage diabetic retinopathy detection algorithms", "offshore floating wind turbine resilience",
        "gravitational wave oscillations from binary neutron stars", "global supply chain cargo throughput",
        "mRNA therapeutic delivery safety profiles", "solid-state electrolyte degradation pathways",
    ]
    real_attributions = [
        "according to an official bulletin published this Tuesday.",
        "as reported in the latest issue of Nature Physics.",
        "following audited data verified by independent regulators.",
        "in formal testimony delivered before the legislative committee.",
        "concluding a multi-year collaborative international survey.",
        "according to financial disclosures filed with market authorities.",
        "per official proceedings of the National Academy of Sciences.",
        "supported by reproducible evidence from multiple observatories.",
    ]

    # --- Fake / Wrong News Combinatorial Elements ---
    fake_prefixes = [
        "SHOCKING BOMBSHELL:", "EXPOSED:", "THEY DON'T WANT YOU TO KNOW:",
        "URGENT ALERT:", "WHISTLEBLOWER LEAKS:", "SECRET CURE BANNED BY MEDIA:",
        "WHAT THE GOVERNMENT CONCEALS:", "UNBELIEVABLE PROOF:", "MAINSTREAM SCIENTISTS ADMIT:",
        "TERRIFYING TRUTH REVEALED:", "ACT NOW BEFORE THIS IS DELETED:", "CONFIDENTIAL MILITARY LEAK:",
    ]
    fake_sources = [
        "Rogue insider", "Anonymous high-ranking general", "Holistic vibrational guru",
        "Deep-state defector", "Suppressed inventor", "Alternative health visionary",
        "Disgruntled space agency technician", "Underground frequency researcher",
        "Ex-pharmaceutical executive turned truth-seeker", "Mysterious telegram channel",
    ]
    fake_actions = [
        "confirms that elite globalists have secretly replaced world leaders with",
        "reveals drinking boiled lemon garlic kerosene permanently cures",
        "admits that giant hollow polar ice walls conceal an ancient race of",
        "proves 5G cell towers emit psychoactive mind-control frequencies causing",
        "exposes that the moon is an artificial surveillance satellite populated by",
        "discloses that municipal water fluoride contains nanotechnology programmed to",
        "uncovers secret government plans to confiscate all citizen bank accounts by",
        "claims eating raw activated baking charcoal cures terminal stage-4",
        "shows military weather superweapons deliberately created yesterday's",
        "verifies that mysterious underground Antarctic stargates are controlled by",
        "admits that the Earth is entirely flat and surrounded by electromagnetic",
        "reveals that a miraculous perpetual motion free-energy generator was banned by",
    ]
    fake_targets = [
        "android clones operating under centralized satellite relays",
        "all forms of viral cancer and arterial plaque within 36 hours",
        "shapeshifting reptilian royalty ruling from subterranean caverns",
        "mass biological obedience and synchronized sleep deprivation",
        "secret military observation posts hiding alien treaty documents",
        "deactivate spiritual intuition and independent decision making",
        "tomorrow midnight under a covert executive martial law protocol",
        "organ failure without any prescription medication or surgery",
        "regional storm patterns to manipulate commodity crop pricing",
        "extraterrestrial overlords demanding immediate mineral tribute",
        "dome containment fields guarded by nuclear naval fleets",
        "oil cartels who arrested the genius inventor in his laboratory",
    ]
    fake_urgencies = [
        "SHARE EVERYWHERE BEFORE CENSORED!!!", "WAKE UP SHEEPLE THE TRUTH IS OUT!!!",
        "DOCTORS ARE FURIOUS THIS GOT LEAKED!", "THE COLLAPSE BEGINS THIS FRIDAY!",
        "THEY WILL SHUT DOWN THE INTERNET TO HIDE THIS!", "SPREAD THIS TO EVERY FAMILY MEMBER NOW!",
        "WATCH THE LIVE BROADCAST BEFORE MAINSTREAM MEDIA TAKES IT DOWN!",
    ]

    dataset = []

    # 1. Generate 5,000 Real News samples (Label 0)
    for i in range(5000):
        ent = random.choice(real_entities)
        act = random.choice(real_actions)
        subj = random.choice(real_subjects)
        attr = random.choice(real_attributions)

        # Style variants
        variant = i % 4
        if variant == 0:
            text = f"{ent} {act} {subj}, {attr}"
        elif variant == 1:
            text = f"Report: {ent} {act} {subj}."
        elif variant == 2:
            text = f"{subj.capitalize()}: {ent} {act} new milestones, {attr}"
        else:
            text = f"Scientific Analysis: {ent} {act} {subj}."

        dataset.append((text, 0))

    # 2. Generate 5,000 Wrong / Fake News samples (Label 1)
    for i in range(5000):
        pre = random.choice(fake_prefixes)
        src = random.choice(fake_sources)
        act = random.choice(fake_actions)
        tgt = random.choice(fake_targets)
        urg = random.choice(fake_urgencies)

        # Style variants
        variant = i % 4
        if variant == 0:
            text = f"{pre} {src} {act} {tgt}! {urg}"
        elif variant == 1:
            text = f"{src.upper()} WARNING: {act} {tgt} - {urg}"
        elif variant == 2:
            text = f"{pre} {act.capitalize()} {tgt}! SPREAD THE WORD!"
        else:
            text = f"{pre} {src} {act} {tgt}."

        dataset.append((text, 1))

    # Shuffle dataset deterministically
    random.shuffle(dataset)
    return dataset


# ---------------------------------------------------------------------------
# 2. Text Preprocessing, Vocabulary & Stylometric Feature Pipeline
# ---------------------------------------------------------------------------

class NewsTextPipeline:
    """Builds subword/word vocabulary and extracts multimodal text + stylometric features."""

    SENSATIONAL_WORDS = {
        "shocking", "bombshell", "exposed", "secret", "banned", "cure", "whistleblower",
        "leaked", "hoax", "truth", "wake", "sheeple", "aliens", "reptilian", "clone",
        "clones", "5g", "flat", "censored", "elites", "nanotech", "urgent", "terrifying",
        "miracle", "hidden", "stargate", "illuminati", "coverup", "conspiracy", "unbelievable",
    }

    CREDIBLE_WORDS = {
        "researchers", "study", "published", "reuters", "federal", "reserve", "percent",
        "official", "report", "confirmed", "department", "data", "clinical", "trial",
        "university", "quarterly", "proceedings", "audited", "journal", "telemetry",
        "regulators", "collaborative", "demonstrated", "analysis", "scientific", "peer-reviewed",
    }

    def __init__(self, max_vocab: int = 4096, max_len: int = 32):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        self.inv_vocab = {0: "<PAD>", 1: "<UNK>"}

    @staticmethod
    def clean_tokens(text: str):
        # Normalize and tokenize
        words = re.findall(r"\b[A-Za-z0-9\-_]{2,}\b", text.lower())
        return words

    def fit(self, texts: list[str]):
        counter = Counter()
        for t in texts:
            counter.update(self.clean_tokens(t))

        for word, _ in counter.most_common(self.max_vocab - 2):
            idx = len(self.vocab)
            self.vocab[word] = idx
            self.inv_vocab[idx] = word

    def text_to_ids(self, text: str) -> list[int]:
        tokens = self.clean_tokens(text)
        ids = [self.vocab.get(tok, 1) for tok in tokens[:self.max_len]]
        # Pad with 0
        if len(ids) < self.max_len:
            ids.extend([0] * (self.max_len - len(ids)))
        return ids

    def extract_stylometrics(self, text: str) -> list[float]:
        """Extracts 6 salient stylometric signals indicating authenticity vs sensationalism."""
        chars = len(text) if len(text) > 0 else 1
        alpha_chars = sum(1 for c in text if c.isalpha())
        upper_chars = sum(1 for c in text if c.isupper())

        caps_ratio = (upper_chars / alpha_chars) if alpha_chars > 0 else 0.0
        excl_density = text.count("!") / (chars / 100.0)
        question_density = text.count("?") / (chars / 100.0)

        tokens_lower = [t.lower() for t in self.clean_tokens(text)]
        n_tok = len(tokens_lower) if len(tokens_lower) > 0 else 1

        sensational_count = sum(1 for t in tokens_lower if t in self.SENSATIONAL_WORDS)
        sensational_score = sensational_count / n_tok

        credible_count = sum(1 for t in tokens_lower if t in self.CREDIBLE_WORDS)
        credible_score = credible_count / n_tok

        length_score = min(len(tokens_lower) / 30.0, 1.0)

        return [
            float(caps_ratio),
            float(excl_density),
            float(question_density),
            float(sensational_score),
            float(credible_score),
            float(length_score),
        ]

    def transform_batch(self, texts: list[str]):
        all_ids = [self.text_to_ids(t) for t in texts]
        all_stylom = [self.extract_stylometrics(t) for t in texts]
        return np.array(all_ids, dtype=np.int64), np.array(all_stylom, dtype=np.float32)

    def to_json(self) -> str:
        return json.dumps({
            "max_vocab": self.max_vocab,
            "max_len": self.max_len,
            "vocab": self.vocab,
        })

    @classmethod
    def from_json(cls, s: str):
        data = json.loads(s)
        pipe = cls(max_vocab=data["max_vocab"], max_len=data["max_len"])
        pipe.vocab = data["vocab"]
        pipe.inv_vocab = {v: k for k, v in pipe.vocab.items()}
        return pipe


# ---------------------------------------------------------------------------
# 3. Neural Network Classifier Architecture (`ox.nn.Module`)
# ---------------------------------------------------------------------------

class FakeNewsClassifier(nn.Module):
    """Deep Neural Network classifier combining text embedding pooling and stylometrics."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        in_features = embed_dim + 6  # 64 semantic embeddings + 6 stylometric features

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.1)

        self.classifier = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, token_ids: ox.Tensor, stylom_features: ox.Tensor) -> ox.Tensor:
        # 1. Semantic Embedding Mean Pooling over sequence length
        emb = self.embedding(token_ids)
        emb_pool = emb.mean(dim=1)  # [batch, embed_dim]

        # 2. Concatenate with Stylometric Features
        combined = ox.cat([emb_pool, stylom_features], dim=1)  # [batch, embed_dim + 6]

        # 3. Deep Feed-Forward Layers
        h = self.fc1(combined)
        h = self.norm1(h).relu()
        h = self.dropout1(h)

        h = self.fc2(h)
        h = self.norm2(h).relu()
        h = self.dropout2(h)

        # 4. Output Logits [batch, 2]: Class 0 = Real, Class 1 = Fake / Wrong
        return self.classifier(h)


# ---------------------------------------------------------------------------
# 4. Evaluation Function for Arbitrary Input News Headlines
# ---------------------------------------------------------------------------

def eval_input(text: str, model: FakeNewsClassifier, pipeline: NewsTextPipeline, verbose: bool = True) -> dict:
    """Evaluates a news headline or excerpt and identifies whether it is wrong/fake news.

    Returns structured analysis with class probability, confidence score, and signals.
    """
    model.eval()

    # Preprocess text
    ids_np, stylom_np = pipeline.transform_batch([text])
    ids_t = ox.from_numpy(ids_np)
    stylom_t = ox.from_numpy(stylom_np)

    # Forward inference
    logits = model(ids_t, stylom_t)
    probs = logits.softmax(1).to_numpy()[0]

    prob_real = float(probs[0])
    prob_fake = float(probs[1])

    is_fake = prob_fake >= 0.5
    confidence = prob_fake if is_fake else prob_real

    # Detected linguistic signals
    signals = []
    stylom = stylom_np[0]
    if stylom[0] > 0.20:
        signals.append(f"Excessive UPPERCASE sensationalism ({stylom[0]*100:.1f}%)")
    if stylom[1] > 0.5:
        signals.append("High exclamation mark frequency ('!')")
    if stylom[3] > 0.05:
        signals.append("Sensational conspiracy trigger words present")
    if stylom[4] > 0.05:
        signals.append("Verified journalistic attribution terms detected")
    if not signals:
        signals.append("Semantic embedding patterns dictate verdict")

    if is_fake:
        verdict = "🚨 WRONG / FAKE NEWS"
        risk_level = "CRITICAL" if confidence > 0.95 else "HIGH" if confidence > 0.80 else "MODERATE"
    else:
        verdict = "✅ CREDIBLE / REAL NEWS"
        risk_level = "LOW"

    result = {
        "text": text,
        "is_fake": is_fake,
        "verdict": verdict,
        "confidence": confidence,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
        "risk_level": risk_level,
        "signals": signals,
    }

    if verbose:
        print("\n" + "-" * 72)
        print(f"INPUT NEWS : \"{text}\"")
        print(f"VERDICT    : {verdict}")
        print(f"CONFIDENCE : {confidence * 100:.2f}%  (P[Fake]={prob_fake*100:.2f}%, P[Real]={prob_real*100:.2f}%)")
        print(f"RISK LEVEL : {risk_level}")
        print("SIGNALS    :")
        for s in signals:
            print(f"  • {s}")
        print("-" * 72)

    return result


# ---------------------------------------------------------------------------
# 5. Training, Testing & Demonstration Workflow
# ---------------------------------------------------------------------------

def train_and_evaluate(save_path: str = "fake_news_detector.safetensors", epochs: int = 4, batch_size: int = 64):
    print("=" * 75)
    print("      Oxideon AI: Wrong / Fake News Identification Engine")
    print("             Trained on a 10,000 Sample Dataset")
    print("=" * 75)

    # 1. Dataset Generation
    t0 = time.time()
    print("\n1. Generating synthetic 10,000 sample balanced news dataset...")
    dataset = generate_10k_news_dataset(seed=42)
    texts, labels = zip(*dataset)
    labels = list(labels)
    real_count = sum(1 for l in labels if l == 0)
    fake_count = sum(1 for l in labels if l == 1)
    print(f"   Total samples : {len(dataset):,}")
    print(f"   Real news     : {real_count:,} (Class 0)")
    print(f"   Fake news     : {fake_count:,} (Class 1 - Wrong news)")
    print(f"   Generation time: {time.time() - t0:.3f}s")

    # 2. Train / Test Split (80% train, 20% test)
    split_idx = int(0.80 * len(dataset))
    train_texts, test_texts = texts[:split_idx], texts[split_idx:]
    train_labels, test_labels = labels[:split_idx], labels[split_idx:]
    print(f"\n2. Data Split:")
    print(f"   Training set   : {len(train_texts):,} samples")
    print(f"   Evaluation set : {len(test_texts):,} samples")

    # 3. Fit Pipeline
    pipeline = NewsTextPipeline(max_vocab=3500, max_len=32)
    pipeline.fit(train_texts)
    print(f"\n3. Tokenizer & Feature Extraction:")
    print(f"   Vocabulary size: {len(pipeline.vocab):,} subwords/words")
    print(f"   Sequence length: {pipeline.max_len} tokens")

    # Transform data
    X_train_ids, X_train_stylom = pipeline.transform_batch(train_texts)
    y_train_np = np.array(train_labels, dtype=np.float32)

    X_test_ids, X_test_stylom = pipeline.transform_batch(test_texts)
    y_test_np = np.array(test_labels, dtype=np.float32)

    # 4. Instantiate Model
    print("\n4. Initializing Deep Neural Network Classifier...")
    model = FakeNewsClassifier(vocab_size=len(pipeline.vocab), embed_dim=64, hidden_dim=128)
    criterion = nn.CrossEntropyLoss()
    optimizer = ox.optim.Adam(model.parameters(), lr=1.5e-3, weight_decay=1e-4)

    print(f"   Total parameters: {len(model.parameters())} tensor layers")
    for name, p in model.named_parameters():
        print(f"     - {name:20s}: shape={str(p.shape):16s}")

    # 5. Training Loop
    print(f"\n5. Training model for {epochs} epochs (batch size = {batch_size})...")
    n_batches = len(train_texts) // batch_size

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct_train = 0
        indices = np.random.permutation(len(train_texts))

        epoch_start = time.time()
        for b in range(n_batches):
            b_idx = indices[b * batch_size : (b + 1) * batch_size]
            b_ids = ox.from_numpy(X_train_ids[b_idx])
            b_stylom = ox.from_numpy(X_train_stylom[b_idx])
            b_y = ox.from_numpy(y_train_np[b_idx])

            model.zero_grad()
            logits = model(b_ids, b_stylom)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.to_numpy().item())
            epoch_loss += loss_val

            preds = np.argmax(logits.to_numpy(), axis=1)
            correct_train += int(np.sum(preds == y_train_np[b_idx]))

        avg_loss = epoch_loss / n_batches
        train_acc = (correct_train / (n_batches * batch_size)) * 100.0
        epoch_time = time.time() - epoch_start
        print(f"   Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.5f} | Train Acc: {train_acc:.2f}% | Time: {epoch_time:.2f}s")

    # 6. Comprehensive Evaluation on 2,000 Test Set
    print("\n6. Running evaluation on 2,000 held-out test samples...")
    model.eval()
    test_batches = len(test_texts) // batch_size
    all_preds = []
    all_targets = []

    for b in range(test_batches):
        b_ids = ox.from_numpy(X_test_ids[b * batch_size : (b + 1) * batch_size])
        b_stylom = ox.from_numpy(X_test_stylom[b * batch_size : (b + 1) * batch_size])
        b_y = y_test_np[b * batch_size : (b + 1) * batch_size]

        logits = model(b_ids, b_stylom)
        preds = np.argmax(logits.to_numpy(), axis=1)
        all_preds.extend(preds)
        all_targets.extend(b_y)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Compute Metrics
    accuracy = np.mean(all_preds == all_targets) * 100.0
    tp = np.sum((all_preds == 1) & (all_targets == 1))
    fp = np.sum((all_preds == 1) & (all_targets == 0))
    tn = np.sum((all_preds == 0) & (all_targets == 0))
    fn = np.sum((all_preds == 0) & (all_targets == 1))

    precision = (tp / (tp + fp)) * 100.0 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"   =====================================================")
    print(f"   TEST SET EVALUATION RESULTS ({len(all_preds):,} samples):")
    print(f"     • Overall Accuracy : {accuracy:.2f}%")
    print(f"     • Precision (Fake) : {precision:.2f}%")
    print(f"     • Recall    (Fake) : {recall:.2f}%")
    print(f"     • F1-Score         : {f1:.2f}%")
    print(f"     • Confusion Matrix : [TN={tn}, FP={fp}] / [FN={fn}, TP={tp}]")
    print(f"   =====================================================")

    # 7. Save Model Weights & Vocabulary
    print(f"\n7. Saving trained model to disk ({save_path})...")
    meta = {
        "model_type": "FakeNewsClassifier",
        "dataset_size": "10000",
        "accuracy": f"{accuracy:.2f}%",
        "f1_score": f"{f1:.2f}%",
        "trained_epochs": str(epochs),
    }
    ox.save_file(model, save_path, metadata=meta)
    vocab_path = os.path.splitext(save_path)[0] + "_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write(pipeline.to_json())
    print(f"   Weights saved to     : {save_path} ({os.path.getsize(save_path) / 1024:.1f} KB)")
    print(f"   Vocabulary saved to  : {vocab_path}")

    # 8. Evaluation on Real-World Input Headlines (with eval input)
    print("\n8. Demonstrating 'eval_input' with Diverse Breaking Headlines:")

    eval_samples = [
        # Real News Excerpts
        "NASA astronomers published peer-reviewed findings confirming water ice plumes erupting from Jupiter's moon Europa.",
        "The Federal Reserve Board announced a 25 basis point benchmark interest rate adjustment following monthly labor statistics.",
        "Clinical researchers at Mayo Clinic concluded a five-year double-blind trial on pediatric immunotherapy tolerance rates.",
        "Department of Energy researchers demonstrated a 14 percent efficiency increase in perovskite photovoltaic solar cells.",

        # Wrong / Fake News Excerpts
        "SHOCKING BOMBSHELL: Rogue insider reveals drinking boiled lemon garlic kerosene permanently cures all viral cancer within 36 hours! SHARE BEFORE CENSORED!!!",
        "WHISTLEBLOWER LEAKS: Secret government treaty proves 5G cell towers emit psychoactive mind-control frequencies causing obedience!",
        "THEY DON'T WANT YOU TO KNOW: Undercover general confirms elite globalists secretly replaced world leaders with android clones!",
        "URGENT ALERT: Mysterious Antarctic ice wall breach exposes ancient reptilian stargate guarded by nuclear submarines!",
    ]

    for sample in eval_samples:
        eval_input(sample, model, pipeline, verbose=True)

    return model, pipeline


def interactive_eval_session(model_path: str = "fake_news_detector.safetensors"):
    """Loads a pre-trained model and launches an interactive eval loop for user input."""
    vocab_path = os.path.splitext(model_path)[0] + "_vocab.json"
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Model or vocabulary not found at '{model_path}'. Running training first...")
        model, pipeline = train_and_evaluate(save_path=model_path, epochs=4)
    else:
        print(f"Loading pre-trained model from '{model_path}'...")
        with open(vocab_path, "r", encoding="utf-8") as f:
            pipeline = NewsTextPipeline.from_json(f.read())
        model = FakeNewsClassifier(vocab_size=len(pipeline.vocab), embed_dim=64, hidden_dim=128)
        model.load_file(model_path)
        print("Model loaded successfully!")

    print("\n" + "=" * 70)
    print("  INTERACTIVE NEWS EVALUATION MODE (Type 'exit' or 'quit' to stop)")
    print("=" * 70)

    while True:
        try:
            user_input = input("\nEnter News Headline or Article excerpt: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting evaluation session. Goodbye!")
                break
            eval_input(user_input, model, pipeline, verbose=True)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Oxideon AI Wrong / Fake News Identifier (10k Dataset)")
    parser.add_argument("--save-path", type=str, default="fake_news_detector.safetensors", help="Path to save/load weights")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt for custom eval inputs")
    parser.add_argument("--eval-text", type=str, default=None, help="Directly evaluate a single headline string")

    args = parser.parse_args()

    if args.interactive:
        interactive_eval_session(args.save_path)
    elif args.eval_text:
        vocab_path = os.path.splitext(args.save_path)[0] + "_vocab.json"
        if os.path.exists(args.save_path) and os.path.exists(vocab_path):
            print(f"Loading pre-trained model from '{args.save_path}'...")
            with open(vocab_path, "r", encoding="utf-8") as f:
                pipeline = NewsTextPipeline.from_json(f.read())
            model = FakeNewsClassifier(vocab_size=len(pipeline.vocab), embed_dim=64, hidden_dim=128)
            model.load_file(args.save_path)
        else:
            model, pipeline = train_and_evaluate(save_path=args.save_path, epochs=args.epochs)
        print("\nEvaluating custom input:")
        eval_input(args.eval_text, model, pipeline, verbose=True)
    else:
        train_and_evaluate(save_path=args.save_path, epochs=args.epochs)


if __name__ == "__main__":
    main()

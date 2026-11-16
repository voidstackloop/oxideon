"""Downloads the tokenizer.json of public models into a scratch directory (not committed) for crossval batch 43.
Usage: py -3.11 tools/crossval/fetch_real_tokenizers.py <out_dir>"""
import os, sys, urllib.request

models = [
    "openai-community/gpt2", "google-bert/bert-base-uncased", "google-bert/bert-base-multilingual-cased", "FacebookAI/roberta-base",
    "google-t5/t5-small", "FacebookAI/xlm-roberta-base", "albert/albert-base-v2", "Qwen/Qwen2.5-0.5B", "microsoft/phi-2",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "NousResearch/Meta-Llama-3-8B", "deepseek-ai/deepseek-coder-1.3b-base", "EleutherAI/gpt-neox-20b",
    "openai/clip-vit-base-patch32", "google-bert/bert-base-cased", "facebook/bart-base", "bigscience/bloom-560m", "NousResearch/Llama-2-7b-hf",
    "google/electra-small-discriminator", "xlnet/xlnet-base-cased", "mistralai/Mistral-7B-v0.1", "Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-135M",
    "stabilityai/stablelm-2-1_6b", "google/flan-t5-small", "microsoft/deberta-v3-base", "camembert/camembert-base", "allenai/OLMo-1B-hf",
]
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
if "--legacy" in sys.argv:  # vocab.json / merges.txt / vocab.txt for the legacy-format test (target/legacy)
    leg = os.path.join(out, "..", "legacy"); os.makedirs(leg, exist_ok=True)
    for repo, tag, f in [("openai-community/gpt2", "gpt2", "vocab.json"), ("openai-community/gpt2", "gpt2", "merges.txt"), ("FacebookAI/roberta-base", "roberta", "vocab.json"),
                         ("FacebookAI/roberta-base", "roberta", "merges.txt"), ("google-bert/bert-base-uncased", "bert", "vocab.txt"), ("google-bert/bert-base-cased", "bertcased", "vocab.txt")]:
        dest = os.path.join(leg, f"{tag}_{f}")
        if not os.path.exists(dest): open(dest, "wb").write(urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/{f}", timeout=60).read()); print("ok  ", tag, f)
for m in models:
    dest = os.path.join(out, m.replace("/", "__") + ".json")
    if os.path.exists(dest): continue
    try:
        data = urllib.request.urlopen(f"https://huggingface.co/{m}/resolve/main/tokenizer.json", timeout=60).read()
        open(dest, "wb").write(data); print("ok  ", m, len(data))
    except Exception as e:
        print("skip", m, e)

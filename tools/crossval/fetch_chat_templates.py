"""Downloads tokenizer_config.json (and chat_template.jinja when present) of public chat models into target/chat_configs
for crossval batch 47. Usage: py -3.11 tools/crossval/fetch_chat_templates.py <out_dir>"""
import json, os, sys, urllib.request

models = [
    "NousResearch/Meta-Llama-3-8B-Instruct", "NousResearch/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "mistralai/Mistral-7B-Instruct-v0.3",
    "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "Qwen/Qwen3-0.6B", "HuggingFaceTB/SmolLM2-135M-Instruct",
    "microsoft/Phi-3-mini-4k-instruct", "microsoft/Phi-3.5-mini-instruct", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "teknium/OpenHermes-2.5-Mistral-7B",
    "HuggingFaceH4/zephyr-7b-beta", "openchat/openchat-3.5-0106", "deepseek-ai/deepseek-coder-1.3b-instruct", "01-ai/Yi-6B-Chat",
    "NousResearch/Hermes-3-Llama-3.1-8B", "allenai/OLMo-2-1124-7B-Instruct", "stabilityai/stablelm-2-zephyr-1_6b", "internlm/internlm2-chat-1_8b",
    "google/gemma-2-2b-it", "unsloth/gemma-2-2b-it", "unsloth/Llama-3.2-1B-Instruct", "unsloth/Llama-3.1-8B-Instruct", "NousResearch/Hermes-2-Pro-Mistral-7B",
    "Qwen/Qwen2-VL-2B-Instruct", "HuggingFaceTB/SmolLM3-3B", "unsloth/gpt-oss-20b", "mistralai/Mistral-Nemo-Instruct-2407", "CohereForAI/c4ai-command-r-v01",
]
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
for m in models:
    dest = os.path.join(out, m.replace("/", "__") + ".json")
    if os.path.exists(dest): continue
    try:
        cfg = json.loads(urllib.request.urlopen(f"https://huggingface.co/{m}/resolve/main/tokenizer_config.json", timeout=60).read())
    except Exception as e:
        print("skip", m, e); continue
    if not cfg.get("chat_template"):
        try: cfg["chat_template"] = urllib.request.urlopen(f"https://huggingface.co/{m}/resolve/main/chat_template.jinja", timeout=60).read().decode("utf-8")
        except Exception: pass
    if not cfg.get("chat_template"):
        try: cfg["chat_template"] = json.loads(urllib.request.urlopen(f"https://huggingface.co/{m}/resolve/main/chat_template.json", timeout=60).read())["chat_template"]
        except Exception: pass
    if not cfg.get("chat_template"): print("no template", m); continue
    json.dump(cfg, open(dest, "w", encoding="utf-8"), ensure_ascii=False); print("ok  ", m)

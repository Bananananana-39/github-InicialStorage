import os
import asyncio
import logging
import logging.config

import numpy as np
import openai

from lightrag import LightRAG, QueryParam
from lightrag.utils import logger, set_verbose_debug

# ─── Configuration ───────────────────────────────────────────────────────────
WORKING_DIR = "./medqa_qwen"
DATA_DIR = "/root/autodl-tmp/LightRAG/benchmark/medqa/corpus/mimic_ex/dataset"

# Set Qwen API key (replace with your actual key or export it as env var)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─── Logging setup ───────────────────────────────────────────────────────────
def configure_logging():
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file = os.path.join(log_dir, "medqa_qwen.log")
    os.makedirs(log_dir, exist_ok=True)

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": "%(levelname)s: %(message)s"},
            "detailed": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "stream": "ext://sys.stderr",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "detailed",
                "filename": log_file,
                "maxBytes": 10 * 1024 * 1024,
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        "loggers": {
            "lightrag": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False,
            },
        },
    })
    logger.setLevel(logging.INFO)
    set_verbose_debug(False)

# ─── Qwen embedding & completion ─────────────────────────────────────────────
async def qwen_embed(texts: list[str]) -> np.ndarray:
    response = await openai.Embedding.acreate(
        model="text-embedding-v1",
        input=texts
    )
    return np.vstack([d["embedding"] for d in response["data"]])

async def qwen_complete(prompt: str, param: QueryParam) -> str:
    response = await openai.ChatCompletion.acreate(
        model="qwen-plus",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=param.max_tokens or 512,
        temperature=param.temperature or 0.7,
    )
    return response.choices[0].message.content

# ─── LightRAG initialization ─────────────────────────────────────────────────
async def initialize_rag() -> LightRAG:
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=qwen_embed,
        llm_model_func=qwen_complete,
    )
    await rag.initialize_storages()
    return rag

# ─── Main logic ──────────────────────────────────────────────────────────────
async def main():
    configure_logging()

    if not openai.api_key:
        print("Please set OPENAI_API_KEY environment variable.")
        return

    if not os.path.isdir(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    rag = await initialize_rag()

    # Index 20 report files
    for i in range(20):
        report_path = os.path.join(DATA_DIR, f"report_{i}.txt")
        if os.path.exists(report_path):
            print(f"Inserting: {report_path}")
            with open(report_path, "r", encoding="utf-8") as f:
                await rag.ainsert(f.read())
        else:
            print(f"Warning: {report_path} not found.")

    print("✅ All reports inserted.")

    # Sample query
    question = "What are the main clinical themes across these reports?"
    for mode in ["naive", "local", "global", "hybrid"]:
        print(f"\n=== Mode: {mode} ===")
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        print(result)

    await rag.finalize_storages()
    print("\nDone.")

if __name__ == "__main__":
    asyncio.run(main())
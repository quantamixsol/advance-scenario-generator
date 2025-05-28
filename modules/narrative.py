import os
import logging
from typing import List
import requests
import streamlit as st
import torch
from transformers import pipeline as hf_pipeline
from openai import OpenAI
from vertexai.preview.generative_models import GenerativeModel
from dotenv import load_dotenv

load_dotenv()

OPENAI_CLIENT = OpenAI()
GEMINI_PRO_MODEL = "gemini-2.5-pro-preview-03-25"

_T5_PIPES: dict[str, any] = {}

def _get_t5(size: str):
    if size not in _T5_PIPES:
        model = f"google/flan-t5-{size}"
        _T5_PIPES[size] = hf_pipeline(
            "text2text-generation",
            model=model,
            device=0 if torch.cuda.is_available() else -1,
        )
    return _T5_PIPES[size]


@st.cache_data(show_spinner=False)
def generate_raw_narrative(prompt: str, engine: str = "t5-small") -> str:
    """Return raw LLM output for prompt using selected engine."""
    if engine.startswith("t5-"):
        size = engine.split("-", 1)[1]
        pipe = _get_t5(size)
        return pipe(prompt, max_length=512, truncation=True)[0]["generated_text"].strip()

    if engine == "gpt-4":
        resp = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial scenario expert."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    if engine.startswith("gemini"):
        model = GenerativeModel(GEMINI_PRO_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 1024,
                "temperature": 0.7,
                "top_p": 1,
                "top_k": 32,
            },
        )
        return resp.candidates[0].content.parts[0].text.strip()

    raise ValueError(f"Unknown engine '{engine}'")


@st.cache_data(show_spinner=False)
def augment_with_news(narrative: str, tickers: List[str]) -> str:
    """Prepend latest headlines for tickers via NewsAPI."""
    api_key = os.getenv("NEWSAPI_KEY")
    headlines: List[str] = []
    if api_key and tickers:
        q = " OR ".join(tickers)
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": q,
            "apiKey": api_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 5,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                data = r.json()
                for art in data.get("articles", [])[:5]:
                    title = art.get("title")
                    if title:
                        headlines.append(title.strip())
            else:
                logging.warning("NewsAPI error %s", r.text)
        except Exception as e:
            logging.warning("NewsAPI request failed: %s", e)

    if headlines:
        update = "## Market Update\n" + "\n".join(f"- {h}" for h in headlines) + "\n\n"
        return update + narrative

    return narrative

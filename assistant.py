# -*- coding: utf-8 -*-
""""تعليمات الحج والعمرة" — the specialized Hajj & Umrah AI assistant (v15).

A ChatGPT-style assistant, but locked to ONE domain: Hajj, Umrah, their
rituals, and every pilgrim-facing service at the two Holy Mosques. Anything
else gets a polite redirect instead of an answer.

Design goals (from the product spec):
  * Answer ONLY questions about Hajj/Umrah rituals & Haramain services;
    politely decline anything else and ask for an on-topic question.
  * Never guess: lean on a small curated knowledge base (knowledge_base.py)
    for grounding, cite the category of official source responsible
    (Ministry of Hajj & Umrah / the Grand Mosque & Prophet's Mosque
    presidency / Nusuk platform), and say so plainly when a detail isn't
    confirmed rather than inventing it.
  * Structured, professional answers: a heading, a short intro, ordered
    points/steps, important warnings, shar'i rulings when relevant (with a
    pointer to official Ifta offices for personal rulings), correct
    duas/adhkar, service locations, and a "related info" close.

Tiering mirrors ai_pipeline.py: reuses ANTHROPIC_API_KEY / OPENAI_API_KEY
(same env vars — no extra configuration) for the highest-quality answers;
without a key, falls back to a template built from knowledge_base.py so the
page is never empty/broken, just less conversational.
"""
import json
import os
import re

import requests

import knowledge_base

LLM_TIMEOUT = 30
MAX_HISTORY_MESSAGES = 12  # keep the request small & the assistant focused

LANG_NAMES = {
    "ar": "Arabic", "en": "English", "tr": "Turkish", "ur": "Urdu",
    "hi": "Hindi", "he": "Hebrew",
}


def _system_prompt(lang: str, kb_context: str) -> str:
    lang_name = LANG_NAMES.get(lang, "Arabic")
    prompt = f"""You are "تعليمات الحج والعمرة" (Hajj & Umrah Guidance), a specialized assistant \
inside a Hajj & Umrah pilgrim-feedback platform. You help pilgrims and prospective pilgrims with \
Hajj, Umrah, and Haramain (the two Holy Mosques) services ONLY.

IN-SCOPE topics (answer these): Hajj rituals, Umrah rituals, Ihram, Tawaf, Sa'i, standing at \
Arafah, Muzdalifah, Mina, stoning the Jamarat, the Hady (sacrifice), shaving/trimming, Tawaf \
al-Ifadah, Tawaf al-Wada', the Miqats, Ihram prohibitions, Fidyah, du'as and adhkar, shar'i \
rulings related to Hajj/Umrah, Grand Mosque services, Prophet's Mosque services, Ifta (fatwa) \
offices, guidance offices, lesson/lecture locations, Qur'an circles, restrooms, ablution areas, \
gates, prayer areas, elderly/disability carts, first aid, health centers, lost & found, crowd \
management, transport services, official Hajj/Umrah apps (e.g. Nusuk), and any other service for \
pilgrims within Makkah, Madinah, or the sacred sites (al-Masha'ir al-Muqaddasah).

OUT OF SCOPE (politely decline): anything unrelated to Hajj/Umrah/Haramain services — general \
chit-chat, coding, unrelated travel, politics, other religions' rituals, etc. When a question is \
out of scope, apologize briefly and ask the person to ask something about Hajj or Umrah instead. \
Do NOT answer the off-topic question even partially.

RELIABILITY RULES (critical):
- Never invent rulings, prices, phone numbers, exact locations, or dates. If you are not certain, \
say so plainly and suggest the person confirm with an official source.
- Ground your answers in what is well-established; official sources you may refer to by name are: \
{knowledge_base.OFFICIAL_SOURCES_AR} ({knowledge_base.OFFICIAL_SOURCES_EN}).
- For personal shar'i rulings (e.g. "did I do X correctly", fidyah for a specific situation), give \
the general rule and explicitly recommend confirming with an official Ifta/guidance office rather \
than issuing a personal fatwa yourself.
- Only give du'as/adhkar you are confident are authentic and correctly worded; if unsure of exact \
wording, describe the general content instead of inventing wording.

ANSWER FORMAT (use Markdown):
- A short bold heading line.
- A one-to-two sentence introduction.
- Organized bullet points, and numbered sequential steps when the answer describes a procedure.
- An "⚠️ تنبيه مهم" / "⚠️ Important" callout for any important warning, if relevant.
- Shar'i rulings when relevant, phrased carefully per the reliability rules above.
- Correct du'as/adhkar when relevant.
- Service locations when relevant.
- End with one short line suggesting a related follow-up topic.

LANGUAGE: Respond in {lang_name}. If the person's message is written in a different language, \
respond in the language they used instead.

{("GROUNDING CONTEXT (verified reference material — prefer this over your own memory when it " \
"applies; it may be partial or empty):\n" + kb_context) if kb_context else ""}"""
    return prompt


def _heuristic_in_scope(text: str) -> bool:
    hay = (text or "").lower()
    return any(kw.lower() in hay for kw in knowledge_base.TOPIC_WORDS)


def llm_configured() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return ""


def _call_anthropic(system_prompt: str, messages: list) -> str:
    key = os.environ["ANTHROPIC_API_KEY"]
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={
            "model": os.environ.get("LLM_MODEL", "claude-haiku-4-5"),
            "max_tokens": 1000,
            "system": system_prompt,
            "messages": messages,
        },
        timeout=LLM_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()
    return "".join(b.get("text", "") for b in data.get("content", [])).strip()


def _call_openai(system_prompt: str, messages: list) -> str:
    key = os.environ["OPENAI_API_KEY"]
    oa_messages = [{"role": "system", "content": system_prompt}] + messages
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "model": os.environ.get("LLM_MODEL", "gpt-4o-mini"),
            "max_tokens": 1000,
            "messages": oa_messages,
        },
        timeout=LLM_TIMEOUT,
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


def _kb_fallback_answer(question: str, lang: str) -> dict:
    """No LLM key configured: answer directly from the knowledge base
    (still grounded, still on-topic-only — just not conversational)."""
    if not _heuristic_in_scope(question):
        msg = {
            "ar": "أستطيع الإجابة فقط عن أسئلة متعلقة بالحج والعمرة وخدمات الحرمين الشريفين. "
                  "تفضل بطرح سؤال في هذا النطاق 🙏",
            "en": "I can only answer questions about Hajj, Umrah, and services at the two Holy "
                  "Mosques. Please ask something in that scope 🙏",
        }.get(lang, None) or {
            "ar": "أستطيع الإجابة فقط عن أسئلة متعلقة بالحج والعمرة وخدمات الحرمين الشريفين.",
        }["ar"]
        return {"reply": msg, "engine": "scope-guard", "out_of_scope": True}

    hits = knowledge_base.retrieve(question, limit=2)
    if not hits:
        msg = {
            "ar": "هذا سؤال متعلق بالحج والعمرة، لكن لا تتوفر لديّ حاليًا معلومة موثوقة كافية "
                  "للإجابة عليه بدقة (لم يتم تفعيل نموذج ذكاء اصطناعي بعد). "
                  f"يُرجى مراجعة {knowledge_base.OFFICIAL_SOURCES_AR} للتأكد.",
            "en": "This is a Hajj/Umrah question, but I don't have enough verified information to "
                  "answer it precisely right now (no AI model is configured yet). Please check "
                  f"{knowledge_base.OFFICIAL_SOURCES_EN}.",
        }.get(lang, None) or (
            "هذا سؤال متعلق بالحج والعمرة، لكن لا تتوفر لديّ حاليًا معلومة موثوقة كافية للإجابة "
            f"عليه بدقة. يُرجى مراجعة {knowledge_base.OFFICIAL_SOURCES_AR}."
        )
        return {"reply": msg, "engine": "kb-fallback-empty", "out_of_scope": False}

    use_ar = lang != "en"
    parts = []
    for e in hits:
        title = e["title_ar"] if use_ar else e["title_en"]
        body = e["body_ar"] if use_ar else e["body_en"]
        parts.append(f"**{title}**\n\n{body}")
    footer = (
        f"\n\n_هذه معلومات عامة من قاعدة معرفة داخلية. للحصول على إجابات أكثر تفصيلًا وسياقًا، "
        f"يمكن لمسؤول الموقع تفعيل الذكاء الاصطناعي التوليدي عبر إضافة مفتاح API. للتأكد من "
        f"التفاصيل الدقيقة راجع {knowledge_base.OFFICIAL_SOURCES_AR}._"
        if use_ar else
        f"\n\n_This is general information from an internal knowledge base. For more detailed, "
        f"conversational answers, the site admin can enable the generative AI tier by adding an "
        f"API key. For precise details, check {knowledge_base.OFFICIAL_SOURCES_EN}._"
    )
    return {"reply": "\n\n---\n\n".join(parts) + footer, "engine": "kb-fallback", "out_of_scope": False}


def answer(history: list, lang: str = "ar") -> dict:
    """history: list of {"role": "user"|"assistant", "content": str}, oldest
    first, ending with the newest user message. Returns
    {"reply": str, "engine": str, "out_of_scope": bool}. Never raises."""
    history = [h for h in (history or [])
               if h.get("role") in ("user", "assistant") and (h.get("content") or "").strip()]
    if not history or history[-1]["role"] != "user":
        return {"reply": "", "engine": "none", "out_of_scope": False}
    history = history[-MAX_HISTORY_MESSAGES:]
    last_question = history[-1]["content"]

    provider = llm_configured()
    if provider:
        try:
            kb_ctx = knowledge_base.context_block(last_question, lang=lang, limit=3)
            system_prompt = _system_prompt(lang, kb_ctx)
            messages = [{"role": h["role"], "content": h["content"]} for h in history]
            if provider == "anthropic":
                text = _call_anthropic(system_prompt, messages)
            else:
                text = _call_openai(system_prompt, messages)
            if text:
                return {"reply": text, "engine": f"llm-{provider}", "out_of_scope": False}
        except Exception as e:
            print(f"[assistant] LLM call failed, falling back to KB: {e}")
    return _kb_fallback_answer(last_question, lang)


def status() -> dict:
    return {"llm_provider": llm_configured(),
            "llm_model": os.environ.get("LLM_MODEL") or
                         ("claude-haiku-4-5" if llm_configured() == "anthropic"
                          else "gpt-4o-mini" if llm_configured() == "openai" else None),
            "kb_topics": len(knowledge_base.KB)}

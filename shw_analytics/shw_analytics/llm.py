"""Thin client for the LTIC AI Gateway (OpenAI-compatible).

Check connectivity and the real model ids with:
    python -m shw_analytics.llm --models
    python -m shw_analytics.llm --ping
"""
import json
import re
import sys

from openai import OpenAI, BadRequestError

from . import config

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_client = None


def client():
    global _client
    if _client is None:
        if not config.API_KEY:
            raise RuntimeError("Set LTIC_API_KEY in your environment (never commit it).")
        _client = OpenAI(base_url=config.GATEWAY_URL, api_key=config.API_KEY)
    return _client


def _extra():
    if config.DISABLE_THINKING:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def strip_think(text):
    return _THINK.sub("", text or "").strip()


def chat(messages, model=None, tools=None, temperature=0.1, max_tokens=None):
    """Returns the raw choice.message. Retries once without extra_body if the
    gateway rejects the thinking switch."""
    kwargs = dict(model=model or config.CHAT_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens or config.MAX_TOKENS)
    if tools:
        kwargs["tools"] = tools
    extra = _extra()
    try:
        resp = client().chat.completions.create(**kwargs, extra_body=extra) if extra \
            else client().chat.completions.create(**kwargs)
    except BadRequestError as e:
        if extra and "tools" not in str(e).lower():
            config.DISABLE_THINKING = False
            resp = client().chat.completions.create(**kwargs)
        else:
            raise
    msg = resp.choices[0].message
    msg.content = strip_think(msg.content)
    msg.finish_reason = resp.choices[0].finish_reason  # "length" = ran out of tokens
    return msg


def parse_json(text):
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    text = strip_think(text).replace("```json", "```")
    if "```" in text:
        text = text.split("```")[1] if text.count("```") >= 2 else text.replace("```", "")
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        raise ValueError(f"No JSON in reply: {text[:200]}")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    return obj


def chat_json(system, user, model=None, retries=2):
    messages = [{"role": "system", "content": system + "\nReply with JSON only, no prose."},
                {"role": "user", "content": user}]
    last = None
    for _ in range(retries + 1):
        reply = chat(messages, model=model).content
        try:
            return parse_json(reply)
        except (ValueError, json.JSONDecodeError) as e:
            last = e
            messages += [{"role": "assistant", "content": reply},
                         {"role": "user", "content": "That was not valid JSON. Reply with the JSON only."}]
    raise last


def embed(texts, model=None):
    resp = client().embeddings.create(model=model or config.EMBED_MODEL, input=list(texts))
    return [d.embedding for d in resp.data]


if __name__ == "__main__":
    if "--models" in sys.argv:
        for m in client().models.list().data:
            print(m.id)
    elif "--ping" in sys.argv:
        print(chat([{"role": "user", "content": "Reply with the word OK."}]).content)
        try:
            tools = [{"type": "function", "function": {"name": "noop", "description": "test",
                      "parameters": {"type": "object", "properties": {}}}}]
            m = chat([{"role": "user", "content": "Call the noop tool."}], tools=tools)
            print("tool calling:", "SUPPORTED" if m.tool_calls else "not used (try SHW_TOOL_MODE=json)")
        except BadRequestError as e:
            print("tool calling: REJECTED by gateway -> set SHW_TOOL_MODE=json\n", e)

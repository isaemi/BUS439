"""Turns on the gateway usage log (and repairs the earlier version that needed httpx).
Run once from the project folder:
    python scripts/install_usage_log.py

Changes only the lines in shw_analytics/llm.py that create the gateway client, so that each
request's tokens and reported cost are written to data/usage_log.csv (never any content).
No extra library is needed. Your other changes to llm.py are kept. Safe to re-run.
"""
import re
import sys
from pathlib import Path

LLM = Path("shw_analytics/llm.py")
PLAIN = "_client = OpenAI(base_url=config.GATEWAY_URL, api_key=config.API_KEY)"


def new_block(indent):
    return (f"{indent}{PLAIN}\n{indent}from . import usage\n"
            f"{indent}usage.attach(_client)     # logs tokens and cost per request (never content); see usage.py\n")


def main():
    if not LLM.exists() or not Path("shw_analytics/usage.py").exists():
        print("NOT INSTALLED: run from the project folder, with shw_analytics/usage.py in place.")
        sys.exit(1)
    s = LLM.read_text()
    if "usage.attach(_client)" in s:
        print("Already installed; nothing to do.")
        return
    old = re.compile(r"(?m)^(\s*)import httpx\n(?:.*\n)*?.*event_hooks=.*\n")       # the earlier version
    plain = re.compile(r"(?m)^(\s*)" + re.escape(PLAIN) + r"[ \t]*\n")
    if len(old.findall(s)) == 1:
        s = old.sub(lambda m: new_block(m.group(1)), s, count=1)
    elif len(plain.findall(s)) == 1:
        s = plain.sub(lambda m: new_block(m.group(1)), s, count=1)
    else:
        print("NOT INSTALLED: could not find the lines that create the gateway client in llm.py. Nothing was changed.")
        sys.exit(1)
    compile(s, str(LLM), "exec")
    LLM.write_text(s)
    print("Installed: gateway requests are logged to data/usage_log.csv, with no extra library needed.")
    print("See totals any time with:  python -m shw_analytics.usage")


if __name__ == "__main__":
    main()

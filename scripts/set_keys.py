"""Write API keys into this project's .env without showing them on screen.

Run from the carecall-agent folder:  python scripts/set_keys.py
Paste each key when asked (it stays hidden) and press Enter. Press Enter alone to skip a key.
"""
import getpass
import re
from pathlib import Path

ENV = Path(__file__).resolve().parents[1] / ".env"
KEYS = ["NEBIUS_API_KEY", "ELEVENLABS_API_KEY", "PINECONE_API_KEY"]

text = ENV.read_text(encoding="utf-8") if ENV.exists() else ""
for name in KEYS:
    value = getpass.getpass(f"Paste {name} (hidden, Enter to skip): ").strip().strip('"').strip("'")
    if not value:
        print(f"  {name}: skipped")
        continue
    line = f"{name}={value}"
    if re.search(rf"^{name}=.*$", text, flags=re.M):
        text = re.sub(rf"^{name}=.*$", lambda m: line, text, flags=re.M)
    else:
        text = text.rstrip("\n") + "\n" + line + "\n"
    ENV.write_text(text, encoding="utf-8")
    print(f"  {name}: saved to .env ({len(value)} characters)")
print(f"Done. Keys are in {ENV}")

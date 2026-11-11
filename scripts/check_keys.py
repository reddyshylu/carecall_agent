"""Show whether each key in .env is set (never prints the values)."""
from pathlib import Path
ENV = Path(__file__).resolve().parents[1] / ".env"
print("Checking:", ENV)
for line in ENV.read_text(encoding="utf-8").splitlines():
    s = line.strip()
    if "=" in s and not s.startswith("#") and s.split("=", 1)[0].endswith("API_KEY"):
        k, v = s.split("=", 1)
        print(f"  {k}: {'NOT SET' if (not v or v.startswith('replace')) else f'set ({len(v)} characters)'}")

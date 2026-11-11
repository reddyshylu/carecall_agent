"""Voice Caller agent: an ElevenLabs voice agent that makes the appointment-reminder call.

How a call runs (no phone line needed, so no Twilio cost):
  1. The app asks ElevenLabs for a short-lived signed URL (your API key never reaches the browser).
  2. A tiny local web page (127.0.0.1 only) opens in a new browser tab. Streamlit's own frames can
     block the microphone, a normal tab does not. The page starts the conversation with ONLY the
     appointment variables (first name, hospital, physician, purpose, date, time).
  3. When the call ends, the page posts the transcript back to the local server; the app can also
     fetch it from the ElevenLabs API by conversation id.
If ElevenLabs is not set up or fails, the nurse can call herself and paste the transcript, or
use a sample transcript for the demo.
"""
from __future__ import annotations

import json
import secrets
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import config

API = "https://api.elevenlabs.io/v1"
CALL_PORT = 8765
JS_CLIENT = "https://unpkg.com/@elevenlabs/client@1.29.0/dist/lib.iife.js"


def configured() -> bool:
    key, agent = config.setting("ELEVENLABS_API_KEY"), config.setting("CARECALL_VOICE_AGENT_ID")
    return bool(key) and not key.startswith(("replace", "your_")) and bool(agent) and not agent.startswith(("replace", "your_"))


def _get(path: str, params: dict | None = None, timeout: int = 20) -> dict:
    url = f"{API}{path}" + (("?" + urllib.parse.urlencode(params)) if params else "")
    req = urllib.request.Request(url, headers={"xi-api-key": config.setting("ELEVENLABS_API_KEY")})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def signed_url() -> str:
    return _get("/convai/conversation/get-signed-url", {"agent_id": config.setting("CARECALL_VOICE_AGENT_ID")})["signed_url"]


def fetch_transcript(conversation_id: str) -> dict:
    """Transcript and status of one conversation from the ElevenLabs API."""
    data = _get(f"/convai/conversations/{urllib.parse.quote(conversation_id)}")
    turns = [{"role": "agent" if t.get("role") == "agent" else "user", "message": t.get("message") or ""}
             for t in data.get("transcript", []) if (t.get("message") or "").strip()]
    return {"status": data.get("status"), "transcript": turns,
            "analysis": (data.get("analysis") or {}).get("data_collection_results", {})}


# ------------------------------------------------------------------ local call page
_CALLS: dict[str, dict] = {}       # token -> {"vars":..., "signed_url":..., "result":...}
_SERVER = None
_LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Hospital A reminder call (demo)</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font-family:Arial,sans-serif;max-width:640px;margin:40px auto;padding:0 16px;background:#f6f8fb;color:#1c2733}
button{font-size:18px;padding:12px 22px;border-radius:8px;border:0;margin-right:8px;cursor:pointer}
#start,#k1{background:#1f6f54;color:#fff}#k2{background:#7a4b00;color:#fff}#stop{background:#b42318;color:#fff}.box{background:#fff;border-radius:10px;padding:16px;margin-top:16px}
.agent{color:#1f4e79}.user{color:#1f6f54}small{color:#667}</style></head><body>
<h2>Reminder call from Hospital A (demo)</h2>
<p><small>You are playing the patient. Synthetic data only. Allow the microphone when the browser asks.</small></p>
<button id="start">Start call</button><button id="stop" disabled>End call</button>
<p><button id="k1" disabled>Press 1 (Yes)</button><button id="k2" disabled>Press 2 (No)</button></p>
<p id="status">Ready.</p><div class="box" id="log"></div>
<script src="__JS__"></script>
<script>
const TOKEN="__TOKEN__", VARS=__VARS__, SIGNED="__SIGNED__";
let conv=null, convId=null, turns=[];
const log=(r,m)=>{turns.push({role:r,message:m});const p=document.createElement('p');p.className=r;p.textContent=(r==='agent'?'Assistant: ':'Patient: ')+m;document.getElementById('log').appendChild(p);};
async function finish(){document.getElementById('status').textContent='Call ended. Saving transcript...';
 await fetch('/done/'+TOKEN,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({conversation_id:convId,transcript:turns})});
 document.getElementById('status').textContent='Saved. Go back to the CareCall app and press "Get call result".';}
document.getElementById('start').onclick=async()=>{
 try{await navigator.mediaDevices.getUserMedia({audio:true});
  conv=await ElevenLabsClient.Conversation.startSession({signedUrl:SIGNED,dynamicVariables:VARS,
   onMessage:(m)=>log(m.source==='ai'?'agent':'user',m.message),
   onStatusChange:(s)=>document.getElementById('status').textContent='Status: '+(s.status||s),
   onDisconnect:()=>{document.getElementById('stop').disabled=true;finish();},
   onError:(e)=>{document.getElementById('status').textContent='Error: '+e;}});
  convId=conv.getId();document.getElementById('start').disabled=true;document.getElementById('stop').disabled=false;
  document.getElementById('k1').disabled=false;document.getElementById('k2').disabled=false;
 }catch(e){document.getElementById('status').textContent='Could not start: '+e;}};
document.getElementById('stop').onclick=async()=>{if(conv){await conv.endSession();}};
const press=(d)=>{if(conv){conv.sendUserMessage(d);log('user',d);}};
document.getElementById('k1').onclick=()=>press('1');document.getElementById('k2').onclick=()=>press('2');
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the terminal quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        token = self.path.rsplit("/", 1)[-1]
        call = _CALLS.get(token)
        if not self.path.startswith("/call/") or not call:
            return self._send(404, b"Unknown or expired call link.")
        page = (PAGE.replace("__JS__", JS_CLIENT).replace("__TOKEN__", token)
                .replace("__VARS__", json.dumps(call["vars"])).replace("__SIGNED__", call["signed_url"]))
        self._send(200, page.encode())

    def do_POST(self):
        token = self.path.rsplit("/", 1)[-1]
        call = _CALLS.get(token)
        if not self.path.startswith("/done/") or not call:
            return self._send(404, b"{}", "application/json")
        length = min(int(self.headers.get("Content-Length", 0)), 200_000)
        try:
            call["result"] = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            call["result"] = {"transcript": [], "conversation_id": None}
        call["signed_url"] = ""            # one use only
        self._send(200, b"{}", "application/json")


def _ensure_server() -> None:
    global _SERVER
    with _LOCK:
        if _SERVER is None:
            _SERVER = ThreadingHTTPServer(("127.0.0.1", CALL_PORT), _Handler)
            threading.Thread(target=_SERVER.serve_forever, daemon=True).start()


def open_call(variables: dict) -> tuple[str, str]:
    """Prepare one call and return the local link the nurse opens in a new tab."""
    _ensure_server()
    token = secrets.token_urlsafe(16)
    _CALLS[token] = {"vars": variables, "signed_url": signed_url(), "result": None}
    return token, f"http://127.0.0.1:{CALL_PORT}/call/{token}"


def call_result(token: str) -> dict | None:
    """The transcript posted by the call page, completed from the API when possible."""
    call = _CALLS.get(token)
    if not call or not call.get("result"):
        return None
    res = call["result"]
    if res.get("conversation_id"):
        try:
            api = fetch_transcript(res["conversation_id"])
            if api["transcript"]:
                return {"conversation_id": res["conversation_id"], "transcript": api["transcript"], "source": "voice (ElevenLabs API)"}
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            pass
    return {"conversation_id": res.get("conversation_id"), "transcript": res.get("transcript", []), "source": "voice (browser)"}

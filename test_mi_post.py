"""Quick test: can the managed identity post to Teams via Graph API?"""
import json
import urllib.request
import urllib.error

try:
    from azure.identity import DefaultAzureCredential
    token = DefaultAzureCredential().get_token("https://graph.microsoft.com/.default").token
    print("TOKEN OK:", token[:30])
except Exception as e:
    print("TOKEN FAIL:", e)
    raise SystemExit(1)

GROUP = "6d55cb22-b8b0-43a4-8ec1-f5df8a966856"
CHAN  = "19:fd1d6a7c24a943fca27378d2272cf65a@thread.tacv2"
url   = f"https://graph.microsoft.com/v1.0/teams/{GROUP}/channels/{CHAN}/messages"

payload = json.dumps({"body": {"contentType": "text", "content": "Managed Identity test post from ACA"}}).encode()
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}, method="POST")

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
        print("SUCCESS! Message ID:", data.get("id"))
except urllib.error.HTTPError as e:
    body = e.read().decode()[:500]
    print(f"HTTP ERROR {e.code}: {e.reason}")
    print(body)
except Exception as e:
    print("FAIL:", e)

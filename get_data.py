"""Download the 7 Delhi CPCB station CSVs (2024-25) from data.opencity.in into data/raw/.
Already-downloaded files are skipped."""
import pathlib

import requests

BASE = "https://data.opencity.in/dataset/0dc7b9fe-9fd4-46ee-a37e-88f0bd6f6362/resource"
FILES = {
    "dtu":      ("3abd2c56-3523-4802-8633-f37055d3b276", "del-dtu-cpcb-2024-25.csv"),
    "ito":      ("890f786d-fb9f-475e-8516-191bfa1b01ea", "del-ito-cpcb-2024-25.csv"),
    "sirifort": ("e562d0c9-cc37-4d5a-a04e-fd95ee9f2f04", "del-sirifort-cpcb-2024-25.csv"),
    "nsit":     ("f7d16626-68d5-4de3-b6e6-e6a1b8831db3", "del-nsit-dwarka-cpcb-2024-25.csv"),
    "ihbas":    ("4cc723ba-bbd1-459f-b4b6-658107defa65", "del-ihbas-cpcb-2024-25.csv"),
    "wazirpur": ("70bfe4b2-b869-4f1c-9c25-f4b768858c1c", "del-wazirpur-dpcc-2024-25.csv"),
    "shadipur": ("e91c68ea-85ac-49f8-8015-dd8b04c32a71", "del-shadipur-cpcb-2024-25.csv"),
}

out = pathlib.Path("data/raw")
out.mkdir(parents=True, exist_ok=True)
for name, (rid, fn) in FILES.items():
    dest = out / fn
    if dest.exists():
        continue
    r = requests.get(f"{BASE}/{rid}/download/{fn}", timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(name, len(r.content) // 1024, "KB")

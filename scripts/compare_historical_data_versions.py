import subprocess, hashlib, tempfile, pathlib
commits = ["HEAD", "e2dee2b", "01501a8", "58d0c6c"]
for c in commits:
    for ins in ["xagusd", "xauusd", "eurusd"]:
        p = f"data/raw/dukascopy/{ins}_1m.csv"
        r = subprocess.run(["git", "show", f"{c}:{p}"], capture_output=True)
        if r.returncode:
            print(c, ins, "not in commit"); continue
        b = r.stdout
        lines = b.count(b"\n") - 1
        first = b.split(b"\n")[1][:25]
        last = b.rstrip().split(b"\n")[-1][:25]
        print(c, ins, lines, first, last, hashlib.sha256(b).hexdigest()[:12])
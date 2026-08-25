"""Embed the final PhantomWiki store, t-SNE to 2D, pick verified annotations."""
import json, re, collections
from pathlib import Path
import numpy as np
from chromadb.utils import embedding_functions
from sklearn.manifold import TSNE
import importlib.util
spec = importlib.util.spec_from_file_location("pwq", "scripts/pw_index_quality.py")
pwq = importlib.util.module_from_spec(spec); spec.loader.exec_module(pwq)

FINAL = "runs/pw1_main/kb_epoch_2.json"
store = json.load(open(FINAL))["store"]
nodes = store["nodes"]
seeds = set(store["origins"])
byid = {n["id"]: n for n in nodes}
ids = [n["id"] for n in nodes]
texts = [n["text"] for n in nodes]
print(f"{len(nodes)} documents")

ef = embedding_functions.DefaultEmbeddingFunction()
vecs = []
for i in range(0, len(texts), 512):
    vecs.extend(ef(texts[i:i+512]))
    print(f"  embedded {min(i+512,len(texts))}/{len(texts)}", flush=True)
X = np.asarray(vecs, dtype=np.float32)
print("t-SNE ...", flush=True)
xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
          max_iter=1000).fit_transform(X)
json.dump({i:[float(a),float(b)] for i,(a,b) in zip(ids,xy)},
          open("/tmp/pwnet_pos.json","w"))

# pick the highest-degree 100%-precision index per family
NAME = pwq.NAME
best = {}
for n in nodes:
    if n.get("flag") != "authored" or not n.get("links"): continue
    mem, fam = pwq.members(n["text"])
    if not mem: continue
    tgt = [set(NAME.findall(byid[t]["text"])) if t in byid else set() for t in n["links"]]
    if all(s & mem for s in tgt):
        if fam not in best or len(n["links"]) > len(best[fam]["links"]):
            best[fam] = n
ann = {}
for fam, n in best.items():
    ann[n["id"]] = {"text": n["text"].rstrip("."),
                    "note": f"{fam}, {len(n['links'])}/{len(n['links'])} correct",
                    "links": n["links"]}
    print("annotate:", fam, "->", n["text"][:50], len(n["links"]), "links")
json.dump(ann, open("/tmp/pwnet_annot.json","w"))
print("done")

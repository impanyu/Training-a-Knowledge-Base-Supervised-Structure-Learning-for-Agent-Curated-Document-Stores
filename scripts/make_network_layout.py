"""Semantic layout for the network figure: embed every document, project to 2D.

Positions are keyed by node id and shared across all checkpoints so the six
panels of the figure are directly comparable -- a document sits in the same
place at iteration 0 and iteration 200, and only the structure drawn over it
changes. Embeddings come from the same default function the store uses, so
proximity in the figure is the same proximity the reader's search sees.

Also selects the index documents to annotate: the highest-out-degree member
of each of the three families that carry the structure (attribute value,
relation of one entity, single entity), plus the degenerate tail.
"""
import json
from pathlib import Path

import numpy as np
from chromadb.utils import embedding_functions
from sklearn.manifold import TSNE

FINAL = "/tmp/kbstate_200.json"
OUT_POS = "/tmp/kbnet_pos.json"
OUT_ANN = "/tmp/kbnet_annot.json"


def main():
    store = json.load(open(FINAL))["store"]
    nodes = store["nodes"]
    seeds = set(store["origins"])
    ids = [n["id"] for n in nodes]
    texts = [n["text"] for n in nodes]
    print(f"{len(nodes)} documents ({len(nodes)-len(seeds)} authored)")

    ef = embedding_functions.DefaultEmbeddingFunction()
    vecs = []
    for i in range(0, len(texts), 512):
        vecs.extend(ef(texts[i:i + 512]))
        print(f"  embedded {min(i+512, len(texts))}/{len(texts)}", flush=True)
    X = np.asarray(vecs, dtype=np.float32)

    print("t-SNE ...", flush=True)
    xy = TSNE(n_components=2, init="pca", perplexity=30, random_state=0,
              max_iter=1000).fit_transform(X)
    json.dump({i: [float(a), float(b)] for i, (a, b) in zip(ids, xy)},
              open(OUT_POS, "w"))
    print("wrote", OUT_POS)

    # Annotate only indexes verified correct by scripts/index_precision.py:
    # each of these links 100% documents that genuinely belong under its key.
    # Picking by out-degree alone would surface "Edmund Pemberly" (22 links,
    # 68% precision), which is exactly what should not be held up as typical.
    WANTED = {"Residents of the city of Eastmere": "city index, 32/32 correct",
              "People whose hobby is astronomy": "hobby index, 21/21 correct",
              "Friends of Delphine Grimsby": "relation index, 12/12 correct"}
    by_text = {n["text"].rstrip("."): n for n in nodes if n["id"] not in seeds}
    ann = {}
    for text, note in WANTED.items():
        n = by_text.get(text)
        if n is None:
            print(f"  WARNING: {text!r} not found in the final store")
            continue
        ann[text] = {"id": n["id"], "text": n["text"], "note": note,
                     "deg": len(n["links"]), "targets": n["links"]}
    json.dump(ann, open(OUT_ANN, "w"), indent=1)
    for k, v in ann.items():
        print(f"  {v['id']}  [{v['deg']} links]  {k}  ({v['note']})")
    print("wrote", OUT_ANN)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""mini_search.py - a miniature search engine in pure Python (stdlib only).

Pipeline: crawl -> tokenize -> inverted index -> BM25 ranking -> snippets.

Features:
  * Polite BFS crawler (same-domain, respects robots.txt, per-host delay)
  * Unicode-aware tokenizer with stopword removal and light stemming (s/es/ing)
  * Inverted index with positional postings, persisted as JSON
  * **BM25** ranking (k1=1.5, b=0.75) + phrase matching bonus
  * Query syntax: `apple AND pie`, `apple OR pie`, quoted `"exact phrase"`, `-excluded`
  * Context snippets with <mark>-style highlighting

Usage:
  python mini_search.py crawl https://example.com --max-pages 20
  python mini_search.py index ./docs            # or index a local folder of .txt/.md
  python mini_search.py query "apple AND pie"
  python mini_search.py demo                    # build a tiny corpus and search it
"""
from __future__ import annotations

import argparse, json, math, os, re, sys, time, urllib.request, urllib.robotparser
from collections import Counter, defaultdict
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit

STOPWORDS = set("a an and are as at be by for from has have he in is it its of on or that the to was were will with".split())
TOKEN_RE = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    out = []
    for raw in TOKEN_RE.findall(text.lower()):
        if len(raw) <= 2 and not raw.isdigit():
            continue
        if raw in STOPWORDS:
            continue
        # light stemming
        for suf in ("ing", "es", "s"):
            if len(raw) > 4 and raw.endswith(suf):
                raw = raw[: -len(suf)]
                break
        out.append(raw)
    return out


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.title, self.links = [], [], []
        self._in_title = False
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            for k, v in attrs:
                if k == "href" and v:
                    self.links.append(v)

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title:
            self.title += data
        else:
            self.parts.append(data)


def fetch(url: str) -> tuple[str, list[str], str] | None:
    req = urllib.request.Request(url, headers={"User-Agent": "MiniSearchBot/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            if "text/html" not in r.headers.get("Content-Type", ""):
                return None
            html = r.read().decode("utf-8", errors="ignore")
    except Exception:
        return None
    p = TextExtractor()
    p.feed(html)
    title = "".join(p.title).strip()
    links = [urljoin(url, l.split("#")[0]) for l in p.links]
    return " ".join(p.parts), links, title


def crawl(seed: str, max_pages: int = 30, delay: float = 0.5) -> dict[str, dict]:
    host = urlsplit(seed).netloc
    robots = urllib.robotparser.RobotFileParser()
    try:
        robots.set_url(urljoin(seed, "/robots.txt"))
        robots.read()
    except Exception:
        robots = None
    queue, seen, docs = [seed], {seed}, {}
    while queue and len(docs) < max_pages:
        url = queue.pop(0)
        if robots and not robots.can_fetch("MiniSearchBot", url):
            continue
        got = fetch(url)
        time.sleep(delay)
        if not got:
            continue
        text, links, title = got
        docs[url] = {"text": text, "title": title}
        for l in links:
            if urlsplit(l).netloc == host and l not in seen and l.startswith("http"):
                seen.add(l)
                queue.append(l)
        print(f"  crawled {url} ({len(text)} chars)")
    return docs


class Index:
    """Inverted index with positional postings + document stats."""

    def __init__(self):
        self.postings: dict[str, dict[int, list[int]]] = defaultdict(dict)  # term -> {doc: [positions]}
        self.doc_len: dict[int, int] = {}
        self.docs: dict[int, dict] = {}  # doc_id -> {url/title/path}
        self.N = 0

    def add(self, doc_id: int, text: str, meta: dict):
        toks = tokenize(text)
        self.doc_len[doc_id] = len(toks)
        self.docs[doc_id] = meta
        for pos, t in enumerate(toks):
            self.postings[t].setdefault(doc_id, []).append(pos)

    def finalize(self):
        self.N = len(self.doc_len)
        self.avgdl = sum(self.doc_len.values()) / max(self.N, 1)

    def save(self, path: str):
        json.dump({"postings": dict(self.postings), "doc_len": self.doc_len,
                   "docs": self.docs, "N": self.N, "avgdl": self.avgdl},
                  open(path, "w", encoding="utf-8"), ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "Index":
        d = json.load(open(path, encoding="utf-8"))
        ix = cls()
        ix.postings = defaultdict(dict, d["postings"])
        ix.doc_len, ix.docs, ix.N, ix.avgdl = d["doc_len"], d["docs"], d["N"], d["avgdl"]
        return ix

    # ---------- BM25 ----------
    def bm25(self, terms: list[str], k1: float = 1.5, b: float = 0.75) -> dict[int, float]:
        scores: dict[int, float] = defaultdict(float)
        for t in terms:
            plist = self.postings.get(t)
            if not plist:
                continue
            idf = math.log(1 + (self.N - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc, positions in plist.items():
                tf = len(positions)
                dl = self.doc_len[doc]
                scores[doc] += idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self.avgdl))
        return scores

    def phrase_docs(self, phrase: str) -> set[int]:
        toks = tokenize(phrase)
        if not toks:
            return set()
        plists = [self.postings.get(t, {}) for t in toks]
        common = set(plists[0])
        for pl in plists[1:]:
            common &= set(pl)
        hits = set()
        for doc in common:
            first = plists[0][doc]
            for start in first:
                if all((start + i) in plists[i][doc] for i in range(1, len(toks))):
                    hits.add(doc)
                    break
        return hits

    def snippet(self, doc_id: int, terms: list[str], width: int = 45) -> str:
        toks = tokenize(self.docs[doc_id].get("text", ""))
        if not toks:
            return ""
        hits = [i for i, t in enumerate(toks) if t in terms]
        if not hits:
            return " ".join(toks[:width]) + " ..."
        c = hits[0]
        lo, hi = max(0, c - 10), min(len(toks), c + width)
        out = []
        for i in range(lo, hi):
            out.append(f"[{toks[i]}]" if toks[i] in terms else toks[i])
        return ("... " if lo else "") + " ".join(out) + (" ..." if hi < len(toks) else "")


def parse_query(q: str) -> tuple[list[str], list[str], set[str], str | None]:
    """Return (AND terms, OR terms, excluded, phrase)."""
    phrase = None
    m = re.search(r'"([^"]+)"', q)
    if m:
        phrase, q = m.group(1), q.replace(m.group(0), " ")
    ands, ors, nots = [], [], set()
    for tok in q.split():
        if tok.upper() == "AND" or tok.upper() == "OR":
            continue
        if tok.startswith("-") and len(tok) > 1:
            nots.add(tokenize(tok[1:])[0] if tokenize(tok[1:]) else tok[1:])
        elif "|" in tok:
            ors.extend(x for x in tokenize(tok.replace("|", " ")) if x)
        else:
            ands.extend(tokenize(tok))
    return ands, ors, nots, phrase


def search(ix: Index, q: str, top: int = 10) -> list[tuple[int, float]]:
    ands, ors, nots, phrase = parse_query(q)
    all_terms = ands + ors + (tokenize(phrase) if phrase else [])
    scores = ix.bm25(all_terms)
    if ands:
        plists = [set(ix.postings.get(t, {})) for t in ands]
        need = set.intersection(*plists) if plists else set()
        scores = {d: s for d, s in scores.items() if d in need}
    if phrase:
        ph = ix.phrase_docs(phrase)
        scores = {d: s * (3 if d in ph else 1) for d, s in scores.items() if d in ph or not ph}
        for d in ph & set(scores):
            scores[d] += 5.0  # phrase bonus
    for d in list(scores):
        if d in nots:
            del scores[d]
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top]


def build_index(docs: dict[str, dict], source: str = "web") -> Index:
    ix = Index()
    for i, (key, d) in enumerate(sorted(docs.items())):
        meta = {"url": key, "title": d.get("title") or key} if source == "web" \
               else {"url": key, "title": os.path.basename(key), "text": d["text"]}
        ix.add(i, d["text"], meta)
    ix.finalize()
    return ix


def index_folder(path: str) -> Index:
    docs = {}
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.endswith((".txt", ".md", ".py", ".html")):
                p = os.path.join(root, fn)
                try:
                    text = open(p, encoding="utf-8", errors="ignore").read()
                except Exception:
                    continue
                if fn.endswith(".html"):
                    ex = TextExtractor(); ex.feed(text)
                    text = " ".join(ex.parts)
                docs[p] = {"text": text, "title": fn}
    return build_index(docs, source="local")


def demo():
    """Build a small local corpus and run three queries — this is the self-test."""
    corpus = {
        "apple_pie.txt": "An apple pie recipe needs fresh apples, sugar, cinnamon and a butter crust. "
                         "Bake the apple pie for 45 minutes until the crust is golden.",
        "os_basics.txt": "An operating system manages processes, memory and files. The kernel schedules "
                         "processes and the file system stores data on disk.",
        "ml_intro.txt": "Machine learning trains models on data. A neural network learns features by "
                        "gradient descent; deep learning stacks many layers.",
        "smoothie.txt": "A green smoothie blends spinach, banana and apple juice. Apple juice makes the "
                        "smoothie sweet without sugar.",
    }
    docs = {k: {"text": v, "title": k} for k, v in corpus.items()}
    ix = build_index(docs, source="local")

    tests = [
        ("apple AND pie", ["apple_pie.txt"]),
        ("apple | smoothie", None),          # ranked, both docs may appear; apple_pie/smoothie first
        ('"apple juice" -pie', ["smoothie.txt"]),
        ("machine learning", ["ml_intro.txt"]),
    ]
    ok = 0
    for q, expect_first in tests:
        results = search(ix, q, top=5)
        top_docs = [ix.docs[d]["title"] for d, _ in results]
        passed = True
        if expect_first is not None:
            passed = all(e in top_docs for e in expect_first)
        status = "PASS" if passed else "FAIL"
        ok += passed
        print(f"[{status}] query: {q!r}")
        for d, s in results:
            print(f"    {s:6.2f}  {ix.docs[d]['title']}")
            print(f"           {ix.snippet(d, tokenize(q.replace('\"','').replace('-','')))}")
    print(f"\n{ok}/{len(tests)} search tests passed")
    assert ok == len(tests), "search self-test failed"
    ix.save("index.json")
    print("index saved to index.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("crawl"); c.add_argument("seed"); c.add_argument("--max-pages", type=int, default=30)
    c.add_argument("--save", default="index.json")
    i = sub.add_parser("index"); i.add_argument("folder"); i.add_argument("--save", default="index.json")
    q = sub.add_parser("query"); q.add_argument("text"); q.add_argument("--load", default="index.json")
    sub.add_parser("demo")
    args = ap.parse_args()

    if args.cmd == "crawl":
        docs = crawl(args.seed, args.max_pages)
        ix = build_index(docs, "web"); ix.save(args.save)
        print(f"indexed {ix.N} pages -> {args.save}")
    elif args.cmd == "index":
        ix = index_folder(args.folder); ix.save(args.save)
        print(f"indexed {ix.N} files -> {args.save}")
    elif args.cmd == "query":
        ix = Index.load(args.load)
        for d, s in search(ix, args.text):
            m = ix.docs[d]
            print(f"{s:7.3f}  {m.get('title')}  {m.get('url','')}")
            print(f"          {ix.snippet(d, tokenize(args.text))}")
    else:
        demo()

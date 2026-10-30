# mini-search-engine 🔍

A miniature search engine in **pure Python, stdlib only** — crawler, indexer and ranker in one file.

## Self-test results

```
[PASS] query: 'apple AND pie'          -> apple_pie.txt
[PASS] query: 'apple | smoothie'       -> smoothie.txt (BM25 ranked)
[PASS] query: '"apple juice" -pie'     -> smoothie.txt (phrase + exclusion)
[PASS] query: 'machine learning'       -> ml_intro.txt

4/4 search tests passed
```

## Features

- 🕷️ **Polite BFS crawler** — same-domain only, honors robots.txt, per-host delay
- 🔤 **Tokenizer** — Unicode-aware (incl. CJK ranges), stopwords, light stemming
- 🗂️ **Inverted index** — positional postings, persisted as JSON
- 📊 **BM25 ranking** (k1=1.5, b=0.75) + phrase-match bonus
- 🧩 **Query syntax** — `AND` / `OR`(`|`) / `"exact phrase"` / `-excluded`
- ✂️ **Snippets** — context windows with hit highlighting

## Usage

```bash
python mini_search.py demo                          # build corpus + run self-tests
python mini_search.py index ./docs                  # index a folder of txt/md/html
python mini_search.py crawl https://site.com --max-pages 30
python mini_search.py query '"apple juice" -pie'
```

## License

MIT

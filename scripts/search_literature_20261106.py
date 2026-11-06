"""Systematic literature search for the ZuCo EEG-to-text claims.

Run with: .venv/Scripts/python.exe scripts/search_literature_20261106.py

Standard library only. Sends no email or personal data (no mailto parameter).

Every HTTP response is cached under --cache, so a re-run costs nothing and gives
the same result. Pass --refresh to query the databases again.

Writes, beside REVIEW.md:
  search_log.json  query, database, date, total count, retrieved count
  records.csv      deduplicated bibliographic records and screening decisions

Screening decisions already in records.csv are kept across runs. Abstracts go only
to a screening sheet in the cache folder and never into the repository.

OpenAlex now gives keyless callers a small shared daily budget. The script stops
calling it when the budget runs low and says so in the log.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SEARCH_DATE = "2026-11-06"
REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "literature_search_20261106"
USER_AGENT = "literature-search-script/1.0"
OPENALEX_MIN_BUDGET_USD = 0.004

OPENALEX_QUERIES = [
    # ZuCo EEG-to-text decoding
    "ZuCo EEG-to-text decoding",
    "EEG to text translation sentence decoding ZuCo",
    "open vocabulary EEG-to-text decoding natural reading",
    "reading EEG semantic decoding sentence embeddings",
    # evaluation critique or noise baseline
    "EEG-to-text noise baseline evaluation critique",
    "EEG-to-text teacher forcing noise inflated performance",
    # data leakage
    "data leakage EEG decoding",
    "brain-to-text data splitting leakage",
    # block or temporal autocorrelation
    "temporal autocorrelation EEG decoding overestimated",
    "block design EEG classification confound",
    # sentence or cross-subject split
    "ZuCo cross-subject split sentence split",
    "unseen subjects unseen sentences EEG text decoding",
    # task or corpus confound
    "ZuCo reading task classification normal reading task-specific reading",
    "EEG reading task corpus confound decoding",
    # retrieval N-way
    "EEG text retrieval contrastive sentence N-way",
    "brain signal sentence retrieval contrastive EEG reading",
    # gaze baseline
    "eye-tracking baseline versus EEG reading decoding",
    "gaze features EEG natural reading classification ZuCo",
    # word-locked
    "fixation-related potentials word decoding natural reading",
    "word-level EEG decoding natural reading fixation",
    # list position or drift
    "list position session drift EEG decoding",
    "EEG nonstationarity session drift brain-to-text decoding",
    # closely similar data and replication
    "MEG brain-to-text decoding reading sentence retrieval",
    "non-invasive language decoding reading failed replication permutation chance",
]

ARXIV_QUERIES = [
    'all:ZuCo',
    'abs:"EEG-to-text"',
    'abs:"EEG to text"',
    'abs:EEG AND abs:leakage AND abs:decoding',
    'abs:EEG AND abs:autocorrelation AND abs:decoding',
    'abs:EEG AND abs:"block design"',
    'abs:EEG AND abs:"eye-tracking" AND abs:reading',
    'abs:EEG AND abs:retrieval AND abs:sentence',
    'abs:EEG AND abs:reading AND abs:decoding',
    'abs:"fixation-related" AND abs:EEG',
    'abs:"brain-to-text" AND abs:EEG',
    'abs:EEG AND abs:drift AND abs:decoding',
    'abs:EEG AND abs:noise AND abs:baseline AND abs:text',
]

S2_QUERIES = [
    "ZuCo EEG-to-text decoding",
    "EEG-to-text noise baseline evaluation",
    "brain-to-text data splitting leakage ZuCo",
    "ZuCo sentence retrieval EEG contrastive",
    "EEG gaze eye-tracking reading decoding ZuCo",
    "EEG list position block drift reading decoding",
    "ZuCo reading task classification",
    "fixation-related EEG word decoding reading",
]

CROSSREF_QUERIES = [
    "ZuCo EEG reading",
    "EEG-to-text decoding",
    "EEG decoding data leakage temporal autocorrelation",
    "EEG eye-tracking natural reading decoding",
]

# ACL Anthology has no public search API. These were run by hand with a web search
# limited to aclanthology.org on the search date. Hits are recorded here so that the
# search stays auditable. Only hits on aclanthology.org are kept.
ACL_WEB_QUERIES: dict[str, list[dict]] = {
    "ZuCo EEG-to-text decoding": [
        {
            "title": "Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder",
            "url": "https://aclanthology.org/2024.acl-long.393/"
        },
        {
            "title": "Rethinking Cross-Subject Data Splitting for Brain-to-Text Decoding",
            "url": "https://aclanthology.org/2025.emnlp-main.289/"
        }
    ],
    "EEG to text noise baseline leakage data splitting evaluation critique": [
        {
            "title": "Rethinking Cross-Subject Data Splitting for Brain-to-Text Decoding",
            "url": "https://aclanthology.org/2025.emnlp-main.289/"
        },
        {
            "title": "Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation",
            "url": "https://aclanthology.org/2025.findings-acl.142/"
        },
        {
            "title": "Is EEG-to-Text Feasible in Real-World Scenarios?",
            "url": "https://aclanthology.org/2026.acl-long.61/"
        }
    ],
    "EEG eye-tracking gaze reading sentence classification ZuCo task": [
        {
            "title": "Relation Classification with Cognitive Attention Supervision",
            "url": "https://aclanthology.org/2021.cmcl-1.26/"
        },
        {
            "title": "Sequence Classification with Human Attention",
            "url": "https://aclanthology.org/K18-1030/"
        },
        {
            "title": "Do Transformer Models Show Similar Attention Patterns to Task-Specific Human Gaze?",
            "url": "https://aclanthology.org/2022.acl-long.296/"
        },
        {
            "title": "Comparing Eye-gaze and Transformer Attention Mechanisms",
            "url": "https://aclanthology.org/2025.gaze4nlp-1.4/"
        },
        {
            "title": "Entity Recognition at First Sight: Improving NER with Eye Movement Information",
            "url": "https://aclanthology.org/N19-1001/"
        }
    ],
    "EEG sentence retrieval brain signals text contrastive reading word-level fixation": [
        {
            "title": "Semantic Brain Signal Decoding through Vector-Quantized Spectrogram Reconstruction for Whisper-Enhanced Text Generation",
            "url": "https://aclanthology.org/2025.findings-acl.142/"
        },
        {
            "title": "Enhancing EEG-to-Text Decoding through Transferable Representations from Pre-trained Contrastive EEG-Text Masked Autoencoder",
            "url": "https://aclanthology.org/2024.acl-long.393/"
        },
        {
            "title": "UniCoRN: Unified Cognitive Signal ReconstructioN bridging cognitive signals and human language",
            "url": "https://aclanthology.org/2023.acl-long.741/"
        },
        {
            "title": "Can Brain Signals Reveal Inner Alignment with Human Languages?",
            "url": "https://aclanthology.org/2023.findings-emnlp.120/"
        },
        {
            "title": "LiNCr 2020 workshop paper, pages 15-27 (title not returned by the search)",
            "url": "https://aclanthology.org/2020.lincr-1.3/"
        }
    ]
}

# General web searches added during the search, after block classification and code-only
# audits turned up that the scholarly databases missed. Logged as a protocol deviation.
WEB_QUERIES: dict[str, list[dict]] = {
    "ZuCo EEG decoding sentence order block position leakage temporal split reading experiment confound": [
        {"title": "Reading Task Classification Using EEG and Eye-Tracking Data", "url": "https://arxiv.org/abs/2112.06310"},
        {"title": "Decoding EEG Brain Activity for Multi-Modal Natural Language Processing", "url": "https://arxiv.org/abs/2102.08655"},
    ],
    "EEG-to-text ZuCo eye-tracking only baseline outperforms EEG fixation duration word length confound retrieval": [
        {"title": "On Creating A Brain-To-Text Decoder", "url": "https://arxiv.org/abs/2501.06326"},
        {"title": "Integrating LLM, EEG, and Eye-Tracking Biomarker Analysis for Word-Level Neural State Classification in Semantic Inference Reading Comprehension", "url": "https://arxiv.org/abs/2309.15714"},
    ],
    "ZuCo EEG reading decoding above chance rigorous held-out subject permutation test no signal failed to decode word embeddings": [
        {"title": "RAG-based EEG-to-Text Translation Using Deep Learning and LLMs", "url": "https://arxiv.org/abs/2605.17503"},
        {"title": "From Word Embedding to Reading Embedding Using Large Language Model, EEG and Eye-tracking", "url": "https://arxiv.org/abs/2401.15681"},
        {"title": "Deep Representation Learning for Open Vocabulary Electroencephalography-to-Text Decoding", "url": "https://arxiv.org/abs/2312.09430"},
    ],
    "EEG-to-text retrieval candidate pool composition inflates accuracy matched distractors reading task ZuCo controlled audit": [
        {"title": "Towards Brain Passage Retrieval -- An Investigation of EEG Query Representations", "url": "https://arxiv.org/abs/2412.06695"},
        {"title": "Joint Text-Audio Alignment for EEG-to-Text Decoding in Chinese Speech Production and Perception", "url": "https://arxiv.org/abs/2607.25626"},
    ],
    "EEG decoding natural reading sentence semantic signal surrogate presentation order trial position slow drift text embedding null result 2026 arXiv": [
        {"title": "What Are We Actually Decoding? Source Attribution for Non-Invasive Brain-to-Language Retrieval", "url": "https://arxiv.org/abs/2605.24524"},
        {"title": "SENSE: Efficient EEG-to-Text via Privacy-Preserving Semantic Retrieval", "url": "https://arxiv.org/abs/2603.17109"},
    ],
}

# GitHub is outside the original protocol. Added on the search date for the same reason.
GITHUB_QUERIES = ["ZuCo EEG", "EEG-to-text", "zuco retrieval"]

# Seeds: (name, list of OpenAlex lookups in order, Semantic Scholar id or None, title).
SEEDS = [
    ("Hollenstein 2023 ZuCo benchmark", "10.3389/fpsyg.2022.1028824", None,
     "The ZuCo benchmark on cross-subject reading task classification with EEG and eye-tracking data"),
    ("Zhang ACL 2026", "10.18653/v1/2026.acl-long.61", None, None),
    ("Rethinking Cross-Subject Data Splitting EMNLP 2025", "10.18653/v1/2025.emnlp-main.289", "arXiv:2312.10987",
     "Rethinking Cross-Subject Data Splitting for Brain-to-Text Decoding"),
    ("Jo et al. noise baselines", "10.48550/arXiv.2405.06459", "arXiv:2405.06459",
     "Are EEG-to-Text Models Working?"),
    ("Beware of Overestimated Decoding Performance", "10.48550/arXiv.2405.17024", "arXiv:2405.17024",
     "Beware of Overestimated Decoding Performance Arising from Temporal Autocorrelations in Electroencephalogram Signals"),
    ("GLIM", "10.48550/arXiv.2505.17099", "arXiv:2505.17099", None),
    ("SemKey", "10.48550/arXiv.2603.03312", "arXiv:2603.03312", None),
    ("Brain-CLIPLM", "10.48550/arXiv.2604.16370", "arXiv:2604.16370", None),
    ("Decoding silent reading", "10.48550/arXiv.2608.20186", "arXiv:2608.20186", None),
    # Extra seeds: anything using ZuCo should cite one of these.
    ("ZuCo 1.0 dataset (extra seed)", "10.1038/sdata.2018.291", None,
     "ZuCo, a simultaneous EEG and eye-tracking resource for natural sentence reading"),
    ("ZuCo 2.0 dataset (extra seed)", "10.48550/arXiv.1912.00903", "arXiv:1912.00903",
     "ZuCo 2.0: A Dataset of Physiological Recordings During Natural Reading and Annotation"),
]


class Http:
    def __init__(self, cache: Path, refresh: bool) -> None:
        self.cache = cache
        self.refresh = refresh
        self.cache.mkdir(parents=True, exist_ok=True)
        self.openalex_remaining = 1.0
        self.openalex_stopped = False

    def get(self, url: str, delay: float = 0.3, retries: int = 3) -> tuple[int, bytes]:
        path = self.cache / (hashlib.sha1(url.encode()).hexdigest() + ".bin")
        if path.exists() and not self.refresh:
            return 200, path.read_bytes()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        status = 0
        for attempt in range(retries):
            time.sleep(delay * (1 + 3 * attempt))
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                    left = resp.headers.get("X-RateLimit-Remaining-USD")
                    if left is not None:
                        self.openalex_remaining = float(left)
                    path.write_bytes(body)
                    return resp.status, body
            except urllib.error.HTTPError as exc:
                status = exc.code
                if status not in (429, 500, 502, 503, 504):
                    return status, b""
            except (urllib.error.URLError, TimeoutError) as exc:
                print("  network error", exc, file=sys.stderr)
        return status, b""

    def openalex(self, path: str, params: dict) -> dict | None:
        if self.openalex_remaining < OPENALEX_MIN_BUDGET_USD:
            self.openalex_stopped = True
            return None
        query = urllib.parse.urlencode(params)
        url = f"https://api.openalex.org/{path}" + (f"?{query}" if query else "")
        status, body = self.get(url, delay=0.15)
        if status != 200:
            print(f"  OpenAlex status {status}: {url}", file=sys.stderr)
            return None
        return json.loads(body)


OA_FIELDS = "id,doi,title,publication_year,type,primary_location,ids,abstract_inverted_index,referenced_works_count,cited_by_count"


def oa_abstract(index: dict | None) -> str:
    if not index:
        return ""
    words = [(pos, word) for word, positions in index.items() for pos in positions]
    return " ".join(word for _, word in sorted(words))


def oa_record(work: dict, source: str) -> dict:
    loc = work.get("primary_location") or {}
    venue = ((loc.get("source") or {}) or {}).get("display_name") or ""
    doi = (work.get("doi") or "").replace("https://doi.org/", "")
    return {
        "id": work["id"].rsplit("/", 1)[-1],
        "title": work.get("title") or "",
        "year": work.get("publication_year") or "",
        "venue": venue,
        "doi_or_url": f"https://doi.org/{doi}" if doi else (loc.get("landing_page_url") or work["id"]),
        "source": source,
        "abstract": oa_abstract(work.get("abstract_inverted_index")),
        "doi": doi.lower(),
    }


def arxiv_id_from(text: str) -> str:
    match = re.search(r"(\d{4}\.\d{4,5})", text or "")
    return match.group(1) if match else ""


def run_openalex(http: Http, log: list, records: list) -> None:
    for i, query in enumerate(OPENALEX_QUERIES, 1):
        qid = f"OA{i:02d}"
        data = http.openalex("works", {"search": query, "per-page": 100, "select": OA_FIELDS})
        if data is None:
            log.append(entry(qid, "OpenAlex", query, None, 0, "not run or failed (budget or HTTP error)"))
            continue
        works = data["results"]
        log.append(entry(qid, "OpenAlex", query, data["meta"]["count"], len(works), "relevance-ranked, first 100"))
        records += [oa_record(w, qid) for w in works]


def run_arxiv(http: Http, log: list, records: list) -> None:
    ns = {"a": "http://www.w3.org/2005/Atom", "o": "http://a9.com/-/spec/opensearch/1.1/"}
    for i, query in enumerate(ARXIV_QUERIES, 1):
        qid = f"AX{i:02d}"
        params = urllib.parse.urlencode({"search_query": query, "start": 0, "max_results": 100,
                                         "sortBy": "relevance"})
        status, body = http.get(f"https://export.arxiv.org/api/query?{params}", delay=3.2)
        if status != 200:
            log.append(entry(qid, "arXiv", query, None, 0, f"HTTP {status}"))
            continue
        root = ET.fromstring(body)
        total = root.findtext("o:totalResults", default="", namespaces=ns)
        items = root.findall("a:entry", ns)
        for item in items:
            url = item.findtext("a:id", default="", namespaces=ns)
            records.append({
                "id": "arXiv:" + arxiv_id_from(url),
                "title": " ".join(item.findtext("a:title", default="", namespaces=ns).split()),
                "year": item.findtext("a:published", default="", namespaces=ns)[:4],
                "venue": "arXiv",
                "doi_or_url": url,
                "source": qid,
                "abstract": " ".join(item.findtext("a:summary", default="", namespaces=ns).split()),
                "doi": f"10.48550/arxiv.{arxiv_id_from(url)}",
            })
        log.append(entry(qid, "arXiv", query, int(total) if total else None, len(items), "relevance-ranked, first 100"))


def s2_record(paper: dict, source: str) -> dict:
    ext = paper.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower()
    arx = ext.get("ArXiv") or ""
    if not doi and arx:
        doi = f"10.48550/arxiv.{arx}"
    return {
        "id": "S2:" + (paper.get("paperId") or ""),
        "title": paper.get("title") or "",
        "year": paper.get("year") or "",
        "venue": paper.get("venue") or "",
        "doi_or_url": f"https://doi.org/{doi}" if doi else (paper.get("url") or ""),
        "source": source,
        "abstract": paper.get("abstract") or "",
        "doi": doi,
    }


def run_s2(http: Http, log: list, records: list) -> None:
    fields = "title,year,venue,externalIds,abstract,url"
    for i, query in enumerate(S2_QUERIES, 1):
        qid = f"S2{i:02d}"
        params = urllib.parse.urlencode({"query": query, "limit": 100, "fields": fields})
        status, body = http.get(f"https://api.semanticscholar.org/graph/v1/paper/search?{params}", delay=4, retries=4)
        if status != 200:
            log.append(entry(qid, "Semantic Scholar", query, None, 0, f"rate-limited or failed, HTTP {status}"))
            continue
        data = json.loads(body)
        papers = data.get("data") or []
        log.append(entry(qid, "Semantic Scholar", query, data.get("total"), len(papers), "first 100"))
        records += [s2_record(p, qid) for p in papers]


def run_crossref(http: Http, log: list, records: list) -> None:
    for i, query in enumerate(CROSSREF_QUERIES, 1):
        qid = f"CR{i:02d}"
        params = urllib.parse.urlencode({"query": query, "rows": 50,
                                         "select": "DOI,title,container-title,issued,abstract"})
        status, body = http.get(f"https://api.crossref.org/works?{params}", delay=1)
        if status != 200:
            log.append(entry(qid, "Crossref", query, None, 0, f"HTTP {status}"))
            continue
        msg = json.loads(body)["message"]
        for item in msg["items"]:
            doi = item["DOI"].lower()
            parts = (item.get("issued") or {}).get("date-parts") or [[None]]
            records.append({
                "id": "CR:" + doi,
                "title": " ".join((item.get("title") or [""])[0].split()),
                "year": parts[0][0] or "",
                "venue": (item.get("container-title") or [""])[0],
                "doi_or_url": f"https://doi.org/{doi}",
                "source": qid,
                "abstract": re.sub(r"<[^>]+>", " ", item.get("abstract") or ""),
                "doi": doi,
            })
        log.append(entry(qid, "Crossref", query, msg["total-results"], len(msg["items"]), "relevance-ranked, first 50"))


def run_acl(log: list, records: list) -> None:
    for i, (query, hits) in enumerate(ACL_WEB_QUERIES.items(), 1):
        qid = f"AC{i:02d}"
        log.append(entry(qid, "ACL Anthology (web search, by hand)", query, None, len(hits),
                         "web search limited to aclanthology.org; no total count available"))
        for hit in hits:
            records.append({
                "id": "ACL:" + hit["url"].rstrip("/").rsplit("/", 1)[-1],
                "title": hit["title"], "year": hit.get("year", ""), "venue": "ACL Anthology",
                "doi_or_url": hit["url"], "source": qid, "abstract": "",
                "doi": "",
            })


def run_web(log: list, records: list) -> None:
    for i, (query, hits) in enumerate(WEB_QUERIES.items(), 1):
        qid = f"WB{i:02d}"
        log.append(entry(qid, "General web search (by hand, deviation)", query, None, len(hits),
                         "only hits new to or relevant for the record set are listed"))
        for hit in hits:
            arx = arxiv_id_from(hit["url"])
            records.append({
                "id": "arXiv:" + arx, "title": hit["title"], "year": "", "venue": "arXiv",
                "doi_or_url": hit["url"], "source": qid, "abstract": "",
                "doi": f"10.48550/arxiv.{arx}",
            })


def run_github(http: Http, log: list, records: list) -> None:
    for i, query in enumerate(GITHUB_QUERIES, 1):
        qid = f"GH{i:02d}"
        params = urllib.parse.urlencode({"q": query, "sort": "updated", "per_page": 100})
        status, body = http.get(f"https://api.github.com/search/repositories?{params}", delay=2)
        if status != 200:
            log.append(entry(qid, "GitHub repositories (deviation)", query, None, 0, f"HTTP {status}"))
            continue
        data = json.loads(body)
        items = data.get("items") or []
        log.append(entry(qid, "GitHub repositories (deviation)", query, data.get("total_count"), len(items),
                         "sorted by last update, first 100; code, not peer reviewed"))
        for repo in items:
            records.append({
                "id": "GH:" + repo["full_name"], "title": repo["full_name"] + ": " + (repo.get("description") or ""),
                "year": (repo.get("created_at") or "")[:4], "venue": "GitHub", "doi_or_url": repo["html_url"],
                "source": qid, "abstract": repo.get("description") or "", "doi": "",
            })


def oa_fetch_ids(http: Http, ids: list[str], source: str) -> list[dict]:
    out = []
    for start in range(0, len(ids), 100):
        chunk = [i.rsplit("/", 1)[-1] for i in ids[start:start + 100]]
        data = http.openalex("works", {"filter": "openalex_id:" + "|".join(chunk), "per-page": 100, "select": OA_FIELDS})
        if data:
            out += [oa_record(w, source) for w in data["results"]]
    return out


def resolve_seed(http: Http, doi: str, title: str | None) -> dict | None:
    fields = OA_FIELDS + ",referenced_works"
    data = http.openalex(f"works/https://doi.org/{doi}", {"select": fields})
    if data and data.get("id"):
        return data
    if title:
        data = http.openalex("works", {"search": title, "per-page": 3, "select": fields})
        if data:
            want = norm_title(title)
            for work in data["results"]:
                if norm_title(work.get("title") or "") == want:
                    return work
    return None


def run_citations(http: Http, log: list, records: list) -> list[dict]:
    seeds_out = []
    for name, doi, s2_id, title in SEEDS:
        row = {"seed": name, "doi": doi}
        work = resolve_seed(http, doi, title)
        cited, refs = [], []
        if work:
            short = work["id"].rsplit("/", 1)[-1]
            row["openalex_id"] = short
            row["openalex_title"] = work.get("title")
            records.append(oa_record(work, "SEED"))
            cursor, total = "*", None
            while cursor:
                data = http.openalex("works", {"filter": f"cites:{short}", "per-page": 200, "cursor": cursor,
                                               "select": OA_FIELDS})
                if not data:
                    break
                total = data["meta"]["count"]
                cited += [oa_record(w, f"citedby:{name}") for w in data["results"]]
                cursor = data["meta"].get("next_cursor") if data["results"] else None
            refs = oa_fetch_ids(http, work.get("referenced_works") or [], f"references:{name}")
            row["openalex_cited_by_total"] = total
        row["openalex_cited_by"] = len(cited)
        row["openalex_references"] = len(refs)
        records += cited + refs

        lookup = s2_id or f"DOI:{doi}"
        s2_cited, s2_refs, s2_status = [], [], []
        for edge, key, bucket in (("citations", "citingPaper", s2_cited), ("references", "citedPaper", s2_refs)):
            params = urllib.parse.urlencode({"fields": "title,year,venue,externalIds,abstract,url", "limit": 500})
            status, body = http.get(f"https://api.semanticscholar.org/graph/v1/paper/{urllib.parse.quote(lookup)}/{edge}?{params}",
                                    delay=4, retries=3)
            s2_status.append(status)
            if status == 200:
                for item in json.loads(body).get("data") or []:
                    if item.get(key) and item[key].get("title"):
                        tag = "citedby" if edge == "citations" else "references"
                        bucket.append(s2_record(item[key], f"s2{tag}:{name}"))
        row["s2_status"] = s2_status
        row["s2_cited_by"] = len(s2_cited)
        row["s2_references"] = len(s2_refs)
        records += s2_cited + s2_refs
        seeds_out.append(row)
        print(row, flush=True)
    log.append(entry("CIT", "OpenAlex + Semantic Scholar", "citation tracing from seeds", None,
                     sum(r["openalex_cited_by"] + r["openalex_references"] + r["s2_cited_by"] + r["s2_references"]
                         for r in seeds_out), "see seeds in this file"))
    return seeds_out


def entry(qid: str, db: str, query: str, count, retrieved: int, note: str) -> dict:
    return {"id": qid, "database": db, "query": query, "date": SEARCH_DATE, "count": count,
            "retrieved": retrieved, "note": note}


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def dedupe(records: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    unique: list[dict] = []
    for rec in records:
        keys = []
        if rec.get("doi"):
            keys.append("doi:" + rec["doi"])
        arx = arxiv_id_from(rec.get("doi_or_url", "")) if "arxiv" in rec.get("doi_or_url", "").lower() else ""
        if arx:
            keys.append("arxiv:" + arx)
        title = norm_title(rec["title"])
        if len(title) > 12:
            keys.append("title:" + title)
        if not keys:
            continue
        hit = next((seen[k] for k in keys if k in seen), None)
        if hit is None:
            hit = dict(rec)
            hit["sources"] = [rec["source"]]
            unique.append(hit)
        else:
            if rec["source"] not in hit["sources"]:
                hit["sources"].append(rec["source"])
            if not hit.get("abstract") and rec.get("abstract"):
                hit["abstract"] = rec["abstract"]
            if not hit.get("venue") or hit.get("venue") == "arXiv":
                hit["venue"] = rec.get("venue") or hit.get("venue")
        for key in keys:
            seen[key] = hit
    return unique


def write_outputs(unique: list[dict], cache: Path) -> None:
    path = OUT / "records.csv"
    old: dict[str, tuple[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                old[norm_title(row["title"])] = (row["screening_decision"], row["reason"])
    fields = ["id", "title", "year", "venue", "doi_or_url", "source", "screening_decision", "reason"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for rec in unique:
            decision, reason = old.get(norm_title(rec["title"]), ("unscreened", ""))
            writer.writerow({
                "id": rec["id"], "title": rec["title"], "year": rec["year"], "venue": rec["venue"],
                "doi_or_url": rec["doi_or_url"], "source": ";".join(rec["sources"]),
                "screening_decision": decision, "reason": reason,
            })
    with (cache / "screening_sheet.jsonl").open("w", encoding="utf-8") as fh:
        for rec in unique:
            fh.write(json.dumps({k: rec[k] for k in ("id", "title", "year", "venue", "sources", "abstract")},
                                ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=Path(
        r"C:\Users\Asus\AppData\Local\Temp\claude\C--Users-Asus-Desktop-New-trends-AI-project1"
        r"\e2cee6a6-7e7f-47ad-a9a4-7326a9f5f258\scratchpad\litsearch_cache"))
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and query again")
    args = parser.parse_args()

    http = Http(args.cache, args.refresh)
    log: list[dict] = []
    records: list[dict] = []
    run_openalex(http, log, records)
    print("OpenAlex budget left", http.openalex_remaining, flush=True)
    seeds = run_citations(http, log, records)
    print("OpenAlex budget left", http.openalex_remaining, flush=True)
    run_arxiv(http, log, records)
    run_crossref(http, log, records)
    run_s2(http, log, records)
    run_acl(log, records)
    run_web(log, records)
    run_github(http, log, records)

    unique = dedupe(records)
    write_outputs(unique, args.cache)
    payload = {
        "search_date": SEARCH_DATE,
        "openalex_budget_ran_out": http.openalex_stopped,
        "queries": log,
        "seeds": seeds,
        "records_found": len(records),
        "records_unique": len(unique),
    }
    (OUT / "search_log.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("found", len(records), "unique", len(unique))


if __name__ == "__main__":
    main()

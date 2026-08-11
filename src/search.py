#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
1. 검색 랭킹 (질의 임베딩 + 색인 로드 + dense 검색)

data/emb/*.npz 를 읽어 질문을 같은 모델로 인코딩하고 코사인 유사도 상위 top_k 를
뽑는다. 이 목록이 2단계 리랭킹의 후보가 된다.

왜 BAAI/bge-m3 인가:
  원문이 한국어·영어·중국어·베트남어·필리핀어·러시아어·우즈베크어로 흩어져 있는데
  질문은 한국어로 들어온다. bge-m3 는 100개 넘는 언어를 한 벡터 공간에 넣도록
  학습돼 있어서 한국어 질문으로 러시아어 원문을 바로 찾는다. 번역을 한 겹 끼우지
  않아도 되고, 뒤에 붙는 리랭커(bge-reranker-v2-m3)와 같은 계열이라 두 단계가
  같은 언어 감각으로 움직인다. 8192토큰까지 받으므로 512토큰 청크는 잘리지 않는다.

문서 임베딩과 반드시 맞춰야 하는 것 (틀리면 조용히 정확도만 떨어진다):
  1. 같은 모델    BAAI/bge-m3, 1024차원
  2. 프리픽스 없음   bge-m3 는 질의에 지시문을 붙이지 않는 대칭 모델이다
  3. float32 재정규화   내적을 그대로 코사인으로 쓰기 위해

문서를 좁히지 않으면(doc=None) 7개 문서를 한 색인으로 합쳐 뒤진다. 전체 청크가
1,100개 남짓이라 전수 비교로 충분하다 (ANN 색인이 필요 없다).

단독 실행:
    python src/search.py "대마재배자는 누구에게 허가를 받나요?"
    python src/search.py --doc ko "마약 밀매의 형량은?"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from corpus import ALL_DOCS, Document, documents, find  # noqa: E402

DEFAULT_MODEL = "BAAI/bge-m3"

# 문서 하나가 82~243청크다. 후보를 10개 뽑아 리랭커에 넘긴다.
DEFAULT_TOP_K = 10


# --------------------------------------------------------------------------
# 자료구조
# --------------------------------------------------------------------------

@dataclass
class Index:
    """검색 대상 청크 전체. 문서 하나일 수도 있고 전부일 수도 있다."""

    name: str                    # 'ko' 또는 'all'
    vectors: np.ndarray          # (N, dim) float32, L2 정규화됨
    texts: np.ndarray            # (N,) 청크 원문
    doc_keys: list[str]          # (N,) 청크가 나온 문서 키
    doc_codes: list[str]         # (N,) 짧은 코드 (인용 표기에 쓴다)
    chunk_indices: np.ndarray    # (N,) int32  문서 안에서 몇 번째 청크인지
    token_starts: np.ndarray     # (N,) int32
    token_ends: np.ndarray       # (N,) int32
    model: str = ""
    n_docs: int = 0

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])


@dataclass
class Hit:
    """검색 결과 한 건."""

    score: float
    text: str
    doc_key: str
    doc_code: str
    doc_title: str
    doc_lang: str                # 원문 언어 이름 ('러시아어'). 답변 프롬프트가 쓴다
    chunk_index: int
    token_start: int
    token_end: int

    @property
    def key(self) -> str:
        """청크를 가리키는 고유 이름. 인용 표기이자 중복 제거의 기준."""
        return f"{self.doc_code}#{self.chunk_index}"

    def preview(self, n: int = 200) -> str:
        one_line = " ".join(self.text.split())
        return one_line[:n] + ("..." if len(one_line) > n else "")


@dataclass
class SearchResult:
    """1단계 결과. 질문 하나로 뽑은 상위 top_k."""

    query: str
    doc: str                                        # 'ko' 또는 'all'
    doc_label: str                                  # 화면에 쓸 이름
    top_k: int
    hits: list[Hit] = field(default_factory=list)   # 점수 내림차순
    n_indexed: int = 0                              # 색인 전체 청크 수
    n_docs: int = 1                                 # 뒤진 문서 수
    elapsed: float = 0.0


# --------------------------------------------------------------------------
# 색인 로드
# --------------------------------------------------------------------------

def _load_one(doc: Document) -> dict:
    """.npz 한 개를 dict 로 읽는다."""
    data = np.load(doc.emb_path)
    vectors = data["embeddings"]
    model = ""
    try:
        model = json.loads(str(data["info"])).get("model", "")
    except (ValueError, KeyError):
        pass
    return {
        "vectors": vectors,
        "texts": data["texts"],
        "chunk_index": data["chunk_index"],
        "token_start": data["token_start"],
        "token_end": data["token_end"],
        "n": int(vectors.shape[0]),
        "model": model,
    }


@lru_cache(maxsize=16)
def load_index(doc: str | None = None) -> Index:
    """
    색인을 만든다. doc 이 None 이거나 'all' 이면 문서 전체를 하나로 합친다.

    Streamlit 이 클릭마다 스크립트를 다시 돌리므로 반드시 캐시한다. 전부 합쳐도
    1,136 x 1024 float32 = 약 4.6MB 라 메모리는 문제가 되지 않는다.
    """
    target = find(doc)
    docs = [target] if target else documents()
    if not docs:
        raise FileNotFoundError("색인이 없습니다. python src/embed.py 를 먼저 돌리세요.")

    vec_parts, text_parts = [], []
    idx_parts, start_parts, end_parts = [], [], []
    doc_keys: list[str] = []
    doc_codes: list[str] = []
    model_name = ""

    for item in docs:
        loaded = _load_one(item)
        vec_parts.append(loaded["vectors"])
        text_parts.append(loaded["texts"])
        idx_parts.append(loaded["chunk_index"])
        start_parts.append(loaded["token_start"])
        end_parts.append(loaded["token_end"])
        doc_keys.extend([item.key] * loaded["n"])
        doc_codes.extend([item.code] * loaded["n"])
        model_name = model_name or loaded["model"]

    return Index(
        name=target.code if target else ALL_DOCS,
        vectors=np.vstack(vec_parts).astype(np.float32),
        texts=np.concatenate(text_parts),
        doc_keys=doc_keys,
        doc_codes=doc_codes,
        chunk_indices=np.concatenate(idx_parts),
        token_starts=np.concatenate(start_parts),
        token_ends=np.concatenate(end_parts),
        model=model_name,
        n_docs=len(docs),
    )


# --------------------------------------------------------------------------
# 질의 임베딩
# --------------------------------------------------------------------------

@lru_cache(maxsize=2)
def load_model(name: str = DEFAULT_MODEL, device: str | None = None):
    """
    임베딩 모델을 올린다. 프로세스당 한 번만 (Streamlit 이 매 클릭마다 스크립트를
    다시 돌기 때문에 캐시가 없으면 클릭마다 모델을 새로 올린다).

    dtype 은 float32 로 고정한다. CPU 에서 bf16 은 AVX512-BF16 이 없으면
    에뮬레이션으로 떨어져 오히려 느리다.
    """
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return SentenceTransformer(name, device=device,
                               model_kwargs={"dtype": "float32"})


def encode_queries(queries: list[str], model_name: str = DEFAULT_MODEL,
                   device: str | None = None) -> np.ndarray:
    """
    질의를 (Q, dim) float32 정규화 벡터로 만든다.

    bge-m3 는 질의와 문서를 같은 방식으로 인코딩한다(대칭). 프리픽스를 붙이면
    오히려 문서 쪽 분포와 어긋나므로 아무것도 붙이지 않는다.
    """
    queries = [q.strip() for q in queries if q and q.strip()]
    if not queries:
        raise ValueError("질의가 비어 있습니다.")

    model = load_model(model_name, device)
    vectors = model.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vectors = vectors.astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


# --------------------------------------------------------------------------
# 1 단계 : 검색 랭킹
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _doc_by_key() -> dict[str, Document]:
    return {d.key: d for d in documents()}


def make_hit(index: Index, row: int, score: float) -> Hit:
    """행 번호 하나를 Hit 으로 만든다."""
    doc = _doc_by_key()[index.doc_keys[row]]
    return Hit(
        score=score,
        text=str(index.texts[row]),
        doc_key=doc.key,
        doc_code=doc.code,
        doc_title=doc.title,
        doc_lang=doc.lang_name,
        chunk_index=int(index.chunk_indices[row]),
        token_start=int(index.token_starts[row]),
        token_end=int(index.token_ends[row]),
    )


def search(query: str, doc: str | None = None, top_k: int = DEFAULT_TOP_K,
           model_name: str = DEFAULT_MODEL,
           device: str | None = None) -> SearchResult:
    """
    질문 하나를 임베딩해 색인 전체와 견주고 상위 top_k 를 돌려준다.

    doc 에 문서 키나 짧은 코드를 주면 그 문서 안에서만 찾고, 주지 않으면 전체
    문서를 뒤진다. 벡터가 전부 L2 정규화돼 있으므로 내적이 곧 코사인 유사도다.
    """
    started = time.time()

    clean = (query or "").strip()
    if not clean:
        raise ValueError("질문이 비어 있습니다.")

    index = load_index(doc)

    scores = encode_queries([clean], model_name, device)[0] @ index.vectors.T
    k = min(top_k, index.size)
    rows = [int(r) for r in np.argsort(-scores)[:k]]

    target = find(doc)
    return SearchResult(
        query=clean,
        doc=target.code if target else ALL_DOCS,
        doc_label=target.label if target else f"전체 문서 {index.n_docs}종",
        top_k=k,
        hits=[make_hit(index, r, float(scores[r])) for r in rows],
        n_indexed=index.size,
        n_docs=index.n_docs,
        elapsed=time.time() - started,
    )


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    parser = argparse.ArgumentParser(description="질문 하나로 색인을 검색한다.")
    parser.add_argument("question", nargs="*", help="검색할 질문")
    parser.add_argument("--doc", default=None,
                        help="문서 키나 짧은 코드 (기본: 전체 문서)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"뽑을 청크 수 (기본: {DEFAULT_TOP_K})")
    args = parser.parse_args()

    question = " ".join(args.question) or "대마재배자는 누구에게 허가를 받나요?"
    sr = search(question, doc=args.doc, top_k=args.top_k)

    print(f"\n질문 : {sr.query}")
    print(f"{sr.doc_label} · 청크 {sr.n_indexed}개 중 상위 {sr.top_k}개 "
          f"({sr.elapsed:.1f}초)\n")
    for rank, hit in enumerate(sr.hits, 1):
        print(f"  {rank:2d} {hit.score:.4f} {hit.key:<8s} {hit.preview(70)}")


if __name__ == "__main__":
    main()

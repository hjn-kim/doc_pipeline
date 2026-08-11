#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
파이프라인 오케스트레이터

단계별 모듈을 순서대로 돌리고 결과를 하나에 담아 돌려준다. 화면 그리는 일은
하지 않는다. 전부 로컬 모델이라 외부 호출이 없다.

    질문
      |
      +-- 1    search.py      검색 랭킹              -> SearchResult
      |
      +-- 2,3  rerank_gpu.py  리랭킹 + 최종 청크 선정 -> RerankResult
      |
      +-- 4    local_llm.py   답변 생성              -> AnswerResult
      |
      +-- 5    grade.py       정답 비교 (정답이 있을 때만)
      |
    PipelineResult

단계마다 실패해도 파이프라인이 멈추지 않는다. 각 결과 객체가 error 를 들고
있으므로 화면에서 어디가 어떻게 실패했는지 보여주면서 나머지는 계속 굴린다.
리랭킹이 죽으면 검색 점수 순서를 그대로 쓴다.

단독 실행:
    python src/main.py "대마재배자는 누구에게 허가를 받나요?"
    python src/main.py --doc rs "고객확인 기준 금액은 얼마인가요?"
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from answer import AnswerResult  # noqa: E402
from corpus import doc_label  # noqa: E402
from grade import GradeResult, grade_answer  # noqa: E402
from rerank import FINAL_TOP_N, RerankResult  # noqa: E402
from search import DEFAULT_TOP_K, SearchResult, search  # noqa: E402


@dataclass
class PipelineResult:
    """전체 결과. 화면은 이것만 보고 그린다."""

    question: str                 # 번호 접두사를 뗀 실제 질문
    raw_question: str             # 사용자가 고른 원래 문자열
    doc: str | None               # 검색 대상 문서 (None 이면 전체)
    doc_name: str                 # 화면에 쓸 이름

    search: SearchResult | None = None       # 1 단계
    rerank: RerankResult | None = None       # 2, 3 단계
    answer: AnswerResult | None = None       # 4 단계
    grade: GradeResult | None = None         # 5 단계 (정답이 있을 때만)
    elapsed: float = 0.0

    @property
    def selected(self):
        """최종 선정된 청크. 리랭킹을 건너뛰었으면 검색 상위로 대체한다."""
        if self.rerank and self.rerank.selected:
            return self.rerank.selected
        if self.search:
            return self.search.hits[:FINAL_TOP_N]
        return []

    def errors(self) -> dict[str, str]:
        """단계 이름 -> 실패 메시지. 화면에서 배너로 띄우기 위한 것."""
        out: dict[str, str] = {}
        if self.rerank and self.rerank.error:
            out["리랭킹"] = self.rerank.error
        if self.answer and self.answer.error:
            out["답변 생성"] = self.answer.error
        if self.grade and self.grade.error:
            out["정답 비교"] = self.grade.error
        return out


def strip_number(question: str) -> str:
    """
    선택 항목 앞의 "1. " 은 화면 표시용 번호다. 질문 내용이 아니므로 떼고 넘긴다.

    붙인 채로 임베딩하면 점수가 조금 깎인다.
    """
    return re.sub(r"^\d+\.\s*", "", question or "").strip()


def run_pipeline(question: str, doc: str | None = None,
                 top_k: int = DEFAULT_TOP_K,
                 final_n: int = FINAL_TOP_N,
                 gold: list[str] | None = None,
                 answer_language: str | None = None,
                 on_stage=None) -> PipelineResult:
    """
    질문 하나를 파이프라인에 통과시킨다.

    doc 은 근거를 찾을 문서다. 문서 키나 짧은 코드를 주면 그 문서 안에서만 찾고,
    주지 않으면 7개 문서를 한 색인으로 합쳐 뒤진다. 질문은 한국어, 근거는 7개
    언어, 답변은 한국어다 (answer_language 로 바꿀 수 있다).

    gold 를 주면 5단계(정답 비교)까지 돈다. data/qa.json 에 적어 둔 정답 후보
    목록이며, 화면에서 질문 번호로 찾아 넘긴다. 비어 있으면 건너뛴다.

    on_stage(단계이름, 결과) 를 주면 단계가 끝날 때마다 부른다. 단계이름은
    "search" / "rerank" / "answer" / "grade" 다. 전체가 20초 넘게 걸리는데 다
    끝나야 첫 카드가 뜨면 멈춘 것처럼 보이기 때문이다.
    """
    # 수 GB 짜리 모델을 올리므로 실제로 파이프라인을 돌 때만 import 한다.
    from local_llm import generate_answer_local
    from rerank_gpu import rerank_cross

    started = time.time()
    clean = strip_number(question)

    def emit(stage: str, payload) -> None:
        if on_stage is not None:
            on_stage(stage, payload)

    # --- 1 단계 : 검색 랭킹 -------------------------------------------------
    sr = search(clean, doc=doc, top_k=top_k)
    emit("search", sr)

    # --- 2, 3 단계 : 리랭킹 + 최종 선정 -------------------------------------
    rr = rerank_cross(clean, sr, top_n=final_n)
    emit("rerank", rr)

    # --- 4 단계 : 답변 생성 -------------------------------------------------
    ans = generate_answer_local(clean, rr.selected,
                                answer_language=answer_language)
    emit("answer", ans)

    # --- 5 단계 : 정답 비교 -------------------------------------------------
    gr = None
    if gold:
        gr = grade_answer(clean, ans.answer if (ans and ans.ok) else "", gold)
        emit("grade", gr)

    return PipelineResult(
        question=clean,
        raw_question=question,
        doc=doc,
        doc_name=doc_label(doc),
        search=sr,
        rerank=rr,
        answer=ans,
        grade=gr,
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

    parser = argparse.ArgumentParser(
        description="질문 하나를 RAG 파이프라인에 통과시킨다.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("question", nargs="*")
    parser.add_argument("--doc", default=None,
                        help="근거를 찾을 문서 키나 짧은 코드 (기본: 전체 문서)\n"
                             "목록은 python src/corpus.py")
    parser.add_argument("--answer-lang", default=None,
                        help="답변을 쓸 언어 이름 (기본: 한국어)")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"검색으로 뽑을 청크 수 (기본: {DEFAULT_TOP_K})")
    parser.add_argument("--final-n", type=int, default=FINAL_TOP_N,
                        help=f"최종 선정 청크 수 (기본: {FINAL_TOP_N})")
    parser.add_argument("--gold", default=None, nargs="*",
                        help="정답 후보. 주면 5단계(정답 비교)까지 돈다\n"
                             "질문 번호로 채점하려면 src/grade.py --run N 을 쓴다")
    args = parser.parse_args()

    question = " ".join(args.question) or "대마재배자는 누구에게 허가를 받나요?"
    result = run_pipeline(question, doc=args.doc, top_k=args.top_k,
                          final_n=args.final_n, gold=args.gold,
                          answer_language=args.answer_lang)

    sr, rr, ans = result.search, result.rerank, result.answer

    print(f"\n질문   : {result.question}")
    print(f"문서   : {result.doc_name}")

    print(f"\n[1] 검색 랭킹    청크 {sr.n_indexed}개 중 상위 {sr.top_k}개 "
          f"({sr.elapsed:.1f}초)")
    for rank, hit in enumerate(sr.hits, 1):
        print(f"    {rank:>2}위 {hit.score:.4f}  {hit.key}")

    print(f"\n[2] 리랭킹       {rr.method}  ({rr.elapsed:.1f}초)")
    for item in rr.ranked[:args.final_n + 3]:
        mark = " <= 선정" if item.rank_after <= args.final_n else ""
        moved = f"{item.moved:+d}" if item.moved else "  "
        print(f"    {item.rank_after:>2}위 (전 {item.rank_before:>2}위 {moved:>3}) "
              f"관련성 {item.percent:>9}  {item.hit.key}{mark}")

    print(f"\n[3] 최종 선정    {len(rr.selected)}개  "
          f"{', '.join(h.key for h in rr.selected)}")

    if not ans.ok:
        print(f"\n[4] 답변 생성    실패: {ans.error}")
    else:
        print(f"\n[4] 답변 생성    ({ans.elapsed:.1f}초, 근거 충분: "
              f"{'예' if ans.enough else '아니오'})")
        print(f"    {ans.answer}")
        if ans.citations:
            print(f"    인용: {', '.join(ans.citations)}")
        if ans.note:
            print(f"    참고: {ans.note}")

    gr = result.grade
    if gr is not None:
        mark = "O" if gr.correct else ("?" if gr.verdict == "판정 불가" else "X")
        print(f"\n[5] 정답 비교    [{mark}] {gr.verdict}")
        print(f"    LLM 정답  : {gr.llm_answer[:110]}")
        print(f"    실제 정답 : {gr.gold_display}")

    for stage, message in result.errors().items():
        print(f"\n[!] {stage} 실패: {message}")

    print(f"\n전체 {result.elapsed:.1f}초")


if __name__ == "__main__":
    main()

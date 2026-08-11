#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
문서 레지스트리 — data/txt 와 data/emb 를 짝지어 준다.

    data/txt/{키}.txt                   원문
    data/emb/{키}_embeddings.npz        bge-m3 로 만든 청크 벡터

파일 이름이 그대로 문서 키다 (예: ko마약류관리에관한법률). 앞 두 글자는 문서를
가리키는 짧은 코드로 쓴다. 인용 표기가 "ko#12" 처럼 짧아야 답변 안에서 읽히기
때문이다.

문서를 추가하는 방법:
    1. data/txt 에 {두글자코드}{제목}.txt 를 넣는다
    2. python src/embed.py 를 돌려 data/emb 를 다시 만든다
    3. (선택) 아래 DOC_TITLES 에 보기 좋은 제목과 언어를 적는다
       적지 않으면 파일 이름에서 뽑아 쓰므로 앱은 그대로 돈다

단독 실행:
    python src/corpus.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TXT_ROOT = ROOT / "data" / "txt"
EMB_ROOT = ROOT / "data" / "emb"
EMB_SUFFIX = "_embeddings.npz"

# 문서를 하나로 좁히지 않고 전부 뒤질 때 쓰는 값.
ALL_DOCS = "all"

# 문서 키 -> (화면에 쓸 제목, 언어 코드, 언어 이름, 한 줄 설명)
#
# 원문 파일 이름은 붙여 쓴 한글이라 그대로 화면에 내면 읽기 힘들다. 여기서 띄어
# 쓴 제목과 원문 언어를 붙여 준다. 원문 언어는 답변 프롬프트에도 들어간다
# (근거가 어느 언어로 적혀 있는지 모델에게 알려 줘야 표기를 옮기지 않는다).
DOC_TITLES: dict[str, tuple[str, str, str, str]] = {
    "ko마약류관리에관한법률": (
        "마약류 관리에 관한 법률", "ko", "한국어",
        "대한민국 법률 제21065호 · 시행 2026. 1. 2.",
    ),
    "en국가마약위협평가": (
        "2025 국가 마약 위협 평가 (NDTA)", "en", "영어",
        "미국 법무부 마약단속국(DEA)·2025년 5월 발간",
    ),
    "ch중화인민공화국형법": (
        "중화인민공화국 형법", "zh", "중국어",
        "1997년 전면개정 · 형법수정안 11까지 반영",
    ),
    "vn부패및경제범죄관련형사사건에서자산회수를위한판결집행절차": (
        "부패·경제범죄 사건 자산회수 집행 실무 안내서", "vi", "베트남어",
        "베트남 법무부 민사판결집행총국 · 미국 INL · UNDP",
    ),
    "pl경찰조사구금심문기소형사사법제도": (
        "동남아시아 고문 방지 문화의 확산", "fil", "필리핀어",
        "고문방지협회(APT) · 인도네시아·필리핀 사례 연구",
    ),
    "rs자금세탁테러자금조달대량살상무기확산자금조달방지국제기준": (
        "FATF 자금세탁·테러자금조달 방지 국제기준", "ru", "러시아어",
        "OECD/FATF 2012년 2월판 러시아어 번역본",
    ),
    "uz범죄학및형사사법": (
        "범죄학 및 형사사법 (학술지 2022)", "uz", "우즈베크어",
        "타슈켄트 국립법과대학 · ISSN 2181-2179",
    ),
}


@dataclass(frozen=True)
class Document:
    """문서 하나. 원문과 색인이 둘 다 있는 것만 만든다."""

    key: str            # 파일 이름에서 온 문서 키 (ko마약류관리에관한법률)
    code: str           # 짧은 코드 (ko). 인용 표기 "ko#12" 의 앞부분
    title: str          # 화면에 쓸 제목
    lang: str           # 원문 언어 코드 (ko / en / zh / vi / fil / ru / uz)
    lang_name: str      # 원문 언어 이름 (한국어)
    note: str           # 한 줄 설명
    txt_path: Path
    emb_path: Path

    @property
    def label(self) -> str:
        """선택 상자에 넣을 한 줄. 제목만으로는 언어가 안 보인다."""
        return f"{self.title} · {self.lang_name}"

    @property
    def has_text(self) -> bool:
        return self.txt_path.is_file()


def _describe(key: str) -> tuple[str, str, str, str]:
    """DOC_TITLES 에 없는 문서도 굴러가게 파일 이름에서 뽑아 쓴다."""
    if key in DOC_TITLES:
        return DOC_TITLES[key]
    return (key[2:] or key, key[:2], key[:2], "")


@lru_cache(maxsize=1)
def load_documents() -> tuple[Document, ...]:
    """
    data/emb 의 .npz 를 기준으로 문서 목록을 만든다.

    색인이 있어야 검색이 되므로 .npz 가 기준이다. 원문 .txt 는 없어도 되고
    (미리보기만 못 쓴다) 코드가 겹치면 뒤에 오는 문서에 숫자를 붙여 구분한다.
    """
    docs: list[Document] = []
    seen: set[str] = set()

    for path in sorted(EMB_ROOT.glob(f"*{EMB_SUFFIX}")):
        key = path.name[: -len(EMB_SUFFIX)]
        title, lang, lang_name, note = _describe(key)

        code = key[:2]
        if code in seen:                       # 앞 두 글자가 겹치는 문서
            code = f"{code}{len(seen)}"
        seen.add(code)

        docs.append(Document(
            key=key,
            code=code,
            title=title,
            lang=lang,
            lang_name=lang_name,
            note=note,
            txt_path=TXT_ROOT / f"{key}.txt",
            emb_path=path,
        ))
    return tuple(docs)


def documents() -> list[Document]:
    """문서 목록. 화면과 파이프라인이 모두 이 순서를 따른다."""
    return list(load_documents())


def find(name: str | None) -> Document | None:
    """
    문서 키 · 짧은 코드 · 제목 중 무엇으로 찾아도 되게 한다.

    None 이나 "all" 은 '전체 문서'라는 뜻이라 None 을 돌려준다.
    """
    if not name or name == ALL_DOCS:
        return None
    for doc in load_documents():
        if name in (doc.key, doc.code, doc.title, doc.label):
            return doc
    raise KeyError(f"모르는 문서입니다: {name}")


def doc_label(name: str | None) -> str:
    """화면 문구용. 전체 검색이면 문서 수를 적어 준다."""
    doc = find(name)
    if doc:
        return doc.label
    return f"전체 문서 {len(load_documents())}종"


# --------------------------------------------------------------------------
# 단독 실행
# --------------------------------------------------------------------------

def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    docs = documents()
    print(f"문서 {len(docs)}개  (원문 {TXT_ROOT}, 색인 {EMB_ROOT})\n")
    print(f"{'코드':<6}{'언어':<16}{'원문':>10}  제목")
    for doc in docs:
        size = f"{doc.txt_path.stat().st_size // 1024:,}KB" if doc.has_text else "-"
        print(f"{doc.code:<6}{doc.lang_name:<16}{size:>10}  {doc.title}")
        if not doc.has_text:
            print(f"{'':<6}[!] 원문이 없습니다: {doc.txt_path}")


if __name__ == "__main__":
    main()

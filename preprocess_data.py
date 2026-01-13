#!/usr/bin/env python3
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Set
from tqdm import tqdm

import requests

# =========================
# Configuration
# =========================
BASE_LIST_URL = "https://platform.aimuse.kr/api/data"
BASE_DETAIL_URL = "https://platform.aimuse.kr/api/data/{}"

DEFAULT_SOLR_SCRIPT = "../../solr/db"
SOLR_QUERY_SCRIPT = os.getenv("SOLR_QUERY_SCRIPT", DEFAULT_SOLR_SCRIPT)
SOLR_CONTAINER_NAME = "solr_ency"

# =========================
# Stage 0. Collect
# =========================
def extract_fields(detail_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "소장품번호": detail_data.get("#소장품번호", ""),
        "description": detail_data.get("description", ""),
        "유물": detail_data.get("유물", ""),
        "시대": detail_data.get("시대", ""),
        "유물_분류": detail_data.get("유물_분류", {}),
        "작가": detail_data.get("작가", []),
        "data_property": detail_data.get("data_property", []),
    }


def collect_data(output_path: str) -> None:
    result: List[Dict[str, Any]] = []

    response = requests.get(BASE_LIST_URL)
    response.raise_for_status()
    data_list = response.json()

    for item in tqdm(data_list, desc="데이터 수집 중"):
        item_id = item.get("소장품번호") or item.get("#소장품번호")
        if not item_id:
            continue

        detail_resp = requests.get(BASE_DETAIL_URL.format(item_id))
        detail_resp.raise_for_status()
        detail_data = detail_resp.json()

        result.append(extract_fields(detail_data))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ 수집 완료: {output_path} ({len(result)}개)")


# =========================
# Solr Utilities
# =========================
def ensure_solr_container_running():
    """Solr Docker 컨테이너가 실행 중인지 확인하고, 없으면 시작"""
    result = subprocess.run(
        ["docker", "ps", "-a", "--filter", f"name={SOLR_CONTAINER_NAME}", "--format", "{{.Status}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    status = result.stdout.strip()
    
    if "Up" in status:
        return True  # 이미 실행 중
    
    # 컨테이너 시작 시도
    print(f"🔄 Solr 컨테이너 시작 중: {SOLR_CONTAINER_NAME}")
    start_result = subprocess.run(
        ["docker", "container", "restart", SOLR_CONTAINER_NAME],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    
    if start_result.returncode == 0:
        print(f"✅ Solr 컨테이너 시작 완료")
        # 컨테이너가 완전히 시작될 때까지 잠시 대기
        import time
        time.sleep(3)
        return True
    else:
        print(f"⚠️  Solr 컨테이너 시작 실패: {start_result.stderr}")
        return False


def run_query_and_parse_json(query: str):
    script_path = os.path.join(SOLR_QUERY_SCRIPT, "query-db.sh")
    if not os.path.exists(script_path):
        return 0, []

    # Solr 스크립트 디렉토리로 이동
    script_dir = SOLR_QUERY_SCRIPT
    original_cwd = os.getcwd()
    
    try:
        os.chdir(script_dir)
        result = subprocess.run(
            ["./query-db.sh", query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    finally:
        os.chdir(original_cwd)

    # 에러 확인
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if stderr:
            print(f"⚠️  Solr 쿼리 에러 (쿼리: {query[:50]}...): {stderr}")
        return 0, []

    output = result.stdout
    try:
        json_start = output.find("{")
        if json_start == -1:
            if output.strip():
                print(f"⚠️  Solr 응답에 JSON이 없습니다 (쿼리: {query[:50]}...): {output[:200]}")
            return 0, []
        response_json = json.loads(output[json_start:])
        num_found = response_json.get("response", {}).get("numFound", 0)
        docs = response_json.get("response", {}).get("docs", [])
        return num_found, docs
    except Exception as e:
        print(f"⚠️  Solr 응답 파싱 에러 (쿼리: {query[:50]}...): {e}")
        return 0, []


# =========================
# Metadata & Search
# =========================
def normalize_headword(text: str) -> str:
    return re.sub(r"[「」『』《》()\[\]{}<>]", "", text or "")


def extract_metadata(item: Dict[str, Any]):
    """db_collection.ipynb와 동일한 로직"""
    raw_headword = item.get("유물", "")
    era = item.get("시대", "")
    hanja = ""
    authors: List[str] = []
    
    for prop in item.get("data_property", []):
        if prop.get("name") == "한자명" and not hanja:
            hanja = prop.get("value", "").split(",")[0].strip()
    
    if "작가" in item:
        authors_from_field = [a["value"] for a in item["작가"] if a.get("name") == "작가명"]
        if authors_from_field:
            authors = authors_from_field
        
        has_author_name = any(a.get("name") == "작가명" for a in item["작가"])
        if not has_author_name:
            for prop in item.get("data_property", []):
                if prop.get("name") == "작가명":
                    authors = [prop.get("value", "").strip()]
                    break
    else:
        for prop in item.get("data_property", []):
            if prop.get("name") == "작가명":
                authors = [prop.get("value", "").strip()]
                break

    return raw_headword, era, hanja, authors


# =========================
# Stage 1. Enrich (fallback)
# =========================
def generate_headword_doc(raw_headword, hanja, authors):
    """db_collection.ipynb와 동일한 로직"""
    def solr_query(q):
        return run_query_and_parse_json(q)
    
    def build_doc_text(doc):
        parts = []
        for key in ["headword", "type", "definition", "summary", "body"]:
            val = doc.get(key)
            if isinstance(val, list):
                parts.append(f"{key}: {', '.join(val)}")
            elif val:
                parts.append(f"{key}: {val}")
        return "\n".join(parts)
    
    def headword_phrase(q):
        s = q.replace('"', r'\"')
        if not s.endswith(" "):
            s += " "
        return f'headword:"{s}"'
    
    def search_with_filters(base_query, hanja, authors):
        if hanja:
            q1 = f'{base_query} AND body:"{hanja}"'
            n1, d1 = solr_query(q1)
            if n1 == 1:
                return build_doc_text(d1[0])
        for author in authors:
            q2 = f'{base_query} AND body:"{author}"'
            n2, d2 = solr_query(q2)
            if n2 == 1:
                return build_doc_text(d2[0])
        if hanja and authors:
            for author in authors:
                q3 = f'{base_query} AND body:"{hanja}" AND body:"{author}"'
                n3, d3 = solr_query(q3)
                if n3 == 1:
                    return build_doc_text(d3[0])
                elif n3 > 1:
                    return build_doc_text(d3[0])
        return ""
    
    def fallback_split_search(raw_headword, hanja, authors):
        tokens = re.findall(r'\w{2,}', raw_headword)
        combos = []
        for i in range(len(tokens)):
            h = tokens[i]
            b = " ".join(tokens[:i] + tokens[i+1:])
            if len(h) >= 2 and len(b.replace(" ", "")) >= 2:
                combos.append((h, b))
        for head, body in combos:
            base_query = f'{headword_phrase(head)} AND body:"{body}"'
            result = search_with_filters(base_query, hanja, authors)
            if result:
                return result
        return ""

    if not raw_headword or not raw_headword.strip():
        return ""

    headword = normalize_headword(raw_headword)
    if not headword:
        return ""

    base_query = headword_phrase(headword)
    num_found, docs = solr_query(base_query)
    if num_found == 1:
        return build_doc_text(docs[0])
    elif num_found > 1:
        result = search_with_filters(base_query, hanja, authors)
        if result:
            return result
    return fallback_split_search(raw_headword, hanja, authors)


def generate_reference_doc(raw_headword, hanja, authors, top_k=3):
    """db_collection.ipynb와 동일한 로직"""
    def solr_query(q):
        return run_query_and_parse_json(q)

    def get_paragraphs_with_headword(body_text, search_terms):
        t = (body_text or "")
        t = t.replace("\n\r", "\n").replace("\r\n", "\n").replace("\r", "\n")
        paras = re.split(r'\n+\s*', t)

        matched_paragraphs = set()
        for para in paras:
            para = para.strip()
            if not para:
                continue
            for term in search_terms:
                if term and term in para:
                    matched_paragraphs.add(para)
                    break
        return list(matched_paragraphs)

    def build_doc_summary(doc):
        parts = []
        for key in ["headword", "type", "definition", "summary", "body"]:
            val = doc.get(key)
            if isinstance(val, list):
                parts.append(f"{key}: {', '.join(val)}")
            elif val:
                parts.append(f"{key}: {val}")
        return "\n".join(parts)

    if not raw_headword or not raw_headword.strip():
        return []

    normalized = normalize_headword(raw_headword)
    if not normalized:
        return []

    search_terms = [
        normalized,
        re.sub(r"[「」『』《》()\[\]{}<>]", "", normalized),
        normalized.replace(" ", "")
    ]

    base_query = f'body:"{normalized}"'
    num_found, docs = solr_query(base_query)
    if num_found == 0:
        return []

    if num_found > top_k:
        filtered_docs = []
        if authors:
            for author in authors:
                filtered_query = f'body:"{normalized}" AND body:"{author}"'
                filtered_num, temp_docs = solr_query(filtered_query)
                if 0 < filtered_num <= top_k:
                    filtered_docs = temp_docs
                    break
        if (not filtered_docs or len(filtered_docs) > top_k) and hanja:
            hanja_query = f'body:"{normalized}" AND body:"{hanja}"'
            hanja_num, temp_docs = solr_query(hanja_query)
            if 0 < hanja_num <= top_k:
                filtered_docs = temp_docs
        if (not filtered_docs or len(filtered_docs) > top_k) and hanja and authors:
            for author in authors:
                combo_query = f'body:"{normalized}" AND body:"{author}" AND body:"{hanja}"'
                combo_num, temp_docs = solr_query(combo_query)
                if 0 < combo_num <= top_k:
                    filtered_docs = temp_docs
                    break
        if filtered_docs:
            docs = filtered_docs
        else:
            docs = docs[:top_k]

    docs = docs[:top_k]

    reference_entries = []
    for doc in docs:
        body = doc.get("body", "")
        if not body:
            continue
        matched_paras = get_paragraphs_with_headword(body, search_terms)
        if matched_paras:
            doc_summary = build_doc_summary(doc)
            combined_text = doc_summary + "\n\n" + "\n\n".join(matched_paras)
            reference_entries.append(combined_text)

    return reference_entries


def enrich_data(input_path: str, output_path: str, top_k: int = 3):
    # Solr 컨테이너 확인 및 시작
    if not ensure_solr_container_running():
        print(f"⚠️  경고: Solr 컨테이너를 시작할 수 없습니다.")
        print("   headword_doc과 reference_doc이 빈 값으로 생성됩니다.")
    
    # Solr 스크립트 확인
    script_path = os.path.join(SOLR_QUERY_SCRIPT, "query-db.sh")
    if not os.path.exists(script_path):
        print(f"⚠️  경고: Solr 스크립트가 없습니다: {script_path}")
        print("   headword_doc과 reference_doc이 빈 값으로 생성됩니다.")
    
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enriched = []
    success_count = 0
    for item in tqdm(data, desc="fallback 생성 중"):
        raw_headword, era, hanja, authors = extract_metadata(item)
        headword_doc = generate_headword_doc(raw_headword, hanja, authors)
        reference_doc = generate_reference_doc(raw_headword, hanja, authors, top_k)
        
        item["headword_doc"] = headword_doc
        item["reference_doc"] = reference_doc
        
        if headword_doc or reference_doc:
            success_count += 1
        
        enriched.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(enriched, f, ensure_ascii=False, indent=2)

    print(f"fallback 생성 완료: {output_path} ({len(enriched)}개)")
    print(f"Solr에서 데이터를 찾은 아이템: {success_count}개 ({success_count/len(enriched)*100:.1f}%)")


# =========================
# Stage 2. Final (원본 로직)
# =========================
def has_bigo(props):
    return any(p.get("name") == "비고" for p in props if isinstance(p, dict))


def has_description(desc):
    if isinstance(desc, list):
        return any(isinstance(d, str) and d.strip() for d in desc)
    if isinstance(desc, str):
        return bool(desc.strip())
    return False


def has_headword_doc(headword_doc):
    if headword_doc is None:
        return False
    if isinstance(headword_doc, str):
        return bool(headword_doc.strip())
    return bool(headword_doc)


def has_reference_doc(reference_doc):
    if reference_doc is None:
        return False
    if isinstance(reference_doc, list):
        return len(reference_doc) > 0
    return bool(reference_doc)


def filter_empty_items(data):
    """
    넷 다 없는 것만 제외:
    - headword_doc이 없고
    - reference_doc이 없고
    - 비고가 없고
    - description이 없으면
    → 제외
    그 외에는 모두 포함
    """
    result = []
    for item in data:
        has_headword = has_headword_doc(item.get("headword_doc"))
        has_reference = has_reference_doc(item.get("reference_doc"))
        has_bigo_val = has_bigo(item.get("data_property", []))
        has_desc_val = has_description(item.get("description"))
        
        if not (has_headword or has_reference or has_bigo_val or has_desc_val):
            continue
        
        result.append(item)
    return result


# =========================
# Stage 2. Final - 긴 설명 줄이기
# =========================
def normalize_newlines(text: str) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text.replace("\n\r", "\n").replace("\r\n", "\n").replace("\r", "\n")


def split_paragraphs(text: str) -> List[str]:
    t = normalize_newlines(text)
    paras = [p.strip() for p in re.split(r"\n+", t)]
    return [p for p in paras if p]


def normalize_for_compare(s: str) -> str:
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = re.sub(r"[「」『』《》()\[\]{}<>]", "", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def extract_headwords_from_text(text: str) -> Set[str]:
    if not text or not isinstance(text, str):
        return set()
    matches = re.findall(r'(?im)^\s*headword\s*:\s*(.+?)\s*$', text)
    heads: List[str] = []
    for m in matches:
        parts = [p.strip() for p in re.split(r'[;,]', m) if p.strip()]
        if not parts:
            parts = [m.strip()]
        heads.extend(parts)
    return {normalize_for_compare(h) for h in heads if h}


def dedup_reference_by_headword(item: Dict[str, Any]) -> None:
    headword_doc = item.get("headword_doc", "")
    reference_doc = item.get("reference_doc", [])

    headwords_in_headword_doc = extract_headwords_from_text(headword_doc)

    if isinstance(reference_doc, list) and reference_doc:
        filtered_refs: List[str] = []
        for ref in reference_doc:
            ref_heads = extract_headwords_from_text(ref)
            # ref(headword)와 headword_doc(headword)의 교집합이 있으면 제외
            if not ref_heads or headwords_in_headword_doc.isdisjoint(ref_heads):
                filtered_refs.append(ref)
        item["reference_doc"] = filtered_refs


def keep_paras_with_headword(paras: List[str], headword: str) -> List[str]:
    if not headword:
        return paras

    hw = str(headword)
    hw_compact = hw.replace(" ", "").lower()

    kept: List[str] = []
    for p in paras:
        p_lower = p.lower()
        p_compact = p.replace(" ", "").lower()
        if hw.lower() in p_lower or hw_compact in p_compact:
            kept.append(p)
    return kept


def filter_body_paragraphs_by_headword(ref_text: str, headword: str) -> Optional[str]:
    if not isinstance(ref_text, str):
        return None

    t = normalize_newlines(ref_text)

    # body: 라인 위치 탐지 (라인 시작 기준, 대소문자 무시)
    m = re.search(r"(?im)^\s*body\s*:\s*", t)
    if not m:
        # body: 태그가 없으면 문자열 전체를 문단 분할/필터링
        paras = split_paragraphs(t)
        kept = keep_paras_with_headword(paras, headword)
        if not kept:
            return None
        return "\n\n".join(kept)

    body_start = m.end()
    header_part = t[:body_start]   # '...body:' 까지 포함
    body_part = t[body_start:]     # body 이후만 문단 분리

    paras = split_paragraphs(body_part)
    kept = keep_paras_with_headword(paras, headword)

    if not kept:
        return None

    # body: 뒤에 필터된 문단만 재조립
    return header_part + "\n\n" + "\n\n".join(kept)


def estimate_tokens(text: str) -> int:
    if not isinstance(text, str):
        return 0
    korean_chars = len(re.findall(r'[가-힣]', text))
    other_chars = len(text) - korean_chars
    estimated_tokens = int(korean_chars / 1.5 + other_chars / 4)
    return estimated_tokens


def keep_only_hw_paragraphs(item: Dict[str, Any], max_length: int = 500, use_tokens: bool = False) -> None:
    headword = item.get("유물", "")
    refs = item.get("reference_doc")
    if not isinstance(refs, list) or not refs:
        return

    new_refs: List[str] = []
    for ref in refs:
        if not isinstance(ref, str):
            new_refs.append(ref)
            continue
            
        should_filter = False
        if use_tokens:
            token_count = estimate_tokens(ref)
            should_filter = token_count > max_length
        else:
            should_filter = len(ref) > max_length
        
        if should_filter:
            updated = filter_body_paragraphs_by_headword(ref, headword)
            if updated:
                new_refs.append(updated)
        else:
            new_refs.append(ref)
    item["reference_doc"] = new_refs


def build_final(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = filter_empty_items(data)

    for item in tqdm(data, desc="reference_doc 중복 제거 중"):
        dedup_reference_by_headword(item)

    for item in tqdm(data, desc="긴 설명 줄이는 중"):
        keep_only_hw_paragraphs(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ final 생성 완료: {output_path} ({len(data)}개)")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--enrich", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--input") #data 폴더 안에 저장
    parser.add_argument("--output") #data 폴더 안에 저장
    parser.add_argument("--top-k", type=int, default=3)

    args = parser.parse_args()

    if args.collect:
        collect_data(args.output)

    elif args.enrich:
        enrich_data(args.input, args.output, args.top_k)

    elif args.final:
        build_final(args.input, args.output)

    else:
        print("옵션을 지정하세요: --collect | --enrich | --final")

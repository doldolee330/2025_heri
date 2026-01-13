#!/usr/bin/env python3
"""
문화유산 해설 생성 REST API
문화유산 ID, 관람객 유형, 질문을 받아서 자연어 해설을 생성하는 API
"""

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import unquote

import torch
import requests
from flask import Flask, request, jsonify
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer

app = Flask(__name__)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "doldol330/2025_heri")
BASE_MODEL_NAME = "google/gemma-3-27b-it"
EVIDENCE_FILE = os.getenv("EVIDENCE_FILE", os.path.join(os.path.dirname(__file__), "data", "aimuse_items_enriched_final.json"))
PROMPTS_FILE = os.getenv("PROMPTS_FILE", os.path.join(os.path.dirname(__file__), "data", "prompts", "prompt_all.txt"))
# Solr 스크립트는 Q4_code 밖의 solr 폴더에 있다고 가정
DEFAULT_SOLR_SCRIPT = "../../solr/db/query-db.sh"
SOLR_QUERY_SCRIPT = os.getenv("SOLR_QUERY_SCRIPT", DEFAULT_SOLR_SCRIPT)

# Global variables
model = None
processor = None
tokenizer = None
evidence_dict = {}
visitor_prompts = {}

# Visitor type mappings
VISITOR_TYPE_MAP = {
    "탐구형": "explore",
    "취미형": "hobby",
    "의무형": "obligated",
    "휴식형": "rest",
    "사교형": "social"
}

VISITOR_TYPE_NAMES = {
    "explore": "탐구형 관람객",
    "hobby": "취미형 관람객",
    "rest": "휴식형 관람객",
    "social": "사교형 관람객",
    "obligated": "의무형 관람객"
}


def load_evidence_data(evidence_file_path: str) -> Dict:
    """Load evidence data from JSON file"""
    global evidence_dict
    
    if evidence_dict:
        return evidence_dict
    
    print(f"Loading evidence data from {evidence_file_path}...")
    with open(evidence_file_path, 'r', encoding='utf-8') as f:
        evidence_data = json.load(f)
    
    evidence_dict = {}
    for item in evidence_data:
        item_id = item.get("id") or item.get("소장품번호")
        if item_id is not None:
            evidence_dict[str(item_id)] = item
    
    print(f"Loaded {len(evidence_dict)} evidence items.")
    return evidence_dict


def load_visitor_prompts(prompts_file: str) -> Dict[str, str]:
    """Load visitor type prompts"""
    global visitor_prompts
    
    if visitor_prompts:
        return visitor_prompts
    
    print(f"Loading visitor prompts from {prompts_file}...")
    
    if os.path.exists(prompts_file):
        with open(prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        sections = re.split(r'##\s+', content)
        
        for section in sections:
            if not section.strip():
                continue
            
            lines = section.strip().split('\n')
            if not lines:
                continue
            
            first_line = lines[0].strip()
            
            if '탐구형' in first_line:
                visitor_prompts['explore'] = '## ' + section.strip()
            elif '취미형' in first_line:
                visitor_prompts['hobby'] = '## ' + section.strip()
            elif '의무형' in first_line:
                visitor_prompts['obligated'] = '## ' + section.strip()
            elif '휴식형' in first_line:
                visitor_prompts['rest'] = '## ' + section.strip()
            elif '사교형' in first_line:
                visitor_prompts['social'] = '## ' + section.strip()
    else:
        visitor_prompts = {
            "explore": "## 탐구형 관람객\n깊이 있는 정보와 전문적인 해설을 제공하세요.",
            "hobby": "## 취미형 관람객\n쉽고 간결한 설명을 제공하세요.",
            "obligated": "## 의무형 관람객\n기본적인 정보만 간단히 제공하세요.",
            "rest": "## 휴식형 관람객\n감성적이고 편안한 설명을 제공하세요.",
            "social": "## 사교형 관람객\n대화형 서술로 상호작용을 유도하세요."
        }
    
    print(f"Loaded prompts for {len(visitor_prompts)} visitor types.")
    return visitor_prompts


def load_model():
    """Load model and processor/tokenizer"""
    global model, processor, tokenizer
    
    if model is not None:
        return model, processor, tokenizer
    
    print(f"Loading model from {MODEL_PATH}...")
    
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        try:
            processor = AutoProcessor.from_pretrained(MODEL_PATH)
        except:
            print("Processor not found in checkpoint, loading from base model...")
            processor = AutoProcessor.from_pretrained(BASE_MODEL_NAME)
        
        print("Model and processor loaded successfully!")
        return model, processor, None
        
    except Exception as e:
        print(f"Error loading with processor: {e}")
        print("Trying with tokenizer only...")
        
        try:
            model = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_PATH,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("Model and tokenizer loaded successfully!")
            return model, None, tokenizer
            
        except Exception as e2:
            print(f"Error loading model: {e2}")
            raise


def run_solr_query(query: str) -> Tuple[int, List[Dict]]:
    """Execute Solr query"""
    if not os.path.exists(SOLR_QUERY_SCRIPT):
        return 0, []
    
    try:
        script_dir = os.path.dirname(SOLR_QUERY_SCRIPT)
        result = subprocess.run(
            [SOLR_QUERY_SCRIPT, query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            cwd=script_dir
        )
        output = result.stdout
        
        json_start = output.find('{')
        if json_start == -1:
            return 0, []
        
        response_json = json.loads(output[json_start:])
        num_found = response_json.get("response", {}).get("numFound", 0)
        docs = response_json.get("response", {}).get("docs", [])
        return num_found, docs
    except Exception as e:
        print(f"Solr query error: {e}")
        return 0, []


def normalize_headword(text: str) -> str:
    """Normalize artifact name (remove brackets)"""
    if not text:
        return ""
    cleaned = re.sub(r"[「」『』《》()\[\]{}<>]", "", text)
    return cleaned.strip()


def extract_metadata(item: Dict) -> Tuple[str, str, str, List[str]]:
    """Extract metadata from item"""
    raw_headword = item.get("유물", "")
    era = item.get("시대", "")
    hanja = ""
    authors = []
    
    for prop in item.get("data_property", []):
        if prop.get("name") == "한자명" and not hanja:
            hanja = prop.get("value", "").split(",")[0].strip()
    
    if "작가" in item:
        authors_from_field = [
            a["value"] for a in item["작가"] 
            if a.get("name") == "작가명"
        ]
        if authors_from_field:
            authors = authors_from_field
    else:
        for prop in item.get("data_property", []):
            if prop.get("name") == "작가명":
                authors = [prop.get("value", "").strip()]
                break
    
    return raw_headword, era, hanja, authors


def generate_headword_doc(item: Dict) -> str:
    """Generate headword_doc"""
    raw_headword, era, hanja, authors = extract_metadata(item)
    headword = normalize_headword(raw_headword)
    
    if not headword:
        return ""
    
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
    
    base_query = headword_phrase(headword)
    num_found, docs = run_solr_query(base_query)
    
    if num_found == 1:
        return build_doc_text(docs[0])
    elif num_found > 1:
        if hanja:
            q1 = f'{base_query} AND body:"{hanja}"'
            n1, d1 = run_solr_query(q1)
            if n1 == 1:
                return build_doc_text(d1[0])
        
        for author in authors:
            q2 = f'{base_query} AND body:"{author}"'
            n2, d2 = run_solr_query(q2)
            if n2 == 1:
                return build_doc_text(d2[0])
    
    return ""


def generate_reference_doc(item: Dict, top_k: int = 3) -> List[str]:
    """Generate reference_doc"""
    raw_headword, era, hanja, authors = extract_metadata(item)
    normalized = normalize_headword(raw_headword)
    
    if not normalized:
        return []
    
    base_query = f'body:"{normalized}"'
    num_found, docs = run_solr_query(base_query)
    
    if num_found == 0:
        return []
    
    if num_found > top_k:
        filtered_docs = []
        if authors:
            for author in authors:
                filtered_query = f'body:"{normalized}" AND body:"{author}"'
                filtered_num, temp_docs = run_solr_query(filtered_query)
                if 0 < filtered_num <= top_k:
                    filtered_docs = temp_docs
                    break
        
        if (not filtered_docs or len(filtered_docs) > top_k) and hanja:
            hanja_query = f'body:"{normalized}" AND body:"{hanja}"'
            hanja_num, temp_docs = run_solr_query(hanja_query)
            if 0 < hanja_num <= top_k:
                filtered_docs = temp_docs
        
        if filtered_docs:
            docs = filtered_docs
        else:
            docs = docs[:top_k]
    
    docs = docs[:top_k]
    
    reference_entries = []
    search_terms = [normalized]
    
    for doc in docs:
        body = doc.get("body", "")
        if not body:
            continue
        
        paras = re.split(r'\n+\s*', body)
        matched_paras = []
        
        for para in paras:
            para = para.strip()
            if not para:
                continue
            for term in search_terms:
                if term and term in para:
                    matched_paras.append(para)
                    break
        
        if matched_paras:
            doc_parts = []
            for key in ["headword", "type", "definition", "summary"]:
                val = doc.get(key)
                if isinstance(val, list):
                    doc_parts.append(f"{key}: {', '.join(val)}")
                elif val:
                    doc_parts.append(f"{key}: {val}")
            
            combined_text = "\n".join(doc_parts) + "\n\n" + "\n\n".join(matched_paras)
            reference_entries.append(combined_text)
    
    return reference_entries


def extract_evidence_text(evidence_item: Dict) -> str:
    """Extract text from evidence document"""
    evidence_parts = []
    
    if evidence_item.get('description'):
        desc = evidence_item['description']
        if isinstance(desc, list):
            desc = '\n'.join([d for d in desc if d])
        evidence_parts.append(f"[유물 설명]\n{desc}")
    
    if evidence_item.get('headword_doc'):
        evidence_parts.append(f"[관련 인물/작품 정보]\n{evidence_item['headword_doc']}")
    
    reference_doc = evidence_item.get('reference_doc', [])
    if reference_doc:
        if isinstance(reference_doc, list) and len(reference_doc) > 0:
            reference_text = '\n'.join([str(ref) for ref in reference_doc if ref])
            if reference_text:
                evidence_parts.append(f"[참고 문헌]\n{reference_text}")
        elif isinstance(reference_doc, str) and reference_doc:
            evidence_parts.append(f"[참고 문헌]\n{reference_doc}")
    
    return '\n\n'.join(evidence_parts) if evidence_parts else "[근거문서 정보를 찾을 수 없습니다]"


def format_messages_for_gemma3(visitor_type: str, artifact: str, question: str, evidence_text: str) -> List[Dict]:
    """Format messages for Gemma3"""
    visitor_key = VISITOR_TYPE_MAP.get(visitor_type, "hobby")
    visitor_name = VISITOR_TYPE_NAMES.get(visitor_key, visitor_type)
    visitor_prompt = visitor_prompts.get(visitor_key, "")
    
    system_content = f"""당신은 {visitor_name}에게 맞춤형 박물관 가이드를 제공하는 AI입니다. 주어진 근거문서를 바탕으로 정확하고 적절한 수준의 답변을 제공하세요.

{visitor_prompt}

근거문서:
이 유물은 '{artifact}'입니다. 다음은 이 유물에 대한 상세 정보입니다.

{evidence_text}"""
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question}
    ]
    
    return messages


def format_messages_for_tokenizer(messages: List[Dict]) -> str:
    """Format messages for tokenizer"""
    formatted = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            formatted += f"<start_of_turn>system\n{content}<end_of_turn>\n"
        elif role == "user":
            formatted += f"<start_of_turn>user\n{content}<end_of_turn>\n"
        elif role == "assistant":
            formatted += f"<start_of_turn>assistant\n{content}<end_of_turn>\n"
    formatted += "<start_of_turn>assistant\n"
    return formatted


def generate_response(visitor_type: str, artifact: str, question: str, evidence_text: str) -> str:
    """Generate response using model"""
    global model, processor, tokenizer
    
    messages = format_messages_for_gemma3(visitor_type, artifact, question, evidence_text)
    
    try:
        if processor is not None:
            chat_text = processor.apply_chat_template(
                messages,
                add_generation_prompt=True
            )

            inputs = processor.tokenizer(
                chat_text,
                return_tensors="pt"
            ).to(model.device)

            input_len = inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    top_k=50,
                )
                generation = generation[0][input_len:]

            decoded = processor.decode(generation, skip_special_tokens=True)
            return decoded.strip()

        elif tokenizer is not None:
            input_text = format_messages_for_tokenizer(messages)
            
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            input_len = inputs["input_ids"].shape[-1]
            
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generation = generation[0][input_len:]
            
            decoded = tokenizer.decode(generation, skip_special_tokens=True)
            return decoded.strip()
            
        else:
            return "모델이 로드되지 않았습니다."
            
    except Exception as e:
        print(f"Generation error: {e}")
        return f"응답 생성 중 오류가 발생했습니다: {str(e)}"


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model_loaded": model is not None})


@app.route('/generate/<heritage_id>/<visitor_type>/<path:question>', methods=['GET'])
def generate_explanation(heritage_id, visitor_type, question):
    """Generate cultural heritage explanation API"""
    try:
        cultural_heritage_id = unquote(heritage_id)
        visitor_type = unquote(visitor_type)
        question = unquote(question)
        
        if not cultural_heritage_id:
            return jsonify({
                "error": "문화유산_ID가 필요합니다.",
                "status": "error"
            }), 400
        
        if not visitor_type or visitor_type not in VISITOR_TYPE_MAP:
            return jsonify({
                "error": f"관람객_유형이 필요합니다. 가능한 값: {', '.join(VISITOR_TYPE_MAP.keys())}",
                "status": "error"
            }), 400
        
        evidence_item = evidence_dict.get(str(cultural_heritage_id))
        
        if not evidence_item:
            try:
                BASE_DETAIL_URL = "https://platform.aimuse.kr/api/data/{}"
                response = requests.get(BASE_DETAIL_URL.format(cultural_heritage_id), timeout=5)
                if response.status_code == 200:
                    detail_data = response.json()
                    evidence_item = {
                        "소장품번호": detail_data.get("#소장품번호", ""),
                        "id": detail_data.get("#소장품번호", ""),
                        "description": detail_data.get("description", ""),
                        "유물": detail_data.get("유물", ""),
                        "시대": detail_data.get("시대", ""),
                        "유물_분류": detail_data.get("유물_분류", {}),
                        "작가": detail_data.get("작가", []),
                        "data_property": detail_data.get("data_property", [])
                    }
                else:
                    return jsonify({
                        "error": f"문화유산 ID '{cultural_heritage_id}'에 해당하는 정보를 찾을 수 없습니다.",
                        "status": "error"
                    }), 404
            except Exception as e:
                return jsonify({
                    "error": f"문화유산 ID '{cultural_heritage_id}'에 해당하는 정보를 찾을 수 없습니다. (API 오류: {str(e)})",
                    "status": "error"
                }), 404
        
        if not evidence_item.get('headword_doc') and not evidence_item.get('reference_doc'):
            evidence_item['headword_doc'] = generate_headword_doc(evidence_item)
            evidence_item['reference_doc'] = generate_reference_doc(evidence_item)
        
        artifact = evidence_item.get('유물', '알 수 없는 유물')
        evidence_text = extract_evidence_text(evidence_item)
        
        response_text = generate_response(visitor_type, artifact, question, evidence_text)

        try:
            answer_match = re.search(r'"답변"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', response_text, re.DOTALL)
            if answer_match:
                response_text = answer_match.group(1).replace('\\"', '"').replace('\\n', '\n')
            else:
                response_text_stripped = response_text.strip()
                if response_text_stripped.startswith('{'):
                    try:
                        parsed = json.loads(response_text_stripped)
                        print(f"Warning: JSON response without '답변' key: {list(parsed.keys())}")
                        response_text = "죄송합니다. 적절한 답변을 생성할 수 없습니다."
                    except:
                        pass
        except Exception as e:
            print(f"JSON parsing error: {e}")
            pass
        
        result = {
            "문화유산_ID": cultural_heritage_id,
            "관람객_유형": visitor_type,
            "유물명": artifact,
            "질문": question,
            "답변": response_text,
            "근거문서_요약": {
                "description": evidence_item.get('description', ''),
                "headword_doc_존재": bool(evidence_item.get('headword_doc')),
                "reference_doc_개수": len(evidence_item.get('reference_doc', [])) if isinstance(evidence_item.get('reference_doc'), list) else (1 if evidence_item.get('reference_doc') else 0)
            },
            "status": "success"
        }
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"API error: {e}")
        return jsonify({
            "error": f"서버 오류가 발생했습니다: {str(e)}",
            "status": "error"
        }), 500


if __name__ == '__main__':
    print("=" * 60)
    print("문화유산 해설 생성 REST API 시작")
    print("=" * 60)
    
    load_evidence_data(EVIDENCE_FILE)
    load_visitor_prompts(PROMPTS_FILE)
    
    try:
        load_model()
    except Exception as e:
        print(f"모델 로드 실패: {e}")
        print("API는 모델 없이 시작됩니다. /generate 엔드포인트는 동작하지 않을 수 있습니다.")
    
    print("=" * 60)
    print("API 준비 완료")
    print("=" * 60)
    print("\nAPI 엔드포인트:")
    print("  - GET  /health     : 헬스 체크")
    print("  - GET  /generate/<heritage_id>/<visitor_type>/<question>   : 해설 생성")
    print("\n예시 요청:")
    print("""
    curl http://localhost:5000/generate/동원1257/탐구형/이%20유물에%20대해%20설명해%20주세요
    """)
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


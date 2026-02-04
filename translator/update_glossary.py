import re
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter
from itertools import islice
import ollama
import boto3  # Necessario per AWS Bedrock

# --- PERCORSI FILE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config", "config.json")
XML_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "lt-lt.xml"))
GLOSSARY_PATH = os.path.join(SCRIPT_DIR, "config", "glossary.json")

def load_config():
    """Carica la configurazione dal file JSON."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configurazione non trovata in {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# Caricamento configurazione globale
CONFIG = load_config()
AI_CHUNK_SIZE = CONFIG.get("chunk_size_glossary", 15)
MIN_FREQUENCY = 3 

def ask_llm_batch(terms):
    """
    Funzione wrapper che smista la richiesta a AWS Bedrock o Ollama
    in base alla configurazione 'llm_provider'.
    """
    provider = CONFIG.get("llm_provider", "ollama").lower()
    
    prompt = f"""
You are an Enterprise Architecture expert.

Given the following list of terms, return ONLY a valid JSON object
where each key is the term and the value is either YES or NO,
depending on whether the term is a specific technical concept
related to Enterprise Architecture or TOGAF.

Terms:
{terms}

Example response:
{{
  "application component": "YES",
  "business value": "YES",
  "random word": "NO"
}}
"""

    if provider == "aws":
        return _call_aws_bedrock(prompt)
    else:
        return _call_ollama(prompt)

def _extract_json_from_text(text):
    """Estrae e parsa il JSON dal testo della risposta, gestendo i blocchi markdown."""
    try:
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        json_content = json_match.group(1) if json_match else text.strip()
        return json.loads(json_content)
    except Exception as e:
        print(f"Errore nel parsing JSON: {e}")
        return {}

def _call_ollama(prompt):
    """Chiamata locale a Ollama."""
    try:
        model = CONFIG["ollama"]["model_name"]
        response = ollama.generate(model=model, prompt=prompt)
        content = response["response"]
        result = _extract_json_from_text(content)
        return {term for term, verdict in result.items() if str(verdict).upper() == "YES"}
    except Exception as e:
        print(f"Errore Ollama: {e}")
        return set()

def _call_aws_bedrock(prompt):
    """Chiamata a AWS Bedrock (Claude)."""
    try:
        aws_cfg = CONFIG["aws"]
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=aws_cfg["region"]
        )
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ],
            "temperature": 0
        })

        response = bedrock.invoke_model(
            body=body,
            modelId=aws_cfg["bedrock_model_id_glossary"]
        )
        
        response_body = json.loads(response.get('body').read())
        content = response_body['content'][0]['text']
        result = _extract_json_from_text(content)
        return {term for term, verdict in result.items() if str(verdict).upper() == "YES"}
    except Exception as e:
        print(f"Errore AWS Bedrock: {e}")
        return set()

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk: break
        yield chunk

def update_glossary():
    if not os.path.exists(XML_PATH):
        print(f"File XML non trovato: {XML_PATH}")
        return

    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    namespaces = {'ns': 'http://www.enterprise-architecture.org/essential/language'}
    
    all_text = []
    for message in root.findall('.//ns:message', namespaces):
        name_element = message.find('ns:name', namespaces)
        if name_element is not None and name_element.text:
            text = re.sub(r'[^a-zA-Z\s]', '', name_element.text.lower())
            all_text.append(text)

    stop_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'your', 'about'}
    words = []
    bigrams = []

    for sentence in all_text:
        tokens = [w for w in sentence.split() if len(w) > 3 and w not in stop_words]
        words.extend(tokens)
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]} {tokens[i+1]}")

    combined_counts = Counter(words) + Counter(bigrams)
    
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"technical_terms": []}

    existing_terms = {item['term'].lower() for item in data['technical_terms']}
    candidates = [t for t, count in combined_counts.items() if count >= MIN_FREQUENCY and t.lower() not in existing_terms]

    added_count = 0
    total_candidates = len(candidates)
    provider_name = CONFIG.get("llm_provider", "ollama")
    
    print(f"--- Modalità: {provider_name.upper()} ---")
    print(f"Analisi di {total_candidates} candidati (chunk size = {AI_CHUNK_SIZE})...")

    for i, chunk in enumerate(chunked(candidates, AI_CHUNK_SIZE), start=1):
        print(f"\n→ Validazione chunk {i}: {chunk}")
        valid_terms = ask_llm_batch(chunk)

        for term in chunk:
            if term in valid_terms:
                print(f" [+] Aggiunto: {term}")
                data['technical_terms'].append({
                    "term": term,
                    "do_not_translate": True,
                    "note": f"Validato via AI ({provider_name})"
                })
                added_count += 1
            else:
                print(f" [-] Scartato: {term}")

        processed = i * AI_CHUNK_SIZE
        remaining = max(total_candidates - processed, 0)
        print(f"⚡ Chunk completato. Rimanenti: {remaining}")

    with open(GLOSSARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nFine! Nuovi termini aggiunti: {added_count}")

if __name__ == "__main__":
    print("Configurazione Attuale:")
    print(f"  Provider: {CONFIG['llm_provider']}")
    if CONFIG['llm_provider'] == 'aws':
        print(f"  Bedrock Model: {CONFIG['aws'].get('bedrock_model_id_glossary')}")
    print(f"  Chunk size: {AI_CHUNK_SIZE}\n")

    update_glossary()
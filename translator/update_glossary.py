import re
from collections import Counter
import xml.etree.ElementTree as ET
import json
import os
from collections import Counter
import ollama  # Assicurati di aver installato: pip install ollama
from itertools import islice

# --- CONFIGURAZIONE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "lt-lt.xml"))
GLOSSARY_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "config", "glossary.json"))
AI_CHUNK_SIZE = 15  # Numero di termini da passare all'AI per ogni chiamata


# Parametri di filtraggio
MIN_FREQUENCY = 3  # Prendi solo termini che appaiono almeno 3 volte
OLLAMA_MODEL = "deepseek-r1:14b"  #deepseek-r1:14b, llama3.1:8b

def ask_ollama_if_ea_batch(terms):
    """
    Chiede a Ollama di classificare una lista di termini.
    Ritorna un set di termini validi per EA/TOGAF.
    Gestisce anche blocchi ```json``` nella risposta.
    """
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

    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt
        )

        content = response["response"]

        # --- Estrazione JSON da eventuali blocchi ```
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1)
        else:
            # fallback: prova a parsare tutto
            json_content = content.strip()

        # Parsing JSON
        result = json.loads(json_content)

        # Ritorna solo i termini validi
        return {term for term, verdict in result.items() if verdict.upper() == "YES"}

    except Exception as e:
        print(f"Errore Ollama (batch): {e}\nContenuto ricevuto:\n{content}")
        return set()

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

def update_glossary():
    if not os.path.exists(XML_PATH): return

    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    namespaces = {'ns': 'http://www.enterprise-architecture.org/essential/language'}
    
    # 1. Raccolta e conteggio frequenze
    all_text = []
    for message in root.findall('.//ns:message', namespaces):
        name_element = message.find('ns:name', namespaces)
        if name_element is not None and name_element.text:
            # Pulizia: solo lettere e spazi, tutto minuscolo
            text = re.sub(r'[^a-zA-Z\s]', '', name_element.text.lower())
            all_text.append(text)

    # 1. Estrazione singole parole (Unigrammi)
    # Escludiamo le "stop words" comuni che non sono termini tecnici
    stop_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'your', 'about'}
    
    words = []
    bigrams = []

    for sentence in all_text:
        tokens = [w for w in sentence.split() if len(w) > 3 and w not in stop_words]
        
        # Aggiungi parole singole
        words.extend(tokens)
        
        # Crea combinazioni di due parole (Bigrammi) - Es: "application" + "component"
        for i in range(len(tokens) - 1):
            bigrams.append(f"{tokens[i]} {tokens[i+1]}")

    # 2. Conta le frequenze di tutto
    combined_counts = Counter(words) + Counter(bigrams)
    
    #  Selezione candidati (Frequenti e non ancora nel glossario)
    if os.path.exists(GLOSSARY_PATH):
        with open(GLOSSARY_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {"technical_terms": []}

    existing_terms = {item['term'].lower() for item in data['technical_terms']}
    
    # Filtriamo per frequenza prima di chiamare l'AI (per risparmiare tempo)
    candidates = [t for t, count in combined_counts.items() if count >= MIN_FREQUENCY and t.lower() not in existing_terms]

    # 3. Validazione con Ollama
    added_count = 0
    total_candidates = len(candidates)
    print(f"Analisi di {total_candidates} candidati con {OLLAMA_MODEL} (chunk size = {AI_CHUNK_SIZE})...")

    for i, chunk in enumerate(chunked(candidates, AI_CHUNK_SIZE), start=1):
        print(f"\n→ Validazione chunk {i}: {chunk}")
        valid_terms = ask_ollama_if_ea_batch(chunk)

        for term in chunk:
            if term in valid_terms:
                print(f" [+] Aggiunto: {term}")
                data['technical_terms'].append({
                    "term": term,
                    "do_not_translate": True,
                    "note": "Validato via AI (TOGAF/EA)"
                })
                added_count += 1
            else:
                print(f" [-] Scartato: {term}")

        # Stampa quanti elementi rimangono
        processed = i * AI_CHUNK_SIZE
        remaining = max(total_candidates - processed, 0)
        print(f"⚡ Chunk completato. Rimanenti da processare: {remaining}")

    # 4. Salvataggio
    with open(GLOSSARY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nFine! Nuovi termini tecnici aggiunti: {added_count}")

if __name__ == "__main__":
    update_glossary()
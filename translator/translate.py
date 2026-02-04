import os
import sys
import re
print(f"Sto usando: {sys.executable}")
import json
import xml.etree.ElementTree as ET
import ollama
import boto3

# --- CONFIGURAZIONE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "it-it.xml"))
DB_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "config", "database.json"))
GLOSSARY_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "config", "glossary.json"))
CONFIG_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "config", "config.json"))

# Log di verifica path avvio
print(f"--- Controllo Percorsi ---")
print(f"Script in: {SCRIPT_DIR}")
print(f"Cerco XML in: {XML_PATH}")
print(f"Esiste l'XML? {'SÌ' if os.path.exists(XML_PATH) else 'NO'}")
print(f"--------------------------\n")


def load_json(path, default=[]):
    if not os.path.exists(path):
        return default
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_config():
    """Carica il file di configurazione."""
    default_config = {
        "enable_glossary_mapping_only": False,
        "enable_generate_translation_only": False,  # <--- NUOVA CONFIGURAZIONE AGGIUNTA
        "llm_provider": "ollama",   # ollama | aws
        "model_name": "mistral-nemo:12b",   # Nome modello Ollama
        "chunk_size_translate": 10, # Batch size
        "bedrock_model_id_translate": "anthropic.claude-3-haiku-20240307-v1:0", 
        "aws_region": "eu-central-1"
    }
    config = load_json(CONFIG_PATH, default=default_config)

    for key, value in default_config.items():
        if key not in config:
            config[key] = value

    return config

def extract_json_from_text(text):
    """
    Pulisce la risposta dell'LLM per estrarre solo il JSON valido.
    Gestisce i blocchi markdown ```json ... ```
    """
    try:
        # Cerca blocchi di codice json
        match = re.search(r"```json\s*(\{.*\}|\[.*\])\s*```", text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Se non trova blocchi, cerca la prima quadra/graffa aperta e l'ultima chiusa
        match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
        if match:
            return match.group(1)
            
        return text # Ritorna il testo grezzo se non trova nulla
    except Exception:
        return text

def translate_texts(texts, system_prompt, glossary_terms=None, batch_mode=True):
    """
    Traduce usando Ollama o AWS Bedrock (Claude).
    """
    provider = CONFIG["llm_provider"]

    # Prompt utente standard
    user_content = f"Translate the following numbered list to Italian. Maintain the numbering format.\n\n{texts}"

    if provider == "ollama":
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_content}
        ]
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        return extract_json_from_text(response["message"]["content"].strip())

    elif provider == "aws":
        # Inizializza Bedrock Runtime
        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=CONFIG["aws_region"]
        )

        # Costruzione Body per modelli Claude 3 (Messaggi API)
        # Nota: Se usi Llama su Bedrock, la struttura del body è diversa. 
        # Questo codice è ottimizzato per Claude (Haiku/Sonnet/Opus).
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "system": system_prompt, # Claude supporta il campo system a livello top
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content}
                    ]
                }
            ],
            "temperature": 0 # Temperatura 0 per traduzioni deterministiche
        })

        try:
            response = client.invoke_model(
                modelId=CONFIG.get("bedrock_model_id_translate", "anthropic.claude-3-haiku-20240307-v1:0"),
                body=body
            )
            
            response_body = json.loads(response.get("body").read())
            result_text = response_body.get("content")[0].get("text")
            
            return extract_json_from_text(result_text)

        except Exception as e:
            print(f"Errore AWS Bedrock: {e}")
            raise e # Rilancia l'errore per gestirlo nel fallback

    else:
        raise ValueError(f"Provider non supportato: {provider}")


def map_glossaries_to_database():
    db = load_json(DB_PATH)
    glossary = load_json(GLOSSARY_PATH, default={"technical_terms": []})

    if not db:
        print("Database vuoto.")
        return

    all_terms = [t['term'] for t in glossary.get("technical_terms", [])]

    if not all_terms:
        return

    updated_count = 0
    for item in db:
        value_text = item.get('value', '')
        item['glossaries'] = [t for t in all_terms if t in value_text]
        if item['glossaries']:
            updated_count += 1

    save_json(DB_PATH, db)
    print(f"Mappatura glossari: {updated_count}/{len(db)}")


def build_system_prompt(glossary_terms, texts_to_check, batch_mode=False):
    # Logica ottimizzata per LLM (sia Ollama che Bedrock)
    
    if isinstance(texts_to_check, list):
        combined_text = " ".join(texts_to_check)
    else:
        combined_text = texts_to_check

    relevant_terms = [term for term in glossary_terms if term in combined_text]
    glossary_str = ", ".join([f"'{term}'" for term in relevant_terms]) if relevant_terms else ""

    base_prompt = (
        "You are an expert Enterprise Architecture translator (English to Italian). "
        "Your task is to translate the provided text naturally but accurately.\n"
        "RULES:\n"
        "1. Do NOT translate technical acronyms (e.g., TOGAF, API, AWS).\n"
        "2. Do NOT change symbols like '#', '@' or numbering.\n"
    )

    if glossary_str:
        base_prompt += f"3. STRICTLY DO NOT TRANSLATE the following terms: [{glossary_str}]. Keep them in English.\n"

    if batch_mode:
        base_prompt += (
            "4. OUTPUT FORMAT: Return ONLY a valid JSON array of strings containing the translations. "
            "Example: [\"Traduzione 1\", \"Traduzione 2\"]. "
            "Do NOT output markdown code blocks or explanations."
        )
    else:
        base_prompt += "4. Output ONLY the translated string."

    return base_prompt


def import_xml_to_json():
    if not os.path.exists(XML_PATH):
        print("XML non trovato.")
        return

    db = load_json(DB_PATH)
    existing = {i['name'] for i in db if i.get("name")}

    namespace = {'ns': 'http://www.enterprise-architecture.org/essential/language'}
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    for msg in root.findall('.//ns:message', namespace):
        name_node = msg.find('ns:name', namespace)
        if name_node is not None and name_node.text:
            name = name_node.text.strip()
            if name not in existing:
                db.append({
                    "name": name,
                    "value": "",
                    "processed": False,
                    "glossaries": []
                })
                existing.add(name)

    save_json(DB_PATH, db)


def translate_items(batch_size=5):
    db = load_json(DB_PATH)
    glossary = load_json(GLOSSARY_PATH, default={"technical_terms": []})
    to_process = [i for i in db if not i['processed']]

    if not to_process:
        return False

    batch = to_process[:batch_size]

    glossary_terms_objs = [
        t['term'] for t in glossary.get("technical_terms", [])
        if t.get("do_not_translate")
    ]
    
    batch_full_text = " ".join([item['name'] for item in batch])
    relevant_glossary_terms = [t for t in glossary_terms_objs if t in batch_full_text]

    texts_list = [item['name'] for item in batch]
    system_prompt = build_system_prompt(relevant_glossary_terms, texts_list, batch_mode=True)
    
    # Per Bedrock mandiamo solo la lista pura, il prompt gestisce il formato
    texts_to_translate = json.dumps(texts_list, ensure_ascii=False) 

    response_content = translate_texts(
        texts_to_translate,
        system_prompt,
        glossary_terms=relevant_glossary_terms,
        batch_mode=True
    )

    try:
        translations = json.loads(response_content)
        
        if len(translations) != len(batch):
            print(f"Warn: Mismatch lunghezze ({len(translations)} vs {len(batch)}). Fallback.")
            raise ValueError("Mismatch batch size")

        for i, item in enumerate(batch):
            item['value'] = translations[i].strip()
            item['processed'] = True

    except Exception as e:
        print(f"Errore batch o JSON malformato: {e}. Risposta grezza: {response_content[:50]}...")
        return translate_items_fallback(batch, relevant_glossary_terms, db)

    save_json(DB_PATH, db)
    # Stampa di progresso
    remaining = len(to_process) - batch_size
    sys.stdout.write(f"\rProcessati: {len(batch)} | Rimanenti: {max(0, remaining)}   ")
    sys.stdout.flush()
    
    return len(to_process) > batch_size


def translate_items_fallback(batch, glossary_terms, db):
    print("\nAvvio Fallback riga per riga...")
    for item in batch:
        system_prompt = build_system_prompt(glossary_terms, item['name'], batch_mode=False)
        item['value'] = translate_texts(
            item['name'],
            system_prompt,
            glossary_terms=glossary_terms,
            batch_mode=False
        ).strip()
        item['processed'] = True

    save_json(DB_PATH, db)
    return True


def update_xml_with_translations():
    if not os.path.exists(XML_PATH):
        print(f"Errore: File XML non trovato in {XML_PATH}")
        return 0

    db = load_json(DB_PATH)
    # Creiamo un dizionario Name -> Value per un accesso rapido
    # NOTA: Rimuovo il filtro 'if i['processed']' se vuoi che copi TUTTO quello che c'è nel DB
    translations = {i['name']: i['value'] for i in db if i.get('value')}

    namespace = {'ns': 'http://www.enterprise-architecture.org/essential/language'}
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    updated_count = 0
    
    # Cerchiamo tutti i blocchi <message>
    for msg in root.findall('.//ns:message', namespace):
        name_node = msg.find('ns:name', namespace)
        value_node = msg.find('ns:value', namespace)
        
        if name_node is not None and name_node.text:
            name = name_node.text.strip()
            # Se il nome esiste nel nostro database JSON
            if name in translations and value_node is not None:
                old_value = value_node.text
                new_value = translations[name]
                
                # Aggiorniamo il valore e incrementiamo il contatore
                value_node.text = new_value
                updated_count += 1

    # Salvataggio con mantenimento dei namespace originali
    ET.register_namespace('', 'http://www.enterprise-architecture.org/essential/language')
    tree.write(XML_PATH, encoding='UTF-8', xml_declaration=True)
    
    return updated_count


# --- MAIN ---
if __name__ == "__main__":
    CONFIG = load_config()
    MODEL_NAME = CONFIG.get("model_name", "mistral-nemo:12b")
    CHUNK_SIZE = CONFIG.get("chunk_size_translate", 5) 

    print("--- Configurazione Attuale ---")
    print(f"Provider: {CONFIG['llm_provider']}")
    print(f"Generate Only: {CONFIG['enable_generate_translation_only']}\n")

    if CONFIG["enable_glossary_mapping_only"]:
        print("Modo: Mapping Glossario...")
        map_glossaries_to_database()

    elif CONFIG["enable_generate_translation_only"]:
        print("Modo: Sostituzione valori da Database a XML...")
        num_sostituiti = update_xml_with_translations()
        print(f"Operazione completata con successo.")
        print(f"TOTALE VOCI SOSTITUITE NELL'XML: {num_sostituiti}")

    else:
        # Flusso standard con Traduzione LLM
        import_xml_to_json()
        print("Inizio traduzione con LLM...")
        while translate_items(CHUNK_SIZE):
            pass
        
        print("\nTraduzione completata. Aggiornamento XML...")
        num_sostituiti = update_xml_with_translations()
        print(f"Voci aggiornate nell'XML: {num_sostituiti}")
        
        map_glossaries_to_database()
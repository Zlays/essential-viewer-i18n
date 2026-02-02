import os
import sys
import json
print(f"Sto usando: {sys.executable}")
import xml.etree.ElementTree as ET
import ollama

# --- CONFIGURAZIONE ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "lt-lt.xml"))
DB_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "config", "database.json"))
GLOSSARY_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "config", "glossary.json"))
MODEL_NAME = "mistral-nemo:12b" # deepseek-r1:14b, translategemma:12bm translategemma:27b, mistral-nemo:12b
CHUNK_SIZE = 5  # Numero di termini da passare all'AI per ogni chiamata

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

def build_system_prompt(glossary_terms, texts_to_check, batch_mode=False):
    """Costruisce il prompt di sistema per la traduzione.
    
    Args:
        glossary_terms: Lista completa dei termini del glossario
        texts_to_check: Testo o lista di testi da controllare per trovare termini rilevanti
        batch_mode: Se True, richiede risposta in formato JSON array
    """
    # Se texts_to_check è una lista, uniscila in un'unica stringa
    if isinstance(texts_to_check, list):
        combined_text = " ".join(texts_to_check)
    else:
        combined_text = texts_to_check
    
    # Filtra solo i termini del glossario che sono presenti nel testo
    relevant_terms = [term for term in glossary_terms if term in combined_text]
    
    glossary_str = ", ".join([f"'{term}'" for term in relevant_terms]) if relevant_terms else ""
    
    base_prompt = (
        f"You are a technical EA/TOGAF translator. Translate from English to Italian. "
        f"Do NOT replace symbols like '#', '@' (e.s: '#' must remain '#', '@' must remain '@'). ")
    
    if glossary_str:
        base_prompt += (
            f"CRITICAL: Keep these terms/symbols EXACTLY as they appear in the original text: {glossary_str}. "
        )
    
    if batch_mode:
        base_prompt += (
            "Respond with ONLY the translations in JSON format as an array, maintaining the same order. "
            "Format: [\"translation1\", \"translation2\", ...]"
        )
    else:
        base_prompt += "Respond ONLY with the translation, no comments."
    
    return base_prompt


def import_xml_to_json():
    """Legge l'XML con Namespace e popola il database JSON."""
    if not os.path.exists(XML_PATH):
        print(f"Errore: File {XML_PATH} non trovato.")
        return

    db = load_json(DB_PATH)
    # Creiamo un set dei nomi esistenti per evitare duplicati e ignorare stringhe vuote
    existing_names = {item['name'] for item in db if item.get('name')}
    
    # Definiamo il namespace trovato nel tuo file XML
    namespace = {'ns': 'http://www.enterprise-architecture.org/essential/language'}

    try:
        tree = ET.parse(XML_PATH)
        root = tree.getroot()

        # Usiamo XPath per trovare tutti i <message> ovunque siano, 
        # usando il prefisso 'ns' per il namespace
        messages = root.findall('.//ns:message', namespace)
        
        new_entries = 0
        for msg in messages:
            name_node = msg.find('ns:name', namespace)
            
            # Verifichiamo che il nodo esista e che abbia del testo (non vuoto)
            if name_node is not None and name_node.text:
                name_text = name_node.text.strip()
                
                if name_text not in existing_names:
                    db.append({
                        "name": name_text,
                        "value": "",
                        "processed": False
                    })
                    existing_names.add(name_text)
                    new_entries += 1
        
        save_json(DB_PATH, db)
        print(f"Importazione dell'XML completata: {new_entries} nuove voci aggiunte nel DB.")
        
    except ET.ParseError as e:
        print(f"Errore durante il parsing dell'XML: {e}")

def translate_items(batch_size=5):
    """Traduce i record non ancora processati in batch con un'unica chiamata AI."""
    db = load_json(DB_PATH)
    glossary = load_json(GLOSSARY_PATH, default={"technical_terms": []})
    
    # Filtra i non processati
    to_process = [i for i in db if not i['processed']]
    
    if not to_process:
        print("Nessun record da tradurre.")
        return False

    # Limitiamo al batch size per non sovraccaricare
    batch = to_process[:batch_size]
    
    # Estraiamo solo i termini da non tradurre
    glossary_terms = [
        t['term'] for t in glossary.get("technical_terms", []) 
        if t.get("do_not_translate")
    ]

    # Prepara la lista dei testi da tradurre
    texts_list = [item['name'] for item in batch]
    
    # Crea il prompt di sistema usando la funzione helper, passando i testi da controllare
    system_prompt = build_system_prompt(glossary_terms, texts_list, batch_mode=True)

    # Crea la lista numerata di frasi da tradurre
    texts_to_translate = "\n".join([f"{idx + 1}. {item['name']}" for idx, item in enumerate(batch)])
    
    print(f"Traducendo {len(batch)} termini in batch...")
    
    try:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': f"Translate these texts to Italian:\n{texts_to_translate}"},
        ]
        response = ollama.chat(model=MODEL_NAME, messages=messages)
        
        # Estrai il contenuto della risposta
        response_content = response['message']['content'].strip()
        
        # Prova a fare il parsing del JSON
        try:
            # Rimuovi eventuali backtick markdown
            if response_content.startswith("```"):
                response_content = response_content.split("```")[1]
                if response_content.startswith("json"):
                    response_content = response_content[4:]
                response_content = response_content.strip()
            
            translations = json.loads(response_content)
            
            # Verifica che il numero di traduzioni corrisponda
            if len(translations) != len(batch):
                print(f"ATTENZIONE: Numero di traduzioni ({len(translations)}) diverso dal batch ({len(batch)})")
                # Fallback: traduzione una per una
                return translate_items_fallback(batch, glossary_terms, db)
            
            # Assegna le traduzioni
            for idx, item in enumerate(batch):
                item['value'] = translations[idx].strip()
                item['processed'] = True
                print(f"{item['name']} → {item['value']}")
                
        except json.JSONDecodeError as e:
            print(f"Errore nel parsing JSON della risposta: {e}")
            print(f"Risposta ricevuta: {response_content}")
            # Fallback: traduzione una per una
            return translate_items_fallback(batch, glossary_terms, db)
            
    except Exception as e:
        print(f"Errore durante la traduzione batch: {e}")
        # Fallback: traduzione una per una
        return translate_items_fallback(batch, glossary_terms, db)

    # Salva il progresso
    save_json(DB_PATH, db)
    
    # Calcola quanti item rimangono da processare
    remaining = len(to_process) - batch_size
    if remaining > 0:
        print(f"\n✓ Batch completato. Rimangono ancora {remaining} item da processare.\n")
    else:
        print(f"\n✓ Tutti gli item sono stati processati!\n")
    
    return len(to_process) > batch_size


def translate_items_fallback(batch, glossary_terms, db):
    """Fallback: traduce gli item uno per uno se il batch fallisce."""
    print("Utilizzo modalità fallback (traduzione singola)...")
    
    for item in batch:
        print(f"Traducendo: {item['name']}...")
        try:
            # Crea il prompt di sistema filtrando i termini rilevanti per questo specifico item
            system_prompt = build_system_prompt(glossary_terms, item['name'], batch_mode=False)
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Translate: {item['name']}"},
            ]
            response = ollama.chat(model=MODEL_NAME, messages=messages)

            item['value'] = response['message']['content'].strip()
            item['processed'] = True
            print(f"Risultato: {item['value']}")
        except Exception as e:
            print(f"Errore con '{item['name']}': {e}")

    save_json(DB_PATH, db)
    return True

def update_xml_with_translations():
    """Aggiorna il file XML originale inserendo le traduzioni nei tag <value>."""
    if not os.path.exists(XML_PATH):
        print(f"Errore: File XML {XML_PATH} non trovato.")
        return
    
    db = load_json(DB_PATH)
    if not db:
        print("Database vuoto, nessuna traduzione da applicare.")
        return
    
    # Crea un dizionario name -> value per lookup veloce
    translations_dict = {item['name']: item['value'] for item in db if item.get('processed')}
    
    if not translations_dict:
        print("Nessuna traduzione processata trovata nel database.")
        return
    
    namespace = {'ns': 'http://www.enterprise-architecture.org/essential/language'}
    
    try:
        # Parse dell'XML
        tree = ET.parse(XML_PATH)
        root = tree.getroot()
        
        # Trova tutti i messaggi
        messages = root.findall('.//ns:message', namespace)
        
        updated_count = 0
        for msg in messages:
            name_node = msg.find('ns:name', namespace)
            value_node = msg.find('ns:value', namespace)
            
            if name_node is not None and name_node.text:
                name_text = name_node.text.strip()
                
                # Se abbiamo una traduzione per questo name
                if name_text in translations_dict:
                    translation = translations_dict[name_text]
                    
                    # Aggiorna il nodo value
                    if value_node is not None:
                        value_node.text = translation
                        updated_count += 1
                        print(f"✓ Aggiornato: {name_text} → {translation}")
        
        # Salva l'XML modificato
        # Registra il namespace per mantenere il prefisso corretto
        ET.register_namespace('', 'http://www.enterprise-architecture.org/essential/language')
        ET.register_namespace('xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        
        tree.write(XML_PATH, encoding='UTF-8', xml_declaration=True)
        
        print(f"\n✓ File XML aggiornato con successo!")
        print(f"  Totale traduzioni applicate: {updated_count}")
        
    except ET.ParseError as e:
        print(f"Errore durante il parsing dell'XML: {e}")
    except Exception as e:
        print(f"Errore durante l'aggiornamento dell'XML: {e}")
        
# --- ESECUZIONE ---
if __name__ == "__main__":
    # 1. Importa dati
    import_xml_to_json()

    # 2. Ciclo di traduzione
    print("Inizio traduzione...")
    while translate_items(batch_size=CHUNK_SIZE):
        print("-" * 20)
    
    print("\nTraduzione completata.")
    
    # 3. Aggiorna l'XML con le traduzioni
    print("\n" + "="*50)
    print("Aggiornamento del file XML...")
    print("="*50 + "\n")
    update_xml_with_translations()
    
    print("\nProcesso completato.")
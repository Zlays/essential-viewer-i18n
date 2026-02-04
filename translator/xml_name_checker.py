import os
import glob
import xml.etree.ElementTree as ET
import re

# --- CONFIGURAZIONE PERCORSI ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH_PATTERN = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "*.xml"))
XML_TARGET_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "it-it.xml"))
OUTPUT_REPORT = os.path.join(SCRIPT_DIR, "report_analisi.md")

NAMESPACE_URL = "http://www.enterprise-architecture.org/essential/language"
NS = {'ns': NAMESPACE_URL}

def clean_text(text):
    """Pulisce il testo da spazi e newline."""
    if text is None: return ""
    return re.sub(r'\s+', ' ', text).strip()

def get_names_from_file(file_path):
    """Estrae i tag <name> ignorando i namespace."""
    names = set()
    try:
        tree = ET.parse(file_path)
        for el in tree.iter():
            tag_local_name = el.tag.split('}')[-1]
            if tag_local_name == 'name':
                valore = clean_text(el.text)
                if valore: names.add(valore)
    except Exception as e:
        print(f"⚠️ Errore lettura {os.path.basename(file_path)}: {e}")
    return names

def update_and_format_xml(target_path, all_missing_tags):
    """Aggiorna, ordina alfabeticamente e formatta l'XML."""
    ET.register_namespace('', NAMESPACE_URL)
    
    try:
        tree = ET.parse(target_path)
        root = tree.getroot()
        
        # 1. Trova il contenitore <strings>
        strings_tag = None
        for el in root.iter():
            if el.tag.split('}')[-1] == 'strings':
                strings_tag = el
                break
        
        if strings_tag is None:
            return False

        # 2. Estrai i messaggi esistenti per evitare duplicati e prepararli al sorting
        existing_messages = []
        existing_names = set()
        
        for msg in list(strings_tag):
            name_node = msg.find("ns:name", NS)
            if name_node is not None:
                name_text = clean_text(name_node.text)
                existing_messages.append((name_text, msg))
                existing_names.add(name_text)

        # 3. Crea i nuovi messaggi per i tag mancanti
        for new_tag in all_missing_tags:
            if new_tag not in existing_names:
                # Crea struttura: <message><name>...</name><value></value></message>
                new_msg = ET.Element(f"{{{NAMESPACE_URL}}}message")
                n = ET.SubElement(new_msg, f"{{{NAMESPACE_URL}}}name")
                n.text = new_tag
                v = ET.SubElement(new_msg, f"{{{NAMESPACE_URL}}}value")
                v.text = ""
                existing_messages.append((new_tag, new_msg))

        # 4. Ordina tutto alfabeticamente per il campo 'name'
        existing_messages.sort(key=lambda x: x[0].lower())

        # 5. Svuota il tag <strings> e reinserisci i nodi ordinati
        strings_tag.clear()
        for _, msg_node in existing_messages:
            strings_tag.append(msg_node)

        # 6. Formattazione (Indentazione)
        # ET.indent è disponibile da Python 3.9+
        if hasattr(ET, 'indent'):
            ET.indent(tree, space="    ", level=0)
        
        # 7. Scrittura su file
        tree.write(target_path, encoding="UTF-8", xml_declaration=True)
        return True

    except Exception as e:
        print(f"❌ Errore durante l'aggiornamento XML: {e}")
        return False

def main():
    all_xml_files = [os.path.normpath(f) for f in glob.glob(XML_PATH_PATTERN)]
    all_other_names = set()
    target_names = set()
    file_inventory = {}
    
    target_exists = os.path.exists(XML_TARGET_PATH)

    # Scansione file
    for file_path in all_xml_files:
        fname = os.path.basename(file_path)
        names = get_names_from_file(file_path)
        file_inventory[fname] = sorted(list(names))
        
        if file_path.lower() == XML_TARGET_PATH.lower():
            target_names = names
        else:
            all_other_names.update(names)

    missing = all_other_names - target_names

    # Esecuzione aggiornamento
    xml_updated = False
    if target_exists and missing:
        xml_updated = update_and_format_xml(XML_TARGET_PATH, missing)

    # Generazione Report Markdown
    try:
        with open(OUTPUT_REPORT, "w", encoding="utf-8") as md:
            md.write("# Report Traduzioni e Allineamento\n\n")
            
            md.write("## 1. Azioni eseguite su It-It.xml\n")
            if not target_exists:
                md.write("> ❌ **ERRORE**: File target non trovato.\n")
            elif xml_updated:
                md.write(f"✅ Il file è stato aggiornato, **ordinato dalla A alla Z** e formattato.\n")
                md.write(f"Aggiunti **{len(missing)}** nuovi tag.\n")
            else:
                md.write("✅ Nessuna azione necessaria: il file è già allineato.\n")
            
            md.write("\n---\n\n")
            md.write("## 2. File analizzati\n")
            for f in sorted(file_inventory.keys()):
                target_label = "🎯" if os.path.join(os.path.dirname(XML_TARGET_PATH), f).lower() == XML_TARGET_PATH.lower() else ""
                md.write(f"- {f} {target_label} ({len(file_inventory[f])} tag)\n")

            md.write("\n---\n\n")
            md.write("## 3. Elenco completo tag per file\n")
            for fname, names in sorted(file_inventory.items()):
                md.write(f"### {fname}\n")
                for n in names: md.write(f"- {n}\n")
                md.write("\n")

        print(f"✅ Operazione completata!")
        print(f"📈 Report: {OUTPUT_REPORT}")
        if xml_updated: print(f"🔤 XML aggiornato e ordinato: {XML_TARGET_PATH}")

    except Exception as e:
        print(f"❌ Errore: {e}")

if __name__ == "__main__":
    main()
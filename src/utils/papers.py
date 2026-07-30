from config import DATA_DIR, MODELS_DIR
from pathlib import Path
import re
import shutil
import tarfile
import arxiv
import os


arxiv_client = arxiv.Client()

TMP_DIR = os.environ.get('TMP_DIR_PATH', f"{DATA_DIR}/papers/.temp")

print(f"DEBUG: TMP_DIR_PATH={TMP_DIR}")

# Kommentare, die nicht mit \ beginnen
COMMENT_REGEX = re.compile(r"(?<!\\)%.*")
INPUT_REGEX = re.compile(r"\\(?:input|include|subfile)\{([^}]+)\}")

ENV_REGEX = re.compile(
    r"\\begin\{(figure|table|abstract|keywords|keyword|titlepage|landscape|IEEEkeywords|equation|split)\*?\}[\s\S]*?\\end\{\1\*?\}",
    re.IGNORECASE
)
APPENDIX_REGEX = re.compile(
    r"(?:\\append(?:ix|ices)|\\begin\{append(?:ix|ices)\}|\\section\*?\{append(?:ix|ices)\})", re.MULTILINE | re.IGNORECASE)
# Befehle die wir behalten wollen
SECTION_REGEX = re.compile(
    r"\\(section|subsection|subsubsection|paragraph)\*?\{([^}]+)\}", re.IGNORECASE)
TEXT_CMD_REGEX = re.compile(
    r"\\(textit|textbf|texttt|emph|blue)\{([^}]+)\}", re.IGNORECASE)
COMPLEX_CMD_REGEX = re.compile(r"~?\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})?")


def download_paper_src(paper_id: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)

    source_filename = f"{paper_id}_source.tar.gz"
    source_path = os.path.join(save_dir, source_filename)

    if os.path.exists(source_path):
        print(f"{paper_id} schon heruntergeladen. Benutze bereits existierende Datei.")
        return source_path

    try:
        search = arxiv.Search(id_list=[paper_id])
        try:
            paper = next(arxiv_client.results(search))
        except StopIteration:
            print(f"Fehler: Paper mit ID {paper_id} nicht gefunden.")
            return None

        print(f"Starte download von Paper: {paper}")
        paper.download_source(dirpath=save_dir, filename=source_filename)
        print(f"Quellcode erfolgreich heruntergeladen: {source_path}")
    except Exception as e:
        print(f"Konnte Quellcode nicht herunterladen: {e}")
        return None

    return source_path


DOCUMENTCLASS_REGEX = re.compile(r'^\s*\\documentclass', re.MULTILINE)


def _find_latex_root_file(directory):
    """
    Findet die Haupt-.tex-Datei.
    Strategie:
    1. Suche ALLE Dateien, die ein nicht-auskommentiertes \\documentclass enthalten.
    2. Wenn es mehrere gibt, nimm die Datei mit der größten Dateigröße (Bytes).
    """

    candidates = []

    print(f"Suche nach Haupt-Datei in: {directory}")

    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".tex"):
                file_path = os.path.join(root, file)
                try:
                    file_size = os.path.getsize(file_path)

                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(4096)

                        if DOCUMENTCLASS_REGEX.search(content):
                            candidates.append((file_path, file_size))

                except Exception as e:
                    print(f"Warnung: Konnte {file_path} nicht prüfen: {e}")

    if not candidates:
        print("Keine Datei mit \\documentclass gefunden.")
        return None

    candidates.sort(key=lambda x: x[1], reverse=True)

    best_candidate = None

    if len(candidates) > 1:
        print("⚠️ Mehrere Start-Dateien gefunden:")
        for c in candidates:
            print(f"   - {os.path.basename(c[0])}: {c[1]} bytes")

        print("Prüfe ob 'main' in einem Namen enthalten ist")

        for c in candidates:
            file_name = os.path.basename(c[0]).lower()
            if "main" in file_name or "paper" in file_name:
                if (("example" in file_name) or ("sample" in file_name) or ("supplement" in file_name) or ("response" in file_name) or ("presentation" in file_name)):
                    continue

                best_candidate = c[0]
                print("'main' oder 'paper' wurde gefunden.")
                break

        if best_candidate is None:
            print("Keine Datei mit 'main' gefunden. Wähle die größte Datei aus. (Falls möglich ohne example im Namen)")
            for c in candidates:
                file_name = os.path.basename(c[0]).lower()
                if not (("example" in file_name) or ("supplement" in file_name) or ("response" in file_name) or ("presentation" in file_name)):
                    best_candidate = c[0]
                    break

            if best_candidate is None:
                best_candidate = candidates[0][0]

    else:
        best_candidate = candidates[0][0]

    print(f"✅ Ausgewählte Root-Datei: {os.path.basename(best_candidate)}")

    return best_candidate


def parse_tex_file(file_path: Path, processed_files=None, root_dir=None):
    """
    Liest eine .tex-Datei rekursiv, folgt \\input-Befehlen auch in Unterordnern.
    """
    if processed_files is None:
        processed_files = set()

    if file_path.suffix.lower() != '.tex':
        file_path = file_path.with_suffix('.tex')
    abs_path = os.path.abspath(file_path)

    if abs_path in processed_files:
        return ""
    if not os.path.exists(abs_path):
        print(f"Warnung: {abs_path} nicht gefunden.")
        return ""

    processed_files.add(abs_path)
    if root_dir is None:
        root_dir = os.path.dirname(abs_path)

    print(f"    ... verarbeite {abs_path}")

    try:
        with open(abs_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        content = COMMENT_REGEX.sub("", content)

        def replace_input(match):
            rel_path = match.group(1).strip()
            if not rel_path.endswith('.tex'):
                rel_path += '.tex'

            # Suche erst relativ zur aktuellen Datei, dann zum Root
            full_path = Path(os.path.join(os.path.dirname(abs_path), rel_path))

            if not full_path.exists():
                # Suche case insensitiv nach dem Pfad
                if full_path.parent.exists():
                    for child in full_path.parent.iterdir():
                        if child.name.lower() == full_path.name.lower():
                            full_path = child
                            break

            if not full_path.exists():
                print(
                    f"       -> Relativer Pfad nicht gefunden ({full_path}), versuche Root...")
                full_path = Path(os.path.join(root_dir, rel_path))
                if full_path.parent.exists():
                    for child in full_path.parent.iterdir():
                        if child.name.lower() == full_path.name.lower():
                            full_path = child
                            break

            if not full_path.exists():
                print(f"Datei: '{full_path}' nicht gefunden")
                return ""

            return parse_tex_file(full_path, processed_files, root_dir)

        content = INPUT_REGEX.sub(replace_input, content)

        return content

    except Exception as e:
        print(f"Fehler bei {abs_path}: {e}")
        return ""


def get_paper_full_tex(paper_path: str, is_folder: bool = False) -> str | None:
    if not is_folder:
        if os.path.exists(TMP_DIR):
            shutil.rmtree(TMP_DIR)

        try:
            with tarfile.open(paper_path, "r:gz") as tar:
                tar.extractall(path=TMP_DIR, filter=f'{DATA_DIR}')

        except tarfile.ReadError:
            print(f"Warnung: {paper_path} ist kein valides tar.gz Archiv.")

            with open(paper_path, 'rb') as f:
                header = f.read(4)

            if header.startswith(b'%PDF'):
                print(
                    "-> Es handelt sich tatsächlich um eine PDF-Datei, keine LaTeX-Quelle.")
                return None
            else:
                print("-> Dateiformat unbekannt oder Datei beschädigt.")
                return None
        except Exception as e:
            print(f"Fehler beim Entpacken des Papers: {e}")
            return None

        # os.remove(paper_path) # Lösche dir tar datei

        main_tex = _find_latex_root_file(TMP_DIR)
    else:
        main_tex = _find_latex_root_file(paper_path)

    if not main_tex:
        return None

    raw_content = parse_tex_file(Path(main_tex), root_dir=TMP_DIR)

    return raw_content


ABSTRACT_ENV_PATTERN = r'\\begin\{abstract\}(.*?)\\end\{abstract\}'
ABSTRACT_SEC_PATTERN = r'\\(?:section|chapter)\*?\{abstract\}(.*?)(?=\\(section|chapter)|$)'


def get_abstract(paper_tex: str) -> str | None:
    """
    Extrahiert den Text zwischen \\begin{abstract} und \\end{abstract}.
    """

    match_env = re.search(ABSTRACT_ENV_PATTERN, paper_tex,
                          re.DOTALL | re.IGNORECASE)

    if match_env:
        return match_env.group(1).strip()

    # Um \section oder \chapter{abstract} zu erkennen
    match_sec = re.search(ABSTRACT_SEC_PATTERN, paper_tex,
                          re.DOTALL | re.IGNORECASE)
    if match_sec:
        return match_sec.group(1).strip()

    # Code um Beispiele wie "\ABSTRACT{...}" zu erkennen
    # Dabei werden die öffnenden und schließenden Klammern gezählt um sicherzugehen, dass nicht zu früh abgebrochen wird
    cmd_pattern = r'\\(?:ABSTRACT|abstract)\s*\{'
    match_cmd = re.search(cmd_pattern, paper_tex)

    if match_cmd:
        start_index = match_cmd.end()

        balance = 1
        result_text = []

        for char in paper_tex[start_index:]:
            if char == '{':
                balance += 1
            elif char == '}':
                balance -= 1

            if balance == 0:
                return "".join(result_text).strip()

            result_text.append(char)

    return None


INTRODUCTION_PATTERN_SECTION = r'\\(section|firstsection)\*?\s*(?:\[[^\]]*\])?\s*\{[^}]*(introduction|background)[^}]*\}'
INTRODUCTION_PATTERN_CHAPTER = r'\\(chapter)\*?\s*(?:\[[^\]]*\])?\s*\{[^}]*(introduction|background)[^}]*\}'
SECTION_START_PATTERN = r'\\(section|firstsection)\*?\s*(?:\[[^\]]*\])?\s*{\s*(?![^}]*\b(?:introduction|background)\b)([^}]+)\}'
CHAPTER_START_PATTERN = r'\\(chapter)\*?\s*(?:\[[^\]]*\])?\s*{\s*(?![^}]*\b(?:introduction|background)\b)([^}]+)\}'


def get_introduction(paper_tex: str) -> str | None:
    """
    Extrahiert den Text der Introduction Section bis zur nächsten Section.
    """
    chapter_keyword = False

    start_match = re.search(INTRODUCTION_PATTERN_SECTION,
                            paper_tex, re.IGNORECASE)
    if not start_match:
        start_match = re.search(
            INTRODUCTION_PATTERN_CHAPTER, paper_tex, re.IGNORECASE)
        chapter_keyword = True

    if start_match:
        # Start (\section{introduction}) finden
        start_index = start_match.end()
        remaining_text = paper_tex[start_index:]

        # 2. Ende finden (nächste Section)
        if chapter_keyword:
            end_match = re.search(CHAPTER_START_PATTERN,
                                  remaining_text, re.IGNORECASE)
        else:
            end_match = re.search(SECTION_START_PATTERN,
                                  remaining_text, re.IGNORECASE)

        if end_match:
            # Text bis zum Start der nächsten Section
            intro_text = remaining_text[:end_match.start()]
        else:
            # Falls keine weitere Section folgt (unwahrscheinlich, aber möglich),
            # nehmen wir den Text bis zum Dokumentende (\end{document})
            end_document_pattern = r'\\end\{document\}'
            doc_end_match = re.search(
                end_document_pattern, remaining_text, re.IGNORECASE)
            if doc_end_match:
                intro_text = remaining_text[:doc_end_match.start()]
            else:
                intro_text = remaining_text

        intro_text = intro_text.strip()

        if intro_text:
            return intro_text
    else:
        first_section_match = re.search(
            SECTION_START_PATTERN, paper_tex, re.IGNORECASE)

        if first_section_match:
            # Wir extrahieren alles vom Anfang bis zum Start der ersten Section
            fallback_text = paper_tex[:first_section_match.start()].strip()

            fallback_text = clean_latex(fallback_text)

            if fallback_text:
                return fallback_text

        return None


def clean_latex(paper_tex: str):
    cleaned_text = str(paper_tex)

    # Alles vor \begin{document} abschneiden
    start_idx = paper_tex.find("\\begin{document}")
    if start_idx != -1:
        cleaned_text = cleaned_text[start_idx:]

    # Entferne alle LaTeX umgebungen die unnötig Platz und Tokens belegen (Tabellen, Figures, etc.)
    cleaned_text = ENV_REGEX.sub("", cleaned_text)

    # Den Text aus "guten" Befehlen behalten (z.b. \section{Introduction})
    # Dafür wird der Befehl entfernt aber der Text in dem Befehl erhalten: "\section{Introduction}"" -> "# Introduction"
    cleaned_text = SECTION_REGEX.sub(r"\n\n# **\2**\n", cleaned_text)
    # Text aus italic etc. behalten
    cleaned_text = TEXT_CMD_REGEX.sub(r"\2", cleaned_text)

    # Appendix entfernen (falls vorhanden) um Tokens zu sparen
    parts = APPENDIX_REGEX.split(cleaned_text)
    if len(parts) > 1:
        cleaned_text = parts[0]

    # Großer Cleanup von allen LaTeX commands
    cleaned_text = COMPLEX_CMD_REGEX.sub("", cleaned_text)

    # Entferne alle verbleibenden leeren Klammern
    cleaned_text = re.sub(r"[\{\}]", "", cleaned_text)

    # Random tilden entfernen (waren ab und zu da)
    cleaned_text = re.sub(r"~", " ", cleaned_text)

    # Normalisiere Leerzeichen und Zeilenumbrüche
    # Mehrfache Leerzeichen
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    # Mehrfache Zeilenumbrüche
    cleaned_text = re.sub(r"\n\s*\n", "\n\n", cleaned_text)

    return cleaned_text.strip()

from config import DATA_DIR, MODELS_DIR
import os
import pandas as pd


def generate_attribution_markdown():
    # Pfade zu den Dateien definieren
    train_path = f"{DATA_DIR}/datasets/.temp/paper_dataset.csv"
    test_path = f"{DATA_DIR}/datasets/.temp/unseen_paper_dataset.csv"
    base_path = f"{DATA_DIR}/datasets/.temp/arxiv_base_dataset.csv"
    output_path = "ATTRIBUTION.md"

    print("Starte Generierung der ATTRIBUTION.md...")

    # 1. Überprüfen, ob alle Quelldateien existieren
    for path in [train_path, test_path, base_path]:
        if not os.path.exists(path):
            print(f"Fehler: Die Datei '{path}' wurde nicht gefunden.")
            return

    # 2. Geladene IDs aus Trainings- und Testsets extrahieren
    try:
        train_df = pd.read_csv(train_path, dtype={"paper_id": str})
        test_df = pd.read_csv(test_path, dtype={"paper_id": str})
        base_df = pd.read_csv(base_path, dtype={"paper_id": str})
    except Exception as e:
        print(f"Fehler beim Lesen der CSV-Dateien: {e}")
        return

    # Alle genutzten IDs zusammenführen und Duplikate entfernen
    used_ids = set()
    if "paper_id" in train_df.columns:
        used_ids.update(train_df["paper_id"].unique())
    if "paper_id" in test_df.columns:
        used_ids.update(test_df["paper_id"].unique())

    print(
        f"Gefundene eindeutige Paper-IDs in den genutzten Datensätzen: {len(used_ids)}"
    )

    # 3. Basis-Datensatz filtern
    # Wir filtern nur die Paper, die tatsächlich in deinen Trainings-/Testdaten gelandet sind
    matched_metadata = base_df[base_df["id"].isin(used_ids)].copy()

    # Fehlende Werte durch leere Strings ersetzen
    matched_metadata["title"] = matched_metadata["title"].fillna(
        "Unknown Title")
    matched_metadata["authors"] = matched_metadata["authors"].fillna(
        "Unknown Authors"
    )
    matched_metadata["license"] = matched_metadata["license"].fillna(
        "Not Specified"
    )

    print(
        f"Metadaten für {len(matched_metadata)} von {len(used_ids)} Papern erfolgreich gematcht."
    )

    # 4. Markdown-Inhalt aufbauen
    markdown_lines = []

    # Titel und Einleitung (deckt Punkt c. und iii. global ab)
    markdown_lines.append("# Data Attribution & Licensing Notice\n")
    markdown_lines.append(
        "This file contains the mandatory attributions, copyright notices, and license notices "
        "for the arXiv papers utilized during the training and evaluation of the AI models in this thesis. "
        "All materials listed below are licensed under their respective Public Licenses (e.g., Creative Commons), "
        "as indicated by the provided license URIs.\n"
    )

    markdown_lines.append("## Index of Attributed Works\n")

    # Tabellenkopf
    # Authors & Copyright deckt i. und ii. ab
    # License URI deckt iii. und c. ab
    # Material Link deckt v. ab
    markdown_lines.append(
        "| Title | Authors & Copyright Notice | License URI | Material Link |"
    )
    markdown_lines.append(
        "| :--- | :--- | :--- | :--- |"
    )

    # Zeilen für die Tabelle generieren
    for _, row in matched_metadata.iterrows():
        p_id = row["id"]
        title = (
            str(row["title"])
            .replace("|", "\\|")
            .replace("\n", " ")
            .strip()
        )
        authors = (
            str(row["authors"])
            .replace("|", "\\|")
            .replace("\n", " ")
            .strip()
        )
        # Die arXiv 'license' Spalte enthält oft direkt die URL (z.B. http://creativecommons.org/licenses/by/4.0/)
        license_val = str(row["license"]).strip()

        # Falls die Lizenz eine URL ist, machen wir sie klickbar, ansonsten geben wir sie als Text aus
        if license_val.startswith("http"):
            license_text = f"[License Text]({license_val})"
        else:
            license_text = f"{license_val} (License URI not explicitly supplied)"

        link = f"https://arxiv.org/abs/{p_id}"

        # Tabellenzeile hinzufügen (mit vorangestelltem Copyright-Hinweis bei den Autoren)
        markdown_lines.append(
            f"| *{title}* | Copyright © {authors} | {license_text} | {link} |"
        )

    # 5. Gesetzlich geforderte CC-Haftungsausschlüsse und Hinweise hinzufügen
    markdown_lines.append("\n## License Disclaimers & Notices\n")

    # Exakter Originaltext der CC BY 4.0 Section 5
    cc_disclaimer = """### Disclaimer of Warranties and Limitation of Liability

a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS, IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION, WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS, ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION, NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT, INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES, COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

c. The disclaimer of warranties and limitation of liability provided above shall be interpreted in a manner that, to the extent possible, most closely approximates an absolute disclaimer and waiver of all liability."""

    markdown_lines.append(cc_disclaimer)
    markdown_lines.append("\n---\n")

    # Hinweise zu Modifikationen und Copyright bleiben wichtig für ML-Datensätze
    markdown_lines.append("### Modifications & Usage Context")
    markdown_lines.append(
        "The raw data extracted from the source materials listed above has been tokenized, "
        "preprocessed, and formatted exclusively to serve as training and evaluation inputs for the machine learning "
        "models described in the accompanying Master's thesis. No semantic modifications to the fundamental integrity "
        "of the original texts were performed."
    )

    markdown_lines.append("\n### Copyright Notice")
    markdown_lines.append(
        "All copyrights remain with their respective owners (the authors indicated in the table above). "
        "The use of these materials within this research framework is permitted under the explicit terms of the "
        "indicated open-access licenses attached to each record."
    )

    # Datei schreiben
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_lines))
        print(
            f"Erfolgreich! Die Datei '{output_path}' wurde mit {len(matched_metadata)} Einträgen erstellt."
        )
    except Exception as e:
        print(f"Fehler beim Schreiben der Ausgabedatei: {e}")


if __name__ == "__main__":
    generate_attribution_markdown()

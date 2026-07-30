from config import DATA_DIR, MODELS_DIR
import pandas as pd
import matplotlib.pyplot as plt

# 1. Konfiguration
csv_dateipfad = f'{DATA_DIR}/abstract_evaluation/model_trade_off.csv'

spalte_name = 'model'
spalte_winrate = 'winrate'
spalte_style = 'style accuracy'
spalte_perplexity = 'median perplexity'

datei_plot1 = f'{DATA_DIR}/abstract_evaluation/tradeoff_winrate.png'
datei_plot2 = f'{DATA_DIR}/abstract_evaluation/tradeoff_perplexity.png'


def erstelle_und_speichere_plots():
    try:
        # 2. Daten laden
        df = pd.read_csv(csv_dateipfad)

        # Bereinigung: Leerzeichen in den Modellnamen entfernen
        df[spalte_name] = df[spalte_name].str.strip()

        # 3. Ground Truth (Zielpunkt) von den restlichen Modellen trennen
        df_gt = df[df[spalte_name] == 'Ground Truth']
        df_models = df[df[spalte_name] != 'Ground Truth']

        if df_gt.empty:
            print("Warnung: 'Ground Truth' wurde in der Spalte 'model' nicht gefunden.")
            return

        # Werte für den Zielpunkt extrahieren
        gt_style = df_gt[spalte_style].values[0]
        gt_perp = df_gt[spalte_perplexity].values[0]
        # Festgelegt auf 100% laut Wunsch
        gt_winrate = df_gt[spalte_winrate].values[0]

        # 4. Figure mit 2 Subplots nebeneinander initialisieren (1 Reihe, 2 Spalten)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # ==========================================
        # PLOT 1: Winrate vs. Style Accuracy
        # ==========================================
        fig1, ax1 = plt.subplots(figsize=(10, 8))

        # Normale Modelle plotten
        ax1.scatter(x=df_models[spalte_style], y=df_models[spalte_winrate],
                    color='royalblue', s=120, alpha=0.7, edgecolors='black', label='Models')

        # Zielpunkt (Ground Truth) plotten
        ax1.scatter(x=gt_style, y=gt_winrate,
                    color='crimson', marker='s', s=120, edgecolors='black', zorder=5, label='Ground truth')

        # Modellnamen für normale Modelle beschriften
        for _, row in df_models.iterrows():
            ax1.annotate(row[spalte_name], (row[spalte_style], row[spalte_winrate]),
                         textcoords="offset points", xytext=(8, 4), ha='left', fontsize=15, weight='bold')

        # Zielpunkt beschriften
        # ax1.annotate('Optimum', (gt_style, gt_winrate),
        #              textcoords="offset points", xytext=(-15, -18), ha='center', fontsize=10, weight='bold', color='crimson')

        # Achsenbeschriftungen & Layout für Plot 1
        ax1.set_xlabel(spalte_style, fontsize=22)
        ax1.set_ylabel(spalte_winrate, fontsize=22)
        ax1.tick_params(axis='both', labelsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Achsenlimits setzen
        ax1.set_xlim(df[spalte_style].min() - 0.075,
                     df[spalte_style].max() + 0.025)
        ax1.set_ylim(df[spalte_winrate].min() - 0.15, 1.05)

        # Zeichnet einen gestrichelten roten Rahmen/Fadenkreuz zum Zielpunkt oben rechts
        ax1.hlines(y=gt_winrate, xmin=ax1.get_xlim()[
                   0], xmax=gt_style, color='crimson', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
        ax1.vlines(x=gt_style, ymin=ax1.get_ylim()[
                   0], ymax=gt_winrate, color='crimson', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
        ax1.legend(loc='lower left', fontsize=18)

        # Plot 1 unabhängig speichern
        fig1.tight_layout()
        fig1.savefig(datei_plot1, dpi=300, bbox_inches='tight')
        print(f"Erster Plot erfolgreich gespeichert unter: {datei_plot1}")

        # ==========================================
        # PLOT 2: Perplexity Median vs. Style Accuracy
        # ==========================================
        fig2, ax2 = plt.subplots(figsize=(10, 8))

        # Normale Modelle plotten
        ax2.scatter(x=df_models[spalte_style], y=df_models[spalte_perplexity],
                    color='darkorange', s=120, alpha=0.7, edgecolors='black', label='Models')

        # Zielpunkt (Ground Truth) plotten
        ax2.scatter(x=gt_style, y=gt_perp,
                    color='crimson', marker='s', s=120, edgecolors='black', zorder=5, label='Ground truth')

        # Modellnamen für normale Modelle beschriften
        for _, row in df_models.iterrows():
            ax2.annotate(row[spalte_name], (row[spalte_style], row[spalte_perplexity]),
                         textcoords="offset points", xytext=(8, 4), ha='left', fontsize=15, weight='bold')

        # Zielpunkt beschriften
        # ax2.annotate('Optimum', (gt_style, gt_perp),
        #              textcoords="offset points", xytext=(0, -18), ha='center', fontsize=10, weight='bold', color='crimson')

        # Achsenbeschriftungen & Layout für Plot 2
        ax2.set_xlabel(spalte_style, fontsize=22)
        ax2.set_ylabel(spalte_perplexity, fontsize=22)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)

        # Achsenlimits setzen
        ax2.set_xlim(df[spalte_style].min() - 0.075,
                     df[spalte_style].max() + 0.025)
        ax2.set_ylim(df[spalte_perplexity].min() - 1.5,
                     df[spalte_perplexity].max() + 0.5)

        # Zeichnet einen gestrichelten roten Rahmen/Fadenkreuz zum Zielpunkt
        ax2.hlines(y=gt_perp, xmin=ax2.get_xlim()[
                   0], xmax=gt_style, color='crimson', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)
        ax2.vlines(x=gt_style, ymin=ax2.get_ylim()[
                   0], ymax=gt_perp, color='crimson', linestyle='--', linewidth=1.5, alpha=0.6, zorder=1)

        ax2.legend(loc='lower left', fontsize=18)

        # Plot 2 unabhängig speichern
        fig2.tight_layout()
        fig2.savefig(datei_plot2, dpi=300, bbox_inches='tight')
        print(f"Zweiter Plot erfolgreich gespeichert unter: {datei_plot2}")

        # Beide Fenster am Bildschirm anzeigen
        plt.show()

    except FileNotFoundError:
        print(f"Fehler: Die Datei '{csv_dateipfad}' wurde nicht gefunden.")
    except KeyError as e:
        print(
            f"Fehler: Die Spalte {e} fehlt in der CSV. Bitte Spaltennamen prüfen.")
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")


if __name__ == "__main__":
    erstelle_und_speichere_plots()

from config import DATA_DIR, MODELS_DIR
import os
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap


def verify_activations_per_layer(file_path_1, file_path_2, file_path_hidden):
    """
    Vergleicht Activation Vectors aus zwei Dateien und setzt sie in Relation
    zu den unbeeinflussten Hidden Activations des Modells.
    """
    print(f"\n--- Lade Dateien für den erweiterten Schicht-Vergleich ---")
    try:
        activations_1 = load_file(file_path_1)
        print(f"✅ Referenz-Datei (Manuell) geladen: {file_path_1}")
        activations_2 = load_file(file_path_2)
        print(f"✅ Vergleichs-Datei (MLP) geladen: {file_path_2}")
        activations_hidden = load_file(file_path_hidden)
        print(f"✅ Hidden Activations geladen: {file_path_hidden}")
    except Exception as e:
        print(f"❌ Fehler beim Laden der Dateien: {e}")
        return None

    # Vektoren anhand des Keys nach Layern gruppieren
    # Erwartetes Key-Format: "{paper_id}_layer_{layer}"
    layer_keys = defaultdict(list)
    for key in activations_1.keys():
        if "_layer_" in key:
            layer = key.split("_layer_")[-1]
            layer_keys[layer].append(key)

    sorted_layers = sorted(layer_keys.keys(), key=lambda x: int(x))

    if not sorted_layers:
        print("❌ Keine passenden Keys mit '_layer_' gefunden.")
        return None

    print(
        f"\n📊 Analyse pro Layer gestartet (Insgesamt {len(sorted_layers)} Layer)")
    print("=" * 85)

    for layer in sorted_layers:
        keys = layer_keys[layer]

        vecs_1 = []
        vecs_2 = []
        vecs_hidden = []
        cos_sims = []

        for key in keys:
            # Nur vergleichen, wenn der Key in ALLEN DREI Dateien existiert
            if key in activations_2 and key in activations_hidden:
                v1_flat = (activations_1[key] * 0.875).flatten()
                v2_flat = (activations_2[key] * 1).flatten()
                vh_flat = activations_hidden[key].flatten()

                vecs_1.append(v1_flat)
                vecs_2.append(v2_flat)
                vecs_hidden.append(vh_flat)

                # Cosine Similarity zwischen Manuell und MLP
                cos_sim = F.cosine_similarity(v1_flat, v2_flat, dim=0)
                cos_sims.append(cos_sim.item())

        if not vecs_1:
            print(
                f"\n⚠️ Layer {layer}: Keine übereinstimmenden Vektoren in allen drei Dateien gefunden.")
            continue

        # Stacking
        stacked_1 = torch.stack(vecs_1)
        stacked_2 = torch.stack(vecs_2)
        stacked_h = torch.stack(vecs_hidden)

        # --- Metriken berechnen ---
        # 1. Referenz (Manuell)
        l2_1 = stacked_1.norm(p=2, dim=-1).mean().item()
        l1_1 = stacked_1.norm(p=1, dim=-1).mean().item()
        max_1 = stacked_1.max().item()
        min_1 = stacked_1.min().item()

        # 2. Vergleich (MLP)
        l2_2 = stacked_2.norm(p=2, dim=-1).mean().item()
        l1_2 = stacked_2.norm(p=1, dim=-1).mean().item()
        max_2 = stacked_2.max().item()
        min_2 = stacked_2.min().item()

        # 3. Hidden Activations (Base)
        l2_h = stacked_h.norm(p=2, dim=-1).mean().item()
        l1_h = stacked_h.norm(p=1, dim=-1).mean().item()
        max_h = stacked_h.max().item()
        min_h = stacked_h.min().item()

        # --- Relative Stärke (Steering / Hidden) ---
        rel_strength_1 = (l2_1 / l2_h) if l2_h != 0 else 0
        rel_strength_2 = (l2_2 / l2_h) if l2_h != 0 else 0

        # --- Durchschnittliche Cosine Similarity ---
        avg_cos_sim = sum(cos_sims) / len(cos_sims)

        # --- Konsolenausgabe ---
        print(
            f"\n--- 🔹 LAYER {layer} (Zugeordnete Vektoren: {len(vecs_1)}) ---")

        # Tabelle für Normen und Extremwerte
        print(
            f"{'Metrik':<18} | {'Manuell (Ref)':<16} | {'MLP (Vergleich)':<16} | {'Hidden (Base)':<16}")
        print("-" * 75)
        print(f"{'Ø L2-Norm':<18} | {l2_1:<16.4f} | {l2_2:<16.4f} | {l2_h:<16.4f}")
        print(f"{'Ø L1-Norm':<18} | {l1_1:<16.4f} | {l1_2:<16.4f} | {l1_h:<16.4f}")
        print(f"{'Max':<18} | {max_1:<16.4f} | {max_2:<16.4f} | {max_h:<16.4f}")
        print(f"{'Min':<18} | {min_1:<16.4f} | {min_2:<16.4f} | {min_h:<16.4f}")

        print("\nDirekter Vergleich & Einfluss:")
        print(f"🤝 Ø Cosinus-Ähnlichkeit (Manuell vs. MLP): {avg_cos_sim:.4f}")

        # Ausgabe der relativen Stärke in Prozent
        print(
            f"⚡ Relative Stärke (Manuell): {rel_strength_1 * 100:>6.2f}% der Hidden-Aktivierung")
        print(
            f"⚡ Relative Stärke (MLP):     {rel_strength_2 * 100:>6.2f}% der Hidden-Aktivierung")


def visualize_steering_vectors(file_path_1, file_path_2, file_path_hidden, layer_to_plot, num_traj_samples=5):
    """
    Visualisiert die Steering-Vektoren für EINEN bestimmten Layer und speichert 
    die Plots automatisch im angegebenen Verzeichnis ab.
    """
    print(f"\n--- Bereite Visualisierung für Layer {layer_to_plot} vor ---")

    # 1. Daten laden
    try:
        activations_1 = load_file(file_path_1)
        activations_2 = load_file(file_path_2)
        activations_hidden = load_file(file_path_hidden)
    except Exception as e:
        print(f"❌ Fehler beim Laden der Dateien: {e}")
        return

    # 2. Vektoren extrahieren (Code bleibt unverändert)
    vecs_1, vecs_2, vecs_h = [], [], []
    layer_str = f"_layer_{layer_to_plot}"

    for key in activations_1.keys():
        if layer_str in key and key in activations_2 and key in activations_hidden:
            v1_flat = (activations_1[key] * 0.875).flatten().cpu().numpy()
            v2_flat = activations_2[key].flatten().cpu().numpy()
            vh_flat = activations_hidden[key].flatten().cpu().numpy()

            vecs_1.append(v1_flat)
            vecs_2.append(v2_flat)
            vecs_h.append(vh_flat)

    if not vecs_1:
        print(f"❌ Keine Vektoren für Layer {layer_to_plot} gefunden.")
        return

    print(
        f"✅ {len(vecs_1)} Vektoren erfolgreich extrahiert. Starte Dimensionsreduktion...")

    X_manual = np.array(vecs_1)
    X_mlp = np.array(vecs_2)
    X_hidden = np.array(vecs_h)

    X_combined_steering = np.vstack([X_manual, X_mlp])

    # --- Figure Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        f'Activation Steering Analyse - Layer {layer_to_plot}', fontsize=16)

    # 1. PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_combined_steering)
    pca_manual, pca_mlp = pca_result[:len(
        X_manual)], pca_result[len(X_manual):]

    axes[0].scatter(pca_manual[:, 0], pca_manual[:, 1],
                    alpha=0.6, label='Manuell (Spiky)', c='blue', s=20)
    axes[0].scatter(pca_mlp[:, 0], pca_mlp[:, 1], alpha=0.6,
                    label='MLP (Dense)', c='red', s=20)
    axes[0].set_title('1. PCA (Globale Varianz)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. UMAP
    print("⏳ Berechne UMAP...")
    reducer = umap.UMAP(random_state=42)
    umap_result = reducer.fit_transform(X_combined_steering)
    umap_manual, umap_mlp = umap_result[:len(
        X_manual)], umap_result[len(X_manual):]

    axes[1].scatter(umap_manual[:, 0], umap_manual[:, 1],
                    alpha=0.6, label='Manuell', c='blue', s=20)
    axes[1].scatter(umap_mlp[:, 0], umap_mlp[:, 1],
                    alpha=0.6, label='MLP', c='red', s=20)
    axes[1].set_title('2. UMAP (Strukturelles Clustering)')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. Trajektorien
    np.random.seed(42)
    indices = np.random.choice(len(X_hidden), min(
        num_traj_samples, len(X_hidden)), replace=False)

    h_sub, m_sub, p_sub = X_hidden[indices], X_manual[indices], X_mlp[indices]
    target_manual, target_mlp = h_sub + m_sub, h_sub + p_sub

    X_traj_combined = np.vstack([h_sub, target_manual, target_mlp])
    pca_traj = PCA(n_components=2)
    traj_2d = pca_traj.fit_transform(X_traj_combined)

    n = len(h_sub)
    h_2d, tm_2d, tp_2d = traj_2d[0:n], traj_2d[n:2*n], traj_2d[2*n:3*n]

    axes[2].scatter(h_2d[:, 0], h_2d[:, 1], c='black',
                    marker='x', s=60, label='Hidden (Base)')

    for i in range(n):
        # Linie: Base -> Manuell (Blau)
        axes[2].plot([h_2d[i, 0], tm_2d[i, 0]], [h_2d[i, 1],
                     tm_2d[i, 1]], color='blue', alpha=0.6, lw=1.5)
        axes[2].scatter(tm_2d[i, 0], tm_2d[i, 1], color='blue',
                        s=30, zorder=5)  # Zielpunkt Manuell

        # Linie: Base -> MLP (Rot)
        axes[2].plot([h_2d[i, 0], tp_2d[i, 0]], [h_2d[i, 1],
                     tp_2d[i, 1]], color='red', alpha=0.6, lw=1.5)
        axes[2].scatter(tp_2d[i, 0], tp_2d[i, 1], color='red',
                        s=30, zorder=5)  # Zielpunkt MLP

    axes[2].plot([], [], color='blue', label='Verschiebung durch Manuell')
    axes[2].plot([], [], color='red', label='Verschiebung durch MLP')
    axes[2].set_title(f'3. Trajektorien (PCA von {len(indices)} Samples)')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()

    # ==========================================
    # NEU: Speichern der Grafik
    # ==========================================
    # 1. Zielverzeichnis definieren und erstellen (falls es nicht existiert)
    save_dir = f"{DATA_DIR}/steering_vectors/comparison"
    os.makedirs(save_dir, exist_ok=True)

    # 2. Dynamischen Dateinamen generieren (enthält Layer und Lambda zur Unterscheidung)
    filename = f"comparison_layer_{layer_to_plot}.png"
    save_path = os.path.join(save_dir, filename)

    # 3. Grafik in hoher Auflösung für die Masterarbeit speichern (dpi=300)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(
        f"💾 Plot erfolgreich in hoher Auflösung gespeichert unter: {save_path}")


if __name__ == "__main__":
    verify_activations_per_layer(
        file_path_1=f"{DATA_DIR}/steering_vectors/test_contrastive_own_18_19_20.safetensors",
        file_path_2=f"{DATA_DIR}/steering_vectors/test_style_control_network_steering_3_lay.safetensors",
        file_path_hidden=f"{DATA_DIR}/steering_vectors/18_19_20_hidden_activations.safetensors"
    )

    # visualize_steering_vectors(
    #     file_path_1=f"{DATA_DIR}/steering_vectors/test_contrastive_own_18_19_20.safetensors",
    #     file_path_2=f"{DATA_DIR}/steering_vectors/test_style_control_network_steering_3_lay.safetensors",
    #     file_path_hidden=f"{DATA_DIR}/steering_vectors/18_19_20_hidden_activations.safetensors",
    #     layer_to_plot=18,
    #     num_traj_samples=50
    # )

    # analyze_layer_differences(f"{DATA_DIR}/steering_vectors/test_set_style_control_network_steering_all_lay.safetensors")

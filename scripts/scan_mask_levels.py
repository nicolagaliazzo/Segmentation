"""
Scan mask levels
================
Scorre tutte le maschere in una cartella e per ciascuna trova i livelli
di intensità (valori unici dei pixel), come fatto in mask_levels_to_png:
  - maschera 2D → np.unique(mask)
  - maschera (H,W,C) con canali identici → np.unique(primo canale)

Uso:
    python scan_mask_levels.py
"""

import os
import glob
import numpy as np

MASKS_DIR = r"C:\Users\laboratorio\Documents\Segmentation\data\images\newimages_1\masks"


def load_mask(path):
    """Carica la maschera (stessa logica di mask_levels_to_png)."""
    path = os.path.abspath(path)
    try:
        import tifffile
        data = tifffile.imread(path)
    except ImportError:
        import cv2
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if data is None:
            return None
    return np.asarray(data)


def get_levels(data):
    """
    Restituisce i livelli (valori unici) e una stringa descrittiva.
    Per 3D (H,W,C) usa il primo canale se i canali sono identici.
    """
    if data is None or data.size == 0:
        return None, "errore lettura"
    if data.ndim == 2:
        levels = np.unique(data).tolist()
        return levels, "2D"
    if data.ndim == 3:
        # Usa primo canale per livelli (come in mask_levels_to_png)
        if data.shape[-1] <= 16:
            ch = np.asarray(data[:, :, 0], dtype=np.intp)
        else:
            ch = np.asarray(data[0], dtype=np.intp)
        levels = np.unique(ch).tolist()
        return levels, f"{data.shape}"
    return None, f"shape {data.shape}"


def main():
    pattern = os.path.join(MASKS_DIR, "*.ome.tif")
    paths = sorted(glob.glob(pattern))
    if not paths:
        pattern = os.path.join(MASKS_DIR, "*.tif")
        paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"Nessuna maschera .ome.tif/.tif in {MASKS_DIR}")
        return

    print(f"Cartella: {MASKS_DIR}")
    print(f"Maschere trovate: {len(paths)}\n")
    print("-" * 80)

    # Raccogli tutti i set di livelli per riassunto
    levels_per_file = []
    all_levels_set = set()

    for path in paths:
        name = os.path.basename(path)
        data = load_mask(path)
        levels, desc = get_levels(data)
        if levels is None:
            print(f"  {name}: {desc}")
            continue
        levels_per_file.append((name, levels, desc))
        all_levels_set.update(levels)
        # Stampa compatta: nome file e livelli
        levels_str = ", ".join(str(v) for v in levels)
        print(f"  {name}")
        print(f"    shape: {desc}  ->  livelli: [{levels_str}]")

    # Riepilogo: quali combinazioni di livelli esistono
    print()
    print("-" * 80)
    print("RIEPILOGO LIVELLI")
    print("-" * 80)
    unique_tuples = {}
    for name, levels, _ in levels_per_file:
        t = tuple(levels)
        if t not in unique_tuples:
            unique_tuples[t] = []
        unique_tuples[t].append(name)

    for levels_tuple, files in sorted(unique_tuples.items(), key=lambda x: (len(x[0]), x[0])):
        levels_str = ", ".join(str(v) for v in levels_tuple)
        print(f"  Livelli [{levels_str}]  ->  {len(files)} file")
        if len(files) <= 3:
            for f in files:
                print(f"    - {f}")
        else:
            for f in files[:2]:
                print(f"    - {f}")
            print(f"    ... e altri {len(files) - 2}")

    print()
    print(f"Valori di intensità presenti in tutta la cartella: {sorted(all_levels_set)}")


if __name__ == "__main__":
    main()

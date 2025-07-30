import os
import numpy as np
from PIL import Image
from glob import glob
from cityscapesscripts.helpers.labels import labels

# Mapping de l'ID brut (dans le masque) vers la catégorie
id_to_group = {l.id: l.category for l in labels if l.id >= 0}

# Mapping des catégories vers classes 0–7 (void = 0)
group_to_class_id = {
    "void": 0,
    "flat": 1,
    "construction": 2,
    "object": 3,
    "nature": 4,
    "sky": 5,
    "human": 6,
    "vehicle": 7,
}


def convert_mask_to_8groups(mask_path):
    """
    Convertit un masque Cityscapes (format labelIds) en un masque 8 classes :
    void=0, flat=1, ..., vehicle=7.
    """
    mask = np.array(Image.open(mask_path))
    new_mask = np.full(mask.shape, 0, dtype=np.uint8)  # Par défaut: void (0)

    for label_id in np.unique(mask):
        group = id_to_group.get(label_id, "void")
        class_id = group_to_class_id.get(group, 0)
        new_mask[mask == label_id] = class_id

    return Image.fromarray(new_mask)


def convert_all_labelIds_to_8classes(input_root, output_root):
    """
    Convertit tous les masques *_labelIds.png dans input_root vers des masques à 8 classes.
    Les résultats sont sauvegardés dans output_root en respectant la structure des dossiers.
    """
    cities = sorted(os.listdir(input_root))

    for city in cities:
        city_input_dir = os.path.join(input_root, city)
        city_output_dir = os.path.join(output_root, city)
        os.makedirs(city_output_dir, exist_ok=True)

        mask_paths = sorted(glob(os.path.join(city_input_dir, "*_labelIds.png")))

        for mask_path in mask_paths:
            new_mask = convert_mask_to_8groups(mask_path)
            filename = os.path.basename(mask_path).replace("_labelIds", "_8class")
            output_path = os.path.join(city_output_dir, filename)

            new_mask.save(output_path)
            print(f"✅ Saved: {output_path}")


from glob import glob
from collections import Counter
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Nouveau mapping strict à 8 classes
group_to_label = {
    0: "void",
    1: "flat",
    2: "construction",
    3: "object",
    4: "nature",
    5: "sky",
    6: "human",
    7: "vehicle"
}

def count_classes_and_plot_pie(folder):
    """
    Compte les classes dans un dossier de masques PNG à 8 classes
    et affiche un camembert de leur distribution.
    
    Args:
        folder (str): Chemin du dossier contenant les masques.
        
    Returns:
        dict: Dictionnaire {class_id: pixel_count}
    """
    counts = Counter()
    paths = sorted(glob(os.path.join(folder, "**/*.png"), recursive=True))

    for path in paths:
        mask = np.array(Image.open(path))
        unique, freq = np.unique(mask, return_counts=True)
        counts.update(dict(zip(unique, freq)))

    # Ne garder que les classes valides définies dans le mapping
    filtered_counts = {k: v for k, v in counts.items() if k in group_to_label}

    labels = [group_to_label[k] for k in filtered_counts.keys()]
    values = list(filtered_counts.values())

    # Affichage du graphique
    plt.figure(figsize=(8, 8))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"Distribution des classes dans : {os.path.basename(folder)}")
    plt.axis("equal")
    plt.show()

    return filtered_counts


import os
import shutil
from glob import glob
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
from collections import defaultdict
import random

def compute_distribution(class_counts):
    total = class_counts.sum()
    if total == 0:
        return np.zeros_like(class_counts, dtype=np.float32)
    return class_counts / total

def augment_image_and_mask(img, mask):
    # Transformations simples (flip horizontal + rotation 0°, 90°, 180°, 270°)
    # pour garder la taille (1024x2048)
    ops = []
    # flip horizontal aléatoire
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
        mask = ImageOps.mirror(mask)
        ops.append("flip")
    # rotation parmi 0,90,180,270
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    if angle != 0:
        img = img.rotate(angle, expand=False)
        mask = mask.rotate(angle, expand=False)
        ops.append(f"rot{angle}")
    return img, mask, ops

def can_add_aug(aug_counts, current_counts, target_dist, margin):
    new_counts = current_counts + aug_counts
    new_dist = compute_distribution(new_counts)
    over_margin = np.any(new_dist > target_dist + margin)
    dist_current = np.linalg.norm(compute_distribution(current_counts) - target_dist)
    dist_new = np.linalg.norm(new_dist - target_dist)
    # Autoriser si ça améliore la distance globale même en dépassant un peu la marge
    if over_margin and dist_new >= dist_current:
        return False
    return True

def create_multiple_balanced_subsets(image_root, mask_root, output_root,
                                     n_images, n_repertoires,
                                     max_aug=2, class_margin=0.1):
    split = "train"
    image_dir = os.path.join(image_root, split)
    mask_dir = os.path.join(mask_root, split)

    mask_paths = sorted(glob(os.path.join(mask_dir, "**", "*_8class.png"), recursive=True))
    mask_info = []

    print(f"🔎 Analyse de {len(mask_paths)} masques...")
    for path in tqdm(mask_paths):
        mask = np.array(Image.open(path))
        counts = np.zeros(8, dtype=np.int64)
        for cls in range(8):
            counts[cls] = np.sum(mask == cls)
        city = os.path.basename(path).split("_")[0]
        mask_info.append({"path": path, "city": city, "counts": counts})

    total_required = n_images * n_repertoires
    if len(mask_info) < total_required:
        print(f"❌ Pas assez d'images disponibles ({len(mask_info)}) pour générer {n_repertoires} répertoires de {n_images}.")
        return

    used_masks = set()
    target_dist = np.ones(8) / 8  # 12.5% par classe

    for i in range(n_repertoires):
        print(f"\n📂 Création du sous-ensemble {i+1}/{n_repertoires}")
        total_class_counts = np.zeros(8, dtype=np.int64)
        selected = []

        # Sélection initiale un peu plus permissive
        remaining_masks = [m for m in mask_info if m["path"] not in used_masks]

        # Trier en favorisant images qui rapprochent la distribution cible
        def score(m):
            new_counts = total_class_counts + m["counts"]
            dist = compute_distribution(new_counts)
            return -np.linalg.norm(dist - target_dist)  # on veut minimiser la distance

        remaining_masks.sort(key=score, reverse=True)

        for m in remaining_masks:
            if len(selected) >= n_images:
                break
            # Autoriser un léger dépassement mais pas trop strict
            new_counts = total_class_counts + m["counts"]
            new_dist = compute_distribution(new_counts)
            if np.all(new_dist <= target_dist + class_margin*2):  # plus permissif
                selected.append(m)
                total_class_counts += m["counts"]
                used_masks.add(m["path"])

        print(f"✅ {len(selected)} images sélectionnées.")
        print(f"📊 Distribution initiale: {compute_distribution(total_class_counts)*100}")

        # --- Phase augmentation ---
        aug_selected = []
        aug_class_counts = np.zeros(8, dtype=np.int64)
        n_aug_added = 0

        # Classes sous-représentées dans la sélection initiale
        dist_init = compute_distribution(total_class_counts)
        under_represented = np.where(dist_init < target_dist)[0]

        # Pour augmenter, on cible images qui ont beaucoup de pixels dans ces classes sous-représentées
        # On crée une liste de candidats pondérés par leur "aide" sur les classes sous-représentées
        aug_candidates = []
        for m in selected:
            score_aug = sum(m["counts"][cls] for cls in under_represented)
            if score_aug > 0:
                aug_candidates.append((score_aug, m))
        aug_candidates.sort(reverse=True, key=lambda x: x[0])

        for score_aug, m in aug_candidates:
            if n_aug_added >= max_aug * len(selected):
                break
            # Charger image + masque
            mask_path = m["path"]
            city = m["city"]
            filename = os.path.basename(mask_path)
            img_filename = filename.replace("_gtFine_8class.png", "_leftImg8bit.png")
            img_path = os.path.join(image_dir, city, img_filename)

            if not os.path.exists(img_path):
                print(f"❌ Image manquante (augmentation) : {img_path}")
                continue

            img = Image.open(img_path)
            mask_img = Image.open(mask_path)

            # Augmenter image et masque
            img_aug, mask_aug, ops = augment_image_and_mask(img, mask_img)

            # Convert mask_aug to numpy to count pixels
            mask_aug_np = np.array(mask_aug)
            aug_counts = np.zeros(8, dtype=np.int64)
            for cls in range(8):
                aug_counts[cls] = np.sum(mask_aug_np == cls)

            # Vérifier si on peut ajouter cette augmentation sans trop dépasser la marge
            if can_add_aug(aug_counts, total_class_counts + aug_class_counts, target_dist, class_margin):
                # Sauvegarder plus tard
                aug_selected.append({
                    "orig": m,
                    "img_aug": img_aug,
                    "mask_aug": mask_aug,
                    "ops": ops,
                    "suffix": f"_aug{n_aug_added+1}"
                })
                aug_class_counts += aug_counts
                n_aug_added += 1

        total_class_counts += aug_class_counts
        print(f"➕ {n_aug_added} images augmentées ajoutées.")
        print(f"📊 Distribution finale: {compute_distribution(total_class_counts)*100}")

        # --- Copier fichiers ---
        subset_dir = os.path.join(output_root, f"{split}_subset_{n_images}_set{i+1}")
        subset_img_root = os.path.join(subset_dir, "images")
        subset_mask_root = os.path.join(subset_dir, "masks")
        os.makedirs(subset_img_root, exist_ok=True)
        os.makedirs(subset_mask_root, exist_ok=True)

        # Copier images originales
        for m in selected:
            mask_path = m["path"]
            city = m["city"]
            filename = os.path.basename(mask_path)
            img_filename = filename.replace("_gtFine_8class.png", "_leftImg8bit.png")
            img_path = os.path.join(image_dir, city, img_filename)

            dest_img_dir = os.path.join(subset_img_root, city)
            dest_mask_dir = os.path.join(subset_mask_root, city)
            os.makedirs(dest_img_dir, exist_ok=True)
            os.makedirs(dest_mask_dir, exist_ok=True)

            shutil.copy2(img_path, os.path.join(dest_img_dir, img_filename))
            shutil.copy2(mask_path, os.path.join(dest_mask_dir, filename))

        # Sauvegarder images augmentées
        for aug in aug_selected:
            orig = aug["orig"]
            city = orig["city"]
            filename = os.path.basename(orig["path"])
            img_filename = filename.replace("_gtFine_8class.png", "_leftImg8bit.png")

            dest_img_dir = os.path.join(subset_img_root, city)
            dest_mask_dir = os.path.join(subset_mask_root, city)
            os.makedirs(dest_img_dir, exist_ok=True)
            os.makedirs(dest_mask_dir, exist_ok=True)

            # Nom fichier augmenté avec suffixe
            new_img_name = img_filename.replace(".png", f"{aug['suffix']}.png")
            new_mask_name = filename.replace(".png", f"{aug['suffix']}.png")

            aug["img_aug"].save(os.path.join(dest_img_dir, new_img_name))
            aug["mask_aug"].save(os.path.join(dest_mask_dir, new_mask_name))

        print(f"📁 Données copiées dans : {subset_dir}")


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Définition des labels originaux (Cityscapes)
labels = [
    ("unlabeled", 0, (0, 0, 0)),
    ("ego vehicle", 1, (0, 0, 0)),
    ("rectification border", 2, (0, 0, 0)),
    ("out of roi", 3, (0, 0, 0)),
    ("static", 4, (0, 0, 0)),
    ("dynamic", 5, (111, 74, 0)),
    ("ground", 6, (81, 0, 81)),
    ("road", 7, (128, 64, 128)),
    ("sidewalk", 8, (244, 35, 232)),
    ("parking", 9, (250, 170, 160)),
    ("rail track", 10, (230, 150, 140)),
    ("building", 11, (70, 70, 70)),
    ("wall", 12, (102, 102, 156)),
    ("fence", 13, (190, 153, 153)),
    ("guard rail", 14, (180, 165, 180)),
    ("bridge", 15, (150, 100, 100)),
    ("tunnel", 16, (150, 120, 90)),
    ("pole", 17, (153, 153, 153)),
    ("polegroup", 18, (153, 153, 153)),
    ("traffic light", 19, (250, 170, 30)),
    ("traffic sign", 20, (220, 220, 0)),
    ("vegetation", 21, (107, 142, 35)),
    ("terrain", 22, (152, 251, 152)),
    ("sky", 23, (70, 130, 180)),
    ("person", 24, (220, 20, 60)),
    ("rider", 25, (255, 0, 0)),
    ("car", 26, (0, 0, 142)),
    ("truck", 27, (0, 0, 70)),
    ("bus", 28, (0, 60, 100)),
    ("caravan", 29, (0, 0, 90)),
    ("trailer", 30, (0, 0, 110)),
    ("train", 31, (0, 80, 100)),
    ("motorcycle", 32, (0, 0, 230)),
    ("bicycle", 33, (119, 11, 32)),
    ("license plate", -1, (0, 0, 142)),
]

def overlay_mask_on_image_34(image_path, mask_path, alpha=0.6):
    """
    Superpose un masque 34 classes Cityscapes colorisé sur une image PNG avec transparence.
    
    Args:
        image_path (str): Chemin vers l'image brute (PNG).
        mask_path (str): Chemin vers le masque 34 classes (PNG).
        alpha (float): Opacité du masque colorisé superposé.
    """

    # 📥 Chargement de l'image et du masque
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    mask_np = np.array(mask)

    # 🖌️ Création du masque colorisé
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    legend_handles = []

    for label_name, class_id, color in labels:
        mask_region = mask_np == class_id
        if np.any(mask_region):  # N'ajoute que les classes présentes
            color_mask[mask_region] = color
            patch_color = np.array(color) / 255.0
            legend_handles.append(
                mpatches.Patch(color=patch_color, label=f"{class_id}: {label_name}")
            )

    color_mask_img = Image.fromarray(color_mask)

    # 🔄 Redimensionnement si besoin
    if color_mask_img.size != image.size:
        color_mask_img = color_mask_img.resize(image.size, resample=Image.NEAREST)

    # 🧩 Fusion transparente
    blended = Image.blend(image, color_mask_img, alpha)

    # 🖼️ Affichage avec légende
    plt.figure(figsize=(12, 12))
    plt.imshow(blended)
    plt.title("Image avec masque 34 classes superposé")
    plt.axis("off")
    if legend_handles:
        plt.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize='small'
        )
    plt.tight_layout()
    plt.show()


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def overlay_mask_on_image_8class(image_path, mask_path, alpha=0.6):
    """
    Superpose un masque 8 classes Cityscapes colorisé sur une image PNG avec transparence.

    Args:
        image_path (str): Chemin vers l'image brute (RGB PNG).
        mask_path (str): Chemin vers le masque à 8 classes (uint8, PNG).
        alpha (float): Opacité du masque colorisé superposé.
    """

    # Chargement des données
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path)
    mask_np = np.array(mask)

    # Palette (fluorescente) pour 8 classes : 0 = void
    class_colors = {
        0: ((0, 0, 0), "void"),
        1: ((255, 0, 0), "flat"),
        2: ((255, 128, 0), "construction"),
        3: ((255, 255, 0), "object"),
        4: ((0, 255, 0), "nature"),
        5: ((0, 255, 255), "sky"),
        6: ((0, 0, 255), "human"),
        7: ((255, 0, 255), "vehicle"),
    }

    # Construction du masque colorisé
    color_mask = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    legend_handles = []

    for class_id, (color, label) in class_colors.items():
        region = mask_np == class_id
        if np.any(region):
            color_mask[region] = color
            legend_handles.append(
                mpatches.Patch(color=np.array(color)/255.0, label=f"{class_id}: {label}")
            )

    color_mask_img = Image.fromarray(color_mask)

    # Resize si nécessaire
    if color_mask_img.size != image.size:
        color_mask_img = color_mask_img.resize(image.size, resample=Image.NEAREST)

    # Fusion avec alpha
    blended = Image.blend(image, color_mask_img, alpha)

    # Affichage
    plt.figure(figsize=(10, 10))
    plt.imshow(blended)
    plt.title("Image avec masque 8 classes superposé")
    plt.axis("off")
    if legend_handles:
        plt.legend(
            handles=legend_handles,
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0.,
            fontsize='small'
        )
    plt.tight_layout()
    plt.show()

import os
import random
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import numpy as np
from glob import glob
from concurrent.futures import ProcessPoolExecutor

def add_random_noise(image):
    np_img = np.array(image)
    noise = np.random.normal(0, 5, np_img.shape).astype(np.int16)
    noisy_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_random_transformations(img, mask):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    if random.random() < 0.5:
        angle = random.choice([90, 180, 270])
        img = img.rotate(angle, resample=Image.BILINEAR)
        mask = mask.rotate(angle, resample=Image.NEAREST)

    if random.random() < 0.5:
        scale = random.uniform(0.9, 1.1)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), resample=Image.BILINEAR)
        mask = mask.resize((new_w, new_h), resample=Image.NEAREST)

        if scale < 1.0:
            pad_w = (w - new_w) // 2
            pad_h = (h - new_h) // 2
            img = ImageOps.expand(img, border=(pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h), fill=0)
            mask = ImageOps.expand(mask, border=(pad_w, pad_h, w - new_w - pad_w, h - new_h - pad_h), fill=255)
        else:
            left = (new_w - w) // 2
            top = (new_h - h) // 2
            img = img.crop((left, top, left + w, top + h))
            mask = mask.crop((left, top, left + w, top + h))

    if random.random() < 0.3:
        factor = random.uniform(0.7, 1.3)
        img = ImageEnhance.Brightness(img).enhance(factor)

    if random.random() < 0.3:
        factor = random.uniform(0.7, 1.5)
        img = ImageEnhance.Contrast(img).enhance(factor)

    if random.random() < 0.2:
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    if random.random() < 0.2:
        img = add_random_noise(img)

    return img, mask

def augment_one_pair(img_path, mask_path, out_img_dir, out_mask_dir, rel_path, n_aug=5):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    sub_dir = os.path.dirname(rel_path)
    os.makedirs(os.path.join(out_img_dir, sub_dir), exist_ok=True)
    os.makedirs(os.path.join(out_mask_dir, sub_dir), exist_ok=True)

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    # ➕ Sauvegarde de l'image et du masque originaux
    orig_img_path = os.path.join(out_img_dir, sub_dir, f"{base_name}_orig.png")
    orig_mask_path = os.path.join(out_mask_dir, sub_dir, f"{base_name}_orig.png")
    img.save(orig_img_path)
    mask.save(orig_mask_path)

    # 🔁 Génération des versions augmentées
    for i in range(n_aug):
        aug_img, aug_mask = apply_random_transformations(img, mask)
        out_img_path = os.path.join(out_img_dir, sub_dir, f"{base_name}_aug{i}.png")
        out_mask_path = os.path.join(out_mask_dir, sub_dir, f"{base_name}_aug{i}.png")
        aug_img.save(out_img_path)
        aug_mask.save(out_mask_path)

def augment_dataset_multicore(img_dir, mask_dir, out_img_dir, out_mask_dir, n_aug_per_image=5, num_workers=8):
    print(f"🔁 Début de l'augmentation avec {num_workers} workers...")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    img_paths = sorted(glob(os.path.join(img_dir, "**/*.png"), recursive=True))
    mask_paths = sorted(glob(os.path.join(mask_dir, "**/*.png"), recursive=True))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for img_path, mask_path in zip(img_paths, mask_paths):
            rel_path = os.path.relpath(img_path, img_dir)
            futures.append(executor.submit(
                augment_one_pair,
                img_path, mask_path,
                out_img_dir, out_mask_dir,
                rel_path,
                n_aug_per_image
            ))

    print("✅ Augmentation terminée.")


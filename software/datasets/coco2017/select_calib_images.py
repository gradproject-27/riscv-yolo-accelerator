
import json
import os
from collections import Counter

ANNOTATIONS = "annotations/instances_val2017.json"
OUTPUT_DIR = "calib_selection"
MAX_IMAGES = 200

# Ignore crowd annotations, but keep small objects so rare classes
# are not accidentally excluded.
MIN_WIDTH = 16
MIN_HEIGHT = 16

with open(ANNOTATIONS, "r") as f:
    coco = json.load(f)

categories = {c["id"]: c["name"] for c in coco["categories"]}
images = {img["id"]: img for img in coco["images"]}

# Collect useful annotations for each image.
image_info = {}

for ann in coco["annotations"]:
    if ann.get("iscrowd", 0):
        continue

    x, y, w, h = ann["bbox"]

    if w < MIN_WIDTH or h < MIN_HEIGHT:
        continue

    image_id = ann["image_id"]
    category_id = ann["category_id"]

    if image_id not in image_info:
        image_info[image_id] = {
            "classes": set(),
            "counts": Counter(),
            "quality": 0.0,
        }

    item = image_info[image_id]
    item["classes"].add(category_id)
    item["counts"][category_id] += 1

    # Prefer larger visible objects, with a capped contribution.
    area = min(w * h, 300 * 300)
    item["quality"] += area ** 0.5

# Greedy class-coverage selection.
selected = []
selected_ids = set()
covered = set()
class_frequency = Counter()

while len(selected) < MAX_IMAGES:
    best_id = None
    best_score = -1

    for image_id, info in image_info.items():
        if image_id in selected_ids:
            continue

        new_classes = info["classes"] - covered

        # First prioritize uncovered classes.
        coverage_score = sum(
            1.0 / (1 + class_frequency[c])
            for c in new_classes
        )

        # Then prefer images with several visible, larger objects.
        object_score = min(info["quality"] / 500.0, 5.0)
        diversity_score = len(info["classes"]) * 0.15

        score = coverage_score * 10 + object_score + diversity_score

        if score > best_score:
            best_score = score
            best_id = image_id

    if best_id is None:
        break

    selected_ids.add(best_id)
    selected.append(best_id)

    info = image_info[best_id]
    covered.update(info["classes"])

    for c, count in info["counts"].items():
        class_frequency[c] += count

os.makedirs(OUTPUT_DIR, exist_ok=True)

selected_images = [images[i] for i in selected]

with open(os.path.join(OUTPUT_DIR, "selected_images.txt"), "w") as f:
    for img in selected_images:
        f.write(img["file_name"] + "\n")

with open(os.path.join(OUTPUT_DIR, "selected_urls.txt"), "w") as f:
    for img in selected_images:
        url = img.get("coco_url")
        if url:
            f.write(url + "\n")

present = set()
for image_id in selected:
    present.update(image_info[image_id]["classes"])

missing = [
    name for cid, name in categories.items()
    if cid not in present
]

print(f"\nSelected images: {len(selected)}")
print(f"Covered classes: {len(present)} / {len(categories)}")

print("\nCovered classes:")
for cid, name in categories.items():
    if cid in present:
        print(f"  [OK] {name}")

print("\nMissing classes:")
if missing:
    for name in missing:
        print(f"  [MISSING] {name}")
else:
    print("  None")

print(f"\nOutput files are in: {OUTPUT_DIR}/")

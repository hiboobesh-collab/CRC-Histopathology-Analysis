import pandas as pd
import imagehash
from PIL import Image
from collections import defaultdict

SPLITS = ["train", "val", "test"]
THRESHOLD = 4

dfs = {
    s: pd.read_csv(f"data/splits/group_{s}.csv")
    for s in SPLITS
}

items = []

for split, df in dfs.items():
    for _, row in df.iterrows():
        items.append((split, row["filepath"]))

print(f"Images to hash: {len(items)}")

# ---------------------------------------------------------
# Compute pHash as 64-bit integers
# ---------------------------------------------------------
records = []

for i, (split, path) in enumerate(items, 1):
    h = imagehash.phash(Image.open(path).convert("RGB"))
    value = int(str(h), 16)
    records.append((split, path, value))

    if i % 1000 == 0:
        print(f"Hashed {i}/{len(items)}")

# ---------------------------------------------------------
# Bucket by each 8-bit chunk of the 64-bit pHash.
# Any pair with Hamming distance <= 4 must share
# at least one 8-bit chunk.
# ---------------------------------------------------------
buckets = [defaultdict(list) for _ in range(8)]

for idx, (_, _, h) in enumerate(records):
    for byte_pos in range(8):
        shift = byte_pos * 8
        chunk = (h >> shift) & 0xFF
        buckets[byte_pos][chunk].append(idx)

print("Hash indexing complete.")
print("Checking cross-split near-duplicates...")

checked = set()
pairs = []

for i, (split_a, path_a, hash_a) in enumerate(records):

    candidates = set()

    for byte_pos in range(8):
        shift = byte_pos * 8
        chunk = (hash_a >> shift) & 0xFF
        candidates.update(buckets[byte_pos][chunk])

    for j in candidates:

        if i == j:
            continue

        split_b, path_b, hash_b = records[j]

        if split_a == split_b:
            continue

        pair_key = tuple(sorted((i, j)))

        if pair_key in checked:
            continue

        checked.add(pair_key)

        distance = (hash_a ^ hash_b).bit_count()

        if distance <= THRESHOLD:
            pairs.append(
                (
                    split_a,
                    path_a,
                    split_b,
                    path_b,
                    distance
                )
            )

print()
print("=" * 70)
print(f"Cross-split pHash pairs with Hamming distance <= {THRESHOLD}: {len(pairs)}")
print("=" * 70)

for pair in pairs[:50]:
    print()
    print("Split A :", pair[0])
    print("Image A :", pair[1])
    print("Split B :", pair[2])
    print("Image B :", pair[3])
    print("Distance:", pair[4])

if not pairs:
    print()
    print("RESULT: No cross-split near-duplicates detected.")
else:
    print()
    print("RESULT: Potential cross-split near-duplicates detected.")
    print("These should be manually inspected before making final claims.")

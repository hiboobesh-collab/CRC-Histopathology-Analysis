from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# Configuration
# ============================================================

METADATA = Path("data/metadata/lc25000_image_groups.csv")
OUTPUT_DIR = Path("data/splits")

RANDOM_STATE = 42

# ============================================================
# Load metadata
# ============================================================

df = pd.read_csv(METADATA)

required_columns = {
    "filename",
    "label",
    "tissue",
    "group_id",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Only use colon images
colon = df[df["tissue"].astype(str).str.lower() == "colon"].copy()

print("=" * 70)
print("LC25000 GROUP-AWARE SPLIT")
print("=" * 70)

print(f"Colon images: {len(colon)}")
print(f"Colon groups: {colon['group_id'].nunique()}")

# ============================================================
# Verify class balance
# ============================================================

print("\nClass distribution:")
print(colon["label"].value_counts())

# ============================================================
# Verify every group belongs to only one class
# ============================================================

group_class_counts = (
    colon.groupby("group_id")["label"]
    .nunique()
)

impure_groups = group_class_counts[group_class_counts > 1]

if len(impure_groups) > 0:
    print("\nERROR: Some groups contain multiple classes!")
    print(impure_groups)
    raise ValueError("Group/class purity check failed.")

print("\nGroup class purity: PASSED")

# ============================================================
# Create one row per group
# ============================================================

groups = (
    colon.groupby("group_id", as_index=False)
    .agg(
        label=("label", "first"),
        image_count=("filename", "count")
    )
)

print(f"\nUnique groups: {len(groups)}")
print("\nGroups per class:")
print(groups["label"].value_counts())

# ============================================================
# Split groups:
#
# 70% TRAIN
# 15% VALIDATION
# 15% TEST
#
# First: 70 / 30
# Then: split remaining 30 into 15 / 15
# ============================================================

train_groups, temp_groups = train_test_split(
    groups,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=groups["label"]
)

val_groups, test_groups = train_test_split(
    temp_groups,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_groups["label"]
)

train_group_ids = set(train_groups["group_id"])
val_group_ids = set(val_groups["group_id"])
test_group_ids = set(test_groups["group_id"])

# ============================================================
# Critical leakage checks
# ============================================================

assert train_group_ids.isdisjoint(val_group_ids)
assert train_group_ids.isdisjoint(test_group_ids)
assert val_group_ids.isdisjoint(test_group_ids)

print("\n" + "=" * 70)
print("GROUP OVERLAP CHECK")
print("=" * 70)

print("Train n Validation:", len(train_group_ids & val_group_ids))
print("Train n Test:", len(train_group_ids & test_group_ids))
print("Validation n Test:", len(val_group_ids & test_group_ids))

# ============================================================
# Assign each image to its split
# ============================================================

def assign_split(group_id):
    if group_id in train_group_ids:
        return "train"
    if group_id in val_group_ids:
        return "val"
    if group_id in test_group_ids:
        return "test"
    raise ValueError(f"Unknown group_id: {group_id}")

colon["split"] = colon["group_id"].apply(assign_split)

# ============================================================
# Construct project file paths
# ============================================================

def make_filepath(row):
    if row["label"] == "colon_aca":
        folder = "colon_aca"
    elif row["label"] == "colon_n":
        folder = "colon_n"
    else:
        raise ValueError(f"Unexpected label: {row['label']}")

    return (
        Path("data")
        / "lung_colon_image_set"
        / "colon_image_sets"
        / folder
        / row["filename"]
    ).as_posix()

colon["filepath"] = colon.apply(make_filepath, axis=1)

# Match the format expected by our Dataset class
colon["class_name"] = colon["label"]

# ============================================================
# Verify every image exists
# ============================================================

missing_files = []

for filepath in colon["filepath"]:
    if not Path(filepath).exists():
        missing_files.append(filepath)

if missing_files:
    print("\nERROR: Missing image files:", len(missing_files))
    print("\n".join(missing_files[:20]))
    raise FileNotFoundError("Some metadata images do not exist.")

print("\nImage existence check: PASSED")

# ============================================================
# Save splits
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

columns = [
    "filepath",
    "label",
    "class_name",
    "filename",
    "group_id",
]

train_df = colon[colon["split"] == "train"][columns].copy()
val_df = colon[colon["split"] == "val"][columns].copy()
test_df = colon[colon["split"] == "test"][columns].copy()

train_df.to_csv(OUTPUT_DIR / "group_train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "group_val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "group_test.csv", index=False)

# ============================================================
# Final report
# ============================================================

print("\n" + "=" * 70)
print("FINAL GROUP-AWARE SPLIT")
print("=" * 70)

for name, split_df, split_groups in [
    ("TRAIN", train_df, train_group_ids),
    ("VALIDATION", val_df, val_group_ids),
    ("TEST", test_df, test_group_ids),
]:
    print(f"\n{name}")
    print(f"  Images : {len(split_df)}")
    print(f"  Groups : {len(split_groups)}")
    print("  Classes:")
    print(split_df["label"].value_counts().to_dict())

print("\nImage percentages:")
print(f"  Train: {len(train_df) / len(colon) * 100:.2f}%")
print(f"  Val  : {len(val_df) / len(colon) * 100:.2f}%")
print(f"  Test : {len(test_df) / len(colon) * 100:.2f}%")

print("\nGroup percentages:")
print(f"  Train: {len(train_group_ids) / len(groups) * 100:.2f}%")
print(f"  Val  : {len(val_group_ids) / len(groups) * 100:.2f}%")
print(f"  Test : {len(test_group_ids) / len(groups) * 100:.2f}%")

print("\nFiles created:")
print(f"  {OUTPUT_DIR / 'group_train.csv'}")
print(f"  {OUTPUT_DIR / 'group_val.csv'}")
print(f"  {OUTPUT_DIR / 'group_test.csv'}")

print("\n" + "=" * 70)
print("GROUP-AWARE SPLIT COMPLETED SUCCESSFULLY")
print("=" * 70)

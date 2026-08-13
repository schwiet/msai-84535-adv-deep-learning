import collections
import glob
import json

from homework.generate_captions import generate_caption
from homework.generate_qa import generate_qa_pairs

files = sorted(glob.glob("data/valid/*_info.json"))
print(f"Loaded {len(files)} validation files")

d = {}
caps = collections.defaultdict(set)
for info_path in files:
    for view in range(10):
        rows = generate_qa_pairs(info_path, view)
        for row in rows:
            d[(row["image_file"], row["question"])] = row["answer"]

        captions = generate_caption(info_path, view)
        for c in captions:
            caps[c["image_file"]].add(c["caption"])

print(f"Rows: {len(d)}")

with open("data/valid_grader/balanced_qa_pairs.json") as f:
    truth = json.load(f)

print(f"Truth Count {len(truth)}")
match_count = 0
mismatch_ct = 0
missing_count = 0
for row in truth:
    matching_gen = d.get((row["image_file"], row["question"]))
    if matching_gen is not None:
        if row["answer"] == matching_gen:
            match_count += 1
        else:
            print(f"Mismatch: ({row['image_file']}, {row['question']}")
            print(f"\tExpected: {row['answer']}\tgenerated: {matching_gen}")
            mismatch_ct += 1
    else:
        print(f"Missing: ({row['image_file']}, {row['question']}")
        missing_count += 1

print(f"QA --\n\nMatches: {match_count}\tMismatch: {mismatch_ct}\tMiss: {missing_count}")

with open("data/valid_grader/all_mc_qas.json") as f:
    truth = json.load(f)

missing_count = 0
for row in truth:
    matching_gen = caps.get(row["image_file"])
    missing_count += 1
    c = row["candidates"][row["correct_index"]]
    if c in matching_gen:
        missing_count -= 1

    else:
        print(f"Missing: ({row['image_file']}, {c}")

print(f"CAP -\n\n\tMiss: {missing_count}")

"""Read first 10 and last 10 records from ffw_records.npz."""
import numpy as np

path = "/puffertank/mjlab/record/ffw_records.npz"
data = dict(np.load(path))

n = next(iter(data.values())).shape[0]
print(f"Total records: {n}\n")
print(f"Keys: {list(data.keys())}")
for k, v in data.items():
    print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

print("\n" + "=" * 60)
print("FIRST 10 RECORDS")
print("=" * 60)
for i in range(min(10, n)):
    print(f"\n--- record {i} ---")
    for k, v in data.items():
        print(f"  {k}: {v[i]}")

print("\n" + "=" * 60)
print("LAST 10 RECORDS")
print("=" * 60)
for i in range(max(0, n - 10), n):
    print(f"\n--- record {i} ---")
    for k, v in data.items():
        print(f"  {k}: {v[i]}")

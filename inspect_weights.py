# inspect_weights.py
import h5py

fn = "models/final_model.weights.h5"   # adjust if your weights file name is different

def walk(name, obj):
    # print dataset names and shapes, focusing on weights like 'kernel' or 'bias'
    if hasattr(obj, "shape"):
        if "kernel" in name.lower() or "weight" in name.lower() or "bias" in name.lower():
            print(f"{name} -> shape: {obj.shape}")

with h5py.File(fn, "r") as f:
    print("Top groups:", list(f.keys()))
    f.visititems(walk)

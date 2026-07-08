"""Audits GPU memory allocation and deallocation for the MOMENT model on the UGR server."""
import torch
import gc
import sys
import time
import paramiko

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

remote_code = """
import torch
import gc
import time

def print_memory(label):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"[{label}] Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
    else:
        print(f"[{label}] CUDA not available")

print_memory("Initial")

try:
    from momentfm import MOMENTPipeline
    print("[MOMENT] Loading backbone...")
    t0 = time.time()
    backbone = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={
            "task_name": "classification",
            "n_channels": 23,
            "seq_len": 512,
            "num_class": 2,
        }
    )
    print(f"[MOMENT] Loaded on CPU in {time.time() - t0:.1f}s")
    print_memory("After load (on CPU)")

    print("[MOMENT] Moving to GPU...")
    backbone = backbone.to("cuda")
    print_memory("After moving to GPU")

    # Run a forward pass
    x = torch.randn(4, 23, 512).to("cuda")
    print_memory("After creating input tensor")
    out = backbone.embed(x_enc=x, reduction="mean")
    print_memory("After forward pass")

    # Delete local tensors
    del x, out
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("After deleting input/output tensors and empty_cache")

    # Delete backbone
    print("[MOMENT] Deleting backbone and cleaning up...")
    del backbone
    gc.collect()
    torch.cuda.empty_cache()
    print_memory("After deleting backbone and empty_cache")

except Exception as e:
    print(f"[ERROR] {e}")
"""

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        # Execute the python script inside our conda environment using srun on titan
        command = (
            "export PATH=\"/opt/anaconda/anaconda3/bin:$PATH\"\n"
            "export PATH=\"/opt/anaconda/bin:$PATH\"\n"
            "eval \"$(conda shell.bash hook)\"\n"
            "conda activate /mnt/homeGPU/egullonl01/conda_tfg\n"
            f"srun -w titan python -c {repr(remote_code)}"
        )
        
        print("Running GPU memory audit on titan...")
        stdin, stdout, stderr = ssh.exec_command(command)
        
        print("--- STDOUT ---")
        print(stdout.read().decode('utf-8'))
        print("--- STDERR ---")
        print(stderr.read().decode('utf-8'))
        
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

if __name__ == "__main__":
    main()

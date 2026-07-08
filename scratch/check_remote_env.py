"""Queries remote UGR GPU server to check deployment directory and installed conda libraries."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"
remote_conda_dir = "/mnt/homeGPU/egullonl01/conda_tfg"

try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password, timeout=10)
    print("[OK] Connected to remote UGR server.")
    
    # Check directory
    print("\nListing remote /mnt/homeGPU/egullonl01/tfg/ contents:")
    stdin, stdout, stderr = ssh.exec_command("ls -la /mnt/homeGPU/egullonl01/tfg/")
    print(stdout.read().decode('utf-8', errors='ignore'))
    
    # Check pip list in conda environment
    print(f"\nChecking pip list in remote Conda environment: {remote_conda_dir}...")
    pip_cmd = (
        "export PATH=\"/opt/anaconda/anaconda3/bin:$PATH\"\n"
        "export PATH=\"/opt/anaconda/bin:$PATH\"\n"
        "eval \"$(conda shell.bash hook)\"\n"
        f"conda activate {remote_conda_dir}\n"
        "pip list"
    )
    stdin, stdout, stderr = ssh.exec_command(pip_cmd)
    out = stdout.read().decode('utf-8', errors='ignore')
    print(out)
    
    ssh.close()
except Exception as e:
    print(f"[ERROR] Connection check failed: {e}")

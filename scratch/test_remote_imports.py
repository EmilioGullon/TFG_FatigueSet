"""Tests importing momentfm, chronos, and timesfm inside the remote conda environment."""
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
    print("[OK] Connected to UGR GPU server.")
    
    # Test importing
    test_cmd = (
        "export PATH=\"/opt/anaconda/anaconda3/bin:$PATH\"\n"
        "export PATH=\"/opt/anaconda/bin:$PATH\"\n"
        "eval \"$(conda shell.bash hook)\"\n"
        f"conda activate {remote_conda_dir}\n"
        "python -c 'import momentfm; import chronos; import timesfm; print(\"SUCCESS: All models imported successfully!\")'"
    )
    
    print("\nRunning remote import check...")
    stdin, stdout, stderr = ssh.exec_command(test_cmd)
    
    out = stdout.read().decode('utf-8', errors='ignore')
    err = stderr.read().decode('utf-8', errors='ignore')
    
    if out:
        print("--- OUTPUT ---")
        print(out)
    if err:
        print("--- ERROR ---")
        print(err)
        
    ssh.close()
except Exception as e:
    print(f"[ERROR] Import check failed: {e}")
    sys.exit(1)

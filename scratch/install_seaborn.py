"""Installs seaborn in the remote Conda environment."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"
remote_conda_dir = "/mnt/homeGPU/egullonl01/conda_tfg"

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        print("[OK] Connected to UGR GPU server.")
        
        install_cmd = (
            "export PATH=\"/opt/anaconda/anaconda3/bin:$PATH\"\n"
            "export PATH=\"/opt/anaconda/bin:$PATH\"\n"
            "eval \"$(conda shell.bash hook)\"\n"
            f"conda activate {remote_conda_dir}\n"
            "pip install seaborn"
        )
        print("\nInstalling seaborn in remote Conda environment...")
        stdin, stdout, stderr = ssh.exec_command(install_cmd)
        print(stdout.read().decode('utf-8', errors='ignore'))
        print(stderr.read().decode('utf-8', errors='ignore'))
        
        ssh.close()
        print("[OK] seaborn installed successfully.")
    except Exception as e:
        print(f"[ERROR] Connection check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

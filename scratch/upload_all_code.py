"""Uploads the updated code files to UGR GPU server."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"
remote_tfg_dir = "/mnt/homeGPU/egullonl01/tfg"

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        sftp = ssh.open_sftp()
        
        files_to_upload = [
            ("fatigueset-lib/fatigueset/models/optimizers.py", f"{remote_tfg_dir}/fatigueset-lib/fatigueset/models/optimizers.py"),
            ("fatigueset-lib/fatigueset/models/foundation.py", f"{remote_tfg_dir}/fatigueset-lib/fatigueset/models/foundation.py"),
            ("Jupyters/09_foundation_models.ipynb", f"{remote_tfg_dir}/Jupyters/09_foundation_models.ipynb"),
            ("Jupyters/experimento_optuna_optimizadores.ipynb", f"{remote_tfg_dir}/Jupyters/experimento_optuna_optimizadores.ipynb"),
            ("Jupyters/run_optuna_server.sh", f"{remote_tfg_dir}/Jupyters/run_optuna_server.sh"),
            ("Jupyters/run_foundation_server.sh", f"{remote_tfg_dir}/Jupyters/run_foundation_server.sh"),
            ("Jupyters/run_timesfm_server.sh", f"{remote_tfg_dir}/Jupyters/run_timesfm_server.sh")
        ]
        
        print("Uploading updated code files:")
        for local_path, remote_path in files_to_upload:
            print(f"- {local_path} -> {remote_path}")
            sftp.put(local_path, remote_path)
            
        sftp.close()
        ssh.close()
        print("[OK] Upload finished successfully.")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

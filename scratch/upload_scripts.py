"""Uploads only the run scripts to the UGR server to update them with correct log paths."""
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
        
        scripts = [
            "run_optuna_server.sh",
            "run_foundation_server.sh",
            "run_timesfm_server.sh"
        ]
        
        print("Uploading updated scripts:")
        for script in scripts:
            local_path = f"Jupyters/{script}"
            remote_path = f"{remote_tfg_dir}/Jupyters/{script}"
            print(f"- {local_path} -> {remote_path}")
            sftp.put(local_path, remote_path)
            
        sftp.close()
        ssh.close()
        print("[OK] Upload finished successfully.")
    except Exception as e:
        print(f"[ERROR] Script upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

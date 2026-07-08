"""Downloads results of the completed remote Foundation Models run."""
import paramiko
import sys
import os

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        sftp = ssh.open_sftp()
        
        # Make sure local output dir exists
        os.makedirs("output", exist_ok=True)
        os.makedirs("Jupyters", exist_ok=True)
        
        # 1. Download resultados_foundation_models.csv
        print("Downloading output/resultados_foundation_models.csv...")
        sftp.get(
            "/mnt/homeGPU/egullonl01/tfg/output/resultados_foundation_models.csv",
            "output/resultados_foundation_models.csv"
        )
        
        # 2. Download taxonomia_modelos.csv
        print("Downloading output/taxonomia_modelos.csv...")
        sftp.get(
            "/mnt/homeGPU/egullonl01/tfg/output/taxonomia_modelos.csv",
            "output/taxonomia_modelos.csv"
        )
        
        # 3. Download executed notebook
        print("Downloading executed Jupyters/09_foundation_models.ipynb...")
        sftp.get(
            "/mnt/homeGPU/egullonl01/tfg/Jupyters/09_foundation_models.ipynb",
            "Jupyters/09_foundation_models.ipynb"
        )
        
        sftp.close()
        ssh.close()
        print("=== Downloads complete! ===")
    except Exception as e:
        print(f"[ERROR] Downloading results failed: {e}")

if __name__ == "__main__":
    main()

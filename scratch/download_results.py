# -*- coding: utf-8 -*-
"""Script to download remote results and logs from the UGR server to local directory."""
import paramiko
import os

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

remote_output_dir = "/mnt/homeGPU/egullonl01/tfg/output"
local_output_dir = "output"

files_to_download = [
    "resultados_foundation_models.csv",
    "resultados_busqueda_optuna.csv",
    "mejores_hiperparametros_optuna.json",
    "comparativa_optuna_optimizadores.csv",
    "optuna_optimizadores_personalizados.png",
    "taxonomia_modelos.csv",
    "taxonomia_modelos_tfg.png"
]

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=15)
        
        sftp = ssh.open_sftp()
        
        # Ensure local output directory exists
        os.makedirs(local_output_dir, exist_ok=True)
        
        print("=== Downloading Output Files ===")
        for f in files_to_download:
            remote_path = f"{remote_output_dir}/{f}"
            local_path = f"{local_output_dir}/{f}"
            try:
                print(f"Downloading: {f}...")
                sftp.get(remote_path, local_path)
            except IOError:
                print(f"[Warning] Could not download {f} (maybe not generated yet).")
        
        print("\n=== Downloading SLURM Logs ===")
        log_files = ["slurm-156246.out", "slurm-156247.out"]
        for log in log_files:
            remote_path = f"/mnt/homeGPU/egullonl01/tfg/Jupyters/{log}"
            local_path = f"Jupyters/{log}"
            try:
                print(f"Downloading log: {log}...")
                sftp.get(remote_path, local_path)
            except IOError:
                print(f"[Warning] Could not download log {log}.")
                
        sftp.close()
        ssh.close()
        print("\n[SUCCESS] DOWNLOAD COMPLETED!")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()

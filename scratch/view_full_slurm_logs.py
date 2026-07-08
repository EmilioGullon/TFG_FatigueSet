"""Reads and prints the entire remote slurm log files for the active jobs."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        for job_name, job_id in [("Optuna", "155622"), ("Foundation Models", "155624")]:
            print(f"\n========================================")
            print(f" FULL LOG FOR {job_name} (Job ID: {job_id})")
            print(f"========================================")
            stdin, stdout, stderr = ssh.exec_command(f"cat /mnt/homeGPU/egullonl01/tfg/Jupyters/slurm-{job_id}.out")
            print(stdout.read().decode('utf-8', errors='ignore'))
            
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Reading full logs failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

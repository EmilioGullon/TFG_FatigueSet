"""Queries SLURM job details using scontrol show job."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        for name, job_id in [("Optuna", "155622"), ("Foundation Models", "155624")]:
            print(f"\n=== scontrol show job {name} ({job_id}) ===")
            stdin, stdout, stderr = ssh.exec_command(f"scontrol show job {job_id}")
            print(stdout.read().decode('utf-8'))
            
        ssh.close()
    except Exception as e:
        print(f"[ERROR] scontrol check failed: {e}")

if __name__ == "__main__":
    main()

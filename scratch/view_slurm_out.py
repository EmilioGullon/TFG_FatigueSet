"""Reads Optuna slurm log to inspect completed trials."""
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
        
        # Read the first 200 lines of the Optuna output
        print("=== FIRST 200 LINES OF OPTUNA LOG ===")
        stdin, stdout, stderr = ssh.exec_command("head -n 200 /mnt/homeGPU/egullonl01/tfg/Jupyters/slurm-155583.out")
        print(stdout.read().decode('utf-8', errors='ignore'))
        
        # Count trial completions
        print("=== TRIAL COMPLETIONS COUNT ===")
        stdin, stdout, stderr = ssh.exec_command("grep -i \"trial \" /mnt/homeGPU/egullonl01/tfg/Jupyters/slurm-155583.out | wc -l")
        print(f"Number of trial lines: {stdout.read().decode('utf-8').strip()}")
        
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Connection check failed: {e}")

if __name__ == "__main__":
    main()

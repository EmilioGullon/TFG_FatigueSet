"""Queries real-time resource usage of active SLURM jobs using sstat."""
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
        
        for name, job_id in [("Optuna", "155622"), ("Foundation Models", "155639")]:
            print(f"\n=== sstat for {name} ({job_id}) ===")
            stdin, stdout, stderr = ssh.exec_command(f"sstat --all -j {job_id}")
            print("STDOUT:")
            print(stdout.read().decode('utf-8'))
            print("STDERR:")
            print(stderr.read().decode('utf-8'))
            
        ssh.close()
    except Exception as e:
        print(f"[ERROR] sstat check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

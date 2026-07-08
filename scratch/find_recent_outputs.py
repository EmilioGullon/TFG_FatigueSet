"""Searches for files modified in the last 2 hours in the remote workspace."""
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
        
        print("Searching for files modified in the last 2 hours under /mnt/homeGPU/egullonl01/tfg/...")
        stdin, stdout, stderr = ssh.exec_command("find /mnt/homeGPU/egullonl01/tfg/ -mmin -120 -type f")
        paths = stdout.read().decode('utf-8').strip().split('\n')
        
        if paths and paths[0]:
            print(f"Found {len(paths)} recently modified files:")
            for p in paths:
                # Skip log files and checkpoints
                if any(x in p for x in ["slurm-", ".ipynb_checkpoints", ".git"]):
                    continue
                print(f"- {p}")
        else:
            print("No files modified in the last 2 hours found.")
            
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""Searches for any slurm-*.out files on the UGR GPU server home partition recursively."""
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
        
        # Search recursively
        print("Searching for slurm-*.out on the server...")
        stdin, stdout, stderr = ssh.exec_command("find /mnt/homeGPU/egullonl01/ -name \"slurm-*.out\"")
        paths = stdout.read().decode('utf-8').strip().split('\n')
        
        if paths and paths[0]:
            print(f"Found {len(paths)} slurm files:")
            for p in paths:
                print(f"- {p}")
        else:
            print("No slurm log files found anywhere under /mnt/homeGPU/egullonl01/.")
            
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

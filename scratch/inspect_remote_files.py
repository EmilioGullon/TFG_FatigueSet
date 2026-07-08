"""Inspects remote home directory and workspace files on UGR server."""
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
        
        # List home directory
        print("=== Files in remote home directory (/mnt/homeGPU/egullonl01/): ===")
        stdin, stdout, stderr = ssh.exec_command("ls -la /mnt/homeGPU/egullonl01/")
        print(stdout.read().decode('utf-8'))
        
        # List workspace directory
        print("\n=== Files in workspace Jupyters/ (/mnt/homeGPU/egullonl01/tfg/Jupyters/): ===")
        stdin, stdout, stderr = ssh.exec_command("ls -la /mnt/homeGPU/egullonl01/tfg/Jupyters/")
        print(stdout.read().decode('utf-8'))
        
        # List output directory
        print("\n=== Files in output/ directory (/mnt/homeGPU/egullonl01/tfg/Jupyters/output/): ===")
        stdin, stdout, stderr = ssh.exec_command("ls -la /mnt/homeGPU/egullonl01/tfg/Jupyters/output/")
        print(stdout.read().decode('utf-8'))
        
        # Check sacct history for recent jobs
        print("\n=== Recent SLURM job history (sacct): ===")
        stdin, stdout, stderr = ssh.exec_command("sacct -u egullonl01 --format=JobID,JobName,State,ExitCode")
        print(stdout.read().decode('utf-8'))
        
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Remote check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""Checks default SSH folder path and searches for slurm log files there."""
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
        
        # 1. Print pwd
        print("Checking default SSH directory (pwd):")
        stdin, stdout, stderr = ssh.exec_command("pwd")
        pwd = stdout.read().decode('utf-8').strip()
        print(f"pwd: {pwd}")
        
        # 2. Check if files exist in this home directory
        print(f"\nListing files in home directory ({pwd}):")
        stdin, stdout, stderr = ssh.exec_command(f"ls -la {pwd}")
        print(stdout.read().decode('utf-8'))
        
        # 3. Find slurm files in home directory
        print("\nSearching for slurm-*.out in home directory:")
        stdin, stdout, stderr = ssh.exec_command(f"find {pwd} -name \"slurm-*.out\"")
        print(stdout.read().decode('utf-8'))
        
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Home check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

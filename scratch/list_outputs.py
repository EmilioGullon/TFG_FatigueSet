"""Lists files in the remote output directory to check for new results."""
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
        
        print("=== Files in /mnt/homeGPU/egullonl01/tfg/output/ ===")
        stdin, stdout, stderr = ssh.exec_command("ls -la --time-style=long-iso /mnt/homeGPU/egullonl01/tfg/output/")
        print(stdout.read().decode('utf-8'))
        
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Listing outputs failed: {e}")

if __name__ == "__main__":
    main()

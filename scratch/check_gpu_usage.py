"""Checks GPU usage on the titan compute node via ssh and nvidia-smi."""
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
        
        # Check titan GPU (runs both Optuna and Foundation Models)
        print("=== titan Node GPU Status ===")
        stdin, stdout, stderr = ssh.exec_command("ssh -o StrictHostKeyChecking=no titan nvidia-smi")
        print(stdout.read().decode('utf-8'))
        
        ssh.close()
    except Exception as e:
        print(f"[ERROR] GPU check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

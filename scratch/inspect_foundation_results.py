"""Reads and prints remote foundation models comparative results CSV."""
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
        
        print("=== Remote resultados_foundation_models.csv Content: ===")
        stdin, stdout, stderr = ssh.exec_command("cat /mnt/homeGPU/egullonl01/tfg/output/resultados_foundation_models.csv")
        print(stdout.read().decode('utf-8'))
        
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Read failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

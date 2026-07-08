"""Checks remote server directories to see if fatigueset dataset already exists."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

print(f"Connecting to {hostname} as {username}...")
try:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname, username=username, password=password, timeout=10)
    print("[OK] Connected to SSH successfully!")
    
    # Check directory contents of home folder and /mnt/homeGPU/egullonl01/
    commands = [
        "ls -l",
        "ls -l /mnt/homeGPU/egullonl01/",
        "df -h /mnt/homeGPU/egullonl01/"
    ]
    
    for cmd in commands:
        print(f"\nRunning command: {cmd}")
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode('utf-8', errors='ignore')
        err = stderr.read().decode('utf-8', errors='ignore')
        if out:
            print("--- OUTPUT ---")
            print(out)
        if err:
            print("--- ERROR ---")
            print(err)
            
    ssh.close()
    print("\n[OK] Connection closed.")
except Exception as e:
    print(f"[FAILED] Error connecting/executing commands: {e}")
    sys.exit(1)

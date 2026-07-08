# -*- coding: utf-8 -*-
import paramiko

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

def run_cmd(ssh, cmd):
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore').strip()
    err = stderr.read().decode('utf-8', errors='ignore').strip()
    return out, err

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=15)
        
        print("=== Queue Status ===")
        out, _ = run_cmd(ssh, "squeue -u egullonl01")
        print(out if out else "No active jobs.")
        
        print("\n=== Output Files in Server ===")
        out, _ = run_cmd(ssh, "ls -lh /mnt/homeGPU/egullonl01/tfg/output")
        print(out)
        
        print("\n=== Recent SLURM Logs ===")
        out, _ = run_cmd(ssh, "ls -lh /mnt/homeGPU/egullonl01/tfg/Jupyters/slurm-*.out")
        print(out)

        ssh.close()
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    main()

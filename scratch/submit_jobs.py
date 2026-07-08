# -*- coding: utf-8 -*-
"""Script to submit the SLURM jobs for Foundation Models and Optuna Optimizadores on the UGR server."""
import paramiko

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

def run_cmd(ssh, cmd):
    print(f"Executing remote command: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='ignore').strip()
    err = stderr.read().decode('utf-8', errors='ignore').strip()
    if out:
        print(f"STDOUT:\n{out}")
    if err:
        print(f"STDERR:\n{err}")
    return out, err

def main():
    try:
        print("Connecting to UGR GPU server via SSH...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=15)
        print("[OK] Connected successfully.\n")

        # Check existing queue
        print("Checking active jobs for user...")
        run_cmd(ssh, "squeue -u egullonl01")

        # Submit jobs
        print("\nSubmitting Foundation Models job...")
        run_cmd(ssh, "sbatch /mnt/homeGPU/egullonl01/tfg/Jupyters/run_foundation_server.sh")

        # Check queue again to verify submission
        print("\nChecking updated queue...")
        run_cmd(ssh, "squeue -u egullonl01")

        ssh.close()
        print("\n[SUCCESS] JOBS SUBMITTED SUCCESSFULLY!")
    except Exception as e:
        print(f"[ERROR] Job submission failed: {e}")

if __name__ == "__main__":
    main()

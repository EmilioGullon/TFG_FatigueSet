"""Checks SLURM queue and reads the tail of slurm log files on the UGR GPU server."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

job_ids = {
    "Optuna": "155622",
    "Foundation Models": "155639"
}

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        # 1. Check queue status
        print("=== SLURM QUEUE STATUS ===")
        stdin, stdout, stderr = ssh.exec_command("squeue -u egullonl01")
        print(stdout.read().decode('utf-8'))
        
        # 2. Check logs for each job
        print("=== SLURM LOG TAILS ===")
        for name, job_id in job_ids.items():
            print(f"\n--- {name} (Job ID: {job_id}) ---")
            
            # Paths to check
            possible_paths = [
                f"/mnt/homeGPU/egullonl01/slurm-{job_id}.out",
                f"/mnt/homeGPU/egullonl01/tfg/Jupyters/slurm-{job_id}.out"
            ]
            
            log_path = None
            for path in possible_paths:
                stdin, stdout, stderr = ssh.exec_command(f"ls {path}")
                if stdout.channel.recv_exit_status() == 0:
                    log_path = path
                    break
            
            if log_path:
                print(f"Log found at: {log_path}")
                stdin, stdout, stderr = ssh.exec_command(f"tail -n 25 {log_path}")
                print(stdout.read().decode('utf-8'))
            else:
                print(f"Log file not found yet in home or Jupyters directories (job is likely starting or pending).")
                
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""Queries scontrol and squeue to monitor the active and pending SLURM jobs on the UGR GPU server."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

job_ids = ["155639"]

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        # 1. Print squeue for current user
        print("=== squeue -u egullonl01 ===")
        stdin, stdout, stderr = ssh.exec_command("squeue -u egullonl01")
        print(stdout.read().decode('utf-8'))
        
        # 2. Print details of specified jobs
        for jid in job_ids:
            print(f"=== Job ID: {jid} scontrol details ===")
            stdin, stdout, stderr = ssh.exec_command(f"scontrol show job {jid}")
            out = stdout.read().decode('utf-8')
            err = stderr.read().decode('utf-8')
            
            if out:
                lines = out.split('\n')
                for line in lines:
                    if any(k in line for k in ["WorkDir=", "StdOut=", "StdErr=", "JobState=", "RunTime=", "Command="]):
                        print(line.strip())
            else:
                print(f"No details found or job completed. stderr: {err.strip()}")
                
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Connection check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

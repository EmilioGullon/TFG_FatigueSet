"""Submits the foundation models job to the UGR GPU server."""
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
        
        # Make script executable just in case
        ssh.exec_command("chmod +x /mnt/homeGPU/egullonl01/tfg/Jupyters/run_foundation_server.sh")
        
        # Submit the job
        print("Submitting Foundation Models job:")
        stdin, stdout, stderr = ssh.exec_command(
            "cd /mnt/homeGPU/egullonl01/tfg/Jupyters/ && sbatch run_foundation_server.sh"
        )
        submit_output = stdout.read().decode('utf-8').strip()
        print(f"Output: {submit_output}")
        
        # Parse job ID
        # Output is usually: "Submitted batch job 155624"
        if "Submitted batch job" in submit_output:
            job_id = submit_output.split()[-1]
            print(f"[OK] Submitted Foundation Models with Job ID: {job_id}")
        else:
            print("[WARNING] Could not parse Job ID.")
            
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Submission failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

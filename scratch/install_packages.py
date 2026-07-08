"""Remote package installer for UGR GPU cluster conda environment."""
import paramiko
import sys

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"
remote_conda_dir = "/mnt/homeGPU/egullonl01/conda_tfg"

def run_remote_pip(ssh, install_args):
    print(f"\nRunning: pip install {install_args}...")
    pip_cmd = (
        "export PATH=\"/opt/anaconda/anaconda3/bin:$PATH\"\n"
        "export PATH=\"/opt/anaconda/bin:$PATH\"\n"
        "eval \"$(conda shell.bash hook)\"\n"
        f"conda activate {remote_conda_dir}\n"
        f"pip install {install_args}"
    )
    stdin, stdout, stderr = ssh.exec_command(pip_cmd)
    
    # Read output progressively
    for line in stdout:
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            sys.stdout.write(line.encode('ascii', errors='ignore').decode('ascii'))
    sys.stdout.flush()
    
    err = stderr.read().decode('utf-8', errors='ignore')
    if err:
        print("\n--- ERROR/WARNING OUTPUT ---")
        print(err)
        
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        print(f"[OK] pip install {install_args} completed successfully.")
        return True
    else:
        print(f"[FAILED] pip install {install_args} exited with status {exit_status}.")
        return False

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        print("[OK] Connected to UGR GPU server.")
        
        # Step 1: Base scientific stack
        success_base = run_remote_pip(ssh, "scipy matplotlib pandas scikit-learn optuna openpyxl jupyter")
        if not success_base:
            print("[ERROR] Base scientific stack failed. Aborting.")
            ssh.close()
            sys.exit(1)
            
        # Step 2: Chronos
        run_remote_pip(ssh, "chronos-forecasting")
        
        # Step 3: TimesFM
        run_remote_pip(ssh, "timesfm")
        
        # Step 4: MOMENT
        run_remote_pip(ssh, "momentfm")
        
        ssh.close()
        print("\n=== STEP-BY-STEP INSTALLATIONS FINISHED ===")
    except Exception as e:
        print(f"[ERROR] Connection check failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

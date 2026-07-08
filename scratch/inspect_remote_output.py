"""Inspects files in remote output directory and prints TimesFM metrics."""
import paramiko
import sys
import json

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

def main():
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        # 1. List output directory recursively
        print("=== Files in remote output/ directory: ===")
        stdin, stdout, stderr = ssh.exec_command("find /mnt/homeGPU/egullonl01/tfg/output/ -type f")
        print(stdout.read().decode('utf-8'))
        
        # 2. Read TimesFM metrics
        print("=== TimesFM Probabilistic Metrics (UGR Server Run): ===")
        stdin, stdout, stderr = ssh.exec_command("cat /mnt/homeGPU/egullonl01/tfg/output/metricas_probabilisticas_timesfm.json")
        metrics_content = stdout.read().decode('utf-8')
        try:
            metrics = json.loads(metrics_content)
            print(json.dumps(metrics, indent=2))
        except Exception:
            print("Raw content:")
            print(metrics_content)
            
        ssh.close()
    except Exception as e:
        print(f"[ERROR] Remote output inspection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

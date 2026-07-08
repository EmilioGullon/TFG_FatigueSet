"""Parses the executing remote notebook experimento_optuna_optimizadores.ipynb and prints cell outputs."""
import paramiko
import sys
import json

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=10)
        
        # Download the notebook file content
        sftp = ssh.open_sftp()
        with sftp.open("/mnt/homeGPU/egullonl01/tfg/Jupyters/experimento_optuna_optimizadores.ipynb", "r") as f:
            nb = json.load(f)
        sftp.close()
        ssh.close()
        
        print("=== EXECUTED CELL OUTPUTS IN EXPERIMENTO_OPTUNA_OPTIMIZADORES.IPYNB ===")
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                source_preview = "".join(cell['source'][:3]).strip().replace('\n', ' ')
                # Print outputs if any
                outputs = cell.get('outputs', [])
                if outputs:
                    print(f"\nCell {i} (Preview: {source_preview}...):")
                    for out in outputs:
                        if out.get('output_type') == 'stream':
                            print("".join(out.get('text', [])).strip())
                        elif out.get('output_type') == 'error':
                            print("  [ERROR]:")
                            print("\n".join(out.get('traceback', [])))
                        elif out.get('output_type') == 'execute_result':
                            print("  [EXECUTE RESULT]:")
                            print("".join(out.get('data', {}).get('text/plain', [])))
                        
    except Exception as e:
        print(f"[ERROR] Remote notebook parsing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

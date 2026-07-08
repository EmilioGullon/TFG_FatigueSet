"""Automated deployment script to transition the FatigueSet codebase and dataset to NGPU-UGR."""
import paramiko
import os
import sys
import shutil
import time

hostname = "ngpu.ugr.es"
username = "egullonl01"
password = "xxegullonl01xx"

remote_tfg_dir = "/mnt/homeGPU/egullonl01/tfg"
remote_conda_dir = "/mnt/homeGPU/egullonl01/conda_tfg"

def safe_write(line):
    """Escribe en stdout ignorando caracteres que la consola local no pueda codificar (evita UnicodeEncodeError)."""
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        # Reintentar eliminando caracteres incompatibles con la codificación de la consola Windows (ej. CP1252)
        sys.stdout.write(line.encode('ascii', errors='ignore').decode('ascii'))

def progress_callback(transferred, total):
    percent = (transferred / total) * 100
    # Print progress every 10%
    if int(percent) % 10 == 0:
        sys.stdout.write(f"\rUploading: {percent:.1f}% ({transferred/(1024*1024):.1f}MB of {total/(1024*1024):.1f}MB)")
        sys.stdout.flush()

def upload_dir(sftp, local_path, remote_path):
    try:
        sftp.mkdir(remote_path)
    except IOError:
        pass
        
    for item in os.listdir(local_path):
        local_item = os.path.join(local_path, item)
        remote_item = os.path.join(remote_path, item).replace("\\", "/")
        
        # Ignorar archivos no deseados
        if any(skip in local_item for skip in ['.venv', '.git', '__pycache__', '.ipynb_checkpoints', '.pytest_cache', 'scratch']):
            continue
            
        if os.path.isdir(local_item):
            upload_dir(sftp, local_item, remote_item)
        else:
            print(f"Uploading file: {item} -> {remote_item}")
            sftp.put(local_item, remote_item)

def main():
    zip_basename = "fatigueset_temp"
    zip_filename = f"{zip_basename}.zip"
    
    try:
        # 1. Conectar al servidor por SSH
        print("1. Connecting to UGR GPU server via SSH...")
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username=username, password=password, timeout=20)
        print("[OK] Connected successfully.")
        
        # 2. Comprobar si el dataset ya existe en el servidor para evitar resubida de 3.2 GB
        print("\n2. Checking if dataset already exists on the server...")
        stdin, stdout, stderr = ssh.exec_command(f"ls {remote_tfg_dir}/fatigueset/metadata.csv")
        dataset_exists = (stdout.channel.recv_exit_status() == 0)
        
        if dataset_exists:
            print("[OK] Dataset already present on the server. Skipping zip and upload steps.")
        else:
            print("Dataset not found on the server. Initiating compression and transfer...")
            # Comprimir el dataset localmente para optimizar la transferencia
            print("Compressing fatigueset dataset (3.2 GB)...")
            t0 = time.time()
            shutil.make_archive(zip_basename, 'zip', 'fatigueset')
            zip_size = os.path.getsize(zip_filename) / (1024 * 1024)
            print(f"[OK] Dataset compressed in {time.time() - t0:.1f}s. Size: {zip_size:.1f} MB.")
            
            # Crear directorios remotos
            print(f"Creating remote directory: {remote_tfg_dir}...")
            ssh.exec_command(f"mkdir -p {remote_tfg_dir}")
            ssh.exec_command(f"mkdir -p {remote_tfg_dir}/fatigueset")
            
            # Conectar por SFTP y subir zip
            sftp = ssh.open_sftp()
            remote_zip_path = f"{remote_tfg_dir}/fatigueset_temp.zip"
            print(f"Uploading {zip_filename} to {remote_zip_path}...")
            t_upload = time.time()
            sftp.put(zip_filename, remote_zip_path, callback=progress_callback)
            print(f"\n[OK] Dataset uploaded in {time.time() - t_upload:.1f}s.")
            sftp.close()
            
            # Descomprimir en el servidor
            print("Extracting dataset on the server...")
            stdin, stdout, stderr = ssh.exec_command(
                f"unzip -q {remote_zip_path} -d {remote_tfg_dir}/fatigueset/ && rm {remote_zip_path}"
            )
            exit_status = stdout.channel.recv_exit_status()
            if exit_status == 0:
                print("[OK] Dataset extracted and cleaned successfully.")
            else:
                print(f"[ERROR] Extraction failed. stderr: {stderr.read().decode('utf-8')}")

        # 3. Subir código actualizado (lib, Jupyters, etc.)
        print("\n3. Transferring updated codebase files...")
        ssh.exec_command(f"mkdir -p {remote_tfg_dir}")
        sftp = ssh.open_sftp()
        
        # Subir el código de fatigueset-lib
        print("Uploading fatigueset-lib package...")
        upload_dir(sftp, "fatigueset-lib", f"{remote_tfg_dir}/fatigueset-lib")
        
        # Subir los notebooks y scripts de SLURM
        print("Uploading Jupyters and configuration files...")
        upload_dir(sftp, "Jupyters", f"{remote_tfg_dir}/Jupyters")
        
        # Subir archivos sueltos en el root
        for f in [".gitignore", "README.md"]:
            if os.path.exists(f):
                sftp.put(f, f"{remote_tfg_dir}/{f}")
                
        sftp.close()
        
        # 4. Crear y configurar el entorno Conda remoto
        print("\n4. Setting up Conda environment on the server...")
        print(f"Checking if conda environment at {remote_conda_dir} exists...")
        stdin, stdout, stderr = ssh.exec_command(f"ls -d {remote_conda_dir}")
        env_exists = (stdout.channel.recv_exit_status() == 0)
        
        if not env_exists:
            print("Conda environment not found. Creating environment python=3.10...")
            conda_cmd = (
                "export PATH=\"/opt/anaconda/anaconda3/bin:$PATH\"\n"
                "export PATH=\"/opt/anaconda/bin:$PATH\"\n"
                f"conda create -p {remote_conda_dir} python=3.10 -y"
            )
            stdin, stdout, stderr = ssh.exec_command(conda_cmd)
            for line in stdout:
                safe_write(line)
            sys.stdout.flush()
        else:
            print("[OK] Conda environment already exists.")
            
        print("\nInstalling packages in the conda environment...")
        pip_cmd = (
            "export PATH=\"/opt/anaconda/anaconda3/bin:$PATH\"\n"
            "export PATH=\"/opt/anaconda/bin:$PATH\"\n"
            "eval \"$(conda shell.bash hook)\"\n"
            f"conda activate {remote_conda_dir}\n"
            "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118\n"
            "pip install scipy matplotlib pandas scikit-learn optuna openpyxl jupyter momentfm chronos-forecasting timesfm"
        )
        stdin, stdout, stderr = ssh.exec_command(pip_cmd)
        
        print("Running package installations (this may take a few minutes)...")
        for line in stdout:
            safe_write(line)
        sys.stdout.flush()
        
        ssh.close()
        print("\n[SUCCESS] DEPLOYMENT AND ENVIRONMENT SET UP FINISHED SUCCESSFULLY!")
        
    except Exception as e:
        print(f"\n[ERROR] Deployment failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpiar archivo zip local si existe
        if os.path.exists(zip_filename):
            print(f"\nCleaning local temporary zip file: {zip_filename}")
            os.remove(zip_filename)

if __name__ == "__main__":
    main()

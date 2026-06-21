# -*- coding: utf-8 -*-
"""
Script de Sanity Check para verificar que todos los modelos se instancien,
realicen una pasada forward/backward y entrenen por 1 época sin errores.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Agregar la librería al path
lib_path = str(Path(__file__).resolve().parent.parent / "fatigueset-lib")
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

from fatigueset.models import (
    RNNFatiga,
    CustomLSTMRegressor,
    CustomGRURegressor,
    CustomCNNLSTMRegressor,
    CustomTCNRegressor,
    CustomTSTransformerRegressor,
    CustomPatchTSTRegressor,
    CustomxLSTMRegressor
)


def main():
    print("=" * 80)
    print("INICIANDO VERIFICACIÓN DE SANIDAD DE MODELOS (SANITY CHECK)")
    print("=" * 80)

    # Parámetros ficticios
    batch_size = 4
    seq_len = 32
    input_size = 15
    output_size = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo de prueba: {device}\n")

    # Definir modelos a verificar
    models_dict = {
        "RNN Clásica": RNNFatiga(input_size=input_size, hidden_size=32, num_layers=1),
        "Custom LSTM": CustomLSTMRegressor(input_size=input_size, hidden_size=32, num_layers=2),
        "Custom GRU": CustomGRURegressor(input_size=input_size, hidden_size=32, num_layers=2),
        "Custom CNN-LSTM": CustomCNNLSTMRegressor(input_size=input_size, hidden_size=32, num_layers=2),
        "Custom TCN": CustomTCNRegressor(input_size=input_size, num_channels=[32, 32, 32]),
        "Custom Transformer": CustomTSTransformerRegressor(input_size=input_size, d_model=32, num_heads=4, num_layers=2),
        "Custom PatchTST": CustomPatchTSTRegressor(input_size=input_size, patch_len=8, stride=4, d_model=32, num_heads=4, num_layers=2),
        "Custom xLSTM (sLSTM)": CustomxLSTMRegressor(input_size=input_size, hidden_size=32, num_layers=2)
    }

    # Crear datos ficticios para 1 época
    x_dummy = torch.randn(20, seq_len, input_size)
    y_dummy = torch.randn(20, output_size)
    dataset = TensorDataset(x_dummy, y_dummy)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    success = True

    for name, model in models_dict.items():
        print(f"Probando {name}...")
        model = model.to(device)
        
        # 1. Contar parámetros
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  - Parámetros entrenables: {num_params:,}")
        
        # 2. Paso forward & backward individual
        try:
            model.train()
            xb, yb = next(iter(dataloader))
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            
            optimizer.zero_grad()
            pred = model(xb)
            
            # Verificar dimensiones
            if pred.shape != (batch_size, output_size):
                print(f"  [ERROR] Forma de salida incorrecta: {pred.shape} (se esperaba {(batch_size, output_size)})")
                success = False
                continue
                
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            
            print("  - Forward & Backward: exitoso (MSE Loss = {:.4f})".format(loss.item()))
            
        except Exception as e:
            print(f"  [ERROR] Fallo en forward/backward paso: {str(e)}")
            success = False
            continue

        # 3. Entrenamiento corto por 1 época
        try:
            epoch_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            print("  - Entrenamiento completo de 1 época: exitoso (Pérdida media = {:.4f})".format(epoch_loss / len(dataloader)))
            print(f"  [OK] {name} verificado con éxito.\n")
            
        except Exception as e:
            print(f"  [ERROR] Fallo en entrenamiento de época: {str(e)}")
            success = False
            
    print("=" * 80)
    if success:
        print("VERIFICACIÓN COMPLETA: ¡Todos los modelos superaron las pruebas de sanidad!")
    else:
        print("VERIFICACIÓN COMPLETA: Hubo errores en la prueba de sanidad de algunos modelos.")
    print("=" * 80)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

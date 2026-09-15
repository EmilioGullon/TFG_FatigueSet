#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generador de la composición de imágenes reales con sobreimpresión HUD biométrica
para la Diapositiva 3 de la presentación del TFG.
Combina 3 fotografías reales auténticas (camionero, cirujano, operador de sala de control)
con HUDs biométricos fotorrealistas de alta resolución (ECG, EDA, EEG, métricas VAS).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def generate_hud_composition():
    fig_dir = os.path.join(os.path.dirname(__file__), '..', 'presentacion', 'figures')
    
    # Cargar imágenes reales
    truck_path = os.path.join(fig_dir, 'real_truck.jpg')
    surgeon_path = os.path.join(fig_dir, 'real_surgeon.jpg')
    control_path = os.path.join(fig_dir, 'real_control.jpg')
    
    img_truck = Image.open(truck_path).convert('RGB')
    img_surgeon = Image.open(surgeon_path).convert('RGB')
    img_control = Image.open(control_path).convert('RGB')
    
    # Dimensiones deseadas para cada panel
    panel_w = 600
    panel_h = 750
    
    def crop_and_resize(im, target_w, target_h):
        w, h = im.size
        target_aspect = target_w / target_h
        current_aspect = w / h
        if current_aspect > target_aspect:
            new_w = int(h * target_aspect)
            left = (w - new_w) // 2
            im = im.crop((left, 0, left + new_w, h))
        else:
            new_h = int(w / target_aspect)
            top = (h - new_h) // 2
            im = im.crop((0, top, w, top + new_h))
        return im.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    p1 = np.array(crop_and_resize(img_truck, panel_w, panel_h)) / 255.0
    p2 = np.array(crop_and_resize(img_surgeon, panel_w, panel_h)) / 255.0
    p3 = np.array(crop_and_resize(img_control, panel_w, panel_h)) / 255.0
    
    # Configurar figura Matplotlib 3 paneles
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.6), dpi=300)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.025)
    
    panels_data = [
        {
            'ax': axes[0],
            'img': p1,
            'sector': "SECTOR TRANSPORTE Y LOGÍSTICA",
            'escenario': "Conductor de Mercancías de Larga Distancia",
            'sensor': "Vestible: Zephyr BioHarness 3.0",
            'senal_nombre': "ECG + Frecuencia Cardíaca (HRV)",
            'metrics': [
                ("Frecuencia Cardíaca:", "88 bpm (Taquicardia compensatoria)"),
                ("Índice LF/HF (HRV):", "2.95 (Sobrecarga simpática)"),
                ("Frecuencia Respiratoria:", "24 rpm (Hiperventilación)")
            ],
            'alert_title': "⚠️ ALERTA: FATIGA FÍSICA CRÍTICA",
            'vas': "VAS Física: 78 / 100",
            'color_theme': '#10b981',  # Esmeralda
            'alert_bg': '#dc2626',      # Rojo alerta
            'signal_type': 'ecg'
        },
        {
            'ax': axes[1],
            'img': p2,
            'sector': "CIRUGÍA Y SANIDAD DE ALTO RIESGO",
            'escenario': "Cirujano en Intervención Prolongada (>4h)",
            'sensor': "Vestible: Empatica E4 Wristband",
            'senal_nombre': "EDA (Conductancia) + Fotopletismografía",
            'metrics': [
                ("Conductancia (EDA):", "8.42 µS (Tónica elevada)"),
                ("Picos Fásicos (SCR):", "14 picos/min (Estrés neurovegetativo)"),
                ("Amplitud de Pulso (BVP):", "-32% (Vasoconstricción periférica)")
            ],
            'alert_title': "⚠️ ALERTA: AGOTAMIENTO PSICOFÍSICO",
            'vas': "VAS Combinada: 72 / 100",
            'color_theme': '#0284c7',  # Azul tecnológico
            'alert_bg': '#ea580c',      # Naranja alerta
            'signal_type': 'eda'
        },
        {
            'ax': axes[2],
            'img': p3,
            'sector': "SALAS DE CONTROL E INDUSTRIA 4.0",
            'escenario': "Operador de Monitorización de Infraestructuras",
            'sensor': "Vestible: Muse S Headband (4 Ch)",
            'senal_nombre': "EEG Frontal (AF7/AF8: Ondas Alfa y Theta)",
            'metrics': [
                ("Potencia Banda Alfa (8-12 Hz):", "+52% (Desactivación cortical)"),
                ("Ratio Theta / Beta:", "4.35 (Somnolencia severa)"),
                ("Tiempo Reacción Motor:", "+42% (Reflejos degradados)")
            ],
            'alert_title': "⚠️ ALERTA: DECAIMIENTO DE VIGILIA",
            'vas': "VAS Mental: 84 / 100",
            'color_theme': '#8b5cf6',  # Violeta tecnológico
            'alert_bg': '#b91c1c',      # Rojo oscuro
            'signal_type': 'eeg'
        }
    ]
    
    for item in panels_data:
        ax = item['ax']
        im = item['img'].copy()
        
        # Aplicar viñeta oscura y degradado inferior/superior para máxima legibilidad del HUD
        h, w, _ = im.shape
        grad_bottom = np.linspace(0, 0.85, int(h * 0.52))[:, None, None]
        im[-int(h * 0.52):, :, :] = im[-int(h * 0.52):, :, :] * (1 - grad_bottom)
        grad_top = np.linspace(0.80, 0, int(h * 0.22))[:, None, None]
        im[:int(h * 0.22), :, :] = im[:int(h * 0.22), :, :] * (1 - grad_top)
        
        ax.imshow(im)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Borde exterior del panel
        rect_border = patches.Rectangle((0, 0), w, h, linewidth=3.5, edgecolor=item['color_theme'], facecolor='none')
        ax.add_patch(rect_border)
        
        # --- Cabecera Superior (Badge Sector) ---
        # Caja translúcida superior
        rect_top = patches.FancyBboxPatch((18, 18), w - 36, 85, boxstyle="round,pad=5,rounding_size=8",
                                          facecolor='#0f172a', alpha=0.90, edgecolor=item['color_theme'], linewidth=1.5)
        ax.add_patch(rect_top)
        ax.text(w / 2.0, 48, item['sector'], ha='center', va='center', fontsize=11, fontweight='bold', color='#f8fafc')
        ax.text(w / 2.0, 78, item['escenario'], ha='center', va='center', fontsize=8.5, color='#94a3b8')
        
        # --- Telemetría en Tiempo Real (Onda Biológica) ---
        # Mini gráfico superpuesto tipo HUD
        hud_box_y = h - 305
        rect_hud = patches.FancyBboxPatch((18, hud_box_y), w - 36, 285, boxstyle="round,pad=6,rounding_size=10",
                                          facecolor='#0b1329', alpha=0.92, edgecolor=item['color_theme'], linewidth=1.8)
        ax.add_patch(rect_hud)
        
        # Título de bioseñal
        ax.text(32, hud_box_y + 26, f"TELEMETRÍA: {item['senal_nombre']}", ha='left', va='center',
                fontsize=8.5, fontweight='bold', color=item['color_theme'])
        ax.text(w - 32, hud_box_y + 26, "EN VIVO (64 Hz)", ha='right', va='center',
                fontsize=7.5, fontweight='bold', color='#ef4444')
        
        # Dibujar onda sintética realista según señal
        wave_x = np.linspace(35, w - 35, 140)
        norm_t = np.linspace(0, 4.0, 140)
        
        if item['signal_type'] == 'ecg':
            # ECG QRS wave
            wave_y = np.zeros_like(norm_t)
            for beat in [0.7, 1.8, 2.9]:
                dt = norm_t - beat
                wave_y += 18.0 * np.exp(-(dt**2)/(2*0.02**2))   # R peak
                wave_y -= 4.0 * np.exp(-((dt+0.04)**2)/(2*0.015**2)) # Q
                wave_y -= 5.0 * np.exp(-((dt-0.05)**2)/(2*0.02**2))  # S
                wave_y += 3.5 * np.exp(-((dt-0.18)**2)/(2*0.06**2))  # T
            wave_y = hud_box_y + 65 - wave_y
        elif item['signal_type'] == 'eda':
            # EDA tonic + phasic peak
            wave_y = 6.0 * (1 - np.exp(-norm_t / 1.5)) + 12.0 * np.exp(-((norm_t - 2.2)**2) / 0.4)
            wave_y = hud_box_y + 75 - wave_y
        else:
            # EEG Alpha rhythm synchronization
            wave_y = 8.0 * np.sin(2 * np.pi * 3.5 * norm_t) + 4.0 * np.sin(2 * np.pi * 1.5 * norm_t)
            wave_y = hud_box_y + 65 - wave_y
            
        ax.plot(wave_x, wave_y, color=item['color_theme'], linewidth=1.6, alpha=0.95)
        # Línea base punteada
        ax.axhline(hud_box_y + 65, xmin=35/w, xmax=(w-35)/w, color='#334155', linestyle=':', linewidth=0.8)
        
        # Métricas clave
        y_text_start = hud_box_y + 115
        for idx, (label, val) in enumerate(item['metrics']):
            ax.text(32, y_text_start + idx * 24, label, ha='left', va='center', fontsize=8.0, color='#cbd5e1')
            ax.text(w - 32, y_text_start + idx * 24, val, ha='right', va='center', fontsize=8.0,
                    fontweight='bold', color='#ffffff')
        
        # Badge Alerta de Fatiga
        rect_alert = patches.FancyBboxPatch((28, hud_box_y + 195), w - 56, 42, boxstyle="round,pad=3,rounding_size=6",
                                            facecolor=item['alert_bg'], alpha=0.95, edgecolor='#ffffff', linewidth=1.0)
        ax.add_patch(rect_alert)
        ax.text(w / 2.0, hud_box_y + 210, item['alert_title'], ha='center', va='center',
                fontsize=9.2, fontweight='bold', color='#ffffff')
        ax.text(w / 2.0, hud_box_y + 226, item['vas'], ha='center', va='center',
                fontsize=8.2, fontweight='bold', color='#fef08a')
        
        # Sensor vestible asociado en el pie
        ax.text(w / 2.0, hud_box_y + 262, f"Dispositivo: {item['sensor']}", ha='center', va='center',
                fontsize=8.0, color='#94a3b8', style='italic')

    # Guardar en presentacion/figures tanto como fatigue_context_illustration.jpg como fatigue_real_hud_triptych.jpg
    out_path1 = os.path.join(fig_dir, 'fatigue_context_illustration.jpg')
    out_path2 = os.path.join(fig_dir, 'fatigue_real_hud_triptych.jpg')
    
    plt.savefig(out_path1, format='jpg', dpi=300, facecolor='#0f172a')
    plt.savefig(out_path2, format='jpg', dpi=300, facecolor='#0f172a')
    plt.close()
    
    print(f"[OK] Imagen HUD compuesta generada con éxito:")
    print(f" -> {out_path1}")
    print(f" -> {out_path2}")

if __name__ == '__main__':
    generate_hud_composition()

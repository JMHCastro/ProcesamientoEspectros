import os
import numpy as np
from scipy.signal import butter, sosfiltfilt

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Diseña un filtro pasabanda Butterworth usando secciones de segundo orden (SOS)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    low = max(0.0001, min(low, 0.9999))
    high = max(low + 0.0001, min(high, 0.9999))
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def ventana_tukey_manual(N, alpha=0.1):
    """Genera una ventana de Tukey (taper) de forma independiente para evitar errores de importación."""
    window = np.ones(N)
    if alpha <= 0:
        return window
    
    n = np.arange(N)
    width = int(alpha * (N - 1) / 2)
    
    if width > 0:
        # Inicio de la ventana
        window[:width+1] = 0.5 * (1 + np.cos(np.pi * (2 * n[:width+1] / (alpha * (N - 1)) - 1)))
        # Final de la ventana
        window[N-width-1:] = 0.5 * (1 + np.cos(np.pi * (2 * (n[N-width-1:] - N + 1) / (alpha * (N - 1)) + 1)))
        
    return window

def aplicar_filtro_y_recorte(data, dt, f_min, f_max, order, taper_pct, ia_inf, ia_sup):
    """Aplica taper, filtro pasabanda Butterworth de fase cero y recorte basado en Intensidad de Arias."""
    n_puntos = len(data)
    
    # 1. Aplicar Taper (ventana de Tukey) en los extremos
    alpha = 2.0 * (taper_pct / 100.0)
    window = ventana_tukey_manual(n_puntos, alpha=alpha)
    data_tapered = data * window
    
    # 2. Filtrado pasabanda
    fs = 1.0 / dt
    sos = butter_bandpass(f_min, f_max, fs, order=order)
    data_filtered = sosfiltfilt(sos, data_tapered)
    
    # 3. Cálculo de la Intensidad de Arias para el recorte
    g = 981.0  # cm/s^2 (asumiendo aceleraciones en cm/s^2)
    acc_cuadrada = data_filtered ** 2
    ia_acumulada = (np.pi / (2.0 * g)) * np.cumsum(acc_cuadrada) * dt
    ia_total = ia_acumulada[-1]
    
    if ia_total > 0:
        ia_norm = ia_acumulada / ia_total
        idx_ini = np.argmax(ia_norm >= (ia_inf / 100.0))
        idx_fin = np.argmax(ia_norm >= (ia_sup / 100.0))
        if idx_fin <= idx_ini:
            idx_fin = n_puntos - 1
    else:
        idx_ini, idx_fin = 0, n_puntos - 1
        
    # Recortar la señal y ajustar el vector de tiempo correspondiente
    recorte_acc = data_filtered[idx_ini:idx_fin + 1]
    tiempo_recortado = np.arange(len(recorte_acc)) * dt
    
    return tiempo_recortado, recorte_acc

def procesar_y_filtrar_acelerogramas():
    # Carpetas de salida
    output_folder_crudos = "Acelerogramas"
    output_folder_filtrados = "Acelerogramas_filtrados"
    
    os.makedirs(output_folder_crudos, exist_ok=True)
    os.makedirs(output_folder_filtrados, exist_ok=True)
    
    # Parámetros extraídos de la tabla
    TAPER_PCT = 5.0        # Taper en %
    F_MIN = 0.1            # Frecuencia mínima (Hz)
    F_MAX = 20.0           # Frecuencia máxima (Hz)
    ORDEN_FILTRO = 4       # Polos utilizados en el filtro
    IA_INF = 0.01          # Límite inferior intensidad de Arias (%)
    IA_SUP = 99.90         # Límite superior intensidad de Arias (%)
    
    # Definir los nombres de los archivos
    archivos_prof_intermedia = [f"Acelerograma_ProfIntermedia_{i}_.txt" for i in range(1, 5)]
    archivos_subduccion = [f"Acelerograma_Subduccion_{i}_.txt" for i in range(1, 9)]
    lista_archivos = archivos_prof_intermedia + archivos_subduccion
    
    archivos_procesados = 0
    
    for nombre_archivo in lista_archivos:
        if not os.path.exists(nombre_archivo):
            print(f"[AVISO] No se encontró el archivo: {nombre_archivo}. Se omitirá.")
            continue
            
        dt = 0.01  # Valor predeterminado
        skip_lines = 0
        
        with open(nombre_archivo, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            if "Dt=" in line:
                try:
                    partes = line.split("Dt=")[1].strip().split()[0]
                    dt = float(partes)
                except Exception:
                    pass
            if "a(t)_h1" in line:
                skip_lines = idx + 1
                break
                
        if skip_lines == 0:
            for idx, line in enumerate(lines):
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        float(parts[0])
                        float(parts[1])
                        skip_lines = idx
                        break
                except ValueError:
                    continue
                    
        try:
            datos = np.loadtxt(nombre_archivo, skiprows=skip_lines)
        except Exception as e:
            print(f"[ERROR] No se pudieron leer los datos de {nombre_archivo}: {e}")
            continue
            
        if datos.ndim != 2 or datos.shape[1] < 2:
            print(f"[AVISO] El archivo {nombre_archivo} no tiene al menos dos columnas.")
            continue
            
        n_puntos = datos.shape[0]
        tiempo_orig = np.arange(n_puntos) * dt
        
        h1 = datos[:, 0]
        h2 = datos[:, 1]
        
        # 1. Guardar archivos crudos originales separados por canal
        base_name = os.path.splitext(nombre_archivo)[0]
        archivo_h1_crudo = os.path.join(output_folder_crudos, f"{base_name}_H1.txt")
        archivo_h2_crudo = os.path.join(output_folder_crudos, f"{base_name}_H2.txt")
        
        header_crudo = "Tiempo(s)\tAceleracion(cm/s^2)"
        np.savetxt(archivo_h1_crudo, np.column_stack((tiempo_orig, h1)), fmt="%.6e", delimiter="\t", header=header_crudo, comments="")
        np.savetxt(archivo_h2_crudo, np.column_stack((tiempo_orig, h2)), fmt="%.6e", delimiter="\t", header=header_crudo, comments="")
        
        # 2. Aplicar filtrado, taper y recorte por Intensidad de Arias para H1 y H2
        t_h1_filt, h1_filtrado = aplicar_filtro_y_recorte(h1, dt, F_MIN, F_MAX, ORDEN_FILTRO, TAPER_PCT, IA_INF, IA_SUP)
        t_h2_filt, h2_filtrado = aplicar_filtro_y_recorte(h2, dt, F_MIN, F_MAX, ORDEN_FILTRO, TAPER_PCT, IA_INF, IA_SUP)
        
        # Guardar en la carpeta "Acelerogramas_filtrados"
        archivo_h1_filt = os.path.join(output_folder_filtrados, f"{base_name}_H1_filtrado.txt")
        archivo_h2_filt = os.path.join(output_folder_filtrados, f"{base_name}_H2_filtrado.txt")
        
        header_filt = f"Tiempo(s)\tAceleracion_Filtrada(cm/s^2) [Taper:{TAPER_PCT}%, Filtro:{F_MIN}-{F_MAX}Hz, Polos:{ORDEN_FILTRO}, Arias:{IA_INF}%-{IA_SUP}%]"
        np.savetxt(archivo_h1_filt, np.column_stack((t_h1_filt, h1_filtrado)), fmt="%.6e", delimiter="\t", header=header_filt, comments="")
        np.savetxt(archivo_h2_filt, np.column_stack((t_h2_filt, h2_filtrado)), fmt="%.6e", delimiter="\t", header=header_filt, comments="")
        
        print(f"[OK] Procesado: {nombre_archivo} -> Filtrado y Recortado (Arias {IA_INF}%-{IA_SUP}%)")
        archivos_procesados += 1

    print(f"\n¡Proceso completado! Se procesaron {archivos_procesados} archivos.")
    print(f" - Crudos: '{output_folder_crudos}/'")
    print(f" - Filtrados y Recortados: '{output_folder_filtrados}/'")

if __name__ == "__main__":
    procesar_y_filtrar_acelerogramas()
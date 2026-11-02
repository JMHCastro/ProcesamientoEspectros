import os
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
    """Genera una ventana de Tukey (taper) de forma independiente."""
    window = np.ones(N)
    if alpha <= 0:
        return window
    
    n = np.arange(N)
    width = int(alpha * (N - 1) / 2)
    
    if width > 0:
        window[:width+1] = 0.5 * (1 + np.cos(np.pi * (2 * n[:width+1] / (alpha * (N - 1)) - 1)))
        window[N-width-1:] = 0.5 * (1 + np.cos(np.pi * (2 * (n[N-width-1:] - N + 1) / (alpha * (N - 1)) + 1)))
        
    return window

def guardar_grafica_comparativa(tiempo_orig, acc_orig, tiempo_filt, acc_filt, nombre_canal, output_folder):
    """Genera y guarda una imagen comparativa entre la señal cruda y la filtrada/recortada."""
    plt.figure(figsize=(10, 5))
    
    # Graficar señal cruda en gris/azul tenue
    plt.plot(tiempo_orig, acc_orig, color='gray', alpha=0.6, label='Crudo (Original)', linewidth=0.8)
    
    # Graficar señal filtrada y recortada en azul fuerte
    plt.plot(tiempo_filt, acc_filt, color='#1f77b4', alpha=0.9, label='Filtrado y Sincronizado', linewidth=1.0)
    
    plt.title(f"Comparativa de Acelerograma: {nombre_canal}", fontsize=12, fontweight='bold')
    plt.xlabel("Tiempo (s)", fontsize=10)
    plt.ylabel("Aceleración ($cm/s^2$)", fontsize=10)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    # Guardar imagen en la carpeta de filtrados
    img_filename = f"{nombre_canal}_comparativa.png"
    img_path = os.path.join(output_folder, img_filename)
    plt.savefig(img_path, dpi=300)
    plt.close()

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
    registros_para_excel = []
    
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
        fs = 1.0 / dt
        
        h1 = datos[:, 0]
        h2 = datos[:, 1]
        
        # 1. Guardar archivos crudos originales separados por canal
        base_name = os.path.splitext(nombre_archivo)[0]
        archivo_h1_crudo = os.path.join(output_folder_crudos, f"{base_name}_H1.txt")
        archivo_h2_crudo = os.path.join(output_folder_crudos, f"{base_name}_H2.txt")
        
        header_crudo = "Tiempo(s)\tAceleracion(cm/s^2)"
        np.savetxt(archivo_h1_crudo, np.column_stack((tiempo_orig, h1)), fmt="%.6e", delimiter="\t", header=header_crudo, comments="")
        np.savetxt(archivo_h2_crudo, np.column_stack((tiempo_orig, h2)), fmt="%.6e", delimiter="\t", header=header_crudo, comments="")
        
        # 2. Aplicar Taper y Filtro Pasabanda a ambas componentes de forma independiente
        alpha = 2.0 * (TAPER_PCT / 100.0)
        window = ventana_tukey_manual(n_puntos, alpha=alpha)
        sos = butter_bandpass(F_MIN, F_MAX, fs, order=ORDEN_FILTRO)
        
        h1_filt = sosfiltfilt(sos, h1 * window)
        h2_filt = sosfiltfilt(sos, h2 * window)
        
        # 3. Calcular la Intensidad de Arias COMBINADA para sincronizar el recorte
        g = 981.0  # cm/s^2
        ia_h1 = (np.pi / (2.0 * g)) * np.cumsum(h1_filt ** 2) * dt
        ia_h2 = (np.pi / (2.0 * g)) * np.cumsum(h2_filt ** 2) * dt
        ia_combinada = ia_h1 + ia_h2  # Energía total acumulada en 2D
        ia_total = ia_combinada[-1]
        
        if ia_total > 0:
            ia_norm = ia_combinada / ia_total
            idx_ini = np.argmax(ia_norm >= (IA_INF / 100.0))
            idx_fin = np.argmax(ia_norm >= (IA_SUP / 100.0))
            if idx_fin <= idx_ini:
                idx_fin = n_puntos - 1
        else:
            idx_ini, idx_fin = 0, n_puntos - 1
            
        # 4. Aplicar los MISMOS índices de recorte a H1 y H2 (Garantiza cero desfase y misma duración)
        h1_recortado = h1_filt[idx_ini:idx_fin + 1]
        h2_recortado = h2_filt[idx_ini:idx_fin + 1]
        tiempo_recortado = np.arange(len(h1_recortado)) * dt

# Nombres de las funciones para ETABS (igual al formato de la imagen)
        func_name_h1 = f"{base_name}__H1_filtrado"
        func_name_h2 = f"{base_name}__H2_filtrado"
        
        # Guardar en la carpeta "Acelerogramas_filtrados"
        archivo_h1_filt = os.path.join(output_folder_filtrados, f"{base_name}_H1_filtrado.txt")
        archivo_h2_filt = os.path.join(output_folder_filtrados, f"{base_name}_H2_filtrado.txt")
        
        header_filt = f"Tiempo(s)\tAceleracion_Filtrada(cm/s^2) [Taper:{TAPER_PCT}%, Filtro:{F_MIN}-{F_MAX}Hz, Arias Combinada:{IA_INF}%-{IA_SUP}%]"
        np.savetxt(archivo_h1_filt, np.column_stack((tiempo_recortado, h1_recortado)), fmt="%.6e", delimiter="\t", header=header_filt, comments="")
        np.savetxt(archivo_h2_filt, np.column_stack((tiempo_recortado, h2_recortado)), fmt="%.6e", delimiter="\t", header=header_filt, comments="")

#  Generar gráficas comparativas PNG para H1 y H2
        guardar_grafica_comparativa(tiempo_orig, h1, tiempo_recortado, h1_recortado, func_name_h1, output_folder_filtrados)
        guardar_grafica_comparativa(tiempo_orig, h2, tiempo_recortado, h2_recortado, func_name_h2, output_folder_filtrados)

# 5. Crear registros en formato ETABS para la exportación a Excel
        for func_name, acc_values in [(func_name_h1, h1_recortado), (func_name_h2, h2_recortado)]:
            guid_list = [""] * len(acc_values)
            guid_list[0] = str(uuid.uuid4())  # Generar GUID en la primera fila de la función

            df_func = pd.DataFrame({
                'Name': func_name,
                'Time sec': tiempo_recortado,
                'Value': acc_values,
                'GUID': guid_list
            })
            registros_para_excel.append(df_func)   

        print(f"[OK] Sincronizado y Procesado: {nombre_archivo} -> Duración: {tiempo_recortado[-1]:.2f}s (Puntos: {len(tiempo_recortado)})")
        archivos_procesados += 1

        # Generar el archivo de Excel con el formato de ETABS
    if registros_para_excel:
        exportar_a_excel_etabs(registros_para_excel, output_folder_filtrados)

    print(f"\n¡Proceso completado! Se procesaron {archivos_procesados} archivos de forma sincronizada.")
    print(f" - Crudos: '{output_folder_crudos}/'")
    print(f" - Filtrados y Sincronizados: '{output_folder_filtrados}/'")

def exportar_a_excel_etabs(lista_dfs,output_folder):
    excel_filename = "Acelerogramas_ETABS_Format.xlsx"
    excel_path = os.path.join(output_folder, excel_filename)
    df_consolidado = pd.concat(lista_dfs, ignore_index=True)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Hoja Consolidada (Ideal para copiar y pegar todo de un solo paso en ETABS)
        df_consolidado.to_excel(writer, sheet_name="ETABS_Consolidado", index=False)      

    print(f"[EXCEL OK] Archivo generado exitosamente: '{excel_path}' (Listo para copiar y pegar a ETABS)")


if __name__ == "__main__":
    procesar_y_filtrar_acelerogramas()
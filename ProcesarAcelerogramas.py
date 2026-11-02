#PROCESAMIENTO ACELEROGRAMAS SASID
#1.-Separa acelerogramas en dos canales (H1 y H2) y los guarda en una subcarpeta "Acelerogramas"
import os
import numpy as np

def procesar_acelerogramas():
    # Definir la subcarpeta de salida
    output_folder = "Acelerogramas"
    os.makedirs(output_folder, exist_ok=True)
    
    # Definir los nombres esperados para los archivos
    # (Puedes ajustar o renombrar las listas según los nombres exactos de tus archivos)
    archivos_prof_intermedia = [f"Acelerograma_ProfIntermedia_{i}_.txt" for i in range(1, 5)]
    archivos_subduccion = [f"Acelerograma_Subduccion_{i}_.txt" for i in range(1, 9)]
    
    lista_archivos = archivos_prof_intermedia + archivos_subduccion
    
    archivos_procesados = 0
    
    for nombre_archivo in lista_archivos:
        if not os.path.exists(nombre_archivo):
            print(f"[AVISO] No se encontró el archivo: {nombre_archivo}. Se omitirá.")
            continue
            
        dt = 0.01  # Valor predeterminado por si no se encuentra en la cabecera
        skip_lines = 0
        
        # Leer el archivo línea por línea para extraer Dt y la línea donde comienzan los datos
        with open(nombre_archivo, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        for idx, line in enumerate(lines):
            # Buscar el paso del tiempo Dt
            if "Dt=" in line:
                try:
                    partes = line.split("Dt=")[1].strip().split()[0]
                    dt = float(partes)
                except Exception:
                    pass
            # Identificar la cabecera de las columnas de datos
            if "a(t)_h1" in line:
                skip_lines = idx + 1
                break
                
        # Si no se encontró la etiqueta 'a(t)_h1', buscar la primera línea con valores numéricos
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
                    
        # Cargar los datos numéricos usando NumPy
        try:
            datos = np.loadtxt(nombre_archivo, skiprows=skip_lines)
        except Exception as e:
            print(f"[ERROR] No se pudieron leer los datos numéricos de {nombre_archivo}: {e}")
            continue
            
        if datos.ndim != 2 or datos.shape[1] < 2:
            print(f"[AVISO] El archivo {nombre_archivo} no tiene al menos dos columnas de datos.")
            continue
            
        n_puntos = datos.shape[0]
        tiempo = np.arange(n_puntos) * dt
        
        h1 = datos[:, 0]
        h2 = datos[:, 1]
        
        # Nombre base sin extensión
        base_name = os.path.splitext(nombre_archivo)[0]
        
        # Guardar Canal 1 (H1)
        archivo_h1 = os.path.join(output_folder, f"{base_name}_H1.txt")
        header_h1 = "Tiempo(s)\tAceleracion_H1(cm/s^2)"
        np.savetxt(archivo_h1, np.column_stack((tiempo, h1)), fmt="%.6e", delimiter="\t", header=header_h1, comments="")
        
        # Guardar Canal 2 (H2)
        archivo_h2 = os.path.join(output_folder, f"{base_name}_H2.txt")
        header_h2 = "Tiempo(s)\tAceleracion_H2(cm/s^2)"
        np.savetxt(archivo_h2, np.column_stack((tiempo, h2)), fmt="%.6e", delimiter="\t", header=header_h2, comments="")
        
        print(f"[OK] Procesado: {nombre_archivo} (Dt = {dt}s) -> Generados H1 y H2")
        archivos_procesados += 1

    print(f"\n¡Proceso finalizado! Se procesaron {archivos_procesados} archivos correctamente en la subcarpeta '{output_folder}'.")

if __name__ == "__main__":
    procesar_acelerogramas()
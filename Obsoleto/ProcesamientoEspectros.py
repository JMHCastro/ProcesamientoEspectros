import os
import pandas as pd

# ==========================================
# CONFIGURACIÓN
# ==========================================
# Lista con los nombres o rutas de tus 12 archivos .txt en orden
# (Asegúrate de colocar los 4 de Profundidad Intermedia y los 8 de Subducción en orden)
archivos_txt = [
    # Profundidad Intermedia (1 al 4)
    "EspectroSA_ProfIntermedia_1_.txt",
    "EspectroSA_ProfIntermedia_2_.txt",
    "EspectroSA_ProfIntermedia_3_.txt",
    "EspectroSA_ProfIntermedia_4_.txt",
    # Subducción (1 al 8)
    "EspectroSA_Subduccion_1_.txt",
    "EspectroSA_Subduccion_2_.txt",
    "EspectroSA_Subduccion_3_.txt",
    "EspectroSA_Subduccion_4_.txt",
    "EspectroSA_Subduccion_5_.txt",
    "EspectroSA_Subduccion_6_.txt",
    "EspectroSA_Subduccion_7_.txt",
    "EspectroSA_Subduccion_8_.txt",
]

# NOTA SOBRE UNIDADES:
# Los archivos de texto vienen en cm/s². Si en tu tabla de Excel los requieres
# como fracción de la gravedad (g), cambia esta variable a True (divide entre 981).
dividir_entre_g = True
g = 981.0  #cm / s²

datos_consolidados = {}
periodos = None

# ==========================================
# PROCESAMIENTO DE ARCHIVOS
# ==========================================
for archivo in archivos_txt:
  if not os.path.exists(archivo):
    print(f"Advertencia: No se encontró el archivo {archivo}")
    continue

  with open(archivo, "r", encoding="utf-8", errors="ignore") as f:
    lineas = f.readlines()

  # Buscar la línea donde comienzan los datos (después de 'T[s]')
  idx_inicio = 0
  for idx, linea in enumerate(lineas):
    if "T[s]" in linea:
      idx_inicio = idx + 1
      break

  t_vals, h1_vals, h2_vals = [], [], []
  for linea in lineas[idx_inicio:]:
    partes = linea.split()
    if len(partes) >= 3:
      t = float(partes[0])
      h1 = float(partes[1])
      h2 = float(partes[2])

      if dividir_entre_g:
        h1 = h1 / g
        h2 = h2 / g

      t_vals.append(t)
      h1_vals.append(h1)
      h2_vals.append(h2)

  # Guardar los periodos una sola vez
  if periodos is None:
    periodos = t_vals

# Limpiar el nombre del archivo para usarlo como prefijo de columna
  # (Ej: 'EspectroSA_ProfIntermedia_1_.txt' se convierte en 'ProfIntermedia_1')
  nombre_base = (
      archivo.replace("EspectroSA_", "").replace(".txt", "").strip("_")
  )

  # Asignar columnas C1 y C2 usando el nombre limpio del espectro
  datos_consolidados[f"{nombre_base}_C1"] = h1_vals
  datos_consolidados[f"{nombre_base}_C2"] = h2_vals

# ==========================================
# CREACIÓN Y EXPORTACIÓN DE LA TABLA
# ==========================================
if periodos:
  df_final = pd.DataFrame({"T(s)": periodos, **datos_consolidados})

  # Archivo de salida separado por tabulaciones (ideal para copiar a Excel)
  archivo_salida = "EspectrosAcelerogramas_Consolidados_Para_Excel.txt"
  df_final.to_csv(archivo_salida, sep="\t", index=False, float_format="%.4f")

  print(f"¡Proceso completado con éxito!")
  print(f"Archivo generado: {archivo_salida}")
  print(
      "Puedes abrir este archivo con un editor de texto o Excel, seleccionar"
      " todo y pegarlo directamente en tu plantilla."
  )
else:
  print(
      "No se pudieron procesar datos. Revisa que los nombres de los archivos"
      " sean correctos."
  )
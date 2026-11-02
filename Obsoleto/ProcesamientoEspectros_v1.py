import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# NUEVAS FUNCIONES: SRSS Y GRÁFICAS
# ==========================================


def agregar_columnas_srss(df, nombres_bases):
  """Calcula la raíz de la suma de los cuadrados (SRSS)

  para cada par de canales C1 y C2.
  """
  for base in nombres_bases:
    c1 = df[f"{base}_C1"]
    c2 = df[f"{base}_C2"]
    # Fórmula: sqrt(C1^2 + C2^2)
    df[base] = np.sqrt(c1**2 + c2**2)
  return df


def generar_graficas_espectros(df, nombres_bases):
  """Genera dos gráficas independientes:

  1. Profundidad Intermedia
  2. Subducción
  """
  # Separar nombres según su categoría
  prof_bases = [b for b in nombres_bases if "ProfIntermedia" in b]
  sub_bases = [b for b in nombres_bases if "Subduccion" in b]

  periodos = df["T(s)"]

  # --- GRÁFICA 1: Profundidad Intermedia ---
  plt.figure(figsize=(9, 6))
  for base in prof_bases:
    plt.plot(periodos, df[base], label=base, linewidth=1.5)
  plt.title(
      "Espectros SRSS - Profundidad Intermedia",
      fontsize=12,
      fontweight="bold",
  )
  plt.xlabel("Periodo T (s)", fontsize=10)
  plt.ylabel("Aceleración Espectral SRSS (g)", fontsize=10)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(fontsize=8, loc="upper right")
  plt.tight_layout()
  plt.savefig("Grafica_Profundidad_Intermedia.png", dpi=300)
  plt.show()

  # --- GRÁFICA 2: Subducción ---
  plt.figure(figsize=(9, 6))
  for base in sub_bases:
    plt.plot(periodos, df[base], label=base, linewidth=1.5)
  plt.title("Espectros SRSS - Subducción", fontsize=12, fontweight="bold")
  plt.xlabel("Periodo T (s)", fontsize=10)
  plt.ylabel("Aceleración Espectral SRSS (g)", fontsize=10)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(fontsize=8, loc="upper right", ncol=2)
  plt.tight_layout()
  plt.savefig("Grafica_Subduccion.png", dpi=300)
  plt.show()

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
nombres_bases = [] 
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

if nombre_base not in nombres_bases:
    nombres_bases.append(nombre_base)

  # Asignar columnas C1 y C2 usando el nombre limpio del espectro
datos_consolidados[f"{nombre_base}_C1"] = h1_vals
datos_consolidados[f"{nombre_base}_C2"] = h2_vals

# ==========================================
# CREACIÓN Y EXPORTACIÓN DE LA TABLA
# ==========================================
if periodos:
  df_final = pd.DataFrame({"T(s)": periodos, **datos_consolidados})

  # ==========================================
  # LLAMADAS A ESPECTRO SRSS Y GRÁFICAS
  # ==========================================
  df_final = agregar_columnas_srss(df_final, nombres_bases)
  generar_graficas_espectros(df_final, nombres_bases)
  # ==========================================

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


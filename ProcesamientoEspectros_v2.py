import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. FUNCIONES DE CÁLCULO Y GRÁFICAS
# ==========================================


def generar_graficas_espectros(df, nombres_prof, nombres_sub):
  """Genera dos gráficas independientes:

  1. Profundidad Intermedia (SRSS)
  2. Subducción (SRSS)
  """
  periodos = df["T(s)"]

  # --- GRÁFICA 1: Profundidad Intermedia ---
  plt.figure(figsize=(9, 6))
  for base in nombres_prof:
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
  for base in nombres_sub:
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
# 2. CONFIGURACIÓN (12 ARCHIVOS)
# ==========================================
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

# Unidades: True para dividir entre g = 981 cm/s^2
dividir_entre_g = True
g = 981.0

datos_consolidados = {}
periodos = None
nombres_bases = []

# ==========================================
# 3. PROCESAMIENTO DE ARCHIVOS
# ==========================================
for archivo in archivos_txt:
  if not os.path.exists(archivo):
    print(f"Advertencia: No se encontró el archivo '{archivo}'")
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

  if periodos is None:
    periodos = t_vals

  # Limpiar el nombre base (ej. ProfIntermedia_1)
  nombre_base = (
      archivo.replace("EspectroSA_", "").replace(".txt", "").strip("_")
  )
  if nombre_base not in nombres_bases:
    nombres_bases.append(nombre_base)

  # Guardar canales C1 y C2
  datos_consolidados[f"{nombre_base}_C1"] = h1_vals
  datos_consolidados[f"{nombre_base}_C2"] = h2_vals

# ==========================================
# 4. REORDENAMIENTO Y CÁLCULO SRSS
# ==========================================
if periodos:
  # Separar bases por categoría
  nombres_prof = [b for b in nombres_bases if "ProfIntermedia" in b]
  nombres_sub = [b for b in nombres_bases if "Subduccion" in b]

  # Construir las columnas en el orden exacto que solicitas:
  # 1. T(s)
  # 2. Todos los canales C1 y C2 (Izquierda)
  columnas_dict = {"T(s)": periodos}
  for base in nombres_bases:
    columnas_dict[f"{base}_C1"] = datos_consolidados[f"{base}_C1"]
    columnas_dict[f"{base}_C2"] = datos_consolidados[f"{base}_C2"]

  df_final = pd.DataFrame(columnas_dict)

  # 3. Agregar a la derecha las columnas SRSS ordenadas (Profundidad Intermedia y luego Subducción)
  for base in nombres_prof + nombres_sub:
    c1 = df_final[f"{base}_C1"]
    c2 = df_final[f"{base}_C2"]
    # Fórmula SRSS: Raíz de la suma de cuadrados
    df_final[base] = np.sqrt(c1**2 + c2**2)

  # Generar las dos gráficas solicitadas
  generar_graficas_espectros(df_final, nombres_prof, nombres_sub)

  # Exportar a TXT para copiar a Excel
  archivo_salida = "EspectrosAcelerogramas_Consolidados_Para_Excel.txt"
  df_final.to_csv(archivo_salida, sep="\t", index=False, float_format="%.4f")

  print(f"¡Proceso completado con éxito!")
  print(f"Archivo generado: {archivo_salida}")
  print(
      "Abre este archivo, copia todo su contenido y pégalo en tu celda superior"
      " en Excel."
  )
else:
  print("No se encontraron datos para procesar.")
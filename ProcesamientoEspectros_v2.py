import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==========================================
# 1. FUNCIONES DE CÁLCULO Y GRÁFICAS
# ==========================================


def generar_graficas_espectros(df, nombres_prof, nombres_sub):
  """Genera dos gráficas independientes:

  1. Profundidad Intermedia (SRSS, Envolvente, Envolvente Escalada y Diseño)
  2. Subducción (SRSS, Envolvente, Envolvente Escalada y Diseño)"""
  
  periodos = df["T(s)"]

  # Color gris para las curvas de los espectros
  color_azul_base ="gray"

  # --- GRÁFICA 1: Profundidad Intermedia ---
  plt.figure(figsize=(9, 6))

  for i,base in enumerate(nombres_prof):
# Variar la transparencia (alpha) desde 0.3 (más claro) hasta 0.7 (más visible)
    opacidad = 0.3 + (0.4 * i / max(1, len(nombres_prof) - 1))
    plt.plot(periodos,df[base], label=base,
      color=color_azul_base, linewidth=1.5, 
      alpha=opacidad, linestyle="-")
  # LÍNEA PARA LA ENVOLVENTE BASE:
  plt.plot(periodos,df["Envolvente_ProfIntermedia"],label="Envolvente Prof.Intermedia",
    color="black",linewidth=2.5,linestyle="--",alpha=0.9)

# ENVOLVENTE ESCALADA (Factor 1.6):
  plt.plot(
      periodos,
      df["EnvolventeEscalada_ProfIntermedia"],
      label=f"ENVOLVENTE x {FACTOR_PROF}",
      color="#2d3436",
      linewidth=2.5,
      linestyle="-",
  )

  # ESPECTROS EPU Y DISEÑO EN ROJO:
  if "EPU" in df.columns:
    plt.plot(
        periodos,
        df["EPU"],
        label="EPU",
        color="#D63031",
        linewidth=2.0,
        linestyle="-",)
  if "Elastico" in df.columns:
    plt.plot(
        periodos,
        df["Elastico"],
        label="Elástico",
        color="#D63031",
        linewidth=2.0,
        linestyle=":",)
  
  plt.title("Espectros - Profundidad Intermedia",
      fontsize=12,fontweight="bold",)
  plt.xlabel("Periodo T (s)", fontsize=10)
  plt.ylabel("Aceleración Espectral SRSS (g)", fontsize=10)
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend(fontsize=8, loc="upper right")
  plt.tight_layout()
  plt.savefig("Grafica_Profundidad_Intermedia.png", dpi=300)
  plt.show()

  # --- GRÁFICA 2: Subducción ---
  plt.figure(figsize=(9, 6))

  for i, base in enumerate(nombres_sub):
    opacidad = 0.35 + (0.55 * i / max(1, len(nombres_sub) - 1))
    plt.plot(
        periodos,
        df[base],
        label=base,
        color=color_azul_base,
        alpha=opacidad,
        linewidth=1.6,
        linestyle="-",
    )

  # Graficar la envolvente base de Subducción en línea negra punteada
  plt.plot(
      periodos,
      df["Envolvente_Subduccion"],
      label="Envolvente Subducción",
      color="#0B1A2E",
      linewidth=2.5,
      linestyle="--",
  )

# ENVOLVENTE ESCALADA (Factor 1.05):
  plt.plot(
      periodos,
      df["EnvolventeEscalada_Subduccion"],
      label=f"ENVOLVENTE x {FACTOR_SUB}",
      color="#2d3436",
      linewidth=2.5,
      linestyle="-",
  )

  # ESPECTROS DE DISEÑO EN ROJO:
  if "EPU" in df.columns:
    plt.plot(
        periodos,
        df["EPU"],
        label="EPU",
        color="#D63031",
        linewidth=2.0,
        linestyle="-",
    )
  if "Elastico" in df.columns:
    plt.plot(
        periodos,
        df["Elastico"],
        label="Elástico",
        color="#D63031",
        linewidth=2.0,
        linestyle=":",
    )

  plt.title(
      "Espectros SRSS y Envolvente - Subducción", fontsize=12, fontweight="bold"
  )
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

# Única definición de los factores de escala
FACTOR_PROF = 1.6
FACTOR_SUB = 1.05

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
# 4. CÁLCULO SRSS, ENVOLVENTES Y ESCALAMIENTO#
#  ==========================================
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
   # Fórmula SRSS correcta según NTC 2023: sqrt((C1^2 + C2^2) / 2)
    df_final[base] = np.sqrt((c1**2 + c2**2)/2.0)

# 4. Calcular y agregar los Espectros Envolvente BASE
  df_final["Envolvente_ProfIntermedia"] = df_final[nombres_prof].max(axis=1)
  df_final["Envolvente_Subduccion"] = df_final[nombres_sub].max(axis=1)
  df_final["Envolvente_General"] = df_final[nombres_prof + nombres_sub].max(axis=1)

# Envolventes escaladas por los factores indicados
  df_final["EnvolventeEscalada_ProfIntermedia"] = (
      df_final["Envolvente_ProfIntermedia"] * FACTOR_PROF)
  df_final["EnvolventeEscalada_Subduccion"] = (
      df_final["Envolvente_Subduccion"] * FACTOR_SUB)


# ==========================================
  # 5. PROCESAMIENTO DEL ESPECTRO DE DISEÑO (EPU y Elástico)
  # ==========================================
  archivo_diseno = "EspectrosDis443.txt"
  if os.path.exists(archivo_diseno):
    with open(archivo_diseno, "r", encoding="utf-8", errors="ignore") as f:
      lineas_dis = f.readlines()

    t_dis, epu_dis, elastico_dis = [], [], []
    for linea in lineas_dis:
      partes = linea.split()
      if len(partes) >= 3:
        try:
          t = float(partes[0])
          epu = float(partes[1]) / g  # Convertir de gals a fracción de g
          elast = float(partes[2]) / g  # Convertir de gals a fracción de g

          t_dis.append(t)
          epu_dis.append(epu)
          elastico_dis.append(elast)
        except ValueError:
          continue  # Salta encabezados de texto si existen

    if t_dis:
      # Interpolación lineal para alinear los periodos del archivo de diseño con los periodos base
      df_final["EPU"] = np.interp(periodos, t_dis, epu_dis)
      df_final["Elastico"] = np.interp(periodos, t_dis, elastico_dis)
    else:
      print(f"Advertencia: No se pudieron extraer datos de {archivo_diseno}")
  else:
    print(f"Advertencia: No se encontró el archivo '{archivo_diseno}'")  


  # ==========================================
  # 6. GENERACIÓN DE GRÁFICAS Y EXPORTACIÓN
  # ==========================================  
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
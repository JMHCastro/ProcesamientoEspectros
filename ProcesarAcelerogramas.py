import os
import glob
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

def taper_y_filtro(x, sos, taper_pct, n_pad=0):
    """Taper (taper_pct % por extremo) -> relleno de ceros (n_pad puntos a cada lado) -> filtro de fase cero."""
    x = x * ventana_tukey_manual(len(x), alpha=2.0 * taper_pct / 100.0)
    if n_pad > 0:
        z = np.zeros(n_pad)
        x = np.concatenate([z, x, z])
    return sosfiltfilt(sos, x)

def guardar_grafica_comparativa(tiempo_orig, acc_orig, tiempo_filt, acc_filt, nombre_canal, output_folder):
    """Genera y guarda una imagen comparativa entre la señal cruda y la filtrada/recortada."""
    plt.figure(figsize=(10, 5))
    plt.plot(tiempo_orig, acc_orig, color='gray', alpha=0.6, label='Crudo (Original)', linewidth=0.8)
    plt.plot(tiempo_filt, acc_filt, color='#1f77b4', alpha=0.9, label='Filtrado y Sincronizado', linewidth=1.0)
    plt.title(f"Comparativa de Acelerograma: {nombre_canal}", fontsize=12, fontweight='bold')
    plt.xlabel("Tiempo (s)", fontsize=10)
    plt.ylabel("Aceleración ($cm/s^2$)", fontsize=10)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.legend(loc='upper right', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{nombre_canal}_comparativa.png"), dpi=300)
    plt.close()

def procesar_y_filtrar_acelerogramas():
    output_folder_crudos = "Acelerogramas"
    output_folder_filtrados = "Acelerogramas_filtrados"

    os.makedirs(output_folder_crudos, exist_ok=True)
    os.makedirs(output_folder_filtrados, exist_ok=True)

    for archivo_viejo in glob.glob(os.path.join(output_folder_filtrados, "*.*")):
        try:
            os.remove(archivo_viejo)
        except Exception:
            pass

    TAPER_PCT = 5.0
    F_MIN = 0.1
    F_MAX = 20.0
    ORDEN_FILTRO = 4
    IA_INF = 0.01
    IA_SUP = 99.90
    # Relleno de ceros a cada lado (Boore, 2005): T_pad = 1.5 * orden / f_min  -> 60 s
    T_PAD = 1.5 * ORDEN_FILTRO / F_MIN

    archivos_prof_intermedia = [f"Acelerograma_ProfIntermedia_{i}_.txt" for i in range(1, 5)]
    archivos_subduccion = [f"Acelerograma_Subduccion_{i}_.txt" for i in range(1, 9)]
    lista_archivos = archivos_prof_intermedia + archivos_subduccion

    archivos_procesados = 0

    for nombre_archivo in lista_archivos:
        if not os.path.exists(nombre_archivo):
            print(f"[AVISO] No se encontró el archivo: {nombre_archivo}. Se omitirá.")
            continue

        dt = 0.01
        skip_lines = 0

        with open(nombre_archivo, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            if "Dt=" in line:
                try:
                    dt = float(line.split("Dt=")[1].strip().split()[0])
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
                        float(parts[0]); float(parts[1])
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
            continue

        n_puntos = datos.shape[0]
        tiempo_orig = np.arange(n_puntos) * dt
        fs = 1.0 / dt
        h1 = datos[:, 0]
        h2 = datos[:, 1]
        base_name = os.path.splitext(nombre_archivo)[0]

        # 1. Guardar archivos crudos originales separados por canal
        header_crudo = "Tiempo(s)\tAceleracion(cm/s^2)"
        np.savetxt(os.path.join(output_folder_crudos, f"{base_name}_H1.txt"),
                   np.column_stack((tiempo_orig, h1)), fmt="%.6e", delimiter="\t", header=header_crudo, comments="")
        np.savetxt(os.path.join(output_folder_crudos, f"{base_name}_H2.txt"),
                   np.column_stack((tiempo_orig, h2)), fmt="%.6e", delimiter="\t", header=header_crudo, comments="")

        sos = butter_bandpass(F_MIN, F_MAX, fs, order=ORDEN_FILTRO)

        # 2. PASADA 1 (solo para ubicar la ventana): taper + filtro sobre el registro completo
        #    y Arias COMBINADA (H1 + H2) -> mismos índices de recorte para ambas componentes.
        h1_tmp = taper_y_filtro(h1, sos, TAPER_PCT)
        h2_tmp = taper_y_filtro(h2, sos, TAPER_PCT)
        g = 981.0  # cm/s^2
        ia_combinada = (np.pi / (2.0 * g)) * np.cumsum(h1_tmp ** 2 + h2_tmp ** 2) * dt
        ia_total = ia_combinada[-1]

        if ia_total > 0:
            ia_norm = ia_combinada / ia_total
            idx_ini = np.argmax(ia_norm >= (IA_INF / 100.0))
            idx_fin = np.argmax(ia_norm >= (IA_SUP / 100.0))
            if idx_fin <= idx_ini:
                idx_ini, idx_fin = 0, n_puntos - 1
        else:
            idx_ini, idx_fin = 0, n_puntos - 1

        # 3. PASADA 2 (la definitiva): recortar el registro ORIGINAL con esa ventana y DESPUÉS
        #    aplicar taper + ceros + filtro. Así la señal arranca y termina en cero, y la
        #    velocidad y el desplazamiento finales tienden a cero (sin deriva).
        n_pad = int(round(T_PAD / dt))
        h1_final = taper_y_filtro(h1[idx_ini:idx_fin + 1], sos, TAPER_PCT, n_pad)
        h2_final = taper_y_filtro(h2[idx_ini:idx_fin + 1], sos, TAPER_PCT, n_pad)
        tiempo_final = np.arange(len(h1_final)) * dt
        # tiempo para graficar alineado con el registro original
        tiempo_graf = tiempo_final + idx_ini * dt - n_pad * dt

        clean_base = base_name.rstrip('_')
        func_name_h1 = f"{clean_base}_H1_filtrado"
        func_name_h2 = f"{clean_base}_H2_filtrado"

        header_filt = "Tiempo(s)\tAceleracion_Filtrada(cm/s^2)"
        np.savetxt(os.path.join(output_folder_filtrados, f"{func_name_h1}.txt"),
                   np.column_stack((tiempo_final, h1_final)), fmt="%.6e", delimiter="\t", header=header_filt, comments="")
        np.savetxt(os.path.join(output_folder_filtrados, f"{func_name_h2}.txt"),
                   np.column_stack((tiempo_final, h2_final)), fmt="%.6e", delimiter="\t", header=header_filt, comments="")

        guardar_grafica_comparativa(tiempo_orig, h1, tiempo_graf, h1_final, func_name_h1, output_folder_filtrados)
        guardar_grafica_comparativa(tiempo_orig, h2, tiempo_graf, h2_final, func_name_h2, output_folder_filtrados)

        # Verificación rápida de deriva (velocidad y desplazamiento al final del registro)
        v = np.cumsum(h1_final) * dt
        u = np.cumsum(v) * dt
        print(f"[{base_name}] ventana {idx_ini*dt:.2f}-{idx_fin*dt:.2f} s | duración final {tiempo_final[-1]:.1f} s | "
              f"H1: v_fin={v[-1]:.4f} cm/s, u_fin={u[-1]:.4f} cm")

        archivos_procesados += 1

    print(f"\n[OK] Se procesaron y filtraron {archivos_procesados} acelerogramas correctamente.")

    try:
        generar_excel_etabs_completo(output_folder_filtrados)
    except Exception as e:
        print(f"\n[ERROR CRÍTICO al generar el Excel de ETABS]: {e}")

# ---------------------------------------------------------------------------
# generar_excel_etabs_completo: SIN CAMBIOS respecto a tu versión
# (Number Steps y Step Size se leen de los archivos, así que se ajustan solos
#  a la nueva duración)
# ---------------------------------------------------------------------------
def generar_excel_etabs_completo(folder_filtrados):
    SF_SUBDUCCION = 1.05
    SF_PROF_INTERMEDIA = 1.6
    CONSTANT_DAMPING = 0.02

    excel_path = os.path.join(folder_filtrados, "Acelerogramas_ETABS_Format.xlsx")

    if os.path.exists(excel_path):
        try:
            os.remove(excel_path)
        except Exception:
            pass

    h1_files = sorted(glob.glob(os.path.join(folder_filtrados, "*_H1_filtrado.txt")))

    if not h1_files:
        print(f"[AVISO] No se encontraron registros válidos en '{folder_filtrados}'.")
        return

    functions_rows = []
    nonlin_rows = []
    damping_rows = []

    print("\n--- Generando Excel Consolidado sin Duplicados ---")
    for h1_path in h1_files:
        basename_h1 = os.path.basename(h1_path)
        base_sismo = basename_h1.replace("_H1_filtrado.txt", "")
        h2_path = h1_path.replace("_H1_filtrado.txt", "_H2_filtrado.txt")

        if not os.path.exists(h2_path):
            continue

        df_h1 = pd.read_csv(h1_path, sep='\t')
        df_h2 = pd.read_csv(h2_path, sep='\t')

        t_h1 = df_h1.iloc[:, 0].values
        acc_h1 = df_h1.iloc[:, 1].values
        t_h2 = df_h2.iloc[:, 0].values
        acc_h2 = df_h2.iloc[:, 1].values

        num_steps = len(t_h1)
        dt = round(t_h1[1] - t_h1[0], 4) if num_steps > 1 else 0.01

        func_name_h1 = f"{base_sismo}_H1_filtrado"
        func_name_h2 = f"{base_sismo}_H2_filtrado"

        guid_h1 = str(uuid.uuid4())
        guid_h2 = str(uuid.uuid4())

        for idx, (t_val, a_val) in enumerate(zip(t_h1, acc_h1)):
            functions_rows.append({'Name': func_name_h1, 'Time sec': t_val, 'Value': a_val, 'GUID': guid_h1 if idx == 0 else ''})
        for idx, (t_val, a_val) in enumerate(zip(t_h2, acc_h2)):
            functions_rows.append({'Name': func_name_h2, 'Time sec': t_val, 'Value': a_val, 'GUID': guid_h2 if idx == 0 else ''})

        if "Subduccion" in base_sismo:
            sf = SF_SUBDUCCION
        else:
            sf = SF_PROF_INTERMEDIA
        case_name = base_sismo.replace("Acelerograma_", "")

        guid_case = str(uuid.uuid4())

        nonlin_rows.append({
            'Name': case_name, 'Initial Condition': 'Nonlinear Case', 'Nonlinear Case': 'P_Sis-NL',
            'Load Type': 'Accel', 'Load Name': 'U1', 'Function': func_name_h1,
            'Load SF': None, 'Trans Accel SF': sf, 'Rot Accel SF': None, 'Time Factor': 1,
            'Arrival Time': 0, 'Angle': 0, 'Modal Case': 'Modal-Ritz', 'Motion Type': 'Transient',
            'Number Steps': num_steps, 'Step Size': dt, 'Static Period': 0, 'Max Substep': 0,
            'Min Substep': 0, 'Force Tolerance': 0.0005, 'Energy Tolerance': 0.0005,
            'Max Iterations': 100, 'Min Iterations': 2, 'Convergence Factor': 1,
            'Design Type': 'Program Determined', 'User Design Type': None, 'GUID': guid_case, 'Notes': None
        })
        nonlin_rows.append({
            'Name': case_name, 'Initial Condition': None, 'Nonlinear Case': None,
            'Load Type': 'Accel', 'Load Name': 'U2', 'Function': func_name_h2,
            'Load SF': None, 'Trans Accel SF': sf, 'Rot Accel SF': None, 'Time Factor': 1,
            'Arrival Time': 0, 'Angle': 0, 'Modal Case': None, 'Motion Type': None,
            'Number Steps': None, 'Step Size': None, 'Static Period': None, 'Max Substep': None,
            'Min Substep': None, 'Force Tolerance': None, 'Energy Tolerance': None,
            'Max Iterations': None, 'Min Iterations': None, 'Convergence Factor': None,
            'Design Type': None, 'User Design Type': None, 'GUID': None, 'Notes': None
        })

        for mode_idx in [1, 2, 3]:
            damping_rows.append({
                'Name': case_name,
                'Consider Max Freq?': None, 'Max Frequency': None,
                'Damping Type': 'Constant' if mode_idx == 1 else None,
                'Constant Damping': CONSTANT_DAMPING if mode_idx == 1 else None,
                'Interpolated By': None, 'Interpolated Time Value': None, 'Interpolated Damping': None,
                'Proportional By': None, 'Mass Coefficient': None, 'Stiffness Coefficient': None,
                'Proportional Time Value 1': None, 'Proportional Damping 1': None,
                'Proportional Time Value 2': None, 'Proportional Damping 2': None,
                'Mode For Ratio': None,
                'Overwrite Mode': mode_idx,
                'Overwrite Damping': 0.00001
            })

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pd.DataFrame(functions_rows).to_excel(writer, sheet_name="Functions - Time History", index=False)
        pd.DataFrame(nonlin_rows).to_excel(writer, sheet_name="Load Cases - TH - Nonlin Modal", index=False)
        pd.DataFrame(damping_rows).to_excel(writer, sheet_name="Load Cases - Damping - Modal", index=False)

    print(f"\n[EXCEL OK] Archivo consolidado guardado en: '{excel_path}'")

if __name__ == "__main__":
    procesar_y_filtrar_acelerogramas()
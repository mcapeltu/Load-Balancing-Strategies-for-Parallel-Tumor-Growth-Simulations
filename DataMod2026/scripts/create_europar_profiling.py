import pandas as pd

# Rutas de entrada
evol_file = "Cuda/RESULTS/crm_tumor_evolution.csv"
kernel_file = "Cuda/RESULTS/evolution_vs_kernel.csv"

# Ruta de salida
output_file = "Cuda/RESULTS/europar_profiling.csv"

# Leer datos
evol = pd.read_csv(evol_file)
kernel = pd.read_csv(kernel_file)

# Comprobar columnas mínimas esperadas
required_evol = {"iteration", "day", "step", "tumor_cells"}
required_kernel = {"iteration", "crm_ms", "frm_ms"}

missing_evol = required_evol - set(evol.columns)
missing_kernel = required_kernel - set(kernel.columns)

if missing_evol:
    raise ValueError(f"Faltan columnas en {evol_file}: {missing_evol}")

if missing_kernel:
    raise ValueError(f"Faltan columnas en {kernel_file}: {missing_kernel}")

# Combinar por iteración
df = evol.merge(
    kernel[["iteration", "crm_ms", "frm_ms"]],
    on="iteration",
    how="inner"
)

# Quedarnos con el estado al final de cada día
# El modelo usa 24 pasos por día
df = df[df["step"] == 24].copy()

# Añadir variables útiles
df["iteration_end_of_day"] = df["iteration"]

df["frm_crm_ratio"] = df["frm_ms"] / df["crm_ms"]

df["kernel_delta_ms"] = df["frm_ms"] - df["crm_ms"]

df["kernel_mean_ms"] = (df["crm_ms"] + df["frm_ms"]) / 2.0

# Seleccionar columnas finales
profiling = df[
    [
        "day",
        "iteration_end_of_day",
        "tumor_cells",
        "crm_ms",
        "frm_ms",
        "frm_crm_ratio",
        "kernel_delta_ms",
        "kernel_mean_ms",
    ]
].sort_values("day")

# Verificaciones
if profiling["day"].duplicated().any():
    raise ValueError("Hay días duplicados en el profiling final.")

print(f"Número de días: {len(profiling)}")
print(f"Primer día: {profiling['day'].min()}")
print(f"Último día: {profiling['day'].max()}")

print("\nPrimeras filas:")
print(profiling.head())

print("\nÚltimas filas:")
print(profiling.tail())

# Guardar CSV
profiling.to_csv(output_file, index=False)

print(f"\nCreado correctamente: {output_file}")

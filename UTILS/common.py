# UTILS/common.py
from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import pandas as pd
import streamlit as st

# Los scorers (ratio, token_set_ratio) viven en rapidfuzz.fuzz.
# El módulo 'process' solo orquesta extract/extractOne con un scorer externo.
try:
    from rapidfuzz import fuzz as rf_fuzz, process as rf_process
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    import difflib
    _RAPIDFUZZ_AVAILABLE = False


# =========================
#  RUTAS Y CONSTANTES
# =========================

# --- Constantes de Dominio y Área ---
DOM_LINEAS: str = "LINEAS"
DOM_COPLES: str = "COPLES"

AREA_CODE_TO_LABEL: dict[int, str] = {1: "L-A", 2: "L-B", 3: "L-C", 4: "Coples"}
AREA_LINEAS: set[int] = {1, 2, 3} # Códigos de área que pertenecen a Líneas
AREA_COPLES: set[int] = {4} # Códigos de área que pertenecen a Coples
DOMAIN_BY_AREA_CODE: dict[int, str] = {
    1: DOM_LINEAS, 2: DOM_LINEAS, 3: DOM_LINEAS, 4: DOM_COPLES
}

# --- Aliases para compatibilidad con UI existente ---
# main.py usa estas etiquetas en el selectbox. Las mantenemos para no romperlo.
LABEL_AREA_LINEAS: str = "Líneas"
LABEL_AREA_COPLES: str = "Coples"


# --- Constantes de configuración de archivos ---
HOJAS_LINEAS: list[str] = ["L-A", "L-B", "L-C"]
COLS_CATALOGO: set[str] = {"Clave", "SubCategoria", "Descripcion", "Origen"}
ORIGEN_VALIDO: set[str] = {"Líneas", "Ambas"}

EXCLUDE_DIRS: set[str] = {".git", ".venv", "build", "dist", "__pycache__", ".mypy_cache"}
EXCEL_PATTERNS: tuple[str, ...] = ("*.xlsx", "*.xls")

CATALOGO_EXACTO = "Claves_de_Rechazo-GPT.xlsx"
CATALOGO_PATRON = "Claves*Rechazo*GPT*.xls*"


def get_paths() -> dict[str, Path]:
    """
    Devuelve rutas canónicas en base a la ubicación de este módulo (UTILS/common.py).
    Se usan nombres de carpeta en mayúsculas para coincidir con la estructura del proyecto
    y asegurar la compatibilidad con sistemas sensibles a mayúsculas/minúsculas como Linux.
    """
    utils_dir = Path(__file__).resolve().parent           # .../UTILS
    root = utils_dir.parent                               # .../Proyecto Cpk's
    paths = {
        "ROOT": root,
        "DATA": root / "DATOS",
        "LINEAS_DIR": root / "DATOS" / "LINEAS",
        "COPLES_DIR": root / "DATOS" / "COPLES",
        "REFS_DIR": root / "REFERENCIAS",
        "CATALOGO_PATH": (root / "REFERENCIAS" / CATALOGO_EXACTO),
    }
    return paths


# =========================
#  UTILIDADES GENERALES
# =========================

def normalize_text(s: str) -> str:
    """
    Normaliza un string: quita tildes, pasa a minúsculas y colapsa espacios.
    """
    if not isinstance(s, str):
        return ""
    # NFD: Descompone caracteres en base + diacrítico
    s = ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')
    s = s.lower().strip()
    s = re.sub(r'\s+', ' ', s)  # Colapsa múltiples espacios a uno solo
    return s


def parse_fecha_str(s: str) -> Optional[date]:
    """
    Convierte un string de fecha a un objeto date.
    - Limpia apóstrofo inicial (común en exportaciones de Excel).
    - Soporta formato 'dd/mm/yyyy'.
    - Devuelve solo la fecha (sin hora).
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    if s.startswith("'"):
        s = s[1:]

    try:
        # pd.to_datetime es más robusto que datetime.strptime
        dt_obj = pd.to_datetime(s, dayfirst=True, errors='coerce')
        return dt_obj.date() if pd.notna(dt_obj) else None
    except (ValueError, TypeError):
        return None


def _is_temp_or_hidden(p: Path) -> bool:
    """Verifica si un archivo es temporal o está oculto."""
    name = p.name
    return name.startswith("~$") or name.startswith(".") or name.endswith(".tmp")


def _excluded_dir(p: Path) -> bool:
    """Verifica si una ruta contiene un directorio excluido."""
    return any(part in EXCLUDE_DIRS for part in p.parts)


def listar_fuentes(
    base: Path,
    mode: Literal["dir", "file"] = "dir",
    patterns: Iterable[str] = EXCEL_PATTERNS,
    recursive: bool = False,
) -> list[Path]:
    """
    Lista archivos Excel según modo:
    - 'dir': escanea una carpeta (no subcarpetas si recursive=False)
    - 'file': devuelve el archivo si existe
    Filtra temporales (~$) y carpetas excluidas.
    """
    base = Path(base)
    if mode == "file":
        return [base] if base.is_file() and not _is_temp_or_hidden(base) else []

    if not base.is_dir():
        return []

    files: list[Path] = []
    glober = base.rglob if recursive else base.glob
    for pat in patterns:
        for p in glober(pat):
            if _is_temp_or_hidden(p) or _excluded_dir(p):
                continue
            files.append(p)
    # únicos y ordenados
    return sorted({p.resolve() for p in files})


def _hash_md5_bytes(data: bytes) -> str:
    """Calcula el hash MD5 de datos en bytes."""
    return hashlib.md5(data).hexdigest()


def _calc_md5(path: Path) -> str | None:
    """Calcula el hash MD5 de un archivo."""
    try:
        with open(path, "rb") as f:
            return _hash_md5_bytes(f.read())
    except Exception:
        return None


def _similarity(a: str, b: str, method: str = "token_set_ratio") -> float:
    """
    Calcula la similitud entre dos strings usando un método específico.
    Usa rapidfuzz si está disponible, con fallback a difflib.
    """
    if not a or not b:
        return 0.0

    if _RAPIDFUZZ_AVAILABLE:
        if method == "token_set_ratio":
            return rf_fuzz.token_set_ratio(a, b)
        # Se pueden añadir otros scorers de rapidfuzz aquí
        return rf_fuzz.ratio(a, b)  # Default a ratio simple

    # Fallback a difflib
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


# =========================
#  CATÁLOGO DE RECHAZO
# =========================

@st.cache_data(show_spinner=False)
def cargar_catalogo_rechazo() -> pd.DataFrame:
    """
    Carga el catálogo desde REFERENCIAS.
    - Intenta nombre exacto 'Claves_de_Rechazo-GPT.xlsx'
    - Fallback: patrón 'Claves*Rechazo*GPT*.xls*'
    - Valida columnas y filtra Origen ∈ {'Líneas','Ambas'}
    """
    paths = get_paths()
    ruta = paths["CATALOGO_PATH"]

    # Fallback si el exacto no existe
    if not ruta.exists():
        candidatos = list(paths["REFS_DIR"].glob(CATALOGO_PATRON))
        if candidatos:
            ruta = candidatos[0]

    if not ruta.exists():
        st.error(f"Catálogo no encontrado en REFERENCIAS: buscado '{CATALOGO_EXACTO}' "
                 f"o patrón '{CATALOGO_PATRON}'.")
        st.stop()

    try:
        cat = pd.read_excel(ruta, sheet_name=0)
    except Exception as e:
        st.error(f"No se pudo leer el catálogo '{ruta.name}': {e}")
        st.stop()

    faltantes = COLS_CATALOGO - set(map(str, cat.columns))
    if faltantes:
        st.error(f"Al catálogo le faltan columnas: {', '.join(sorted(faltantes))}")
        st.stop()

    # Normaliza
    cat["Clave"] = cat["Clave"].astype(str).str.strip()
    cat["Origen"] = cat["Origen"].astype(str).str.strip()
    cat["SubCategoria"] = cat["SubCategoria"].astype(str).str.strip()
    cat["Descripcion"] = cat["Descripcion"].astype(str).str.strip()
    cat["DescripcionNorm"] = cat["Descripcion"].apply(normalize_text)

    # Filtra por origen válido
    cat = cat[cat["Origen"].isin(ORIGEN_VALIDO)].copy()

    if cat.empty:
        st.warning("El catálogo filtrado por Origen ('Líneas','Ambas') quedó vacío.")
    return cat


@st.cache_data(show_spinner=False)
def cargar_overrides_rechazo() -> pd.DataFrame:
    """
    Carga un archivo de overrides opcional para forzar la clasificación.
    El archivo debe estar en REFERENCIAS/overrides_rechazos.csv
    Columnas: Dominio, SourceCol, DescPattern, ClaveCatalogo, Forzar, Nota
    """
    paths = get_paths()
    overrides_path = paths["REFS_DIR"] / "overrides_rechazos.csv"

    if not overrides_path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(overrides_path)
        # Validar columnas esenciales
        required_cols = {"Dominio", "Forzar"}
        if not required_cols.issubset(df.columns):
            st.warning(f"El archivo de overrides '{overrides_path.name}' no tiene las columnas requeridas: {required_cols}.")
            return pd.DataFrame()
        
        # Normalizar para facilitar la comparación
        df['Dominio'] = df['Dominio'].str.upper().str.strip()
        df['Forzar'] = df['Forzar'].str.upper().str.strip()
        return df
    except Exception as e:
        st.warning(f"No se pudo cargar el archivo de overrides '{overrides_path.name}': {e}")
        return pd.DataFrame()

# =========================
#  FUZZY MATCHING
# =========================

def match_descripcion_to_catalog(
    desc: str,
    catalog_df: pd.DataFrame,
    dominio: str,
    threshold_loose: int = 70
) -> list[dict[str, Any]]:
    """
    Busca la mejor coincidencia para una descripción en el catálogo de rechazo
    usando fuzzy matching.

    - Filtra el catálogo por `dominio` ('LINEAS', 'COPLES') o 'Ambas'.
    - Usa `rapidfuzz` si está instalado (preferido), o `difflib` como fallback.
    - Devuelve una lista de diccionarios con los mejores matches (hasta 2).
    - El índice devuelto por el motor de fuzzy matching es posicional por diseño.
    """
    if not desc or not isinstance(desc, str) or catalog_df.empty:
        return []

    # 1. Filtrar catálogo por el dominio relevante
    filtro_origen = (catalog_df["Origen"] == dominio) | (catalog_df["Origen"] == "Ambas")
    cat_filtrado_orig = catalog_df[filtro_origen]
    if cat_filtrado_orig.empty:
        return []

    # Resetear el índice para garantizar que el índice devuelto por fuzzy sea posicional
    cat_filtrado = cat_filtrado_orig.reset_index(drop=True)

    # 2. Preparar las opciones para el matching
    # Usar la columna pre-normalizada del catálogo y convertir a lista para un comportamiento predecible.
    opciones = cat_filtrado["DescripcionNorm"].tolist()
    desc_norm = normalize_text(desc)
    match_list = []

    # 3. Ejecutar fuzzy matching
    if _RAPIDFUZZ_AVAILABLE:
        # rapidfuzz.extract devuelve (choice, score, index)
        resultados = rf_process.extract(desc_norm, opciones, score_cutoff=threshold_loose, limit=2) # type: ignore
        for _match_desc_norm, score, index in resultados:
            fila = cat_filtrado.iloc[index]
            match_list.append({
                "Clave": fila["Clave"], "DescripcionCatalogo": fila["Descripcion"],
                "SubCategoria": fila["SubCategoria"], "Origen": fila["Origen"], "Score": score,
            })

    else: # Fallback a difflib
        # difflib.get_close_matches devuelve una lista de strings
        matches_str = difflib.get_close_matches(desc_norm, opciones, n=2, cutoff=threshold_loose / 100)
        for match_desc_norm in matches_str:
            try:
                index = opciones.index(match_desc_norm)
                fila = cat_filtrado.iloc[index]
                score = difflib.SequenceMatcher(None, desc_norm, match_desc_norm).ratio() * 100
                match_list.append({
                    "Clave": fila["Clave"], "DescripcionCatalogo": fila["Descripcion"],
                    "SubCategoria": fila["SubCategoria"], "Origen": fila["Origen"], "Score": score,
                })
            except ValueError:
                continue

    return match_list


# =========================
#  EXTRACCIÓN DE EVENTOS (FORMATO LARGO)
# =========================


def _detect_header_row_in_sheet(path: Path, sheet_name: str, *, scan_rows: int = 5) -> int:
    """
    Determina la fila del encabezado. Para los archivos actuales, es siempre la fila 2 (índice 1).
    La firma se mantiene para futura extensibilidad (p. ej. escanear N filas).
    """
    # Lógica simplificada: para los archivos de este proyecto, el encabezado está siempre en la fila 2.
    return 1


def _read_sheet_raw(path: Path, sheet_name: str) -> pd.DataFrame | None:
    """
    Lee una hoja de Excel completa como datos crudos, sin encabezado (header=None).
    Esto es útil para acceder a filas específicas (como la fila 1 de descripciones)
    por su índice posicional, ya que `df_raw.iat[0, j]` corresponde a la celda en la fila 1.
    """
    try:
        return pd.read_excel(path, sheet_name=sheet_name, header=None)
    except Exception:
        return None


def _get_dynamic_cols(df_data: pd.DataFrame, last_fixed_col: str = "OEE") -> list[str]:
    """
    Identifica las columnas dinámicas (claves de rechazo/TI) en un DataFrame.
    Por defecto, son todas las columnas a la derecha de la columna `last_fixed_col`.
    """
    try:
        last_fixed_col_idx = df_data.columns.get_loc(last_fixed_col)
        return df_data.columns[last_fixed_col_idx + 1:].tolist()
    except KeyError:
        # Fallback si la columna fija no existe: no se pueden determinar columnas dinámicas.
        return []


def _classify_dynamic_column(
    dyn_col_name: str,
    desc_superior: str,
    df_data_col: pd.Series,
    dominio: str,
    catalog_df: pd.DataFrame,
    overrides_df: pd.DataFrame,
    # --- Mode parameters ---
    mode: str,
    desc_threshold: float,
    desc_validation_mode: str,
    valid_keys_map: dict[str, dict],
    # --- Fuzzy mode parameters ---
    threshold_high: float,
    threshold_low: float,
    fraction_tolerance: float,
) -> dict[str, Any]:
    """
    Clasifica una única columna dinámica y devuelve un diccionario de auditoría.
    Soporta modo 'must_match' (estricto) y 'fuzzy' (legado).
    """
    audit_info = {
        "SourceCol": dyn_col_name, "desc_row0": desc_superior, "decision_final": "TI",
        "reason_code": "DEFAULT_TI", "match_method": None, "score_top1": None, "score_top2": None,
        "ClaveCatalogo": None, "DescripcionCatalogo": None, "OrigenCatalogo": None, "message": None,
    }
    TI_TOKEN_REGEX = r"\b(hr|hora|min|paro|setup|cambio|mantenimiento)\b"

    # --- MODO ESTRICTO: MUST_MATCH ---
    if mode == "must_match":
        audit_info["match_method"] = "key_match"
        norm_key = normalize_text(dyn_col_name)
        
        if norm_key not in valid_keys_map:
            audit_info["reason_code"] = "MUST_MATCH_FAIL"
            return audit_info

        key_info = valid_keys_map[norm_key]
        catalog_desc_norm = key_info["DescripcionNorm"]
        desc_superior_norm = normalize_text(desc_superior)

        score = _similarity(desc_superior_norm, catalog_desc_norm, method="token_set_ratio")
        audit_info["score_top1"] = score

        if score >= desc_threshold:
            audit_info["decision_final"] = "RECHAZO"
            audit_info["reason_code"] = "DESC_MATCH_OK"
        else:
            if desc_validation_mode == "warn":
                audit_info["decision_final"] = "RECHAZO"
                audit_info["reason_code"] = "DESC_MISMATCH_WARN"
                audit_info["message"] = f"Advertencia: La descripción '{desc_superior}' para la clave '{dyn_col_name}' tuvo un score bajo ({score:.1f}%) pero fue aceptada."
            else: # "enforce"
                audit_info["reason_code"] = "DESC_MISMATCH_ENFORCE"
                audit_info["message"] = f"Info: La columna '{dyn_col_name}' fue descartada. Su descripción '{desc_superior}' no coincide con la del catálogo (score: {score:.1f}%)."

        if audit_info["decision_final"] == "RECHAZO":
            audit_info.update({
                "ClaveCatalogo": key_info["Clave"], "DescripcionCatalogo": key_info["Descripcion"],
                "SubCategoria": key_info["SubCategoria"], "OrigenCatalogo": key_info["Origen"],
            })
        return audit_info

    # --- MODO LEGADO: FUZZY ---
    elif mode == "fuzzy":
        # 1. Overrides (máxima prioridad)
        if not overrides_df.empty:
            for _, override in overrides_df.iterrows():
                if override["Dominio"] == dominio:
                    match_sc = pd.isna(override["SourceCol"]) or override["SourceCol"] == dyn_col_name
                    match_desc = pd.isna(override["DescPattern"]) or re.search(override["DescPattern"], desc_superior, re.IGNORECASE)
                    if match_sc and match_desc:
                        audit_info["decision_final"] = override["Forzar"]
                        audit_info["reason_code"] = f"OVERRIDE_{audit_info['decision_final']}"
                        audit_info["match_method"] = "override"
                        audit_info["message"] = f"Decisión por override en columna '{dyn_col_name}'."
                        if audit_info["decision_final"] == "RECHAZO" and pd.notna(override["ClaveCatalogo"]):
                            cat_entry = catalog_df[catalog_df["Clave"] == override["ClaveCatalogo"]].iloc[0]
                            audit_info.update({
                                "ClaveCatalogo": cat_entry["Clave"], "DescripcionCatalogo": cat_entry["Descripcion"],
                                "SubCategoria": cat_entry["SubCategoria"], "OrigenCatalogo": cat_entry["Origen"], "score_top1": 100.0
                            })
                        return audit_info

        # 2. Match por Clave Exacta
        cat_exact_match = catalog_df[(catalog_df["Clave"] == dyn_col_name) & ((catalog_df["Origen"] == dominio) | (catalog_df["Origen"] == "Ambas"))]
        if not cat_exact_match.empty:
            entry = cat_exact_match.iloc[0]
            audit_info.update({
                "decision_final": "RECHAZO", "reason_code": "MATCH_CLAVE_EXACTA", "match_method": "clave_exacta",
                "ClaveCatalogo": entry["Clave"], "DescripcionCatalogo": entry["Descripcion"],
                "SubCategoria": entry["SubCategoria"], "OrigenCatalogo": entry["Origen"], "score_top1": 100.0
            })
            return audit_info

        # 3. Fuzzy Match y Zona Gris
        matches = match_descripcion_to_catalog(desc_superior, catalog_df, dominio, threshold_loose=int(threshold_low))
        audit_info["match_method"] = "fuzzy"
        if not matches:
            audit_info["reason_code"] = "NO_MATCH"
            return audit_info

        top1 = matches[0]
        top2 = matches[1] if len(matches) > 1 else None
        audit_info.update({"score_top1": top1["Score"], "score_top2": top2["Score"] if top2 else None})

        if top1["Score"] >= threshold_high:
            audit_info["decision_final"] = "RECHAZO"
            audit_info["reason_code"] = "MATCH_FUZZY_ALTO"
            audit_info.update({ "ClaveCatalogo": top1["Clave"], "DescripcionCatalogo": top1["DescripcionCatalogo"], "SubCategoria": top1["SubCategoria"], "OrigenCatalogo": top1["Origen"] })
        elif top1["Score"] > threshold_low:  # Zona Gris
            if re.search(TI_TOKEN_REGEX, normalize_text(desc_superior)):
                audit_info["reason_code"] = "ZG_TI_TOKEN"
            else:
                pzas_series_temp = pd.to_numeric(df_data_col, errors='coerce').fillna(0)
                non_zero_pzas = pzas_series_temp[pzas_series_temp > 0]
                if not non_zero_pzas.empty and (non_zero_pzas.round() - non_zero_pzas).abs().mean() <= fraction_tolerance:
                    audit_info["decision_final"] = "RECHAZO"
                    audit_info["reason_code"] = "ZG_MAYORIA_ENTEROS"
                    audit_info.update({
                        "ClaveCatalogo": top1["Clave"], "DescripcionCatalogo": top1["DescripcionCatalogo"], "SubCategoria": top1["SubCategoria"], "OrigenCatalogo": top1["Origen"]
                    })
                else:
                    audit_info["reason_code"] = "ZG_INCONCLUSO_TI"
        else:
            audit_info["reason_code"] = "MATCH_FUZZY_BAJO"
        return audit_info

    raise ValueError(f"Modo de clasificación desconocido: '{mode}'")


def _extract_rechazos_long_from_data(
    df_data: pd.DataFrame,
    df_raw: pd.DataFrame,
    dominio: str,
    catalog_df: pd.DataFrame,    
    overrides_df: pd.DataFrame,
    mode: str,
    desc_threshold: float,
    desc_validation_mode: str,
    valid_keys_map: dict[str, dict],
    threshold_high: float,
    threshold_low: float,
    enforce_integer_pzas: bool,
    fraction_tolerance: float,
    file_context: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """
    Extrae los datos de rechazo de las columnas dinámicas y los transforma a formato largo.
    - La descripción de la clave se toma SIEMPRE de la fila 1 (índice 0) del archivo raw.
    - Devuelve un DataFrame con columnas de contexto y del evento de rechazo.
    """
    messages = []
    dynamic_cols = _get_dynamic_cols(df_data)
    if not dynamic_cols or df_raw.shape[0] < 2:
        return pd.DataFrame(), messages

    all_rechazos = []
    # Mapa de nombre de columna en df_data a su índice de columna en df_raw.
    # Los encabezados de df_data están en la fila 2 (índice 1) de df_raw.
    raw_headers_map = {val: idx for idx, val in df_raw.iloc[1].items() if pd.notna(val)}

    for dyn_col_name in dynamic_cols:
        raw_col_idx = raw_headers_map.get(dyn_col_name)
        if raw_col_idx is None:
            continue

        # La descripción de la clave está SIEMPRE en la fila 1 (índice 0) de la misma columna.
        desc_superior = df_raw.iat[0, raw_col_idx]
        if pd.isna(desc_superior) or not isinstance(desc_superior, str) or not desc_superior.strip():
            continue

        audit_info = _classify_dynamic_column(
            dyn_col_name, desc_superior, df_data[dyn_col_name], dominio, catalog_df,
            overrides_df, mode, desc_threshold, desc_validation_mode, valid_keys_map,
            threshold_high, threshold_low, fraction_tolerance
        )

        if audit_info["message"]:
            messages.append(audit_info["message"])

        if audit_info["decision_final"] != "RECHAZO":
            continue

        # Si la decisión fue RECHAZO pero no se asignó un match (ej. override sin clave), buscar el mejor posible
        if audit_info["ClaveCatalogo"] is None:
            best_match_list = match_descripcion_to_catalog(desc_superior, catalog_df, dominio, threshold_loose=int(threshold_low))
            if not best_match_list: continue
            best_match = best_match_list[0]
            audit_info.update({
                "ClaveCatalogo": best_match["Clave"], "DescripcionCatalogo": best_match["DescripcionCatalogo"],
                "SubCategoria": best_match["SubCategoria"], "OrigenCatalogo": best_match["Origen"],
                "score_top1": best_match["Score"]
            })

        # --- Preparación para Melt ---
        pzas_series = pd.to_numeric(df_data[dyn_col_name], errors='coerce').fillna(0)
        if enforce_integer_pzas:
            pzas_series = pzas_series.round()

        filas_con_rechazo = pzas_series[pzas_series > 0]
        if filas_con_rechazo.empty:
            continue

        temp_df = df_data.loc[filas_con_rechazo.index].copy()

        # Añadir columnas del evento de rechazo
        temp_df["SourceCol"] = dyn_col_name
        temp_df["desc_row0"] = desc_superior
        temp_df["ClaveCatalogo"] = audit_info["ClaveCatalogo"]
        temp_df["DescripcionCatalogo"] = audit_info["DescripcionCatalogo"]
        temp_df["SubCategoria"] = audit_info["SubCategoria"]
        temp_df["OrigenCatalogo"] = audit_info["OrigenCatalogo"]
        temp_df["match_score"] = audit_info["score_top1"]
        temp_df["Pzas"] = filas_con_rechazo # Valor del evento
        temp_df["reason_code"] = audit_info["reason_code"]

        # Definir columnas a mantener para el formato largo
        columnas_base = ["Fecha", "Turno", "Maquina", "Diametro", "Libraje", "Acero", "Rosca", "Area", "TotalPiezas", "PzasRech"]
        columnas_evento = ["SourceCol", "desc_row0", "ClaveCatalogo", "DescripcionCatalogo", "SubCategoria", "OrigenCatalogo", "match_score", "Pzas", "reason_code"]
        columnas_a_mantener = [c for c in columnas_base if c in temp_df.columns] + columnas_evento
        all_rechazos.append(temp_df[columnas_a_mantener])

    df_result = pd.concat(all_rechazos, ignore_index=True) if all_rechazos else pd.DataFrame()
    return df_result, messages


# =========================
#  CONSOLIDACIÓN + MANIFIESTO
# =========================

def _leer_hoja_excel(ruta: Path, hoja: str) -> pd.DataFrame | None:
    """
    Lee una hoja específica de un archivo Excel y añade metadatos.
    Usa la fila 2 (índice 1) como encabezado, según la estructura de archivos del proyecto.
    """
    try:
        # Para este proyecto, el encabezado real está siempre en la fila 2 (índice 1).
        header_idx = _detect_header_row_in_sheet(ruta, hoja)
        df = pd.read_excel(ruta, sheet_name=hoja, header=header_idx)
        if df is None or df.empty:
            return None
        df["__archivo_origen__"] = ruta.name
        df["__hoja_origen__"] = hoja
        df["__ruta_relativa__"] = str(ruta)
        return df
    except Exception:
        return None


def _manifest_row(ruta: Path) -> dict:
    """Crea una fila inicial para el manifiesto de carga."""
    st_mtime = None
    try:
        st_mtime = ruta.stat().st_mtime
    except Exception:
        pass

    return {
        "archivo": ruta.name,
        "ruta": str(ruta),
        "modificado": st_mtime,
        "tam_bytes": ruta.stat().st_size if st_mtime else 0,
        "ok": False,
        "sheets": [],
        "filas": 0,
        "motivo_falla": None,
    }


@st.cache_data(show_spinner=False)
def cargar_lineas_con_manifiesto(recursive: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida TODAS las hojas L-A/B/C de todos los .xlsx en datos/lineas.
    Devuelve (df_unificado, manifest_df)
    """
    paths = get_paths()
    carpeta = paths["LINEAS_DIR"]

    # --- INICIO DEBUG ---
    # Estos mensajes aparecerán en los logs de Streamlit Cloud
    print("--- DEBUG: Iniciando carga para LINEAS ---")
    print(f"Ruta base del proyecto (ROOT): {paths['ROOT']}")
    print(f"Intentando acceder a la carpeta: {carpeta}")
    if not carpeta.exists():
        print(f"ERROR DE RUTA: La ruta NO EXISTE: {carpeta}")
    elif not carpeta.is_dir():
        print(f"ERROR DE RUTA: La ruta EXISTE pero NO ES UNA CARPETA: {carpeta}")
    else:
        print(f"ÉXITO DE RUTA: La ruta es una carpeta válida: {carpeta}")
    # --- FIN DEBUG ---

    fuentes = listar_fuentes(carpeta, mode="dir", recursive=recursive)

    frames: list[pd.DataFrame] = []
    manifest: list[dict] = []
    print(f"DEBUG: listar_fuentes encontró {len(fuentes)} archivo(s): {[f.name for f in fuentes]}")
    for ruta in fuentes:
        info = _manifest_row(ruta)
        sheets_ok: list[str] = []
        filas_tot = 0

        try:
            # Leer solo hojas esperadas
            for hoja in HOJAS_LINEAS:
                df = _leer_hoja_excel(ruta, hoja)
                if df is None:
                    continue
                frames.append(df)
                sheets_ok.append(hoja)
                filas_tot += len(df)

            if filas_tot > 0:
                info["ok"] = True
                info["sheets"] = sheets_ok
                info["filas"] = filas_tot
            else:
                info["motivo_falla"] = "Sin hojas L-A/B/C con filas"

        except Exception as e:
            info["motivo_falla"] = f"Error general: {e}"

        manifest.append(info)

    df_unificado = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest_df = pd.DataFrame(manifest)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(["ok", "archivo"], ascending=[False, True])

    return df_unificado, manifest_df


@st.cache_data(show_spinner=False)
def cargar_coples_con_manifiesto(recursive: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Consolida TODOS los archivos "BDD_COP_*.xlsx" en datos/coples/.
    Para cada archivo, lee su primera hoja asumiendo que el header está en la fila 2.
    Devuelve (df_unificado, manifest_df)
    """
    paths = get_paths()
    carpeta = paths["COPLES_DIR"]

    # --- INICIO DEBUG ---
    # Estos mensajes aparecerán en los logs de Streamlit Cloud
    print("--- DEBUG: Iniciando carga para COPLES ---")
    print(f"Ruta base del proyecto (ROOT): {paths['ROOT']}")
    print(f"Intentando acceder a la carpeta: {carpeta}")
    if not carpeta.exists():
        print(f"ERROR DE RUTA: La ruta NO EXISTE: {carpeta}")
    elif not carpeta.is_dir():
        print(f"ERROR DE RUTA: La ruta EXISTE pero NO ES UNA CARPETA: {carpeta}")
    else:
        print(f"ÉXITO DE RUTA: La ruta es una carpeta válida: {carpeta}")
    # --- FIN DEBUG ---

    # 1. Descubrimiento de fuentes específico para Coples
    fuentes = listar_fuentes(carpeta, mode="dir", recursive=recursive, patterns=("BDD_COP_*.xlsx",))

    frames: list[pd.DataFrame] = []
    manifest: list[dict] = []
    print(f"DEBUG: listar_fuentes encontró {len(fuentes)} archivo(s): {[f.name for f in fuentes]}")
    for ruta in fuentes:
        info = _manifest_row(ruta)
        
        try:
            # 2. Lectura por archivo (single-sheet)
            try:
                xls = pd.ExcelFile(ruta)
                hojas = list(xls.sheet_names)
                if not hojas:
                    info["motivo_falla"] = "El archivo no contiene hojas."
                    manifest.append(info)
                    continue
            except Exception as e:
                info["motivo_falla"] = f"No se pudo abrir: {e}"
                manifest.append(info)
                continue
            
            hoja_a_leer = hojas[0]
            
            # 3. Header y lectura de datos
            df = _leer_hoja_excel(ruta, hoja_a_leer)

            if df is None or df.empty:
                info["motivo_falla"] = f"La hoja '{hoja_a_leer}' está vacía o no se pudo leer."
                manifest.append(info)
                continue

            # Aplanar MultiIndex si pandas lo crea por error, y limpiar nombres
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
            else:
                df.columns = [str(col).strip() for col in df.columns]

            # 4. Columna Area y tipos
            if 'Area' not in df.columns:
                df['Area'] = 4
            else:
                df['Area'] = pd.to_numeric(df['Area'], errors='coerce').fillna(4)
            df['Area'] = df['Area'].astype(int)

            frames.append(df)
            
            # 5. Actualizar manifiesto
            info["ok"] = True
            info["sheets"] = [hoja_a_leer]
            info["filas"] = len(df)

        except Exception as e:
            info["motivo_falla"] = f"Error general: {e}"

        manifest.append(info)

    # 5. Consolidación y retorno
    df_unificado = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    manifest_df = pd.DataFrame(manifest)
    if not manifest_df.empty:
        manifest_df = manifest_df.sort_values(["ok", "archivo"], ascending=[False, True])

    return df_unificado, manifest_df


# =========================
#  FUNCIÓN DE CARGA UNIFICADA
# =========================

def _resolve_domain(area_input: Any) -> str:
    """Helper interno para determinar el dominio a partir de varios tipos de entrada."""
    if isinstance(area_input, str):
        area_str = area_input.upper().strip()
        if area_str in (DOM_LINEAS, LABEL_AREA_LINEAS.upper()):
            return DOM_LINEAS
        if area_str in (DOM_COPLES, LABEL_AREA_COPLES.upper()):
            return DOM_COPLES
        # Check if it's a label like "L-A"
        for code, label in AREA_CODE_TO_LABEL.items():
            if area_str == label.upper():
                return DOMAIN_BY_AREA_CODE[code]

    elif isinstance(area_input, int):
        if area_input in DOMAIN_BY_AREA_CODE:
            return DOMAIN_BY_AREA_CODE[area_input]

    elif isinstance(area_input, (set, list, tuple)):
        # Si se pasa un conjunto como AREA_LINEAS o AREA_COPLES
        if area_input == AREA_LINEAS:
            return DOM_LINEAS
        if area_input == AREA_COPLES:
            return DOM_COPLES

    raise ValueError(f"Entrada de área no reconocida: '{area_input}'")


def cargar_area(area: Any, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Punto de entrada único y flexible para cargar datos de un área.

    Acepta múltiples formatos para `area`:
    - Dominio (str): "LINEAS" o "COPLES".
    - Etiqueta de UI (str): "Líneas" o "Coples".
    - Etiqueta de sub-área (str): "L-A", "L-B", etc.
    - Código de área (int): 1, 2, 3 para líneas; 4 para coples.
    - Conjunto de códigos (set): AREA_LINEAS o AREA_COPLES.

    Argumentos (kwargs) se pasan a las funciones de carga subyacentes (ej. `recursive=True`).

    Devuelve una tupla (df_produccion, df_eventos) para mantener compatibilidad.
    """
    dominio = _resolve_domain(area)

    if dominio == DOM_LINEAS:
        df_consolidado, _ = cargar_lineas_con_manifiesto(**kwargs)
        return df_consolidado, df_consolidado # Devuelve el mismo df para ambos

    if dominio == DOM_COPLES:
        df_consolidado, _ = cargar_coples_con_manifiesto(**kwargs)
        return df_consolidado, df_consolidado # Devuelve el mismo df para ambos

    # Esta línea es teóricamente inalcanzable gracias a _resolve_domain
    raise RuntimeError(f"Lógica de dominio no implementada para '{dominio}'")


# Nota: Tras este cambio de firma, se recomienda limpiar la caché de la app
# para asegurar que los nuevos parámetros se apliquen correctamente.
@st.cache_data(show_spinner="Cargando y clasificando rechazos...")
def cargar_rechazos_long_area(
    *,
    dominio: str,
    mode: Literal["must_match", "fuzzy"] = "must_match",
    desc_threshold: float = 85.0,
    desc_validation_mode: Literal["enforce", "warn"] = "enforce",
    threshold_high: float = 92.0,
    threshold_low: float = 82.0,
    enforce_integer_pzas: bool = True, # Mantener True para asegurar Pzas enteras
    fraction_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """
    Recorre todos los libros/hojas de un dominio (LINEAS o COPLES) y devuelve
    un DataFrame en formato LONG con todas las piezas de rechazo clasificadas.

    - Usa la fila 2 (índice 1) como encabezado de datos.
    - Usa la fila 1 (índice 0) para obtener la descripción de la clave de rechazo.
    - Realiza un fuzzy match de la descripción contra el catálogo de rechazos.

    Args:
        dominio (str): El dominio a procesar, debe ser DOM_LINEAS o DOM_COPLES.
        mode (str): 'must_match' (estricto) o 'fuzzy' (legado).
        desc_threshold (float): Umbral para validar la descripción dentro de una clave.
        desc_validation_mode (str): 'enforce' o 'warn' si la descripción no coincide.
        threshold_high (float): Umbral superior para el modo fuzzy.
        threshold_low (float): Umbral inferior para el modo fuzzy.
        enforce_integer_pzas (bool): Si es True, aplica lógica para redondear o reclasificar columnas.
        fraction_tolerance (float): Tolerancia para considerar una fracción como imprecisión numérica.

    Returns:
        pd.DataFrame: Un DataFrame con los rechazos en formato largo. Puede ser vacío.

    Raises:
        ValueError: Si el dominio no es reconocido o no se encuentran archivos.
    """
    paths = get_paths()
    catalog_df = cargar_catalogo_rechazo()
    overrides_df = cargar_overrides_rechazo()
    
    # --- Pre-build map for strict mode ---
    valid_keys_map = {}
    if mode == "must_match":
        cat_filtrado = catalog_df[(catalog_df["Origen"] == dominio) | (catalog_df["Origen"] == "Ambas")]
        for _, row in cat_filtrado.iterrows():
            norm_key = normalize_text(row["Clave"])
            if norm_key not in valid_keys_map:
                valid_keys_map[norm_key] = row.to_dict()

    all_rechazos_long = []

    if dominio == DOM_LINEAS:
        carpeta = paths["LINEAS_DIR"]
        fuentes = listar_fuentes(carpeta, mode="dir", recursive=False)
        hojas_a_leer = HOJAS_LINEAS
    elif dominio == DOM_COPLES:
        carpeta = paths["COPLES_DIR"]
        fuentes = listar_fuentes(carpeta, mode="dir", recursive=False)
        hojas_a_leer = None  # Para coples, se leen todas las hojas del archivo
    else:
        raise ValueError(f"Dominio '{dominio}' no reconocido. Use DOM_LINEAS o DOM_COPLES.")

    if not fuentes:
        raise ValueError(f"No se encontraron archivos Excel en la carpeta '{carpeta}' para el dominio '{dominio}'.")
    messages = []

    for path in fuentes:
        sheets_in_file = hojas_a_leer
        if sheets_in_file is None:  # Caso COPLES, detectar hojas dinámicamente
            try:
                sheets_in_file = pd.ExcelFile(path).sheet_names
            except Exception:
                continue  # Saltar archivo si no se puede abrir

        for sheet in sheets_in_file:
            df_raw = _read_sheet_raw(path, sheet)
            if df_raw is None or df_raw.empty:
                continue

            header_idx = _detect_header_row_in_sheet(path, sheet)
            try:
                df_data = pd.read_excel(path, sheet_name=sheet, header=header_idx)
            except Exception:
                continue  # Saltar hoja si no se puede leer con el encabezado

            if df_data.empty:
                continue

            rech_long_parcial, partial_messages = _extract_rechazos_long_from_data(
                df_data=df_data,
                df_raw=df_raw,
                dominio=dominio,
                catalog_df=catalog_df,
                overrides_df=overrides_df,
                mode=mode,
                desc_threshold=desc_threshold,
                desc_validation_mode=desc_validation_mode,
                valid_keys_map=valid_keys_map,
                threshold_high=threshold_high,
                threshold_low=threshold_low,
                enforce_integer_pzas=enforce_integer_pzas,
                fraction_tolerance=fraction_tolerance,
                file_context={"path": path, "sheet": sheet}
            )
            messages.extend(partial_messages)

            if not rech_long_parcial.empty:
                rech_long_parcial["Archivo"] = path.name
                rech_long_parcial["Hoja"] = sheet
                all_rechazos_long.append(rech_long_parcial)

    # 3. Consolidación y aplicación de contrato de salida
    if not all_rechazos_long:
        schema_cols = [
            "Fecha", "Turno", "Maquina", "Area", "AreaLabel", "Dominio", "Diametro",
            "Libraje", "Acero", "Rosca", "TotalPiezas", "PzasRech",
            "ClaveCatalogo", "DescripcionCatalogo", "SubCategoria", "OrigenCatalogo", "reason_code",
            "match_score", "desc_row0", "SourceCol", "Pzas", "Archivo", "Hoja"
        ]
        empty_df = pd.DataFrame(columns=schema_cols)
        empty_df.attrs["rechazos_long_messages"] = messages
        return empty_df

    final_df = pd.concat(all_rechazos_long, ignore_index=True)
    final_df["Dominio"] = dominio

    # 4. Estandarización final del esquema antes de devolver
    if "PzasRech" not in final_df.columns:
        final_df["PzasRech"] = 0

    final_df['ClaveCatalogo'] = final_df['ClaveCatalogo'].astype(str).str.strip()
    if 'SubCategoria' not in final_df.columns:
        final_df['SubCategoria'] = 'Sin subcategoría'
    else:
        final_df['SubCategoria'] = final_df['SubCategoria'].fillna('Sin subcategoría')

    final_df['AreaLabel'] = final_df['Area'].map(AREA_CODE_TO_LABEL).fillna('Desconocido')

    if "Pzas" in final_df.columns:
        final_df["Pzas"] = final_df["Pzas"].clip(lower=0)
        final_df["Pzas"] = final_df["Pzas"].astype(pd.Int64Dtype())

    final_df.attrs["rechazos_long_messages"] = messages
    return final_df

# =========================
#  AUDITORÍA DE MAPEO
# =========================

@st.cache_data(show_spinner="Generando auditoría de mapeo...")
def get_mapping_audit(
    dominio: str,
    threshold_high: float,
    threshold_low: float,
    mode: str = "must_match",
    desc_threshold: float = 85.0,
    desc_validation_mode: str = "enforce",
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con la auditoría detallada del mapeo de columnas dinámicas.
    No duplica la lógica de carga, sino que reutiliza el clasificador de columnas.
    """
    paths = get_paths()
    catalog_df = cargar_catalogo_rechazo()
    overrides_df = cargar_overrides_rechazo()

    # --- Pre-build map for strict mode ---
    valid_keys_map = {}
    if mode == "must_match":
        cat_filtrado = catalog_df[(catalog_df["Origen"] == dominio) | (catalog_df["Origen"] == "Ambas")]
        for _, row in cat_filtrado.iterrows():
            norm_key = normalize_text(row["Clave"])
            if norm_key not in valid_keys_map:
                valid_keys_map[norm_key] = row.to_dict()

    audit_records = []

    if dominio == DOM_LINEAS:
        carpeta = paths["LINEAS_DIR"]
        fuentes = listar_fuentes(carpeta, mode="dir", recursive=False)
        hojas_a_leer = HOJAS_LINEAS
    elif dominio == DOM_COPLES:
        carpeta = paths["COPLES_DIR"]
        fuentes = listar_fuentes(carpeta, mode="dir", recursive=False)
        hojas_a_leer = None
    else:
        return pd.DataFrame()

    for path in fuentes:
        sheets_in_file = hojas_a_leer or (pd.ExcelFile(path).sheet_names if path.exists() else [])
        for sheet in sheets_in_file:
            df_raw = _read_sheet_raw(path, sheet)
            if df_raw is None or df_raw.empty: continue

            header_idx = _detect_header_row_in_sheet(path, sheet)
            try:
                df_data = pd.read_excel(path, sheet_name=sheet, header=header_idx)
                if df_data.empty: continue
            except Exception: continue

            for dyn_col_name in _get_dynamic_cols(df_data):
                raw_col_idx = {v: k for k, v in df_raw.iloc[1].items()}.get(dyn_col_name)
                if raw_col_idx is None: continue
                desc_superior = df_raw.iat[0, raw_col_idx]
                audit_result = _classify_dynamic_column(
                    dyn_col_name, desc_superior, df_data[dyn_col_name], dominio, catalog_df,
                    overrides_df, mode, desc_threshold, desc_validation_mode, valid_keys_map,
                    threshold_high, threshold_low, 1e-6
                )
                audit_result.update({"Archivo": path.name, "Hoja": sheet})
                audit_records.append(audit_result)

    return pd.DataFrame(audit_records)

# =========================
#  HELPERS UI (opcional)
# =========================

def get_messages_from_df(df: pd.DataFrame) -> list[str]:
    """
    Extrae los mensajes de diagnóstico adjuntos a un DataFrame.
    Es una función no cacheada para ser llamada desde la UI.
    """
    return df.attrs.get("rechazos_long_messages", [])

def render_manifest(manifest_df: pd.DataFrame, title: str = "Manifiesto de archivos") -> None:
    """
    Renderiza un DataFrame de manifiesto de archivos en Streamlit.
    """
    if manifest_df is None or manifest_df.empty:
        st.info("No se registraron archivos.")
        return
    st.subheader(title)
    cols = ["archivo", "ok", "sheets", "filas", "modificado", "tam_bytes"]
    st.dataframe(manifest_df[cols], use_container_width=True, height=320)
    excl = manifest_df.loc[~manifest_df["ok"], ["archivo", "motivo_falla"]]
    if not excl.empty:
        st.caption("Archivos detectados pero no incorporados:")
        st.dataframe(excl, use_container_width=True, height=200)


# =========================
#  API PÚBLICA DEL MÓDULO
# =========================

__all__ = [
    # Constantes de Dominio y Área
    "DOM_LINEAS", "DOM_COPLES", "AREA_CODE_TO_LABEL", "AREA_LINEAS", "AREA_COPLES",
    "DOMAIN_BY_AREA_CODE",
    # Funciones de Carga de Datos
    "cargar_area",
    "cargar_catalogo_rechazo",
    "cargar_overrides_rechazo",
    "get_mapping_audit",
    "cargar_rechazos_long_area",
    "cargar_lineas_con_manifiesto",
    "cargar_coples_con_manifiesto",
    # Funciones de Utilidad
    "get_paths",
    "listar_fuentes",
    "normalize_text",
    "parse_fecha_str",
    "match_descripcion_to_catalog",
    # Funciones de UI
    "get_messages_from_df",
    "render_manifest",
    # Constantes de compatibilidad (etiquetas)
    "LABEL_AREA_LINEAS", "LABEL_AREA_COPLES",
]


# =========================
#  BLOQUE DE PRUEBA
# =========================

if __name__ == "__main__":
    print("--- Probando Constantes ---")
    print(f"DOM_LINEAS: {DOM_LINEAS}")
    print(f"AREA_LINEAS (códigos): {AREA_LINEAS}")
    print(f"AREA_CODE_TO_LABEL: {AREA_CODE_TO_LABEL}")
    print(f"DOMAIN_BY_AREA_CODE: {DOMAIN_BY_AREA_CODE}")
    print(f"LABEL_AREA_LINEAS: {LABEL_AREA_LINEAS}")
    print(f"LABEL_AREA_COPLES: {LABEL_AREA_COPLES}")
    print("\n--- Probando Utilidades ---")
    test_fecha_str = "'16/04/2025"
    parsed_date = parse_fecha_str(test_fecha_str)
    print(f"parse_fecha_str('{test_fecha_str}') -> {parsed_date} (Tipo: {type(parsed_date)})")

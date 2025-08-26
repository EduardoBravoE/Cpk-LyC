# from __future__ import annotations
# utils/insights.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Any, Dict, List, Literal, Optional, Union
from io import BytesIO
from datetime import date

# Reutilizar utilidades de common.py
from UTILS.common import (
    normalize_text,
    parse_fecha_str,
    AREA_CODE_TO_LABEL,
    DOMAIN_BY_AREA_CODE,
    DOM_LINEAS, DOM_COPLES, # Para filtros
    get_mapping_audit,
    cargar_rechazos_long_area,
)

def compute_basic_kpis(df, pieza_col="Piezas", clave_col="Clave"):
    if df is None or df.empty:
        return {"produccion_total": 0, "rechazos_total": 0, "tasa_rechazo": 0.0}
    prod = df.get(pieza_col, pd.Series([0]*len(df))).fillna(0).sum()
    # Rechazos: asume filas donde 'EsRechazoValido' True o por lógica de negocio
    rech = df.get("EsRechazoValido", pd.Series(False)).astype(bool)
    piezas_rech = df.loc[rech, pieza_col].fillna(0).sum() if pieza_col in df.columns else 0
    tasa = (piezas_rech / prod) if prod > 0 else 0.0
    return {
        "produccion_total": int(prod),
        "rechazos_total": int(piezas_rech),
        "tasa_rechazo": float(round(tasa, 4)),
    }

def compute_cpk_grouped(df, group_cols, mean_col, lsl_col, usl_col):
    """
    Requiere que df tenga columnas de especificación: LSL/USL por 'group'.
    Retorna DataFrame con Cp y Cpk.
    """
    if df.empty:
        return pd.DataFrame()

    def _cpk(grp):
        mu = grp[mean_col].astype(float).mean()
        sigma = grp[mean_col].astype(float).std(ddof=1)
        lsl = grp[lsl_col].astype(float).iloc[0]
        usl = grp[usl_col].astype(float).iloc[0]
        if sigma == 0 or pd.isna(sigma):
            return pd.Series({"mu": mu, "sigma": sigma, "Cp": np.nan, "Cpk": np.nan})
        cp = (usl - lsl) / (6 * sigma)
        cpk = min((usl - mu) / (3 * sigma), (mu - lsl) / (3 * sigma))
        return pd.Series({"mu": mu, "sigma": sigma, "Cp": cp, "Cpk": cpk})

    out = df.groupby(group_cols).apply(_cpk).reset_index()
    return out

def generate_insights_text(kpis, top_prod_df=None, worst_cpk_df=None, threshold_cpk=1.33):
    prod = kpis["produccion_total"]
    tasa = kpis["tasa_rechazo"] * 100
    txt = []
    txt.append(f"📌 Producción total: **{prod:,}** piezas · Tasa de rechazo: **{tasa:.2f}%**.")
    if worst_cpk_df is not None and not worst_cpk_df.empty:
        bad = worst_cpk_df[worst_cpk_df["Cpk"] < threshold_cpk]
        if not bad.empty:
            n = len(bad)
            txt.append(f"⚠️ Se detectaron **{n}** combinaciones con **Cpk < {threshold_cpk}**. Prioriza estas para acciones correctivas.")
    if top_prod_df is not None and not top_prod_df.empty:
        top = top_prod_df.iloc[0]
        txt.append(f"🏭 El producto más crítico por volumen es **{top.get('Producto','(N/A)')}** con **{int(top.get('Piezas',0)):,}** piezas.")
    txt.append("💡 Recomendación: revisa causas raíz de rechazos dominantes y verifica calibraciones de las máquinas con Cpk bajo.")
    return "\n\n".join(txt)


# =============================================================================
#  HELPERS INTERNOS PARA RENOMBRADO DE COLUMNAS
# =============================================================================

# Constante a nivel de módulo para columnas esenciales
_ESSENTIAL_COLS = [
    "Fecha", "Turno", "Maquina", "Diametro", "Libraje", "Acero", "Rosca",
    "TotalPiezas", "PzasRech", "Area"
]

def diagnose_columns(df: pd.DataFrame) -> dict[str, Any]:
    """
    Analiza las columnas de un DataFrame crudo y diagnostica el proceso de renombrado.

    Devuelve un diccionario con:
    - original_cols: Lista de columnas originales.
    - normalized_cols: Lista de columnas normalizadas.
    - mapping_applied: Diccionario del renombrado que se aplicaría.
    - essential_missing_after_mapping: Columnas esenciales que faltarían.
    - present_after_mapping: Columnas esenciales que estarían presentes.
    """
    if df.empty:
        return {
            "original_cols": [],
            "normalized_cols": [],
            "mapping_applied": {},
            "essential_missing_after_mapping": _ESSENTIAL_COLS,
            "present_after_mapping": [],
        }

    original_cols = df.columns.tolist()
    normalized_cols = [_normalize_colname(c) for c in original_cols]
    df_renamed, mapping_applied = _auto_rename_columns(df)
    renamed_cols_set = set(df_renamed.columns)
    missing = [col for col in _ESSENTIAL_COLS if col not in renamed_cols_set]
    present = [col for col in _ESSENTIAL_COLS if col in renamed_cols_set]

    return {
        "original_cols": original_cols,
        "normalized_cols": normalized_cols,
        "mapping_applied": mapping_applied,
        "essential_missing_after_mapping": missing,
        "present_after_mapping": present,
    }

def _coerce_fecha_series(s: pd.Series) -> pd.Series:
    """
    Convierte una serie a datetime64[ns] de forma robusta.
    - Maneja strings, apóstrofos, números de serie de Excel y formatos mixtos.
    """
    # Trabajar sobre una copia para no modificar la serie original en su lugar
    s_work = s.copy()

    # 1. Manejar valores numéricos (fechas de serie de Excel)
    numerics = pd.to_numeric(s_work, errors='coerce')
    is_numeric = numerics.notna()
    if is_numeric.any():
        s_work.loc[is_numeric] = pd.to_datetime(
            numerics[is_numeric], unit='D', origin='1899-12-30', errors='coerce'
        )

    # 2. Manejar valores de texto (los que no eran numéricos)
    is_str_like = ~is_numeric
    if is_str_like.any():
        s_cleaned = s_work[is_str_like].astype(str).str.strip().str.lstrip("'")
        
        # Intentar formato estricto primero
        parsed_strict = pd.to_datetime(s_cleaned, format="%d/%m/%Y", errors="coerce")
        
        # Para los que fallaron, intentar formato laxo
        failed_strict_mask = parsed_strict.isna()
        if failed_strict_mask.any():
            parsed_strict.loc[failed_strict_mask] = pd.to_datetime(
                s_cleaned[failed_strict_mask], dayfirst=True, errors='coerce'
            )
        s_work.loc[is_str_like] = parsed_strict

    # 3. Forzar el tipo final a datetime64[ns]
    return pd.to_datetime(s_work, errors='coerce')

def _normalize_colname(raw: str) -> str:
    """Normaliza un nombre de columna para comparación: sin tildes, minúsculas, sin espacios/guiones."""
    if not isinstance(raw, str):
        return ""
    # Usar la base de common.normalize_text (quita tildes, baja a minúsculas, colapsa espacios)
    normalized = normalize_text(raw)
    # Quitar todos los espacios, guiones y guiones bajos para una comparación robusta
    normalized = re.sub(r'[\s_-]+', '', normalized)
    return normalized


def _build_synonym_map() -> Dict[str, str]:
    """Crea un mapa de sinónimos de columnas (normalizadas) a nombres canónicos."""
    return {
        "fecha": "Fecha",
        "turno": "Turno",
        "maquina": "Maquina",
        "maq": "Maquina",
        "diametro": "Diametro",
        "diam": "Diametro",
        "libraje": "Libraje",
        "acero": "Acero",
        "rosca": "Rosca",
        "totalpiezas": "TotalPiezas",
        "pzasrech": "PzasRech",
        "piezasrech": "PzasRech",
        "rechazo": "PzasRech",
        "rechazadas": "PzasRech",
        "pzasok": "PzasOK",
        "piezasok": "PzasOK",
        "ok": "PzasOK",
        "buenas": "PzasOK",
        "area": "Area",
    }


def _auto_rename_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Intenta renombrar columnas a un formato canónico usando un mapa de sinónimos."""
    synonym_map = _build_synonym_map()
    rename_mapping = {}
    for col in df.columns:
        normalized_col = _normalize_colname(col)
        if normalized_col in synonym_map:
            canonical_name = synonym_map[normalized_col]
            # Solo renombrar si el nombre actual no es ya el canónico
            if col != canonical_name:
                rename_mapping[col] = canonical_name

    df_renamed = df.rename(columns=rename_mapping)
    return df_renamed, rename_mapping


def ensure_catalog_description(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Garantiza la presencia de una columna de descripción estandarizada ('DescripcionCatalogo').

    Busca alias comunes (case-insensitive), renombra al estándar y, si no encuentra
    ninguno, crea la columna con un valor por defecto. Es idempotente.

    Devuelve:
        - El DataFrame con la columna estandarizada.
        - El nombre de la columna estandarizada ('DescripcionCatalogo').
    """
    df_copy = df.copy()
    canonical_name = 'DescripcionCatalogo'
    
    # Lista de alias en orden de prioridad
    aliases = ['DescripcionCatalogo', 'DescCatalogo', 'Descripción', 'Descripcion', 'Descripcion_Catalogo', 'Desc_Catalogo']
    
    # Mapeo de columnas lower-case a su nombre original para búsqueda case-insensitive
    lower_to_original_map = {str(c).lower(): c for c in df_copy.columns}

    found_col_original_name = None
    for alias in aliases:
        if alias.lower() in lower_to_original_map:
            found_col_original_name = lower_to_original_map[alias.lower()]
            break

    if found_col_original_name and found_col_original_name != canonical_name:
        df_copy.rename(columns={found_col_original_name: canonical_name}, inplace=True)
    elif not found_col_original_name and canonical_name not in df_copy.columns:
        df_copy[canonical_name] = "—"
        
    return df_copy, canonical_name

# =============================================================================
#  NUEVAS FUNCIONES DE ANÁLISIS
# =============================================================================

def prepare_df_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estandariza y enriquece el DataFrame de producción para análisis.

    - Estandariza columnas clave: Fecha (date sin hora), Turno (str), Maquina (str),
      Diametro, Libraje, Acero, Rosca, TotalPiezas (num), PzasRech (num), Area (int).
    - Crea columnas derivadas:
      - AreaLabel (map desde AREA_CODE_TO_LABEL)
      - Dominio (map desde DOMAIN_BY_AREA_CODE -> "LINEAS" o "COPLES")
      - ComboProducto = concat(normalizados: Diametro|Libraje|Acero|Rosca) para agrupaciones.
    - No descarta filas con TotalPiezas = 0; conserva esas filas.

    Args:
        df (pd.DataFrame): DataFrame de producción crudo.

    Returns:
        pd.DataFrame: DataFrame preparado para análisis.

    Raises:
        ValueError: Si faltan columnas esenciales.
    """
    if df.empty:
        return pd.DataFrame()

    # Renombrado automático de columnas para robustez
    df_renamed, mapping_applied = _auto_rename_columns(df)

    missing_cols = [col for col in _ESSENTIAL_COLS if col not in df_renamed.columns]
    if missing_cols:
        error_msg = f"Columnas esenciales faltantes para el análisis: {', '.join(missing_cols)}"
        if mapping_applied:
            # Mostrar un resumen del renombrado para ayudar a depurar
            mapping_summary = "\n".join([f"  - '{k}' -> '{v}'" for k, v in list(mapping_applied.items())[:10]])
            error_msg += f"\n\nSe aplicó el siguiente renombrado automático (parcial):\n{mapping_summary}"
        raise ValueError(error_msg)

    df_copy = df_renamed.copy()

    # 1. Estandarizar tipos y normalizar texto
    df_copy["Fecha"] = _coerce_fecha_series(df_copy["Fecha"])
    # Normalizar columnas de texto, que ya son strings por la lógica de renombrado
    # y la carga de pandas, pero `astype(str)` es una garantía extra.
    df_copy["Turno"] = df_copy["Turno"].astype(str).apply(normalize_text)
    df_copy["Maquina"] = df_copy["Maquina"].astype(str).apply(normalize_text)
    df_copy["Diametro"] = df_copy["Diametro"].astype(str).apply(normalize_text)
    df_copy["Libraje"] = df_copy["Libraje"].astype(str).apply(normalize_text)
    df_copy["Acero"] = df_copy["Acero"].astype(str).apply(normalize_text)
    df_copy["Rosca"] = df_copy["Rosca"].astype(str).apply(normalize_text)
    df_copy["TotalPiezas"] = pd.to_numeric(df_copy["TotalPiezas"], errors='coerce').fillna(0)
    df_copy["PzasRech"] = pd.to_numeric(df_copy["PzasRech"], errors='coerce').fillna(0)
    # Manejar PzasOK si existe después del renombrado
    if "PzasOK" in df_copy.columns:
        df_copy["PzasOK"] = pd.to_numeric(df_copy["PzasOK"], errors='coerce').fillna(0)
    df_copy["Area"] = pd.to_numeric(df_copy["Area"], errors='coerce').fillna(-1).astype(int)

    # 2. Crear columnas derivadas
    df_copy["AreaLabel"] = df_copy["Area"].map(AREA_CODE_TO_LABEL).fillna("Desconocido")
    df_copy["Dominio"] = df_copy["Area"].map(DOMAIN_BY_AREA_CODE).fillna("Desconocido")

    # ComboProducto: concatenar normalizados
    df_copy["ComboProducto"] = df_copy["Diametro"] + "|" + \
                               df_copy["Libraje"] + "|" + \
                               df_copy["Acero"] + "|" + \
                               df_copy["Rosca"]

    return df_copy


def apply_filters(
    df: pd.DataFrame,
    *,
    diametros: Optional[List[str]] = None,
    librajes: Optional[List[str]] = None,
    aceros: Optional[List[str]] = None,
    roscas: Optional[List[str]] = None,
    fechas: Optional[tuple[date, date]] = None,
    turnos: Optional[List[str]] = None,
    maquinas: Optional[List[str]] = None,
    dominio: Optional[Union[str, List[str]]] = None,
    areacodes: Optional[Union[int, Set[int]]] = None
) -> pd.DataFrame:
    """
    Aplica filtros a un DataFrame preparado para análisis.

    Args:
        df (pd.DataFrame): DataFrame de producción preparado (de `prepare_df_for_analysis`).
        diametros (Optional[List[str]]): Lista de diámetros a incluir.
        librajes (Optional[List[str]]): Lista de librajes a incluir.
        aceros (Optional[List[str]]): Lista de aceros a incluir.
        roscas (Optional[List[str]]): Lista de roscas a incluir.
        fechas (Optional[tuple[date, date]]): Tupla (fecha_inicio, fecha_fin) para filtrar.
        turnos (Optional[List[str]]): Lista de turnos a incluir.
        maquinas (Optional[List[str]]): Lista de máquinas a incluir.
        dominio (Optional[Union[str, List[str]]]): Dominio(s) ("LINEAS", "COPLES") a incluir.
        areacodes (Optional[Union[int, Set[int]]]): Código(s) de área (1, 2, 3, 4) a incluir.

    Returns:
        pd.DataFrame: DataFrame filtrado.
    """
    filtered_df = df.copy()

    # Validar columnas necesarias para los filtros
    required_filter_cols = {
        "Diametro", "Libraje", "Acero", "Rosca", "Fecha", "Turno", "Maquina",
        "Dominio", "Area"
    }
    for col in required_filter_cols:
        if col not in filtered_df.columns:
            raise ValueError(f"Columna '{col}' necesaria para el filtro no encontrada en el DataFrame. "
                             "Asegúrate de que el DataFrame fue preparado con `prepare_df_for_analysis`.")

    if diametros:
        filtered_df = filtered_df[filtered_df["Diametro"].isin([normalize_text(d) for d in diametros])]
    if librajes:
        filtered_df = filtered_df[filtered_df["Libraje"].isin([normalize_text(l) for l in librajes])]
    if aceros:
        filtered_df = filtered_df[filtered_df["Acero"].isin([normalize_text(a) for a in aceros])]
    if roscas:
        filtered_df = filtered_df[filtered_df["Rosca"].isin([normalize_text(r) for r in roscas])]
    if fechas:
        start_date, end_date = fechas
        # Compara solo la parte de la fecha de la columna datetime de forma inclusiva
        filtered_df = filtered_df[filtered_df["Fecha"].dt.date.between(start_date, end_date, inclusive="both")]
    if turnos:
        filtered_df = filtered_df[filtered_df["Turno"].isin([normalize_text(t) for t in turnos])]
    if maquinas:
        filtered_df = filtered_df[filtered_df["Maquina"].isin([normalize_text(m) for m in maquinas])]

    if dominio:
        if isinstance(dominio, str):
            dominio = [dominio.upper()]
        else:
            dominio = [d.upper() for d in dominio]
        filtered_df = filtered_df[filtered_df["Dominio"].isin(dominio)]

    if areacodes:
        if isinstance(areacodes, int):
            areacodes = {areacodes}
        filtered_df = filtered_df[filtered_df["Area"].isin(areacodes)]

    return filtered_df


def apply_filters_long(
    df_long: pd.DataFrame,
    *,
    fechas: Optional[tuple[date, date]] = None,
    maquinas: Optional[List[str]] = None,
    turnos: Optional[List[str]] = None,
    diametros: Optional[List[str]] = None,
    librajes: Optional[List[str]] = None,
    aceros: Optional[List[str]] = None,
    roscas: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Aplica filtros a un DataFrame de rechazos en formato largo.
    """
    if df_long.empty:
        return df_long

    filtered_df = df_long.copy()

    # Normalizar columnas de texto para la comparación
    for col in ["Maquina", "Turno", "Diametro", "Libraje", "Acero", "Rosca"]:
        if col in filtered_df.columns:
            filtered_df[col] = filtered_df[col].astype(str).apply(normalize_text)

    if fechas:
        start_date, end_date = fechas
        s_fecha = pd.to_datetime(filtered_df["Fecha"], errors="coerce")
        filtered_df = filtered_df[s_fecha.dt.date.between(start_date, end_date, inclusive="both")]
    if maquinas:
        filtered_df = filtered_df[filtered_df["Maquina"].isin([normalize_text(m) for m in maquinas])]
    if turnos:
        filtered_df = filtered_df[filtered_df["Turno"].isin([normalize_text(t) for t in turnos])]
    if diametros:
        filtered_df = filtered_df[filtered_df["Diametro"].isin([normalize_text(d) for d in diametros])]
    if librajes:
        filtered_df = filtered_df[filtered_df["Libraje"].isin([normalize_text(l) for l in librajes])]
    if aceros:
        filtered_df = filtered_df[filtered_df["Acero"].isin([normalize_text(a) for a in aceros])]
    if roscas:
        filtered_df = filtered_df[filtered_df["Rosca"].isin([normalize_text(r) for r in roscas])]

    return filtered_df


def _normalize_rechazos_long_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura que el DataFrame de rechazos en formato largo tenga un esquema consistente.
    Renombra columnas desde alias conocidos y crea columnas esenciales faltantes
    con valores por defecto para evitar KeyErrors.
    """
    if df.empty:
        return df

    df_copy = df.copy()

    # Mapa de sinónimos: {alias: nombre_canónico}
    synonym_map = {
        'Clave': 'ClaveCatalogo', 'Clave_Catalogo': 'ClaveCatalogo', 'clavecatalogo': 'ClaveCatalogo',
        'Descripcion': 'DescripcionCatalogo', 'Descripción': 'DescripcionCatalogo', 'Descripcion_Catalogo': 'DescripcionCatalogo',
        'SubCategoría': 'SubCategoria', 'Subcategory': 'SubCategoria', 'Sub_Categoria': 'SubCategoria', 'subcategoria': 'SubCategoria',
    }

    # Renombrar columnas existentes basadas en sinónimos
    rename_dict = {col: synonym_map[col] for col in df_copy.columns if col in synonym_map}
    if rename_dict:
        df_copy.rename(columns=rename_dict, inplace=True)

    # Garantizar la existencia de columnas esenciales para la agrupación
    if 'ClaveCatalogo' not in df_copy.columns:
        df_copy['ClaveCatalogo'] = 'Sin Clave'
    if 'DescripcionCatalogo' not in df_copy.columns:
        df_copy['DescripcionCatalogo'] = 'Sin Descripción'
    if 'SubCategoria' not in df_copy.columns:
        df_copy['SubCategoria'] = 'Sin subcategoría'
    if 'Pzas' not in df_copy.columns:
        df_copy['Pzas'] = 0

    # Garantizar tipos de datos correctos
    for col in ['ClaveCatalogo', 'DescripcionCatalogo', 'SubCategoria', 'Maquina', 'Turno']:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(str).str.strip().fillna('')

    df_copy['Pzas'] = pd.to_numeric(df_copy['Pzas'], errors='coerce').fillna(0).astype('Int64')

    return df_copy


def compute_top_claves_rechazo(
    df_rech_long: pd.DataFrame,
    *,
    top_n: int = 10,
    desglose: str = "Global"
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula el top N de claves de rechazo, con opción de desglose.
    Es tolerante a la ausencia de columnas como 'SubCategoria', 'Maquina' o 'Turno'.
    """
    if df_rech_long.empty:
        return pd.DataFrame(), {"total_pzas": 0, "top_n": top_n, "desglose": desglose}

    # 1. Normalizar el esquema para garantizar la existencia de columnas base
    df_norm = _normalize_rechazos_long_schema(df_rech_long)
    # Garantizar la columna de descripción estandarizada
    df_norm, desc_col_name = ensure_catalog_description(df_norm)

    # 2. Construir dinámicamente las columnas de agrupación
    group_cols = ["ClaveCatalogo", desc_col_name, "SubCategoria"]
    if desglose == "Máquina" and "Maquina" in df_norm.columns:
        group_cols.append("Maquina")
    elif desglose == "Turno" and "Turno" in df_norm.columns:
        group_cols.append("Turno")

    # 3. Realizar la agregación
    df_top = (
        df_norm.groupby(group_cols, as_index=False, dropna=False)
        .agg(Pzas=("Pzas", "sum"))
        .sort_values("Pzas", ascending=False)
        .head(top_n)
    )

    meta = {
        "total_pzas": df_norm["Pzas"].sum(),
        "top_n": top_n,
        "desglose": desglose,
    }
    return df_top, meta


def build_top_claves_rechazo_figure(df_top: pd.DataFrame, desglose: str) -> go.Figure:
    """
    Construye una figura Plotly para el top de claves de rechazo usando plotly.express.
    Es tolerante a la ausencia de la columna de desglose para colorear y de descripción.
    """
    if df_top.empty:
        fig = go.Figure()
        fig.update_layout(title="Top Claves de Rechazo", annotations=[dict(text="No hay datos para mostrar", xref="paper", yref="paper", showarrow=False, font=dict(size=20))])
        return fig

    # 1. Estandarización previa del DataFrame
    df_top_std, desc_col = ensure_catalog_description(df_top)
    if 'ClaveCatalogo' not in df_top_std.columns:
        df_top_std['ClaveCatalogo'] = 'N/A'
    if 'Pzas' not in df_top_std.columns:
        df_top_std['Pzas'] = 0
    df_top_std['Pzas'] = pd.to_numeric(df_top_std['Pzas'], errors='coerce').fillna(0)

    # Crear 'EtiquetaClave' de forma vectorizada
    df_top_std['EtiquetaClave'] = df_top_std['ClaveCatalogo'].astype(str)
    non_empty_desc_mask = (df_top_std[desc_col].notna()) & (df_top_std[desc_col].astype(str).str.strip().ne('')) & (df_top_std[desc_col].astype(str).str.strip().ne('—'))
    df_top_std.loc[non_empty_desc_mask, 'EtiquetaClave'] = df_top_std['ClaveCatalogo'].astype(str) + " - " + df_top_std[desc_col].astype(str)

    # 2. Determinar la columna de color y degradar si es necesario
    color_col = None
    title_desglose = desglose
    if desglose == "Máquina":
        if "Maquina" in df_top_std.columns:
            color_col = "Maquina"
        else:
            title_desglose = "Global (Máquina no disponible)"
            if "SubCategoria" in df_top_std.columns:
                color_col = "SubCategoria"
    elif desglose == "Turno":
        if "Turno" in df_top_std.columns:
            color_col = "Turno"
        else:
            title_desglose = "Global (Turno no disponible)"
            if "SubCategoria" in df_top_std.columns:
                color_col = "SubCategoria"
    else:  # Global
        if "SubCategoria" in df_top_std.columns:
            color_col = "SubCategoria"

    # Ordenar el DataFrame para que la barra más grande quede arriba
    df_top_std = df_top_std.sort_values("Pzas", ascending=True)

    # 3. Usar plotly.express.bar
    fig = px.bar(
        df_top_std,
        x="Pzas",
        y="EtiquetaClave",
        orientation='h',
        color=color_col,
        text="Pzas",
        title=f"Top Claves de Rechazo por {title_desglose}",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        labels={
            "Pzas": "Total Piezas Rechazadas",
            "EtiquetaClave": "Clave de Rechazo",
            color_col: desglose if color_col and desglose in ["Máquina", "Turno"] else "Subcategoría"
        }
    )

    # 4. Ajustes finales de layout y formato
    fig.update_layout(
        xaxis_title="Total Piezas Rechazadas",
        yaxis_title="Clave de Rechazo",
        legend_title_text=desglose if color_col and desglose in ["Máquina", "Turno"] else "Subcategoría",
        hovermode="y unified",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    fig.update_traces(
        texttemplate='%{x:,.0f}',
        textposition='outside'
    )

    return fig


def compute_producto_critico(
    df: pd.DataFrame,
    *,
    top_n: int = 10,
    detail_by: Optional[Literal["Turno", "Maquina"]] = None
) -> pd.DataFrame:
    """
    Calcula el "producto crítico" basándose en producción y rechazo.

    Args:
        df (pd.DataFrame): DataFrame de producción preparado.
        top_n (int): Número de productos críticos a devolver.
        detail_by (Optional[Literal["Turno", "Maquina"]]): Columna para desagregar.

    Returns:
        pd.DataFrame: DataFrame con los productos críticos y sus métricas.

    Raises:
        ValueError: Si faltan columnas esenciales o `detail_by` es inválido.
    """
    required_cols = ["ComboProducto", "TotalPiezas", "PzasRech"]
    if detail_by:
        if detail_by not in df.columns:
            raise ValueError(f"Columna '{detail_by}' para desagregación no encontrada.")
        required_cols.append(detail_by)

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas esenciales faltantes para el cálculo de producto crítico: {', '.join(missing_cols)}")

    group_cols = ["ComboProducto"]
    if detail_by:
        group_cols.append(detail_by)

    grouped_df = df.groupby(group_cols).agg(
        Produccion=("TotalPiezas", "sum"),
        RechazoPzas=("PzasRech", "sum")
    ).reset_index()

    # Calcular IndiceRechazo
    def calculate_indice_rechazo(row):
        if row["Produccion"] > 0:
            return row["RechazoPzas"] / row["Produccion"]
        elif row["Produccion"] == 0 and row["RechazoPzas"] == 0:
            return 0.0
        else:  # Produccion == 0 and RechazoPzas > 0
            return np.nan # Indeterminado

    grouped_df["IndiceRechazo"] = grouped_df.apply(calculate_indice_rechazo, axis=1)

    # Ordenar y seleccionar top_n
    # Primero, identificar los top_n ComboProducto por Produccion y luego IndiceRechazo
    top_combos_base = grouped_df.groupby("ComboProducto").agg(
        TotalProduccion=("Produccion", "sum"),
        TotalRechazo=("RechazoPzas", "sum")
    ).reset_index()
    top_combos_base["IndiceRechazoBase"] = top_combos_base.apply(
        lambda r: r["TotalRechazo"] / r["TotalProduccion"] if r["TotalProduccion"] > 0 else (0.0 if r["TotalRechazo"] == 0 else np.nan),
        axis=1
    )
    top_combos_base = top_combos_base.sort_values(
        by=["TotalProduccion", "IndiceRechazoBase"],
        ascending=[False, False]
    ).head(top_n)

    # Filtrar el DataFrame agrupado original para incluir solo los top_n combos
    df_critico = grouped_df[grouped_df["ComboProducto"].isin(top_combos_base["ComboProducto"])].copy()

    # Re-ordenar el df_critico para que los combos aparezcan en el orden del top_combos_base
    df_critico["_combo_order"] = df_critico["ComboProducto"].astype("category").cat.set_categories(
        top_combos_base["ComboProducto"], ordered=True
    )
    df_critico = df_critico.sort_values(by=["_combo_order", "Produccion"], ascending=[True, False]).drop(columns="_combo_order")

    return df_critico


def compute_debug_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calcula estadísticas de depuración útiles para el análisis del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame de producción preparado.

    Returns:
        Dict[str, Any]: Diccionario con las estadísticas.
    """
    stats = {
        "filas_totales": len(df),
        "filas_total_0": 0,
        "produccion_total": 0,
        "rechazo_total": 0,
        "combos_unicos": 0,
        "maquinas_unicas": 0,
        "turnos_unicos": 0,
    }

    if df.empty:
        return stats

    if "TotalPiezas" in df.columns:
        stats["filas_total_0"] = (df["TotalPiezas"] == 0).sum()
        stats["produccion_total"] = df["TotalPiezas"].sum()
    if "PzasRech" in df.columns:
        stats["rechazo_total"] = df["PzasRech"].sum()
    if "ComboProducto" in df.columns:
        stats["combos_unicos"] = df["ComboProducto"].nunique()
    if "Maquina" in df.columns:
        stats["maquinas_unicas"] = df["Maquina"].nunique()
    if "Turno" in df.columns:
        stats["turnos_unicos"] = df["Turno"].nunique()

    return stats


def build_producto_critico_figure(
    df_critico: pd.DataFrame,
    *,
    title: str = "Productos Críticos",
    yaxis_bar: str = "Produccion",
    yaxis_line: str = "IndiceRechazo"
) -> go.Figure:
    """
    Genera una figura Plotly para visualizar productos críticos.

    Args:
        df_critico (pd.DataFrame): DataFrame resultante de `compute_producto_critico`.
        title (str): Título de la figura.
        yaxis_bar (str): Columna para el eje Y de las barras (Produccion).
        yaxis_line (str): Columna para el eje Y de la línea (IndiceRechazo).

    Returns:
        plotly.graph_objects.Figure: Figura Plotly.

    Raises:
        ValueError: Si faltan columnas esenciales en `df_critico`.
    """
    required_cols = ["ComboProducto", yaxis_bar, yaxis_line]
    missing_cols = [col for col in required_cols if col not in df_critico.columns]
    if missing_cols:
        raise ValueError(f"Columnas esenciales faltantes para la figura: {', '.join(missing_cols)}")

    if df_critico.empty:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="No hay datos para mostrar", xref="paper", yref="paper", showarrow=False, font=dict(size=20))])
        return fig

    # Determinar el color si hay desagregación
    color_col = None
    if "Turno" in df_critico.columns:
        color_col = "Turno"
    elif "Maquina" in df_critico.columns:
        color_col = "Maquina"

    fig = go.Figure()

    # Barras para Produccion
    fig.add_trace(go.Bar(
        x=df_critico["ComboProducto"],
        y=df_critico[yaxis_bar],
        name="Producción",
        marker_color='skyblue',
        hovertemplate="<b>%{x}</b><br>Producción: %{y:,.0f}<extra></extra>",
        marker_pattern_shape=color_col # Usar patrón si hay desagregación
    ))

    # Línea para IndiceRechazo (eje secundario)
    fig.add_trace(go.Scatter(
        x=df_critico["ComboProducto"],
        y=df_critico[yaxis_line] * 100, # Mostrar como porcentaje
        name="Índice de Rechazo",
        mode="lines+markers",
        yaxis="y2",
        line=dict(color='red', width=2),
        marker=dict(size=8, symbol='circle'),
        hovertemplate="<b>%{x}</b><br>Índice Rechazo: %{y:.2f}%<extra></extra>"
    ))

    # Configuración del layout
    fig.update_layout(
        title_text=title,
        xaxis_title="Producto",
        yaxis_title="Producción (Piezas)",
        yaxis2=dict(
            title="Índice de Rechazo (%)",
            overlaying="y",
            side="right",
            range=[0, df_critico[yaxis_line].max() * 100 * 1.1 if not df_critico.empty else 100] # Ajustar rango
        ),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.5)'),
        hovermode="x unified",
        barmode='group' # Asegura que las barras se agrupen si hay desagregación
    )

    # Si hay desagregación, usar color por la columna de detalle
    if color_col:
        # Plotly Express es mejor para esto, pero con go.Figure se puede hacer con múltiples traces
        # Para simplificar, si hay detail_by, se podría considerar hacer un facet_col o usar colores distintos
        # para cada valor de detail_by. Para este ejemplo, el pattern_shape ya da una indicación.
        # Una implementación más robusta con detail_by implicaría iterar sobre los valores únicos de detail_by
        # y añadir un trace por cada uno.
        pass # La implementación actual ya usa marker_pattern_shape

    return fig


def build_rechazos_debug_payload(
    df_rech_filtrado: pd.DataFrame,
    df_rech_long: pd.DataFrame | None = None
) -> dict:
    """
    Genera un payload de depuración para el DataFrame de rechazos.

    Esta función es pura (no usa Streamlit) y es tolerante a la falta de columnas.

    Args:
        df_rech_filtrado (pd.DataFrame): El DataFrame de rechazos ya filtrado por la UI.
        df_rech_long (pd.DataFrame | None): El DataFrame de rechazos original (sin filtrar)
                                             del cual se extraerán los mensajes de diagnóstico.

    Returns:
        dict: Un diccionario con estadísticas y DataFrames de muestra para depuración.
            - "total_filas": int
            - "filas_frac": int (conteo de filas con Pzas no enteras)
            - "pzas_dtype": str
            - "muestra_frac": DataFrame con hasta 200 ejemplos de filas con Pzas fraccionarias.
            - "top_fuentes": DataFrame con las 50 principales fuentes de residuos decimales.
            - "messages": Lista de mensajes de diagnóstico del proceso de carga.
    """
    # Initialize payload with default empty values
    payload = {
        "total_filas": 0,
        "filas_frac": 0,
        "pzas_dtype": "N/A",
        "muestra_frac": pd.DataFrame(),
        "top_fuentes": pd.DataFrame(),
        "messages": [],
    }

    if df_rech_long is not None and hasattr(df_rech_long, 'attrs'):
        payload["messages"] = df_rech_long.attrs.get("rechazos_long_messages", [])

    if df_rech_filtrado.empty:
        return payload

    df = df_rech_filtrado.copy()
    payload["total_filas"] = len(df)

    if "Pzas" not in df.columns:
        return payload

    pzas_series = pd.to_numeric(df["Pzas"], errors='coerce')
    payload["pzas_dtype"] = str(pzas_series.dtype)

    # Calculate fractional mask safely
    frac_mask = (pzas_series.round() != pzas_series) & pzas_series.notna()
    payload["filas_frac"] = int(frac_mask.sum())

    if payload["filas_frac"] > 0:
        # 1. Generate sample of fractional rows
        muestra_cols = ["Archivo", "Hoja", "SourceCol", "desc_row0", "ClaveCatalogo", "DescripcionCatalogo", "SubCategoria", "Pzas"]
        existing_muestra_cols = [col for col in muestra_cols if col in df.columns]
        payload["muestra_frac"] = df.loc[frac_mask, existing_muestra_cols].head(200)

        # 2. Generate top sources of decimal residues
        df["ResiduoDecimal"] = (pzas_series - pzas_series.round()).abs()
        group_cols = ["SourceCol", "desc_row0", "ClaveCatalogo"]
        existing_group_cols = [col for col in group_cols if col in df.columns]
        if existing_group_cols:
            payload["top_fuentes"] = df[frac_mask].groupby(existing_group_cols, dropna=False).agg(ResiduoDecimal=("ResiduoDecimal", "sum")).sort_values("ResiduoDecimal", ascending=False).reset_index().head(50)

    return payload

# =============================================================================
#  ANÁLISIS DE SENSIBILIDAD DE UMBRALES
# =============================================================================

def _get_top_n_keys(df_rech_long: pd.DataFrame, top_n: int) -> list[str]:
    """Helper to get the top N rejection keys by Pzas."""
    if df_rech_long.empty or 'ClaveCatalogo' not in df_rech_long.columns:
        return []
    
    top_keys_df = (
        df_rech_long.groupby("ClaveCatalogo")
        .agg(Pzas=("Pzas", "sum"))
        .sort_values("Pzas", ascending=False)
        .head(top_n)
    )
    return top_keys_df.index.tolist()


def build_threshold_comparison(
    df_audit_t1: pd.DataFrame,
    df_audit_t2: pd.DataFrame,
    rech_long_t1: pd.DataFrame,
    rech_long_t2: pd.DataFrame,
    top_n: int
) -> dict:
    """
    Compara los resultados de mapeo y el Top-N de rechazos entre dos umbrales.

    Args:
        df_audit_t1: DataFrame de auditoría para el umbral 1.
        df_audit_t2: DataFrame de auditoría para el umbral 2.
        rech_long_t1: DataFrame de rechazos long para el umbral 1.
        rech_long_t2: DataFrame de rechazos long para el umbral 2.
        top_n: Número de claves de rechazo a comparar.

    Returns:
        Un diccionario con la comparación, listo para la UI.
    """
    payload = {
        "summary_t1": {}, "summary_t2": {}, "top_n_t1": [], "top_n_t2": [],
        "jaccard_top_n": 0.0, "keys_in": [], "keys_out": []
    }

    # 1. Resumen de auditoría
    if not df_audit_t1.empty and 'reason_code' in df_audit_t1.columns:
        payload["summary_t1"] = df_audit_t1['reason_code'].value_counts().to_dict()
    if not df_audit_t2.empty and 'reason_code' in df_audit_t2.columns:
        payload["summary_t2"] = df_audit_t2['reason_code'].value_counts().to_dict()

    # 2. Comparación de Top-N
    top_n_t1 = _get_top_n_keys(rech_long_t1, top_n)
    top_n_t2 = _get_top_n_keys(rech_long_t2, top_n)
    payload["top_n_t1"] = top_n_t1
    payload["top_n_t2"] = top_n_t2

    set_t1 = set(top_n_t1)
    set_t2 = set(top_n_t2)

    intersection = len(set_t1.intersection(set_t2))
    union = len(set_t1.union(set_t2))
    payload["jaccard_top_n"] = intersection / union if union > 0 else 0.0
    
    payload["keys_in"] = sorted(list(set_t2 - set_t1))
    payload["keys_out"] = sorted(list(set_t1 - set_t2))

    return payload


def scan_thresholds_for_stability(
    dominio: str,
    thresholds: list[float],
    top_n: int
) -> pd.DataFrame:
    """
    Escanea un rango de umbrales para analizar la estabilidad del mapeo.

    Args:
        dominio: Dominio a analizar ('LINEAS' o 'COPLES').
        thresholds: Lista de umbrales altos a probar.
        top_n: Número de claves de rechazo para el análisis de estabilidad.

    Returns:
        Un DataFrame con métricas de estabilidad para cada umbral.
    """
    scan_results = []
    cached_data = {}

    # Pre-calcular todos los datos necesarios para evitar llamadas redundantes
    thresholds_to_scan = sorted(list(set(thresholds + [t + 2 for t in thresholds])))
    for t_high in thresholds_to_scan:
        if t_high not in cached_data:
            t_low = max(70.0, t_high - 10.0)
            audit_df = get_mapping_audit(dominio=dominio, threshold_high=t_high, threshold_low=t_low)
            rech_long_df = cargar_rechazos_long_area(dominio=dominio, threshold_high=t_high, threshold_low=t_low)
            cached_data[t_high] = {
                "audit": audit_df,
                "top_n_keys": _get_top_n_keys(rech_long_df, top_n)
            }

    for t_high in thresholds:
        data_t = cached_data.get(t_high)
        data_t_plus_2 = cached_data.get(t_high + 2)

        if not data_t: continue

        # Calcular Jaccard
        jaccard = 0.0
        if data_t_plus_2:
            set_t = set(data_t["top_n_keys"])
            set_t_plus_2 = set(data_t_plus_2["top_n_keys"])
            intersection = len(set_t.intersection(set_t_plus_2))
            union = len(set_t.union(set_t_plus_2))
            jaccard = intersection / union if union > 0 else 1.0 if not union else 0.0

        # Calcular porcentajes de clasificación
        audit_df = data_t["audit"]
        total_cols = len(audit_df)
        pct_fuertes = 0.0
        pct_grises = 0.0
        if total_cols > 0 and 'reason_code' in audit_df.columns:
            strong_reasons = {"MATCH_CLAVE_EXACTA", "OVERRIDE_RECHAZO", "MATCH_FUZZY_ALTO"}
            pct_fuertes = audit_df['reason_code'].isin(strong_reasons).sum() / total_cols
            pct_grises = audit_df['reason_code'].str.startswith("ZG", na=False).sum() / total_cols

        scan_results.append({
            "T": t_high,
            "Jaccard_T_Tplus2": jaccard,
            "pct_fuertes": pct_fuertes,
            "pct_grises": pct_grises
        })

    return pd.DataFrame(scan_results)


def suggest_threshold_high(df_scan: pd.DataFrame) -> float:
    """
    Sugiere un umbral alto óptimo basado en un escaneo de estabilidad.
    
    Args:
        df_scan (pd.DataFrame): DataFrame resultante de `scan_thresholds_for_stability`.

    Returns:
        float: El umbral alto recomendado.

    Justificación de la Heurística:
    El umbral ideal es aquel que maximiza la consistencia de los resultados (alta estabilidad)
    y minimiza la ambigüedad en las decisiones de mapeo (baja zona gris).
    1.  **Criterio Principal**: Se busca el umbral más bajo que inicia una "meseta de estabilidad",
        definida como una ventana de 3 puntos consecutivos donde la similitud Jaccard se
        mantiene >= 90% y el porcentaje de casos en la zona gris es <= 15%.
    2.  **Fallback**: Si no se encuentra una meseta ideal, se selecciona el umbral que ofrece
        el mejor "compromiso", calculado como `Jaccard - pct_grises`. Esto penaliza
        la ambigüedad mientras premia la estabilidad.
    3.  **Default**: Si no hay datos, se retorna un valor por defecto seguro (92.0).
    """
    if df_scan.empty:
        return 92.0  # Default si no hay datos

    df = df_scan.copy()
    
    df['is_stable_jaccard'] = df['Jaccard_T_Tplus2'] >= 0.90
    df['is_low_gray'] = df['pct_grises'] <= 0.15

    window_size = 3
    df['is_sustained_stability'] = (
        df['is_stable_jaccard'].rolling(window=window_size, min_periods=1).min().astype(bool) &
        df['is_low_gray']
    )

    candidates = df[df['is_sustained_stability']]
    if not candidates.empty:
        return candidates['T'].iloc[0]

    # Fallback: si no hay un punto ideal, buscar el mejor compromiso
    df['score'] = df['Jaccard_T_Tplus2'] - df['pct_grises']
    best_fallback_T = df.loc[df['score'].idxmax()]['T']
    
    return best_fallback_T if pd.notna(best_fallback_T) else 92.0


def compute_clave_drilldown_data(
    df_rech_long: pd.DataFrame,
    clave: str,
    group_by: Literal["Dia", "Maquina"] = "Dia"
) -> pd.DataFrame:
    """
    Prepara los datos para un drill-down de una clave de rechazo específica.

    Filtra los rechazos para una clave dada y los agrupa por día o por máquina
    para analizar su contribución a lo largo del tiempo o por fuente.

    Args:
        df_rech_long (pd.DataFrame): DataFrame de rechazos en formato largo.
        clave (str): La ClaveCatalogo a analizar.
        group_by (Literal["Dia", "Maquina"]): Dimensión para la agrupación.

    Returns:
        pd.DataFrame: Un DataFrame con los datos agregados, listo para graficar.
    """
    if df_rech_long.empty or 'ClaveCatalogo' not in df_rech_long.columns:
        return pd.DataFrame()

    df_norm = _normalize_rechazos_long_schema(df_rech_long)
    df_clave = df_norm[df_norm['ClaveCatalogo'] == clave].copy()

    if df_clave.empty:
        return pd.DataFrame()

    if group_by == "Dia":
        if 'Fecha' not in df_clave.columns:
            return pd.DataFrame()
        df_clave['Fecha'] = pd.to_datetime(df_clave['Fecha'], errors='coerce')
        df_clave = df_clave.dropna(subset=['Fecha'])
        
        drilldown_df = (
            df_clave.groupby(pd.Grouper(key='Fecha', freq='D'))
            .agg(Pzas=("Pzas", "sum"))
            .reset_index()
        )
        return drilldown_df.sort_values("Fecha")

    elif group_by == "Maquina":
        if 'Maquina' not in df_clave.columns:
            return pd.DataFrame()
            
        drilldown_df = (
            df_clave.groupby("Maquina")
            .agg(Pzas=("Pzas", "sum"))
            .reset_index()
        )
        return drilldown_df.sort_values("Pzas", ascending=False)
    
    return pd.DataFrame()


def build_clave_drilldown_figure(
    df_drilldown: pd.DataFrame,
    clave: str,
    group_by: Literal["Dia", "Maquina"]
) -> go.Figure:
    """
    Construye una figura Plotly para el drill-down de una clave de rechazo.

    Args:
        df_drilldown (pd.DataFrame): Datos agregados desde `compute_clave_drilldown_data`.
        clave (str): La ClaveCatalogo que se está analizando (para el título).
        group_by (Literal["Dia", "Maquina"]): Dimensión usada para la agrupación.

    Returns:
        go.Figure: Una figura de Plotly (gráfico de barras).
    """
    if df_drilldown.empty:
        fig = go.Figure()
        fig.update_layout(title=f"Drill-Down para Clave: {clave}", annotations=[dict(text="No hay datos para mostrar", xref="paper", yref="paper", showarrow=False, font=dict(size=20))])
        return fig

    title = f"Contribución por {group_by} para Clave: {clave}"
    x_axis = "Fecha" if group_by == "Dia" else "Maquina"
    
    fig = px.bar(df_drilldown, x=x_axis, y="Pzas", title=title, labels={x_axis: group_by, "Pzas": "Piezas Rechazadas"})
    fig.update_traces(texttemplate='%{y:,.0f}', textposition='outside')
    fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    return fig

# =============================================================================
#  HELPERS PARA TIEMPO IMPRODUCTIVO (TI)
# =============================================================================

def compute_ti_unmapped_dynamics(df_rech_long: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica y extrae las dinámicas de Tiempo Improductivo (TI) no mapeadas.

    Filtra el DataFrame de rechazos en formato largo para encontrar entradas que no
    fueron mapeadas a una clave de rechazo válida, considerándolas como TI en horas.

    Args:
        df_rech_long (pd.DataFrame): DataFrame de rechazos en formato largo, que
                                     debe incluir la columna 'Mensaje' con el
                                     código de la razón del mapeo.

    Returns:
        pd.DataFrame: Un DataFrame que contiene solo las filas de TI, con la columna
                      'Pzas' renombrada a 'TI_Horas' y columnas de contexto.
    """
    if df_rech_long.empty or 'Mensaje' not in df_rech_long.columns:
        return pd.DataFrame()

    df_norm = _normalize_rechazos_long_schema(df_rech_long.copy())

    # Razones que indican que una columna no es un rechazo mapeado, sino un TI.
    # Estas son columnas numéricas que no encontraron un match en el catálogo de rechazos.
    ti_reasons = {"NO_MATCH", "ZG_SCORE_BAJO"}

    df_ti = df_norm[df_norm['Mensaje'].isin(ti_reasons)].copy()

    if df_ti.empty:
        return pd.DataFrame()

    # Renombrar 'Pzas' a 'TI_Horas' para claridad semántica
    df_ti.rename(columns={"Pzas": "TI_Horas"}, inplace=True)

    # Seleccionar columnas relevantes para el reporte de TI
    ti_cols = [
        "Fecha", "Area", "Maquina", "Turno", "SourceCol", "TI_Horas", "Mensaje"
    ]
    existing_ti_cols = [col for col in ti_cols if col in df_ti.columns]

    return df_ti[existing_ti_cols]


def build_ti_kpis(df_ti_dynamics: pd.DataFrame) -> dict:
    """
    Calcula KPIs clave a partir de las dinámicas de Tiempo Improductivo (TI).

    Args:
        df_ti_dynamics (pd.DataFrame): DataFrame resultante de `compute_ti_unmapped_dynamics`.

    Returns:
        dict: Un diccionario con KPIs, incluyendo el total de horas de TI y
              desgloses por área y máquina.
    """
    kpis = {'total_ti_horas': 0.0, 'ti_por_area': pd.DataFrame(), 'ti_por_maquina': pd.DataFrame()}

    if df_ti_dynamics.empty or 'TI_Horas' not in df_ti_dynamics.columns:
        return kpis

    kpis['total_ti_horas'] = df_ti_dynamics['TI_Horas'].sum()

    if 'Area' in df_ti_dynamics.columns:
        df_with_label = df_ti_dynamics.copy()
        df_with_label["AreaLabel"] = df_with_label["Area"].map(AREA_CODE_TO_LABEL).fillna("Desconocido")
        kpis['ti_por_area'] = df_with_label.groupby('AreaLabel').agg(TI_Horas=('TI_Horas', 'sum')).sort_values('TI_Horas', ascending=False).reset_index()

    if 'Maquina' in df_ti_dynamics.columns:
        kpis['ti_por_maquina'] = df_ti_dynamics.groupby('Maquina').agg(TI_Horas=('TI_Horas', 'sum')).sort_values('TI_Horas', ascending=False).reset_index()

    return kpis

# =============================================================================
#  HELPERS DE EXPORTACIÓN
# =============================================================================

def export_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Convierte un DataFrame a una cadena de bytes en formato CSV (UTF-8 con BOM).

    Args:
        df (pd.DataFrame): El DataFrame a exportar.

    Returns:
        bytes: El contenido del archivo CSV listo para ser descargado.
    """
    if df.empty:
        return b""
    # Usar utf-8-sig para que Excel maneje correctamente los caracteres especiales
    return df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')


def export_dfs_to_xlsx_bytes(
    dfs_map: Dict[str, pd.DataFrame],
    messages: Optional[List[str]] = None
) -> bytes:
    """
    Exporta uno o más DataFrames a un archivo XLSX en memoria.

    Permite incluir una hoja adicional con mensajes de proceso.

    Args:
        dfs_map (Dict[str, pd.DataFrame]): Un diccionario donde las claves son los
                                           nombres de las hojas y los valores son
                                           los DataFrames a exportar.
        messages (Optional[List[str]]): Una lista de mensajes para incluir en una
                                        hoja separada llamada 'Mensajes de Proceso'.

    Returns:
        bytes: El contenido del archivo XLSX listo para ser descargado.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs_map.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        if messages:
            messages_df = pd.DataFrame(messages, columns=["Mensajes de Proceso"])
            messages_df.to_excel(writer, sheet_name="Mensajes de Proceso", index=False)
            
    return output.getvalue()
# =============================================================================
#  API PÚBLICA DEL MÓDULO
# =============================================================================

__all__ = [
    "compute_basic_kpis", # Existente
    "compute_cpk_grouped", # Existente
    "generate_insights_text", # Existente
    "prepare_df_for_analysis",
    "apply_filters",
    "compute_producto_critico",
    "compute_debug_stats",
    "build_producto_critico_figure",
    "apply_filters_long",
    "compute_top_claves_rechazo",
    "build_top_claves_rechazo_figure",
    "diagnose_columns",
    "build_rechazos_debug_payload",
    "build_threshold_comparison",
    "scan_thresholds_for_stability",
    "suggest_threshold_high",
    "compute_clave_drilldown_data",
    "build_clave_drilldown_figure",
    "compute_ti_unmapped_dynamics",
    "build_ti_kpis",
    "export_df_to_csv_bytes",
    "export_dfs_to_xlsx_bytes",
]

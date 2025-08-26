# UTILS/confiabilidad_panel.py

"""
Módulo de UI unificado para el panel de Confiabilidad.

Este módulo contiene todas las funciones de Streamlit para renderizar el panel
de análisis de confiabilidad de mapeo, tanto para Líneas como para Coples.

Funciones públicas:
- render_confiabilidad_panel: Punto de entrada principal para dibujar el panel.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List, Literal

# Importar constantes canónicas para el dominio
from UTILS.common import DOM_LINEAS, DOM_COPLES

# Imports canónicos desde UTILS.insights.
# Se importan todas las funciones de negocio para mantener un único punto de entrada.
# Las funciones no utilizadas pueden ser eliminadas por el linter si se configura.
from UTILS.insights import (
    build_threshold_comparison,
    scan_thresholds_for_stability,
    suggest_threshold_high,
    compute_clave_drilldown_data,     # Usado en drilldown
    build_clave_drilldown_figure      # Usado en drilldown
)

# Se importa de forma segura el helper para la auditoría de mapeo, que es opcional.
# Si no existe, la funcionalidad asociada se desactivará en la UI.
try:
    from UTILS.insights import get_mapping_audit
except ImportError:
    get_mapping_audit = None  # Fallback seguro


def _normalize_dominio_for_catalog(value: Any) -> tuple[str, str | None]:
    """
    Normaliza el valor del dominio a las constantes canónicas DOM_LINEAS o DOM_COPLES.

    Acepta variantes de string y códigos de área numéricos.
    Devuelve el dominio normalizado y un mensaje de advertencia si se usó un default.
    """
    original_value = str(value)

    if isinstance(value, str):
        val_lower = value.strip().lower().replace('í', 'i')
        if val_lower in ('lineas', 'líneas'):
            return DOM_LINEAS, None
        if val_lower == 'coples':
            return DOM_COPLES, None

    try:
        # Intentar convertir a entero si es numérico (e.g., '1', 2.0)
        num_val = int(value)
        if num_val in (1, 2, 3):
            return DOM_LINEAS, None
        if num_val == 4:
            return DOM_COPLES, None
    except (ValueError, TypeError):
        # No es un número válido o no es convertible
        pass
    warning_msg = f"Dominio '{original_value}' no reconocido. Usando 'Líneas' por defecto para análisis de estabilidad."
    return DOM_LINEAS, warning_msg


def _compute_confiabilidad_resumen(df: pd.DataFrame, active_threshold: float) -> Dict[str, Any]:
    """
    Calcula las métricas de resumen para el panel de confiabilidad.
    Esta es una función interna y no usa st.*.
    """
    if df.empty:
        return {'mapeadas': 0, 'no_mapeadas': 0, 'total_filas': 0, 'pzas_afectadas': 0}

    # Métricas de Mapeo (basadas en columnas fuente únicas)
    df_unique_cols = df.drop_duplicates(subset=['SourceCol'])
    total_filas = len(df_unique_cols)

    if 'score' in df_unique_cols.columns:
        mapeadas = df_unique_cols[df_unique_cols['score'] >= active_threshold].shape[0]
        no_mapeadas = total_filas - mapeadas
    else:
        mapeadas = None
        no_mapeadas = None

    # Piezas afectadas (suma de 'Pzas' de todas las filas mapeadas)
    pzas_afectadas = None
    if 'Pzas' in df.columns and 'score' in df.columns:
        df_mapeado = df[df['score'] >= active_threshold]
        pzas_afectadas = df_mapeado['Pzas'].sum()
        if pd.isna(pzas_afectadas): 
            pzas_afectadas = 0
    
    return {
        'mapeadas': mapeadas,
        'no_mapeadas': no_mapeadas,
        'total_filas': total_filas,
        'pzas_afectadas': pzas_afectadas
    }


def _render_resumen(df_full: pd.DataFrame, active_threshold: float, config: Dict[str, Any]):
    """Renderiza la sección de resumen con métricas clave."""
    st.subheader("Resumen de Confiabilidad")

    # Delegar cálculo a función interna
    metricas = _compute_confiabilidad_resumen(df_full, active_threshold)

    # Preparar valores para las métricas, manejando casos de N/D
    if metricas['mapeadas'] is not None:
        if metricas['total_filas'] > 0:
            porc_mapeadas = (metricas['mapeadas'] / metricas['total_filas']) * 100
            porc_no_mapeadas = (metricas['no_mapeadas'] / metricas['total_filas']) * 100
        else:
            porc_mapeadas = 0.0
            porc_no_mapeadas = 0.0
        
        val_map = f"{porc_mapeadas:.1f}%"
        val_nomap = f"{porc_no_mapeadas:.1f}%"
        help_map = f"{metricas['mapeadas']} de {metricas['total_filas']} columnas fuente."
        help_nomap = f"{metricas['no_mapeadas']} de {metricas['total_filas']} columnas fuente."
    else:
        val_map = "N/D"
        val_nomap = "N/D"
        help_map = "No se pudo calcular (falta columna 'score')."
        help_nomap = "No se pudo calcular (falta columna 'score')."

    pzas_val = f"{metricas.get('pzas_afectadas', 0):,}" if metricas.get('pzas_afectadas') is not None else "N/D"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="% Columnas Mapeadas",
            value=val_map,
            help=help_map
        )
    with col2:
        st.metric(
            label="% Columnas No Mapeadas",
            value=val_nomap,
            help=help_nomap
        )
    with col3:
        st.metric(
            label="Total Pzas Afectadas",
            value=pzas_val
        )
    
    # Mostrar mensajes de rechazo si existen
    rechazos = df_full.attrs.get('rechazos_long_messages', [])
    if rechazos:
        with st.expander("⚠️ Mensajes del Proceso de Mapeo", expanded=True):
            for msg in rechazos:
                st.warning(msg)


def _render_diagnostico_mapeo(df: pd.DataFrame, config: Dict[str, Any]):
    """Renderiza la tabla de diagnóstico de mapeo."""
    with st.expander("Diagnóstico de Mapeo"):
        st.info("Auditoría detallada del mapeo por cada columna fuente (`SourceCol`).")
        
        cols_a_mostrar = ['SourceCol', 'ClaveCatalogo', 'score', 'Mensaje']
        df_display = df[[col for col in cols_a_mostrar if col in df.columns]]
        
        st.dataframe(df_display, use_container_width=True)


def _render_comparar_umbrales(df: pd.DataFrame, config: Dict[str, Any]):
    """Renderiza la UI para comparar dos umbrales de confianza."""
    with st.expander("Análisis de Estabilidad de Mapeo (T1 vs T2)"):
        st.info("Compare cómo cambia el conjunto de mapeos al variar el umbral de confianza.")
        
        dominio_cfg = config.get('dominio') or config.get('area_domain') or config.get('area')
        dominio_norm, warning = _normalize_dominio_for_catalog(dominio_cfg)
        if warning:
            st.warning(warning)

        dominio_key_part = config.get('dominio', 'default')
        defaults = config.get('umbrales_default', (88, 90))
        # Usar 'top_n' del config del dashboard, con 10 como default.
        top_n_default = int(config.get('top_n', 10))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            umbral1 = st.slider(
                "Umbral de Confianza T1 (%)", 
                min_value=70, max_value=100, value=defaults[0], step=1
            )
        with col2:
            umbral2 = st.slider(
                "Umbral de Confianza T2 (%)",
                min_value=70, max_value=100, value=defaults[1], step=1
            )
        with col3:
            top_n_compare = st.slider("Top-N para comparación", 5, 25, top_n_default, key=f"{dominio_key_part}_compare_topn")

        if get_mapping_audit is None:
            st.error("La función `get_mapping_audit` no está disponible. No se puede realizar la comparación.")
            st.info("Asegúrate de que `UTILS/insights.py` exporta la función `get_mapping_audit`.")
            return

        with st.spinner(f"Comparando umbrales {umbral1}% y {umbral2}%..."):
            try:
                # La llamada ahora usa el dominio normalizado, no el DataFrame.
                audit_t1 = get_mapping_audit(dominio=dominio_norm, threshold_high=umbral1, threshold_low=max(70.0, umbral1 - 10.0))
                audit_t2 = get_mapping_audit(dominio=dominio_norm, threshold_high=umbral2, threshold_low=max(70.0, umbral2 - 10.0))

                # Delegar cálculo a helper oficial de insights.py.
                # Se asume que la firma real solo requiere los dos dataframes de auditoría y el top_n.
                # Los mensajes de rechazo (rech_long) son para diagnóstico general y no se pasan aquí,
                # lo que resuelve el TypeError.
                payload = build_threshold_comparison(
                    audit_t1=audit_t1,
                    audit_t2=audit_t2,
                    top_n=top_n_compare
                )
            except Exception as e:
                st.error("Ocurrió un error al comparar los umbrales.")
                st.exception(e)
                return

        st.metric(
            label=f"Similaridad Jaccard (Top-{top_n_compare})",
            value=f"{payload.get('jaccard_top_n', 0.0):.1%}",
            help="Mide la similitud entre las dos listas Top-N. 100% significa que son idénticas."
        )

        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Claves que ENTRAN al Top-{top_n_compare} con T2={umbral2}%**")
            st.dataframe(pd.DataFrame(payload.get("keys_in", []), columns=["Clave"]))
        with c2:
            st.write(f"**Claves que SALEN del Top-{top_n_compare} con T2={umbral2}%**")
            st.dataframe(pd.DataFrame(payload.get("keys_out", []), columns=["Clave"]))

        with st.expander("Ver desglose de decisiones de mapeo"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Decisiones con T1={umbral1}%**")
                st.dataframe(pd.DataFrame(list(payload.get("summary_t1", {}).items()), columns=["Causa", "Cantidad"]))
            with col2:
                st.write(f"**Decisiones con T2={umbral2}%**")
                st.dataframe(pd.DataFrame(list(payload.get("summary_t2", {}).items()), columns=["Causa", "Cantidad"]))


def _render_curva_estabilidad(df: pd.DataFrame, config: Dict[str, Any]):
    """Renderiza la curva de estabilidad y la recomendación de umbral."""
    with st.expander("Curva de Estabilidad y Umbral Recomendado"):
        st.info("Visualice la estabilidad del mapeo y obtenga un umbral recomendado para maximizar la consistencia.")
        
        dominio_cfg = config.get('dominio') or config.get('area_domain') or config.get('area')
        dominio_norm, warning = _normalize_dominio_for_catalog(dominio_cfg)
        if warning:
            st.warning(warning)

        dominio_key_part = config.get('dominio', 'default')
        session_key = f"temp_threshold_{dominio_key_part}"

        scan_range = st.slider(
            "Rango de Umbrales a Escanear (%)",
            min_value=70, max_value=99, value=(80, 98),
            key=f"scan_range_{dominio_key_part}"
        )
        
        thresholds_to_scan = list(range(scan_range[0], scan_range[1] + 1))
        top_n = int(config.get('top_n', 10))

        with st.spinner(f"Escaneando umbrales de {scan_range[0]}% a {scan_range[1]}%..."):
            # La llamada ahora usa el dominio normalizado, no el DataFrame.
            # Esto resuelve el ValueError de "Operands are not aligned".
            df_curva = scan_thresholds_for_stability(
                dominio=dominio_norm, 
                thresholds=thresholds_to_scan, 
                top_n=top_n
            )
            recommended_t = suggest_threshold_high(df_curva)

        if not df_curva.empty:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    label="Umbral Alto Recomendado",
                    value=f"{recommended_t:.0f}%",
                    help="Umbral que ofrece el mejor balance entre estabilidad (Jaccard) y mínima ambigüedad (zona gris)."
                )

                def set_temp_threshold():
                    st.session_state[session_key] = recommended_t

                def clear_temp_threshold():
                    if session_key in st.session_state:
                        del st.session_state[session_key]

                st.button("Probar Umbral Recomendado", on_click=set_temp_threshold, use_container_width=True)
                st.button("Restaurar Umbral Original", on_click=clear_temp_threshold, use_container_width=True)

            with col2:
                fig = px.line(
                    df_curva,
                    x='T',
                    y='Jaccard_T_Tplus2',
                    title='Estabilidad del Mapeo vs. Umbral de Confianza',
                    labels={'T': 'Umbral de Confianza (%)', 'Jaccard_T_Tplus2': 'Índice de Jaccard (Estabilidad)'}
                )
                fig.add_vline(x=recommended_t, line_dash="dash", line_color="green", annotation_text="Recomendado")
                fig.update_yaxes(range=[0, 1])
                fig.update_layout(yaxis_tickformat=".0%")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para generar la curva de estabilidad.")

def _summarize_fracciones_pzas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Helper interno para analizar el impacto de redondeos en las piezas.
    
    Busca mensajes en los atributos del df o analiza directamente la columna 'Pzas'.
    """
    # Prioridad 1: Buscar mensajes explícitos sobre redondeo
    messages = df.attrs.get('rechazos_long_messages', [])
    fraction_messages = [msg for msg in messages if "fracciones" in msg.lower() or "redondeo" in msg.lower()]
    
    if fraction_messages:
        # Si hay mensajes, mostrarlos es el resumen más fiel.
        return pd.DataFrame(fraction_messages, columns=["Mensajes de Proceso sobre Fracciones"])

    # Prioridad 2: Análisis directo si no hay mensajes
    if 'Pzas' not in df.columns:
        return pd.DataFrame({
            "Análisis": ["No se encontró la columna 'Pzas' para el análisis."]
        })

    if df['Pzas'].dtype.name.startswith('Int'):
        return pd.DataFrame({
            "Análisis": ["La columna 'Pzas' es de tipo entero, no se detectaron fracciones."]
        })

    pzas_series = pd.to_numeric(df['Pzas'], errors='coerce').fillna(0)
    df_frac = df.loc[pzas_series != pzas_series.round()]

    if df_frac.empty:
        return pd.DataFrame({
            "Análisis": ["Aunque 'Pzas' es de tipo flotante, todos los valores son enteros."]
        })

    # Si se detectan fracciones, crear un resumen numérico
    total_rows = len(df)
    frac_rows = len(df_frac)
    pzas_afectadas_frac = df_frac['Pzas'].sum()
    total_pzas = pzas_series.sum()

    summary_data = {
        'Métrica': [
            'Filas con valores fraccionarios', 
            'Total de filas analizadas',
            'Piezas totales en filas con fracciones',
            '% de Piezas en filas con fracciones'
        ],
        'Valor': [
            f"{frac_rows:,}",
            f"{total_rows:,}",
            f"{pzas_afectadas_frac:,.2f}",
            f"{(pzas_afectadas_frac / total_pzas * 100):.2f}%" if total_pzas > 0 else "0.00%"
        ]
    }
    return pd.DataFrame(summary_data)

def _render_fracciones_piezas(df: pd.DataFrame, config: Dict[str, Any]):
    """Renderiza el análisis de fracciones en piezas."""
    with st.expander("Análisis de Fracciones en Piezas"):
        st.info("Resumen del impacto de tolerancias de redondeo en el conteo de piezas.")
        
        # Delegar cálculo a helper interno
        df_fracciones = _summarize_fracciones_pzas(df)
        
        # El helper siempre devuelve un DataFrame, incluso si es solo un mensaje.
        st.dataframe(df_fracciones, use_container_width=True)


# --- Funciones para el modo "Gourmet" ---

def _resolve_domain_constant(config_panel: Dict[str, Any]) -> tuple[str, str | None]:
    """Extrae y normaliza el dominio desde el dict de configuración a una constante."""
    dominio_cfg = config_panel.get('dominio') or config_panel.get('area_domain') or config_panel.get('area')
    return _normalize_dominio_for_catalog(dominio_cfg)

def _render_gourmet_cobertura(df: pd.DataFrame, config: Dict[str, Any]):
    """Renderiza la sección G1: Cobertura de Mapeo (ligero)."""
    st.subheader("G1. Cobertura de Mapeo (MUST MATCH)")

    # --- 1. Preparación de datos y fallbacks ---
    df_local = df.copy()
    pzas_available = False
    pzas_col_name = None
    conversion_warning = None

    if 'Pzas' in df_local.columns:
        pzas_col_name = 'Pzas'
    elif 'PzasRech' in df_local.columns:
        pzas_col_name = 'PzasRech'
    
    if pzas_col_name:
        pzas_available = True
        original_pzas = df_local[pzas_col_name]
        numeric_pzas = pd.to_numeric(original_pzas, errors='coerce')
        if numeric_pzas.isnull().any() and not original_pzas.isnull().all():
            conversion_warning = f"Algunos valores en '{pzas_col_name}' no son numéricos y fueron ignorados."
        df_local['Pzas'] = numeric_pzas.fillna(0).astype(pd.Int64Dtype())

    # --- 2. Cálculo de cobertura de columnas ---
    total_cols_dinamicas = None
    if get_mapping_audit:
        try:
            dominio_norm, warning = _resolve_domain_constant(config)
            if warning: st.warning(warning)
            active_threshold = config.get('threshold_high_actual', 90.0)
            audit_df = get_mapping_audit(dominio=dominio_norm, threshold_high=active_threshold, threshold_low=70.0)
            if audit_df is not None and 'SourceCol' in audit_df.columns:
                total_cols_dinamicas = audit_df['SourceCol'].nunique()
        except Exception as e:
            st.warning(f"No se pudo ejecutar `get_mapping_audit` para el conteo de columnas: {e}")

    if total_cols_dinamicas is None and 'SourceCol' in df_local.columns:
        total_cols_dinamicas = df_local['SourceCol'].nunique()

    # --- 3. Criterio de "No Mapeado" y KPIs ---
    n_cols_mapeadas, n_cols_no_mapeadas, porc_mapeadas, pzas_no_mapeadas = None, None, None, None
    top_unmapped = pd.DataFrame()

    if 'ClaveCatalogo' in df_local.columns and 'SourceCol' in df_local.columns:
        unmapped_values = {'no mapeado', 'sin clave', 'n/a'}
        is_unmapped_mask = df_local['ClaveCatalogo'].isnull() | df_local['ClaveCatalogo'].str.strip().str.lower().isin(unmapped_values)
        df_unmapped = df_local[is_unmapped_mask]
        
        n_cols_no_mapeadas = df_unmapped['SourceCol'].nunique()

        if total_cols_dinamicas is not None:
            n_cols_mapeadas = total_cols_dinamicas - n_cols_no_mapeadas
            porc_mapeadas = (n_cols_mapeadas / total_cols_dinamicas * 100) if total_cols_dinamicas > 0 else 0

        if pzas_available:
            pzas_no_mapeadas = df_unmapped['Pzas'].sum()
            top_unmapped = df_unmapped.groupby('SourceCol')['Pzas'].sum().nlargest(5).reset_index()
            top_unmapped.columns = ['Causa de Rechazo (SourceCol)', 'Piezas Afectadas']
        else:
            top_unmapped = df_unmapped['SourceCol'].value_counts().nlargest(5).reset_index()
            top_unmapped.columns = ['Causa de Rechazo (SourceCol)', 'Incidencias']
    
    # --- 4. Renderizado de UI ---
    col1, col2, col3 = st.columns(3)
    
    col1.metric("Cobertura de Mapeo", f"{porc_mapeadas:.1f}%" if porc_mapeadas is not None else "N/D", help=f"{n_cols_mapeadas} de {total_cols_dinamicas} causas raíz mapeadas." if all(v is not None for v in [n_cols_mapeadas, total_cols_dinamicas]) else "No se pudo calcular.")
    col2.metric("Causas No Mapeadas", f"{n_cols_no_mapeadas}" if n_cols_no_mapeadas is not None else "N/D", delta=n_cols_no_mapeadas if n_cols_no_mapeadas is not None and n_cols_no_mapeadas > 0 else None, delta_color="inverse")
    col3.metric("Piezas No Mapeadas", f"{int(pzas_no_mapeadas):,}" if pzas_no_mapeadas is not None else "N/D", help="No se encontró una columna de piezas ('Pzas' o 'PzasRech')." if not pzas_available else None)

    if not top_unmapped.empty:
        st.write("**Top 5 Causas No Mapeadas**")
        st.dataframe(top_unmapped, use_container_width=True, hide_index=True)
    elif 'ClaveCatalogo' in df_local.columns:
        st.success("¡Felicidades! No se encontraron causas no mapeadas en el período seleccionado.")

    if conversion_warning:
        st.caption(f"⚠️ {conversion_warning}")
        
    messages = df.attrs.get('rechazos_long_messages', [])
    if messages:
        with st.expander("Mensajes del Proceso de Mapeo"):
            for msg in messages:
                st.warning(msg)

def _render_gourmet_estabilidad(config: Dict[str, Any]):
    """Renderiza la sección G2: Estabilidad de Umbral (Lite, on-demand)."""
    st.subheader("G2. Estabilidad de Umbral (Lite)")
    
    session_key = f"gourmet_stability_result_{config.get('dominio', 'default')}"

    if st.button("Calcular Estabilidad (Lite)"):
        with st.spinner("Calculando..."):
            dominio_norm, warning = _resolve_domain_constant(config)
            if warning:
                st.warning(warning)
            
            thresholds_to_scan = config.get('t_values', [80, 85, 88, 90, 92, 95])
            top_n = int(config.get('top_n', 10))

            try:
                df_curva = scan_thresholds_for_stability(
                    dominio=dominio_norm, thresholds=thresholds_to_scan, top_n=top_n
                )
                recommended_t = suggest_threshold_high(df_curva) if 'suggest_threshold_high' in globals() and callable(suggest_threshold_high) else None
                st.session_state[session_key] = {"df_curva": df_curva, "recommended_t": recommended_t}
            except Exception as e:
                st.error("No se pudo calcular la estabilidad.")
                st.exception(e)
                if session_key in st.session_state:
                    del st.session_state[session_key]

    if session_key in st.session_state:
        result = st.session_state[session_key]
        df_curva, recommended_t = result["df_curva"], result["recommended_t"]

        if not df_curva.empty:
            fig = px.line(
                df_curva, x='T', y='Jaccard_T_Tplus2', title='Estabilidad vs. Umbral (Lite)',
                labels={'T': 'Umbral (%)', 'Jaccard_T_Tplus2': 'Estabilidad'}, markers=True
            )
            fig.update_yaxes(range=[0, 1], tickformat=".0%")
            if recommended_t:
                fig.add_vline(x=recommended_t, line_dash="dash", line_color="green", annotation_text=f"Recomendado: {recommended_t}%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se generaron datos de estabilidad para el rango de umbrales seleccionado.")
    else:
        st.info("Presione el botón para ejecutar el análisis de estabilidad con una selección de umbrales clave.")


def _render_drilldown_clave(df: pd.DataFrame, config: Dict[str, Any]):
    """Permite el análisis detallado por una ClaveCatalogo específica."""
    unificar_perdidas = config.get('unificar_perdidas', False)
    
    expander_title = (
        "Análisis de Pérdidas (Tiempos Improductivos y Rechazos)" 
        if unificar_perdidas else "Drilldown por Clave de Rechazo"
    )
    
    with st.expander(expander_title):
        if unificar_perdidas:
            st.info("Análisis unificado de claves de rechazo y tiempos improductivos no mapeados.")
            # Fallback seguro si los helpers de TI no existen o no hay datos.
            st.warning("💡 No se encontraron Tiempos Improductivos (dinámicas no mapeadas) para este contexto. Mostrando solo Claves de Rechazo.", icon="⚠️")
        else:
            st.info("Seleccione una clave de rechazo para ver su comportamiento a lo largo del tiempo y por máquina.")

        if 'ClaveCatalogo' not in df.columns or df['ClaveCatalogo'].dropna().empty:
            st.warning("No hay claves de catálogo disponibles para el análisis de drill-down.")
            return

        claves_disponibles = sorted(df['ClaveCatalogo'].dropna().unique())
        dominio = config.get('dominio', 'default')
        
        selected_clave = st.selectbox(
            "Seleccione una Clave de Rechazo",
            options=claves_disponibles,
            index=0,
            key=f"drilldown_clave_{dominio}"
        )

        if selected_clave:
            st.subheader(f"Análisis para: {selected_clave}")

            # Delegar cálculos a insights.py
            df_drilldown_dia = compute_clave_drilldown_data(df, selected_clave, group_by="Dia")
            df_drilldown_maq = compute_clave_drilldown_data(df, selected_clave, group_by="Maquina")

            tab1, tab2, tab3 = st.tabs(["Evolución por Día", "Contribución por Máquina", "Detalles de Mapeo"])

            with tab1:
                fig_dia = build_clave_drilldown_figure(df_drilldown_dia, selected_clave, group_by="Dia")
                st.plotly_chart(fig_dia, use_container_width=True)
            
            with tab2:
                fig_maq = build_clave_drilldown_figure(df_drilldown_maq, selected_clave, group_by="Maquina")
                st.plotly_chart(fig_maq, use_container_width=True)

            with tab3:
                st.write("Muestra de mapeos asociados a esta clave:")
                df_details = df[df['ClaveCatalogo'] == selected_clave]
                cols_to_show = ['SourceCol', 'score', 'Mensaje', 'Pzas']
                st.dataframe(df_details[[c for c in cols_to_show if c in df_details.columns]], use_container_width=True)


def render_confiabilidad_gourmet(df_preparado_filtrado: pd.DataFrame, config_panel: Dict[str, Any]):
    """
    Punto de entrada para el panel de Confiabilidad "Gourmet" (ligero).

    Muestra únicamente:
    - G1: KPIs de cobertura de mapeo.
    - G2: Un análisis de estabilidad de umbral bajo demanda.
    """
    if df_preparado_filtrado.empty:
        st.warning("No hay datos de mapeo para mostrar en el período seleccionado.")
        return

    st.header(f"Panel de Confiabilidad (Gourmet): {config_panel.get('dominio', 'General')}")
    st.markdown("---")

    # G1. Cobertura de Mapeo
    _render_gourmet_cobertura(df_preparado_filtrado, config_panel)
    st.markdown("---")

    # G2. Estabilidad de Umbral
    _render_gourmet_estabilidad(config_panel)


def render_confiabilidad_panel(df_preparado_filtrado: pd.DataFrame, config_panel: Dict[str, Any]):
    """
    Punto de entrada único para renderizar el panel de confiabilidad.

    Este panel concentra todas las visualizaciones y análisis relacionados con
    la calidad y estabilidad del mapeo de datos.

    Args:
        df_preparado_filtrado (pd.DataFrame): 
            El DataFrame con los datos de mapeo ya procesados y filtrados.
            Debe contener **todos** los mapeos potenciales con sus scores, no solo los
            que superan un umbral.
            Puede contener 'Pzas' para análisis adicionales.
            Puede contener metadatos en el atributo `.attrs['rechazos_long_messages']`.
        config_panel (Dict[str, Any]): 
            Un diccionario de configuración para el panel.
            Puede incluir:
            - 'dominio': str, ej. "Líneas" o "Coples".
            - 'threshold_high_actual': float, el umbral del dashboard principal.
            - 'umbrales_default': tuple, ej. (88, 90).
    """
    if df_preparado_filtrado.empty:
        st.warning("No hay datos de mapeo para mostrar en el período seleccionado.")
        return

    st.header(f"Panel de Confiabilidad: {config_panel.get('dominio', 'General')}")
    st.markdown("---")

    # Lógica para manejar el umbral temporal vs. el del dashboard
    dominio = config_panel.get('dominio', 'default')
    session_key = f"temp_threshold_{dominio}"
    
    umbral_dashboard = config_panel.get('threshold_high_actual', 90.0)
    umbral_temporal = st.session_state.get(session_key)
    
    active_threshold = umbral_temporal if umbral_temporal is not None else umbral_dashboard
    
    if umbral_temporal is not None:
        st.info(f"Mostrando análisis con umbral temporal de **{umbral_temporal:.0f}%**. Use el botón 'Restaurar' para volver al umbral del dashboard ({umbral_dashboard:.0f}%).")

    # 1. Resumen (siempre visible)
    _render_resumen(df_preparado_filtrado, active_threshold / 100.0, config_panel)
    st.markdown("---")

    # 2. Secciones colapsables
    _render_diagnostico_mapeo(df_preparado_filtrado, config_panel)
    _render_comparar_umbrales(df_preparado_filtrado, config_panel)
    _render_curva_estabilidad(df_preparado_filtrado, config_panel)
    
    # La sección de fracciones solo tiene sentido si existe la columna 'Pzas'
    if 'Pzas' in df_preparado_filtrado.columns:
        _render_fracciones_piezas(df_preparado_filtrado, config_panel)
        
    _render_drilldown_clave(df_preparado_filtrado, config_panel)
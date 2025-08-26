# -*- coding: utf-8 -*-
"""
UTILS/lineas_dashboard.py
Dashboard de Líneas (tubos largos) refactorizado para usar UTILS.insights.

- Carga sus propios datos usando UTILS.common.cargar_area.
- Toda la lógica de preparación, filtrado y cálculo se delega a UTILS.insights.
- Los filtros en la barra lateral usan claves únicas para preservar el estado.

Autor: Eduardo + M365 Copilot
"""

from __future__ import annotations
from datetime import date

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Importaciones centralizadas
from UTILS.common import DOM_LINEAS, cargar_area, cargar_rechazos_long_area
from UTILS.confiabilidad_panel import render_confiabilidad_gourmet
from UTILS.insights import (
    prepare_df_for_analysis,
    apply_filters,
    apply_filters_long,
    compute_producto_critico,
    compute_top_claves_rechazo,
    build_producto_critico_figure,
    build_top_claves_rechazo_figure,
    compute_debug_stats,
    diagnose_columns,
    compute_ti_unmapped_dynamics,
    build_ti_kpis,
    export_df_to_csv_bytes,
    )


def render_lineas_dashboard():
    """
    Renderiza el dashboard completo para el área de Líneas.

    Esta función no recibe argumentos. Carga y procesa los datos internamente,
    y renderiza los componentes de la UI de Streamlit.
    """
    st.header("Análisis de Producción y Pérdidas - Líneas")

    # --- 1. Carga y preparación de datos ---
    with st.spinner("Cargando y preparando datos de Líneas..."):
        try:
            # `cargar_area` busca y consolida los archivos Excel en la ruta de datos.
            # Devuelve una tupla (df, df), tomamos el primero para producción.
            df_raw, _ = cargar_area(DOM_LINEAS)
        except Exception as e:
            st.error(f"No se pudieron cargar los datos de producción para Líneas. Verifica que los archivos existan en la carpeta de datos.")
            st.exception(e)
            return

        df_prepared = prepare_df_for_analysis(df_raw)

    if df_prepared.empty:
        st.warning("No hay datos disponibles para el área de Líneas.")
        return

    # --- 2. Sidebar de filtros ---
    st.sidebar.title("Filtros - Líneas")

    # Botón de recarga
    if st.sidebar.button("Recargar datos (limpiar caché)", key="lineas_recargar"):
        st.cache_data.clear()
        st.rerun()

    # Filtros de fecha
    s_fecha = pd.to_datetime(df_prepared["Fecha"], errors="coerce")
    min_ts = s_fecha.min()
    max_ts = s_fecha.max()
    # Convertir a objetos `date` de Python, con fallback a hoy si hay NaT
    min_date = min_ts.date() if pd.notna(min_ts) else date.today()
    max_date = max_ts.date() if pd.notna(max_ts) else date.today()

    sel_fechas = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="lineas_fechas"
    )
    # Asegurar que siempre sea una tupla de 2
    if len(sel_fechas) != 2:
        sel_fechas = (min_date, max_date)

    # Filtros de producto y proceso
    def get_options(col_name):
        return sorted(df_prepared[col_name].dropna().unique())

    sel_diam = st.sidebar.multiselect("Diámetro", get_options("Diametro"), key="lineas_diam")
    sel_lib = st.sidebar.multiselect("Libraje", get_options("Libraje"), key="lineas_lib")
    sel_acero = st.sidebar.multiselect("Acero", get_options("Acero"), key="lineas_acero")
    sel_rosca = st.sidebar.multiselect("Rosca", get_options("Rosca"), key="lineas_rosca")
    sel_turno = st.sidebar.multiselect("Turno", get_options("Turno"), key="lineas_turno")
    sel_maq = st.sidebar.multiselect("Máquina", get_options("Maquina"), key="lineas_maq")

    # Controles de visualización
    st.sidebar.title("Controles de Gráfica")
    desglose_map = {"Global": None, "Por Turno": "Turno", "Por Máquina": "Maquina"}
    sel_desglose_label = st.sidebar.selectbox("Desglose", list(desglose_map.keys()), key="lineas_desglose")
    detail_by_value = desglose_map[sel_desglose_label]

    top_n_value = st.sidebar.slider("Top N", min_value=5, max_value=30, value=10, step=1, key="lineas_top_n")

    # Debug toggle
    debug_mode = st.sidebar.checkbox("Mostrar datos crudos (debug)", key="lineas_debug_cb")

    # --- 3. Aplicación de filtros ---
    df_filtered = apply_filters(
        df_prepared,
        fechas=sel_fechas,
        diametros=sel_diam or None,
        librajes=sel_lib or None,
        aceros=sel_acero or None,
        roscas=sel_rosca or None,
        turnos=sel_turno or None,
        maquinas=sel_maq or None,
        dominio=DOM_LINEAS
    )

    if df_filtered.empty:
        st.info("No se encontraron resultados con los filtros seleccionados.")
        return

    st.subheader("Análisis de Productos Críticos")
    # --- 4. Cálculo de "Producto Crítico" ---
    df_critico = compute_producto_critico(
        df_filtered,
        top_n=top_n_value,
        detail_by=detail_by_value
    )

    # --- 5. Gráfica ---
    if df_critico.empty:
        st.info("No hay datos de productos críticos para mostrar con los filtros actuales.")
    else:
        with st.container():
            fig = build_producto_critico_figure(
                df_critico,
                title=f"Top {top_n_value} Productos Críticos - Líneas"
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_data = export_df_to_csv_bytes(df_critico)
            st.download_button(
                label="📥 Exportar Producto Crítico (CSV)",
                data=csv_data,
                file_name=f"producto_critico_lineas_{date.today()}.csv",
                mime="text/csv",
                key="lineas_export_critico"
            )

    # --- Análisis de Pérdidas (TI y Rechazos) ---
    st.subheader("Análisis de Pérdidas (Tiempos Improductivos y Rechazos)")

    # Controles para el análisis
    c1, c2, c3 = st.columns(3)
    top_n_rechazo = c1.slider("Top N Claves de Rechazo", 5, 25, 10, key="lineas_rechazo_top_n")
    desglose_rechazo = c2.selectbox("Desglose de Rechazos por", ["Global", "Máquina", "Turno"], key="lineas_rechazo_desglose")
    threshold_high_rechazo = c3.slider("Umbral de Similitud Alto (%)", 70, 100, 92, key="lineas_rechazo_threshold_high", help="Puntaje para aceptar un match automáticamente.")
    threshold_low_rechazo = max(70.0, threshold_high_rechazo - 10.0)

    # Carga y filtrado de datos de pérdidas (contiene tanto rechazos como TI potenciales)
    df_rechazos_long = cargar_rechazos_long_area(
        dominio=DOM_LINEAS,
        threshold_high=threshold_high_rechazo,
        threshold_low=threshold_low_rechazo
    )

    if df_rechazos_long.empty:
        st.info("No se encontraron datos de rechazo o TI para analizar con el umbral actual.")
    else:
        # Aplicar los mismos filtros del sidebar
        df_rechazos_filtrado = apply_filters_long(
            df_rechazos_long,
            fechas=sel_fechas,
            diametros=sel_diam or None,
            librajes=sel_lib or None,
            aceros=sel_acero or None,
            roscas=sel_rosca or None,
            turnos=sel_turno or None,
            maquinas=sel_maq or None,
        )

        if df_rechazos_filtrado.empty:
            st.info("No hay datos de pérdidas que coincidan con los filtros seleccionados.")
        else:
            tab_ti, tab_rechazos = st.tabs(["Tiempos Improductivos (Horas)", "Rechazos (Piezas)"])

            with tab_ti:
                # --- TI (dinámicas no mapeadas) ---
                df_ti_dynamics = compute_ti_unmapped_dynamics(df_rechazos_filtrado)
                ti_kpis = build_ti_kpis(df_ti_dynamics)

                if ti_kpis['total_ti_horas'] > 0:
                    st.metric(
                        label="Total Horas Improductivas (no mapeadas)",
                        value=f"{ti_kpis['total_ti_horas']:.2f} hrs",
                        help="Suma de valores de columnas numéricas que no pudieron ser mapeadas a una clave de rechazo."
                    )
                    with st.expander("Ver desglose de TI por máquina y fuente"):
                        st.write("**Contribución por Máquina**")
                        st.dataframe(ti_kpis['ti_por_maquina'])
                        
                        # Tooltip con SourceCol
                        df_source_cols = df_ti_dynamics.groupby('SourceCol').agg(TI_Horas=('TI_Horas', 'sum')).sort_values('TI_Horas', ascending=False).reset_index()
                        st.write("**Contribución por columna de origen (`SourceCol`)**")
                        st.dataframe(df_source_cols)
                else:
                    st.success("No se encontraron Tiempos Improductivos (dinámicas no mapeadas) con los filtros actuales.")

            with tab_rechazos:
                # Calcular y mostrar resultados en contenedores hermanos
                df_top_rechazos, meta = compute_top_claves_rechazo(
                    df_rechazos_filtrado, top_n=top_n_rechazo, desglose=desglose_rechazo
                )

                # Contenedor 1: Gráfica
                with st.expander("📊 Gráfica de Top Claves de Rechazo", expanded=True):
                    fig_rechazos = build_top_claves_rechazo_figure(df_top_rechazos, desglose_rechazo)
                    st.plotly_chart(fig_rechazos, use_container_width=True)
                    st.caption(f"Total de piezas rechazadas (con filtros aplicados): {int(meta['total_pzas']):,}")

                # Contenedor 2: Tabla
                with st.expander("Ver tabla de Top Rechazos"):
                    st.dataframe(df_top_rechazos)

                    csv_data_top = export_df_to_csv_bytes(df_top_rechazos)
                    st.download_button(
                        label="📥 Exportar Top Claves (CSV)",
                        data=csv_data_top,
                        file_name=f"top_claves_rechazo_lineas_{date.today()}.csv",
                        mime="text/csv",
                        key="lineas_export_top_claves"
                    )

                # --- Integración del Panel de Confiabilidad Unificado ---

                config_panel = {
                    "dominio": DOM_LINEAS, # Pasa la constante canónica
                    "threshold_high_actual": threshold_high_rechazo,
                    "top_n": top_n_rechazo, # Reutiliza el slider del top_n de rechazos
                    "t_values": [80, 85, 88, 90, 92], # Lista fija para el modo Gourmet
                }
                
                render_confiabilidad_gourmet(df_rechazos_filtrado, config_panel)

    # --- 7. Debug (toggle) ---
    if debug_mode:
        with st.expander("Datos Crudos y Estadísticas (Debug)"):
            # --- Bloque de Diagnóstico de Columnas ---
            with st.expander("Diagnóstico de columnas (Líneas)"):
                st.write("Este bloque analiza las columnas del archivo Excel **antes** de cualquier procesamiento.")
                
                # Ejecutar diagnóstico en el DataFrame crudo
                diag_results = diagnose_columns(df_raw)

                st.write("**1. Columnas originales encontradas en el archivo:**")
                st.write(diag_results["original_cols"])

                st.write("**2. Mapeo de renombrado aplicado:**")
                if diag_results["mapping_applied"]:
                    st.json(diag_results["mapping_applied"])
                else:
                    st.info("No se aplicó ningún renombrado. Las columnas ya tenían nombres canónicos o no se encontraron sinónimos.")

                st.write("**3. Estado de columnas esenciales DESPUÉS del mapeo:**")
                if diag_results["essential_missing_after_mapping"]:
                    st.warning(f"Faltan las siguientes columnas esenciales: `{diag_results['essential_missing_after_mapping']}`")
                else:
                    st.success("Todas las columnas esenciales están presentes.")
                
                st.write("**4. Muestra de datos crudos (primeras 3 filas):**")
                st.dataframe(df_raw.head(3))

                if not df_prepared.empty:
                    st.write("**5. Columnas finales en el DataFrame preparado:**")
                    st.write(df_prepared.columns.tolist())
                    
                    st.write("**6. Diagnóstico de la columna 'Fecha':**")
                    st.write(f"- Tipo de dato (dtype): `{df_prepared['Fecha'].dtype}`")
                    st.write("- Primeros 5 valores (como string):")
                    st.code(df_prepared["Fecha"].head().astype(str).to_string())

            st.subheader("Datos Filtrados (200 primeras filas)")
            st.dataframe(df_filtered.head(200))

            st.subheader("Estadísticas de Depuración (sobre datos filtrados)")
            debug_stats = compute_debug_stats(df_filtered)
            st.json(debug_stats)

            st.subheader("DataFrame de Productos Críticos (para la gráfica)")
            st.dataframe(df_critico)

# --- Wrapper para compatibilidad con el antiguo main.py ---
def mostrar_dashboard_lineas(df_produccion=None, df_eventos=None):
    """
    Punto de entrada para el dashboard de Líneas, compatible con main.py.
    Ignora los argumentos y llama a la función de renderizado principal.
    """
    render_lineas_dashboard()

# --- 8. API pública ---
__all__ = ["render_lineas_dashboard", "mostrar_dashboard_lineas"]

# --- 9. Prueba mínima ---
if __name__ == "__main__":
    st.set_page_config(
        page_title="Test - Dashboard Líneas",
        layout="wide"
    )
    render_lineas_dashboard()

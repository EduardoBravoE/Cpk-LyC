# -*- coding: utf-8 -*-
"""
UTILS/coples_dashboard.py
Dashboard de Coples refactorizado para usar UTILS.insights.

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
from UTILS.common import (
    DOM_COPLES,
    cargar_coples_con_manifiesto,
    render_manifest,
    cargar_rechazos_long_area,
    cargar_rechazos_con_cache_inteligente,
)
from UTILS.confiabilidad_panel import render_confiabilidad_panel
from UTILS.insights import (
    prepare_df_for_analysis,
    apply_filters,
    apply_filters_long,
    compute_producto_critico,
    compute_top_claves_rechazo,
    compute_debug_stats,
    build_producto_critico_figure,
    build_top_claves_rechazo_figure,
    compute_ti_unmapped_dynamics,
    build_ti_kpis,
    export_df_to_csv_bytes,
)


def render_coples_dashboard(datos_rechazos_cache: dict[str, pd.DataFrame] = None):
    """
    Renderiza el dashboard completo para el área de Coples.

    Esta función no recibe argumentos. Carga y procesa los datos internamente,
    y renderiza los componentes de la UI de Streamlit.

    Args:
        datos_rechazos_cache: Datos de rechazos precargados (opcional)
    """
    st.header("Dashboard de Coples - Análisis de Producción y Rechazos")

    # --- 1. Carga y preparación de datos ---
    with st.spinner("Cargando datos de producción..."):
        # Usamos la función con manifiesto para obtener más información de depuración.
        df_raw, manifest_df = cargar_coples_con_manifiesto(recursive=False)

    if df_raw.empty:
        st.error(
            "No se encontraron datos de producción para Coples. Revisa el manifiesto de carga de arriba "
            "y verifica que los archivos Excel estén en la carpeta 'DATOS/COPLES' en tu repositorio de GitHub."
        )
        return
    else:
        df_prepared = prepare_df_for_analysis(df_raw)
        st.success(f"✅ Datos de COPLES listos: {len(df_raw):,} registros")

    # --- 2. Sidebar de filtros ---
    st.sidebar.title("Filtros - Coples")

    # Botón de recarga
    if st.sidebar.button("Recargar datos (limpiar caché)", key="coples_recargar"):
        st.cache_data.clear()

    # Filtros de fecha
    s_fecha = pd.to_datetime(df_prepared["Fecha"], format="%d/%m/%Y", errors="coerce")
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
        key="coples_fechas",
    )
    # Asegurar que siempre sea una tupla de 2
    if len(sel_fechas) != 2:
        sel_fechas = (min_date, max_date)

    # Filtros de producto y proceso
    def get_options(col_name):
        return sorted(df_prepared[col_name].dropna().unique())

    sel_diam = st.sidebar.multiselect(
        "Diámetro", get_options("Diametro"), key="coples_diam"
    )
    sel_lib = st.sidebar.multiselect(
        "Libraje", get_options("Libraje"), key="coples_lib"
    )
    sel_acero = st.sidebar.multiselect(
        "Acero", get_options("Acero"), key="coples_acero"
    )
    sel_rosca = st.sidebar.multiselect(
        "Rosca", get_options("Rosca"), key="coples_rosca"
    )
    sel_turno = st.sidebar.multiselect(
        "Turno", get_options("Turno"), key="coples_turno"
    )
    sel_maq = st.sidebar.multiselect(
        "Máquina", get_options("Maquina"), key="coples_maq"
    )

    # Controles de visualización
    st.sidebar.title("Controles de Gráfica")
    desglose_map = {"Global": None, "Por Turno": "Turno", "Por Máquina": "Maquina"}
    sel_desglose_label = st.sidebar.selectbox(
        "Desglose", list(desglose_map.keys()), key="coples_desglose"
    )
    detail_by_value = desglose_map[sel_desglose_label]

    top_n_value = st.sidebar.slider(
        "Top N", min_value=5, max_value=30, value=10, step=1, key="coples_top_n"
    )

    # Debug toggle
    debug_mode = st.sidebar.checkbox(
        "Mostrar datos crudos (debug)", key="coples_debug_cb"
    )

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
        dominio=DOM_COPLES,
    )

    if df_filtered.empty:
        st.info("No se encontraron resultados con los filtros seleccionados.")
        return

    # --- 4. Cálculo de "Producto Crítico" ---
    df_critico = compute_producto_critico(
        df_filtered, top_n=top_n_value, detail_by=detail_by_value
    )

    # --- Debug info sobre datos ---
    if debug_mode:
        st.subheader("🔍 Debug: Análisis de Datos de COPLES")
        
        # Debug 1: Datos después de la carga inicial
        st.write("**1. Datos después de carga inicial:**")
        st.write(f"- Total registros cargados: {len(df_raw):,}")
        if not df_raw.empty and "Area" in df_raw.columns:
            area_counts_raw = df_raw["Area"].value_counts().sort_index()
            st.write(f"- Distribución por Area (crudo): {dict(area_counts_raw)}")
        
        # Debug 2: Datos después de preparación
        st.write("**2. Datos después de preparación:**")
        st.write(f"- Total registros preparados: {len(df_prepared):,}")
        if not df_prepared.empty:
            area_counts_prep = df_prepared["AreaLabel"].value_counts()
            st.write(f"- Distribución por AreaLabel: {dict(area_counts_prep)}")
            productos_unicos = df_prepared["ComboProducto"].nunique()
            st.write(f"- COPLES: {len(df_prepared):,} registros, {productos_unicos} productos únicos")
        
        # Debug 3: Datos después de filtros
        st.write("**3. Datos después de aplicar filtros:**")
        st.write(f"- Total registros filtrados: {len(df_filtered):,}")
        if not df_filtered.empty:
            area_counts_filt = df_filtered["AreaLabel"].value_counts()
            st.write(f"- Distribución por AreaLabel después de filtros: {dict(area_counts_filt)}")
            productos_unicos_filt = df_filtered["ComboProducto"].nunique()
            st.write(f"- COPLES: {len(df_filtered):,} registros, {productos_unicos_filt} productos únicos")
            productos_list = sorted(df_filtered["ComboProducto"].unique())
            st.write(f"  - Productos: {productos_list[:5]}..." if len(productos_list) > 5 else f"  - Productos: {productos_list}")

        # Debug 4: Datos del producto crítico
        st.write("**4. Datos después del cálculo de Producto Crítico:**")
        st.write(f"- TOP {top_n_value} productos calculados: {len(df_critico):,}")
        if not df_critico.empty:
            productos_criticos = sorted(df_critico["ComboProducto"].unique())
            st.write(f"- Productos críticos: {productos_criticos}")

    # Crear dos secciones principales usando tabs
    tab1, tab2 = st.tabs(["📊 Análisis de Producción", "⚠️ Análisis de Rechazos"])

    # ===========================================
    # SECCIÓN 1: ANÁLISIS DE PRODUCCIÓN
    # ===========================================
    with tab1:
        st.subheader("📈 Productos Críticos - Análisis de Producción")

        # Mostramos el manifiesto en un expander dentro de esta sección
        with st.expander("Ver Manifiesto de Carga de Archivos"):
            render_manifest(manifest_df, title="Archivos de Coples Encontrados")

        # --- Gráfica de Producto Crítico ---
        if df_critico.empty:
            st.info(
                "No hay datos de productos críticos para mostrar con los filtros actuales."
            )
        else:
            # CORRECCIÓN: Eliminar checkbox cosmético innecesario
            # COPLES solo tiene una área, no necesita filtro de área
            st.markdown("### 📊 Productos Críticos de COPLES")
            
            # Usar todos los datos de COPLES directamente
            df_critico_filtered = df_critico.copy()

            with st.container():
                # SOLUCIÓN: Agrupar por ComboProducto para evitar múltiples puntos rojos
                # Si hay desglose por Turno o Máquina, necesitamos re-agrupar para la visualización
                if len(df_critico_filtered) > 0:
                    # Verificar si tenemos la columna RechazoPzas para calcular correctamente
                    if "RechazoPzas" in df_critico_filtered.columns:
                        df_for_chart = (
                            df_critico_filtered.groupby("ComboProducto")
                            .agg({
                                "Produccion": "sum",
                                "RechazoPzas": "sum"
                            })
                            .reset_index()
                        )
                        # Calcular IndiceRechazo real del período: Total Rechazos / Total Producción
                        df_for_chart["IndiceRechazo"] = df_for_chart.apply(
                            lambda row: row["RechazoPzas"] / row["Produccion"] 
                            if row["Produccion"] > 0 else 0.0, axis=1
                        )
                    else:
                        # Si no tenemos RechazoPzas, intentar reconstruir desde IndiceRechazo
                        df_for_chart = df_critico_filtered.groupby("ComboProducto").apply(
                            lambda group: pd.Series({
                                "Produccion": group["Produccion"].sum(),
                                "RechazoPzas": (group["IndiceRechazo"] * group["Produccion"]).sum(),
                            })
                        ).reset_index()
                        # Calcular IndiceRechazo real del período total
                        df_for_chart["IndiceRechazo"] = df_for_chart.apply(
                            lambda row: row["RechazoPzas"] / row["Produccion"] 
                            if row["Produccion"] > 0 else 0.0, axis=1
                        )
                else:
                    df_for_chart = df_critico_filtered.copy()
                
                fig = build_producto_critico_figure(
                    df_for_chart,
                    title=f"Top {top_n_value} Productos Críticos - Coples",
                )
                st.plotly_chart(fig, width="stretch")

                csv_data = export_df_to_csv_bytes(df_critico_filtered)
                st.download_button(
                    label="📥 Exportar Producto Crítico (CSV)",
                    data=csv_data,
                    file_name=f"producto_critico_coples_{date.today()}.csv",
                    mime="text/csv",
                    key="coples_export_critico",
                )

    # ===========================================
    # SECCIÓN 2: ANÁLISIS DE RECHAZOS
    # ===========================================
    with tab2:
        st.subheader("🔍 Análisis de Rechazos y Mapeo")

        # Filtro de áreas específico para la sección de rechazos (mismo que en producción)
        st.markdown("### 🎯 Filtro de Área de Producción")
        col1, col2, col3 = st.columns(3)

        with col1:
            coples_rech_selected = st.checkbox(
                "Coples",
                value=True,
                key="coples_rechazo_area_coples",
                help="Incluir datos del área de Coples",
            )
        with col2:
            st.empty()  # Espacio vacío para mantener consistencia visual
        with col3:
            st.empty()  # Espacio vacío para mantener consistencia visual

        # Convertir selección a códigos de área
        selected_areas_rech = []
        if coples_rech_selected:
            selected_areas_rech.append(4)  # Coples

        # Controles para el análisis de rechazos
        c1, c2, c3 = st.columns(3)
        top_n_rechazo = c1.slider(
            "Top N Claves de Rechazo", 5, 25, 10, key="coples_rechazo_top_n"
        )
        desglose_rechazo = c2.selectbox(
            "Desglose de Rechazos por",
            ["Global", "Máquina", "Turno", "Turno + Máquina"],
            key="coples_rechazo_desglose",
        )
        threshold_high_rechazo = c3.slider(
            "Umbral de Similitud Alto (%)",
            70,
            100,
            92,
            key="coples_rechazo_threshold_high",
            help="Puntaje para aceptar un match automáticamente.",
        )
        threshold_low_rechazo = max(70.0, threshold_high_rechazo - 10.0)

        # Carga y filtrado de datos de pérdidas (contiene tanto rechazos como TI potenciales)
        df_rechazos_long = cargar_rechazos_con_cache_inteligente(
            dominio=DOM_COPLES,
            threshold_high=threshold_high_rechazo,
            threshold_low=threshold_low_rechazo,
            datos_rechazos_cache=datos_rechazos_cache,
        )

        if df_rechazos_long.empty:
            st.info(
                "No se encontraron datos de rechazo o TI para analizar con el umbral actual."
            )
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

            # Aplicar filtro de áreas a los rechazos si no está seleccionada
            if len(selected_areas_rech) < 1 and "Area" in df_rechazos_filtrado.columns:
                try:
                    df_rechazos_filtrado = df_rechazos_filtrado[
                        df_rechazos_filtrado["Area"].isin(selected_areas_rech)
                    ]
                except Exception as e:
                    st.error(f"❌ Error al aplicar filtro de áreas en coples: {e}")
                    st.write("🔍 Debug: Continuando sin filtro de áreas")

            if df_rechazos_filtrado.empty:
                st.info(
                    "No hay datos de pérdidas que coincidan con los filtros seleccionados."
                )
            else:
                # Gráfico de Top Claves de Rechazo
                df_top_rechazos, meta = compute_top_claves_rechazo(
                    df_rechazos_filtrado, top_n=top_n_rechazo, desglose=desglose_rechazo
                )

                # Gráfico de Top Claves de Rechazo (ocupando todo el ancho)
                st.subheader("📊 Top Claves de Rechazo")
                fig_rechazos = build_top_claves_rechazo_figure(
                    df_top_rechazos, desglose_rechazo
                )
                st.plotly_chart(fig_rechazos, width="stretch")
                st.caption(
                    f"Total de piezas rechazadas (con filtros activos): {int(meta['total_pzas']):,}"
                )

                # Tabla de Top Rechazos (en expander debajo)
                with st.expander(
                    "📋 Ver Tabla Detallada de Top Rechazos", expanded=False
                ):
                    st.dataframe(df_top_rechazos)

                    csv_data_top = export_df_to_csv_bytes(df_top_rechazos)
                    st.download_button(
                        label="📥 Exportar Top Claves (CSV)",
                        data=csv_data_top,
                        file_name=f"top_claves_rechazo_coples_{date.today()}.csv",
                        mime="text/csv",
                        key="coples_export_top_claves",
                    )

                # --- Panel de Confiabilidad de Mapeo de Claves de Rechazo ---
                st.markdown("---")
                config_panel = {
                    "dominio": DOM_COPLES,  # Pasa la constante canónica
                    "threshold_high_actual": threshold_high_rechazo,
                    "top_n": top_n_rechazo,  # Reutiliza el slider del top_n de rechazos
                    "t_values": [80, 85, 88, 90, 92],  # Lista fija para el modo Gourmet
                }
                render_confiabilidad_panel(df_rechazos_filtrado, config_panel)

    # --- Debug (toggle) ---
    if debug_mode:
        with st.expander("Datos Crudos y Estadísticas (Debug)"):
            st.subheader("Datos Filtrados (200 primeras filas)")
            st.dataframe(df_filtered.head(200))

            st.subheader("Estadísticas de Depuración (sobre datos filtrados)")
            debug_stats = compute_debug_stats(df_filtered)
            st.json(debug_stats)

            st.subheader("DataFrame de Productos Críticos (para la gráfica)")
            st.dataframe(df_critico)


# --- Wrapper para compatibilidad con main.py ---
def mostrar_dashboard_coples(df_produccion=None, df_eventos=None):
    """
    Punto de entrada para el dashboard de Coples, compatible con main.py.
    Ignora los argumentos y llama a la función de renderizado principal.
    """
    render_coples_dashboard()


# --- 8. API pública ---
__all__ = ["render_coples_dashboard", "mostrar_dashboard_coples"]

# --- 9. Prueba mínima ---
if __name__ == "__main__":
    st.set_page_config(page_title="Test - Dashboard Coples", layout="wide")
    render_coples_dashboard()

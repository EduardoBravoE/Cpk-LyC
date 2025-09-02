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
from UTILS.common import (
    DOM_LINEAS,
    cargar_lineas_con_manifiesto,
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
    build_producto_critico_figure,
    build_top_claves_rechazo_figure,
    compute_debug_stats,
    diagnose_columns,
    compute_ti_unmapped_dynamics,
    build_ti_kpis,
    export_df_to_csv_bytes,
    )

# Toggle de depuración en UI: cambiar a True solo durante desarrollo local si se necesita
DEBUG_UI = False


def _show_debug(msg: str):
    """Muestra mensajes de debug en la UI solo si DEBUG_UI == True.

    Esto evita que los usuarios vean mensajes de diagnóstico en producción.
    """
    if DEBUG_UI:
        st.write(msg)


def render_lineas_dashboard(datos_rechazos_cache: dict[str, pd.DataFrame] = None):
    """
    Renderiza el dashboard completo para el área de Líneas.

    Esta función no recibe argumentos. Carga y procesa los datos internamente,
    y renderiza los componentes de la UI de Streamlit.

    Args:
        datos_rechazos_cache: Datos de rechazos precargados (opcional)
    """
    st.header("Dashboard de Líneas - Análisis de Producción y Rechazos")

    # --- 1. Carga y preparación de datos ---
    with st.spinner("Cargando datos de producción..."):
        # Usamos la función con manifiesto para obtener más información de depuración.
        # Esta función devuelve los datos y una tabla con el estado de la carga de cada archivo.
        df_raw, manifest_df = cargar_lineas_con_manifiesto(recursive=False)

    if df_raw.empty:
        st.error(
            "No se encontraron datos de producción para Líneas. Revisa el manifiesto de carga de arriba "
            "y verifica que los archivos Excel estén en la carpeta 'DATOS/LINEAS' en tu repositorio de GitHub."
        )
        return
    else:
        df_prepared = prepare_df_for_analysis(df_raw)
        st.success(f"✅ Datos de LÍNEAS listos: {len(df_raw):,} registros")

    # --- 2. Sidebar de filtros ---
    st.sidebar.title("Filtros - Líneas")

    # Botón de recarga
    if st.sidebar.button("Recargar datos (limpiar caché)", key="lineas_recargar"):
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

    # --- 4. Cálculo de "Producto Crítico" ---
    df_critico = compute_producto_critico(
        df_filtered,
        top_n=top_n_value,
        detail_by=detail_by_value
    )

    # Crear dos secciones principales usando tabs
    tab1, tab2 = st.tabs(["📊 Análisis de Producción", "⚠️ Análisis de Rechazos"])

    # ===========================================
    # SECCIÓN 1: ANÁLISIS DE PRODUCCIÓN
    # ===========================================
    with tab1:
        st.subheader("📈 Productos Críticos - Análisis de Producción")

        # Mostramos el manifiesto en un expander dentro de esta sección
        with st.expander("Ver Manifiesto de Carga de Archivos"):
            render_manifest(manifest_df, title="Archivos de Líneas Encontrados")

        # --- Gráfica de Producto Crítico ---
        if df_critico.empty:
            st.info("No hay datos de productos críticos para mostrar con los filtros actuales.")
        else:
            # Filtro de líneas específico para el dashboard de líneas
            st.markdown("### 🎯 Filtro de Líneas de Producción")
            col1, col2, col3 = st.columns(3)

            with col1:
                linea_a = st.checkbox("L-A", value=True, key="lineas_linea_a",
                                    help="Incluir datos de la línea L-A")
            with col2:
                linea_b = st.checkbox("L-B", value=True, key="lineas_linea_b",
                                    help="Incluir datos de la línea L-B")
            with col3:
                linea_c = st.checkbox("L-C", value=True, key="lineas_linea_c",
                                    help="Incluir datos de la línea L-C")

            # Convertir selección a códigos de área
            selected_lineas = []
            if linea_a: selected_lineas.append(1)  # L-A
            if linea_b: selected_lineas.append(2)  # L-B
            if linea_c: selected_lineas.append(3)  # L-C

            # Aplicar filtro de líneas si no están todas seleccionadas
            if len(selected_lineas) < 3:
                df_critico_filtered = df_critico[df_critico["Area"].isin(selected_lineas)]
                # Nota: El filtro de rechazos se aplicará más tarde en la sección de rechazos
            else:
                df_critico_filtered = df_critico

            with st.container():
                fig = build_producto_critico_figure(
                    df_critico_filtered,
                    title=f"Top {top_n_value} Productos Críticos - Líneas"
                )
                st.plotly_chart(fig, use_container_width=True)

                csv_data = export_df_to_csv_bytes(df_critico_filtered)
                st.download_button(
                    label="📥 Exportar Producto Crítico (CSV)",
                    data=csv_data,
                    file_name=f"producto_critico_lineas_{date.today()}.csv",
                    mime="text/csv",
                    key="lineas_export_critico"
                )

    # ===========================================
    # SECCIÓN 2: ANÁLISIS DE RECHAZOS
    # ===========================================
    with tab2:
        st.subheader("🔍 Análisis de Rechazos y Mapeo")

        # Filtro de líneas específico para la sección de rechazos (mismo que en producción)
        st.markdown("### 🎯 Filtro de Líneas de Producción")
        col1, col2, col3 = st.columns(3)

        with col1:
            linea_a_rech = st.checkbox("L-A", value=True, key="lineas_rechazo_linea_a",
                                     help="Incluir datos de la línea L-A")
        with col2:
            linea_b_rech = st.checkbox("L-B", value=True, key="lineas_rechazo_linea_b",
                                     help="Incluir datos de la línea L-B")
        with col3:
            linea_c_rech = st.checkbox("L-C", value=True, key="lineas_rechazo_linea_c",
                                     help="Incluir datos de la línea L-C")

        # Convertir selección a códigos de área
        selected_lineas_rech = []
        if linea_a_rech: selected_lineas_rech.append(1)  # L-A
        if linea_b_rech: selected_lineas_rech.append(2)  # L-B
        if linea_c_rech: selected_lineas_rech.append(3)  # L-C

        # Controles para el análisis de rechazos
        c1, c2, c3 = st.columns(3)
        top_n_rechazo = c1.slider("Top N Claves de Rechazo", 5, 25, 10, key="lineas_rechazo_top_n")
        desglose_rechazo = c2.selectbox("Desglose de Rechazos por", ["Global", "Máquina", "Turno"], key="lineas_rechazo_desglose")
        threshold_high_rechazo = c3.slider("Umbral de Similitud Alto (%)", 70, 100, 92, key="lineas_rechazo_threshold_high", help="Puntaje para aceptar un match automáticamente.")
        threshold_low_rechazo = max(70.0, threshold_high_rechazo - 10.0)

        # Carga y filtrado de datos de pérdidas (contiene tanto rechazos como TI potenciales)
        df_rechazos_long = cargar_rechazos_con_cache_inteligente(
            dominio=DOM_LINEAS,
            threshold_high=threshold_high_rechazo,
            threshold_low=threshold_low_rechazo,
            datos_rechazos_cache=datos_rechazos_cache
        )

        _show_debug(f"🔍 Debug: df_rechazos_long.shape = {df_rechazos_long.shape}")
        if "Area" in df_rechazos_long.columns:
            _show_debug(f"🔍 Debug: Áreas en rechazos = {sorted(df_rechazos_long['Area'].unique())}")
        _show_debug(f"🔍 Debug: selected_lineas_rech = {selected_lineas_rech}")

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

            # Aplicar filtro de líneas a los rechazos si no están todas seleccionadas
            if len(selected_lineas_rech) < 3 and "Area" in df_rechazos_filtrado.columns:
                try:
                    _show_debug(f"🔍 Debug: Aplicando filtro de líneas. Antes: {len(df_rechazos_filtrado)} registros")
                    _show_debug(f"🔍 Debug: Filtrando por áreas: {selected_lineas_rech}")

                    # Convertir áreas a enteros para comparación consistente
                    df_rechazos_filtrado_copy = df_rechazos_filtrado.copy()
                    df_rechazos_filtrado_copy["Area"] = pd.to_numeric(df_rechazos_filtrado_copy["Area"], errors='coerce').fillna(-1).astype(int)

                    area_counts = df_rechazos_filtrado_copy["Area"].value_counts()
                    _show_debug(f"🔍 Debug: Distribución de áreas en datos: {dict(area_counts)}")

                    df_rechazos_filtrado = df_rechazos_filtrado_copy[df_rechazos_filtrado_copy["Area"].isin(selected_lineas_rech)]
                    _show_debug(f"🔍 Debug: Después del filtro: {len(df_rechazos_filtrado)} registros")
                except Exception as e:
                    st.error(f"❌ Error al aplicar filtro de áreas: {e}")
                    st.write("🔍 Debug: Continuando sin filtro de áreas")
            else:
                _show_debug(f"🔍 Debug: No se aplica filtro de líneas (todas seleccionadas o sin columna Area)")
                if "Area" in df_rechazos_filtrado.columns:
                    try:
                        # Convertir áreas a enteros para mostrar distribución
                        df_temp = df_rechazos_filtrado.copy()
                        df_temp["Area"] = pd.to_numeric(df_temp["Area"], errors='coerce').fillna(-1).astype(int)
                        area_counts = df_temp["Area"].value_counts()
                        _show_debug(f"🔍 Debug: Distribución de áreas en datos: {dict(area_counts)}")
                    except Exception as e:
                        _show_debug(f"🔍 Debug: Error al mostrar distribución de áreas: {e}")

            _show_debug(f"🔍 Debug: df_rechazos_filtrado.shape final = {df_rechazos_filtrado.shape}")

            if df_rechazos_filtrado.empty:
                st.info("No hay datos de pérdidas que coincidan con los filtros seleccionados.")
            else:
                try:
                    # Gráfico de Top Claves de Rechazo
                    _show_debug(f"🔍 Debug: Llamando compute_top_claves_rechazo con {len(df_rechazos_filtrado)} registros")
                    _show_debug(f"🔍 Debug: Columnas disponibles: {list(df_rechazos_filtrado.columns)}")

                    df_top_rechazos, meta = compute_top_claves_rechazo(
                        df_rechazos_filtrado, top_n=top_n_rechazo, desglose=desglose_rechazo
                    )
                    _show_debug(f"🔍 Debug: compute_top_claves_rechazo retornó {len(df_top_rechazos)} registros")
                    _show_debug(f"🔍 Debug: df_top_rechazos columnas: {list(df_top_rechazos.columns) if not df_top_rechazos.empty else 'VACÍO'}")
                    _show_debug(f"🔍 Debug: meta = {meta}")

                    if df_top_rechazos.empty:
                        st.warning("⚠️ No se encontraron claves de rechazo para mostrar")
                    else:
                        # Gráfico de Top Claves de Rechazo (ocupando todo el ancho)
                        st.subheader("📊 Top Claves de Rechazo")
                        _show_debug(f"🔍 Debug: Llamando build_top_claves_rechazo_figure con {len(df_top_rechazos)} registros")
                        _show_debug(f"🔍 Debug: df_top_rechazos preview:")
                        if DEBUG_UI:
                            st.dataframe(df_top_rechazos.head())

                        fig_rechazos = build_top_claves_rechazo_figure(df_top_rechazos, desglose_rechazo)
                        _show_debug(f"🔍 Debug: build_top_claves_rechazo_figure retornó: {type(fig_rechazos)}")

                        # Mostrar estadísticas básicas de Pzas para diagnóstico
                        try:
                            _show_debug(f"🔍 Debug: Pzas describe: {df_top_rechazos['Pzas'].describe().to_dict()}")
                        except Exception:
                            _show_debug("🔍 Debug: No se pudo calcular describe() de Pzas")

                        # Fallback: si la figura es None o no tiene trazas, construir un bar sencillo
                        has_traces = bool(fig_rechazos and hasattr(fig_rechazos, 'data') and len(fig_rechazos.data) > 0)
                        if not has_traces:
                            # Construir silent fallback sin mostrar warnings/errores al usuario
                            try:
                                if 'DescripcionCatalogo' in df_top_rechazos.columns:
                                    labels = (
                                        df_top_rechazos['ClaveCatalogo'].astype(str)
                                        + ' - '
                                        + df_top_rechazos['DescripcionCatalogo'].astype(str)
                                    )
                                else:
                                    labels = df_top_rechazos['ClaveCatalogo'].astype(str)

                                import plotly.graph_objects as _go

                                fallback_fig = _go.Figure()
                                fallback_fig.add_trace(
                                    _go.Bar(
                                        x=df_top_rechazos['Pzas'].astype(float),
                                        y=labels,
                                        orientation='h',
                                        marker=dict(color='crimson'),
                                        text=df_top_rechazos['Pzas'].astype(int),
                                        textposition='outside'
                                    )
                                )
                                fallback_fig.update_layout(title=f"Top Claves de Rechazo - {desglose_rechazo}", xaxis_title='Pzas', yaxis_title='Clave')
                                st.plotly_chart(fallback_fig, use_container_width=True)
                            except Exception:
                                # Si el fallback falla, renderizar la figura original si existe (silencioso)
                                if fig_rechazos:
                                    st.plotly_chart(fig_rechazos, use_container_width=True)
                        else:
                            _show_debug(f"🔍 Debug: Figura tiene {len(fig_rechazos.data)} trazas")
                            if hasattr(fig_rechazos, 'data') and len(fig_rechazos.data) > 0:
                                _show_debug(f"🔍 Debug: Primera traza tiene {len(fig_rechazos.data[0].x) if hasattr(fig_rechazos.data[0], 'x') else 'N/A'} puntos")
                            st.plotly_chart(fig_rechazos, use_container_width=True)

                        st.caption(f"Total de piezas rechazadas (con filtros aplicados): {int(meta['total_pzas']):,}")

                except Exception as e:
                    st.error(f"❌ Error al generar gráfico de rechazos: {e}")
                    st.write("🔍 Debug: Mostrando primeras 5 filas de df_rechazos_filtrado:")
                    st.dataframe(df_rechazos_filtrado.head())
                    if "ClaveCatalogo" in df_rechazos_filtrado.columns:
                        st.write(f"🔍 Debug: Valores únicos en ClaveCatalogo: {df_rechazos_filtrado['ClaveCatalogo'].unique()[:10]}")
                    if "Pzas" in df_rechazos_filtrado.columns:
                        st.write(f"🔍 Debug: Valores en Pzas: {df_rechazos_filtrado['Pzas'].describe()}")
                    st.write(f"🔍 Debug: Tipos de datos en columna Area: {df_rechazos_filtrado['Area'].dtype if 'Area' in df_rechazos_filtrado.columns else 'Columna Area no existe'}")

                # Tabla de Top Rechazos (en expander debajo)
                with st.expander("📋 Ver Tabla Detallada de Top Rechazos", expanded=False):
                    st.dataframe(df_top_rechazos)

                    csv_data_top = export_df_to_csv_bytes(df_top_rechazos)
                    st.download_button(
                        label="📥 Exportar Top Claves (CSV)",
                        data=csv_data_top,
                        file_name=f"top_claves_rechazo_lineas_{date.today()}.csv",
                        mime="text/csv",
                        key="lineas_export_top_claves"
                    )

                # --- Panel de Confiabilidad de Mapeo de Claves de Rechazo ---
                st.markdown("---")
                config_panel = {
                    "dominio": DOM_LINEAS, # Pasa la constante canónica
                    "threshold_high_actual": threshold_high_rechazo,
                    "top_n": top_n_rechazo, # Reutiliza el slider del top_n de rechazos
                    "t_values": [80, 85, 88, 90, 92], # Lista fija para el modo Gourmet
                }

                render_confiabilidad_panel(df_rechazos_filtrado, config_panel)

    # --- Debug (toggle) ---
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
            # st.subheader("Análisis de Productos Críticos")

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

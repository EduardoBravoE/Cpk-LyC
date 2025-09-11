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


def _normalize_area_col(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza la columna 'Area' a valores tipo 'L-A', 'L-B', 'L-C'.
    Si la columna es numérica, la convierte.
    Si no existe, la agrega vacía.
    """
    df = df.copy()
    if "Area" in df.columns:
        # Si es numérica, conviértela
        if pd.api.types.is_numeric_dtype(df["Area"]):
            df["Area"] = df["Area"].map({1: "L-A", 2: "L-B", 3: "L-C"})
        else:
            # Si ya es string, asegúrate que sean los valores esperados
            df["Area"] = df["Area"].replace({1: "L-A", 2: "L-B", 3: "L-C"})
    else:
        df["Area"] = None
    return df


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
        df_raw, manifest_df = cargar_lineas_con_manifiesto(recursive=False)

    if df_raw.empty:
        st.error(
            "No se encontraron datos de producción para Líneas. Revisa el manifiesto de carga de arriba "
            "y verifica que los archivos Excel estén en la carpeta 'DATOS/LINEAS' en tu repositorio de GitHub."
        )
        return
    else:
        df_prepared = prepare_df_for_analysis(df_raw)
        df_prepared = _normalize_area_col(df_prepared)  # <-- Normaliza aquí
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
        key="lineas_fechas",
    )
    # Asegurar que siempre sea una tupla de 2
    if len(sel_fechas) != 2:
        sel_fechas = (min_date, max_date)

    # Filtros de producto y proceso
    def get_options(col_name):
        return sorted(df_prepared[col_name].dropna().unique())

    sel_diam = st.sidebar.multiselect(
        "Diámetro", get_options("Diametro"), key="lineas_diam"
    )
    sel_lib = st.sidebar.multiselect(
        "Libraje", get_options("Libraje"), key="lineas_lib"
    )
    sel_acero = st.sidebar.multiselect(
        "Acero", get_options("Acero"), key="lineas_acero"
    )
    sel_rosca = st.sidebar.multiselect(
        "Rosca", get_options("Rosca"), key="lineas_rosca"
    )
    sel_turno = st.sidebar.multiselect(
        "Turno", get_options("Turno"), key="lineas_turno"
    )
    sel_maq = st.sidebar.multiselect(
        "Máquina", get_options("Maquina"), key="lineas_maq"
    )

    # Controles de visualización
    st.sidebar.title("Controles de Gráfica")
    desglose_map = {"Global": None, "Por Turno": "Turno", "Por Máquina": "Maquina"}
    sel_desglose_label = st.sidebar.selectbox(
        "Desglose", list(desglose_map.keys()), key="lineas_desglose"
    )
    detail_by_value = desglose_map[sel_desglose_label]

    top_n_value = st.sidebar.slider(
        "Top N", min_value=5, max_value=30, value=10, step=1, key="lineas_top_n"
    )

    # Debug toggle
    debug_mode = st.sidebar.checkbox(
        "Mostrar datos crudos (debug)", key="lineas_debug_cb"
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
        dominio=DOM_LINEAS,
    )

    if df_filtered.empty:
        st.info("No se encontraron resultados con los filtros seleccionados.")
        return

    # --- Debug info sobre datos ---
    if debug_mode:
        st.subheader("🔍 Debug: Análisis de Datos de LÍNEAS")
        
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
            
            # Mostrar productos únicos por área
            for area in ["L-A", "L-B", "L-C"]:
                area_data = df_prepared[df_prepared["AreaLabel"] == area]
                if not area_data.empty:
                    productos_unicos = area_data["ComboProducto"].nunique()
                    st.write(f"- {area}: {len(area_data):,} registros, {productos_unicos} productos únicos")
        
        # Debug 3: Datos después de filtros
        st.write("**3. Datos después de aplicar filtros:**")
        st.write(f"- Total registros filtrados: {len(df_filtered):,}")
        if not df_filtered.empty:
            area_counts_filt = df_filtered["AreaLabel"].value_counts()
            st.write(f"- Distribución por AreaLabel después de filtros: {dict(area_counts_filt)}")
            
            # Mostrar productos únicos por área después de filtros
            for area in ["L-A", "L-B", "L-C"]:
                area_data = df_filtered[df_filtered["AreaLabel"] == area]
                if not area_data.empty:
                    productos_unicos = area_data["ComboProducto"].nunique()
                    st.write(f"- {area}: {len(area_data):,} registros, {productos_unicos} productos únicos")
                    # Mostrar los productos únicos
                    productos_list = sorted(area_data["ComboProducto"].unique())
                    st.write(f"  - Productos: {productos_list[:5]}..." if len(productos_list) > 5 else f"  - Productos: {productos_list}")

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

        # --- Filtro de líneas específico para el dashboard de líneas ---
        st.markdown("### 🎯 Filtro de Líneas de Producción")
        col1, col2, col3 = st.columns(3)

        with col1:
            linea_a = st.checkbox(
                "L-A",
                value=True,
                key="lineas_linea_a",
                help="Incluir datos de la línea L-A",
            )
        with col2:
            linea_b = st.checkbox(
                "L-B",
                value=True,
                key="lineas_linea_b",
                help="Incluir datos de la línea L-B",
            )
        with col3:
            linea_c = st.checkbox(
                "L-C",
                value=True,
                key="lineas_linea_c",
                help="Incluir datos de la línea L-C",
            )

        # Determinar líneas seleccionadas para el cálculo
        selected_lineas = []
        if linea_a:
            selected_lineas.append("L-A")
        if linea_b:
            selected_lineas.append("L-B")
        if linea_c:
            selected_lineas.append("L-C")

        # --- Re-calcular Producto Crítico con filtro de líneas aplicado ---
        if selected_lineas:
            # Filtrar datos por las líneas seleccionadas ANTES de calcular TOP N
            df_filtered_by_lines = df_filtered[df_filtered["AreaLabel"].isin(selected_lineas)]
            
            # Calcular TOP N específico para las líneas seleccionadas
            df_critico = compute_producto_critico(
                df_filtered_by_lines, top_n=top_n_value, detail_by=detail_by_value
            )
            df_critico = _normalize_area_col(df_critico)  # <-- Normaliza aquí

            # --- Limpieza contundente de columnas numéricas ---
            for col in ["Produccion", "IndiceRechazo"]:
                if col in df_critico.columns:
                    df_critico[col] = pd.to_numeric(df_critico[col], errors="coerce")
            # Eliminar filas donde Produccion o IndiceRechazo sean nulos o no numéricos
            df_critico = df_critico.dropna(subset=["Produccion", "IndiceRechazo"])
            df_critico = df_critico[(df_critico["Produccion"].apply(lambda x: isinstance(x, (int, float)))) & (df_critico["IndiceRechazo"].apply(lambda x: isinstance(x, (int, float))))]
            
            df_critico_filtered = df_critico.copy()
            
            # Debug info específico para producto crítico
            if debug_mode:
                st.write("**4. Datos después del cálculo de Producto Crítico:**")
                st.write(f"- Líneas seleccionadas: {selected_lineas}")
                st.write(f"- Registros para cálculo: {len(df_filtered_by_lines):,}")
                st.write(f"- TOP {top_n_value} productos calculados: {len(df_critico_filtered):,}")
                if not df_critico_filtered.empty:
                    productos_criticos = sorted(df_critico_filtered["ComboProducto"].unique())
                    st.write(f"- Productos críticos: {productos_criticos}")
                    
                    # Mostrar distribución por área en el resultado final
                    if "Area" in df_critico_filtered.columns:
                        area_counts_critico = df_critico_filtered["Area"].value_counts()
                        st.write(f"- Distribución final por Area: {dict(area_counts_critico)}")
        else:
            # Si no hay líneas seleccionadas, mostrar mensaje
            st.warning("⚠️ Selecciona al menos una línea de producción para ver los productos críticos.")
            df_critico_filtered = pd.DataFrame()

        # --- Gráfica de Producto Crítico ---
        if df_critico_filtered.empty:
            if selected_lineas:
                st.info(
                    f"No hay datos de productos críticos para mostrar para las líneas seleccionadas: {', '.join(selected_lineas)}"
                )
            else:
                st.info("Selecciona al menos una línea de producción.")
        else:
                import traceback
                with st.container():
                    try:
                        # Forzar tipos y limpiar nulos antes de graficar
                        for col in ["Produccion", "IndiceRechazo"]:
                            if col in df_critico_filtered.columns:
                                df_critico_filtered[col] = pd.to_numeric(df_critico_filtered[col], errors="coerce")
                        # Eliminar filas donde Produccion o IndiceRechazo sean nulos o no numéricos
                        df_critico_filtered = df_critico_filtered.dropna(subset=["Produccion", "IndiceRechazo"])
                        # Opcional: eliminar filas donde Produccion o IndiceRechazo sean strings no convertidos
                        df_critico_filtered = df_critico_filtered[(df_critico_filtered["Produccion"].apply(lambda x: isinstance(x, (int, float)))) & (df_critico_filtered["IndiceRechazo"].apply(lambda x: isinstance(x, (int, float))))]
                        if "ComboProducto" in df_critico_filtered.columns:
                            df_critico_filtered["ComboProducto"] = df_critico_filtered["ComboProducto"].fillna("N/A").astype(str)
                        
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
                            title=f"Top {top_n_value} Productos Críticos - Líneas",
                        )
                        st.plotly_chart(fig, width="stretch")
                    except Exception as e:
                        # Mostrar error solo en modo debug
                        if DEBUG_UI:
                            st.code(traceback.format_exc())
                            st.dataframe(df_critico_filtered.head())
                    csv_data = export_df_to_csv_bytes(df_critico_filtered)
                    st.download_button(
                        label="📥 Exportar Producto Crítico (CSV)",
                        data=csv_data,
                        file_name=f"producto_critico_lineas_{date.today()}.csv",
                        mime="text/csv",
                        key="lineas_export_critico",
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
            linea_a_rech = st.checkbox(
                "L-A",
                value=True,
                key="lineas_rechazo_linea_a",
                help="Incluir datos de la línea L-A",
            )
        with col2:
            linea_b_rech = st.checkbox(
                "L-B",
                value=True,
                key="lineas_rechazo_linea_b",
                help="Incluir datos de la línea L-B",
            )
        with col3:
            linea_c_rech = st.checkbox(
                "L-C",
                value=True,
                key="lineas_rechazo_linea_c",
                help="Incluir datos de la línea L-C",
            )

        selected_lineas_rech = []
        if linea_a_rech:
            selected_lineas_rech.append("L-A")
        if linea_b_rech:
            selected_lineas_rech.append("L-B")
        if linea_c_rech:
            selected_lineas_rech.append("L-C")

        # Controles para el análisis de rechazos
        c1, c2, c3 = st.columns(3)
        top_n_rechazo = c1.slider(
            "Top N Claves de Rechazo", 5, 25, 10, key="lineas_rechazo_top_n"
        )
        desglose_rechazo = c2.selectbox(
            "Desglose de Rechazos por",
            ["Global", "Máquina", "Turno", "Turno + Máquina"],
            key="lineas_rechazo_desglose",
        )
        threshold_high_rechazo = c3.slider(
            "Umbral de Similitud Alto (%)",
            70,
            100,
            92,
            key="lineas_rechazo_threshold_high",
            help="Puntaje para aceptar un match automáticamente.",
        )
        threshold_low_rechazo = max(70.0, threshold_high_rechazo - 10.0)

        # Carga y filtrado de datos de pérdidas (contiene tanto rechazos como TI potenciales)
        with st.spinner("Cargando rechazos para LINEAS con umbrales personalizados..."):
            try:
                # Agregar debug para identificar el problema
                if debug_mode:
                    st.write("🔄 **Debug:** Iniciando carga de rechazos...")
                    st.write(f"- Dominio: {DOM_LINEAS}")
                    st.write(f"- Threshold High: {threshold_high_rechazo}")
                    st.write(f"- Threshold Low: {threshold_low_rechazo}")
                    st.write(f"- Cache disponible: {datos_rechazos_cache is not None}")
                
                # Intentar usar cache primero si está disponible
                if (datos_rechazos_cache and 
                    "lineas" in datos_rechazos_cache and 
                    not datos_rechazos_cache["lineas"].empty):
                    df_rechazos_long = datos_rechazos_cache["lineas"]
                    if debug_mode:
                        st.write("✅ **Debug:** Usando datos de rechazos desde cache")
                else:
                    # Si no hay cache, intentar carga normal
                    df_rechazos_long = cargar_rechazos_con_cache_inteligente(
                        dominio=DOM_LINEAS,
                        threshold_high=threshold_high_rechazo,
                        threshold_low=threshold_low_rechazo,
                        datos_rechazos_cache=datos_rechazos_cache,
                    )
                
                if debug_mode:
                    st.write(f"✅ **Debug:** Rechazos cargados - {len(df_rechazos_long):,} registros")
                
                df_rechazos_long = _normalize_area_col(
                    df_rechazos_long
                )  # <-- Normaliza aquí
                
                if debug_mode:
                    st.write("✅ **Debug:** Normalización de área completada")
                    
            except Exception as e:
                st.error(f"❌ Error al cargar rechazos: {e}")
                st.write("🔍 **Debug:** Error en la carga de rechazos")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())
                
                # Crear DataFrame vacío para continuar
                df_rechazos_long = pd.DataFrame()
                
        if not df_rechazos_long.empty:
            st.success("✅ Rechazos para LINEAS cargados con umbrales personalizados.")
        else:
            st.warning("⚠️ No se pudieron cargar datos de rechazos o no hay datos disponibles.")
            st.info("💡 **Sugerencia:** Verifica que existen archivos de datos de rechazos en la carpeta REFERENCIAS.")

        _show_debug(f"🔍 Debug: df_rechazos_long.shape = {df_rechazos_long.shape}")
        if "Area" in df_rechazos_long.columns:
            _show_debug(
                f"🔍 Debug: Áreas en rechazos = {sorted(df_rechazos_long['Area'].unique())}"
            )
        _show_debug(f"🔍 Debug: selected_lineas_rech = {selected_lineas_rech}")

        if df_rechazos_long.empty:
            st.info(
                "No se encontraron datos de rechazo o TI para analizar con el umbral actual."
            )
            # Mostrar mensaje para ayudar al usuario
            st.markdown("""
            **Posibles causas:**
            - No existen archivos de datos de rechazos en la carpeta REFERENCIAS
            - Los archivos están corruptos o en formato incorrecto
            - Los umbrales de similitud son muy restrictivos
            
            **Soluciones:**
            - Verifica la carpeta REFERENCIAS
            - Ajusta los umbrales de similitud
            - Recarga la página para limpiar el cache
            """)
        else:
            # Debug: Mostrar estado después de la carga
            if debug_mode:
                st.write(f"**Debug Rechazos Paso 1:** Datos cargados - {len(df_rechazos_long):,} registros")
                if "Area" in df_rechazos_long.columns:
                    area_counts = df_rechazos_long["Area"].value_counts()
                    st.write(f"- Distribución por Area: {dict(area_counts)}")
            
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

            # Debug: Mostrar estado después de filtros generales
            if debug_mode:
                st.write(f"**Debug Rechazos Paso 2:** Después de filtros generales - {len(df_rechazos_filtrado):,} registros")

            # Filtrar por líneas seleccionadas
            if debug_mode:
                st.write(f"**Debug Rechazos Paso 3:** Líneas seleccionadas para rechazos: {selected_lineas_rech}")
                if "Area" in df_rechazos_filtrado.columns:
                    areas_disponibles = sorted(df_rechazos_filtrado["Area"].unique())
                    st.write(f"- Áreas disponibles en datos: {areas_disponibles}")
                    
            df_rechazos_filtrado = df_rechazos_filtrado[
                df_rechazos_filtrado["Area"].isin(selected_lineas_rech)
            ]

            # Debug: Mostrar estado después de filtro de líneas
            if debug_mode:
                st.write(f"**Debug Rechazos Paso 4:** Después de filtro de líneas - {len(df_rechazos_filtrado):,} registros")

            # ...existing code...

            if df_rechazos_filtrado.empty:
                st.warning("⚠️ No hay datos de pérdidas que coincidan con los filtros seleccionados.")
                if debug_mode:
                    st.write("**Debug:** Posibles causas del DataFrame vacío:")
                    st.write(f"- Líneas seleccionadas: {selected_lineas_rech}")
                    st.write(f"- Áreas en datos originales: {sorted(df_rechazos_long['Area'].unique()) if 'Area' in df_rechazos_long.columns else 'N/A'}")
            else:
                try:
                    # Gráfico de Top Claves de Rechazo
                    _show_debug(
                        f"🔍 Debug: Llamando compute_top_claves_rechazo con {len(df_rechazos_filtrado)} registros"
                    )
                    _show_debug(
                        f"🔍 Debug: Columnas disponibles: {list(df_rechazos_filtrado.columns)}"
                    )

                    df_top_rechazos, meta = compute_top_claves_rechazo(
                        df_rechazos_filtrado,
                        top_n=top_n_rechazo,
                        desglose=desglose_rechazo,
                    )
                    _show_debug(
                        f"🔍 Debug: compute_top_claves_rechazo retornó {len(df_top_rechazos)} registros"
                    )
                    _show_debug(
                        f"🔍 Debug: df_top_rechazos columnas: {list(df_top_rechazos.columns) if not df_top_rechazos.empty else 'VACÍO'}"
                    )
                    _show_debug(f"🔍 Debug: meta = {meta}")

                    if df_top_rechazos.empty:
                        st.warning("⚠️ No se encontraron claves de rechazo para mostrar")
                    else:
                        # Gráfico de Top Claves de Rechazo (ocupando todo el ancho)
                        st.subheader("📊 Top Claves de Rechazo")
                        _show_debug(
                            f"🔍 Debug: Llamando build_top_claves_rechazo_figure con {len(df_top_rechazos)} registros"
                        )
                        _show_debug(f"🔍 Debug: df_top_rechazos preview:")
                        if DEBUG_UI:
                            st.dataframe(df_top_rechazos.head())

                        fig_rechazos = build_top_claves_rechazo_figure(
                            df_top_rechazos, desglose_rechazo
                        )
                        _show_debug(
                            f"🔍 Debug: build_top_claves_rechazo_figure retornó: {type(fig_rechazos)}"
                        )

                        # Mostrar estadísticas básicas de Pzas para diagnóstico
                        try:
                            _show_debug(
                                f"🔍 Debug: Pzas describe: {df_top_rechazos['Pzas'].describe().to_dict()}"
                            )
                        except Exception:
                            _show_debug(
                                "🔍 Debug: No se pudo calcular describe() de Pzas"
                            )

                        # Fallback: si la figura es None o no tiene trazas, construir un bar sencillo
                        has_traces = False
                        if fig_rechazos and hasattr(fig_rechazos, "data"):
                            try:
                                has_traces = len(fig_rechazos.data) > 0
                            except (AttributeError, TypeError):
                                has_traces = False
                                
                        if not has_traces:
                            # Construir silent fallback sin mostrar warnings/errores al usuario
                            try:
                                if "DescripcionCatalogo" in df_top_rechazos.columns:
                                    labels = (
                                        df_top_rechazos["ClaveCatalogo"].astype(str)
                                        + " - "
                                        + df_top_rechazos["DescripcionCatalogo"].astype(
                                            str
                                        )
                                    )
                                else:
                                    labels = df_top_rechazos["ClaveCatalogo"].astype(
                                        str
                                    )

                                import plotly.graph_objects as _go

                                fallback_fig = _go.Figure()
                                fallback_fig.add_trace(
                                    _go.Bar(
                                        x=df_top_rechazos["Pzas"].astype(float),
                                        y=labels,
                                        orientation="h",
                                        marker=dict(color="crimson"),
                                        text=df_top_rechazos["Pzas"].astype(int),
                                        textposition="outside",
                                    )
                                )
                                fallback_fig.update_layout(
                                    title=f"Top Claves de Rechazo - {desglose_rechazo}",
                                    xaxis_title="Pzas",
                                    yaxis_title="Clave",
                                )
                                st.plotly_chart(fallback_fig, width="stretch")
                            except Exception:
                                # Si el fallback falla, renderizar la figura original si existe (silencioso)
                                if fig_rechazos:
                                    st.plotly_chart(fig_rechazos, width="stretch")
                        else:
                            try:
                                _show_debug(f"🔍 Debug: Figura de rechazos creada exitosamente")
                                _show_debug(f"🔍 Debug: Tipo de figura: {type(fig_rechazos)}")
                            except Exception:
                                _show_debug("🔍 Debug: Error accediendo a propiedades de la figura")
                                
                            st.plotly_chart(fig_rechazos, width="stretch")

                        st.caption(
                            f"Total de piezas rechazadas (con filtros aplicados): {int(meta['total_pzas']):,}"
                        )

                except Exception as e:
                    st.error(f"❌ Error al generar gráfico de rechazos: {e}")
                    st.write(
                        "🔍 Debug: Mostrando primeras 5 filas de df_rechazos_filtrado:"
                    )
                    st.dataframe(df_rechazos_filtrado.head())
                    if "ClaveCatalogo" in df_rechazos_filtrado.columns:
                        st.write(
                            f"🔍 Debug: Valores únicos en ClaveCatalogo: {df_rechazos_filtrado['ClaveCatalogo'].unique()[:10]}"
                        )
                    if "Pzas" in df_rechazos_filtrado.columns:
                        st.write(
                            f"🔍 Debug: Valores en Pzas: {df_rechazos_filtrado['Pzas'].describe()}"
                        )
                    st.write(
                        f"🔍 Debug: Tipos de datos en columna Area: {df_rechazos_filtrado['Area'].dtype if 'Area' in df_rechazos_filtrado.columns else 'Columna Area no existe'}"
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
                        file_name=f"top_claves_rechazo_lineas_{date.today()}.csv",
                        mime="text/csv",
                        key="lineas_export_top_claves",
                    )

                # --- Panel de Confiabilidad de Mapeo de Claves de Rechazo ---
                st.markdown("---")
                config_panel = {
                    "dominio": DOM_LINEAS,  # Pasa la constante canónica
                    "threshold_high_actual": threshold_high_rechazo,
                    "top_n": top_n_rechazo,  # Reutiliza el slider del top_n de rechazos
                    "t_values": [80, 85, 88, 90, 92],  # Lista fija para el modo Gourmet
                }

                render_confiabilidad_panel(df_rechazos_filtrado, config_panel)

    # --- Debug (toggle) ---
    if debug_mode:
        with st.expander("Datos Crudos y Estadísticas (Debug)"):
            # --- Bloque de Diagnóstico de Columnas ---
            with st.expander("Diagnóstico de columnas (Líneas)"):
                st.write(
                    "Este bloque analiza las columnas del archivo Excel **antes** de cualquier procesamiento."
                )

                # Ejecutar diagnóstico en el DataFrame crudo
                diag_results = diagnose_columns(df_raw)

                st.write("**1. Columnas originales encontradas en el archivo:**")
                st.write(diag_results["original_cols"])

                st.write("**2. Mapeo de renombrado aplicado:**")
                if diag_results["mapping_applied"]:
                    st.json(diag_results["mapping_applied"])
                else:
                    st.info(
                        "No se aplicó ningún renombrado. Las columnas ya tenían nombres canónicos o no se encontraron sinónimos."
                    )

                st.write("**3. Estado de columnas esenciales DESPUÉS del mapeo:**")
                if diag_results["essential_missing_after_mapping"]:
                    st.warning(
                        f"Faltan las siguientes columnas esenciales: `{diag_results['essential_missing_after_mapping']}`"
                    )
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
    st.set_page_config(page_title="Test - Dashboard Líneas", layout="wide")
    render_lineas_dashboard()

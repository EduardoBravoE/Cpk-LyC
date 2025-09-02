# -*- coding: utf-8 -*-
"""
main.py v2.0
Orquestador principal de la aplicación Streamlit.

Este script no contiene lógica de negocio. Su única responsabilidad es:
1. Configurar la página y la barra lateral.
2. Enrutar a la función de renderizado del dashboard apropiado (Líneas o Coples)
   basándose en la selección del usuario.

Autor: Eduardo + M365 Copilot
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import time

# 1. Imports mínimos y explícitos con compatibilidad
# ===================================================
from UTILS.common import DOM_LINEAS, DOM_COPLES
from UTILS.common import cargar_lineas_con_manifiesto, cargar_coples_con_manifiesto
from UTILS.common import precargar_datos_produccion, obtener_datos_cacheados, cargar_datos_con_feedback, precargar_datos_rechazos, cargar_rechazos_con_cache_inteligente

# Intenta importar las nuevas funciones 'render_*', con fallback a las antiguas 'mostrar_*'
try:
    from UTILS.lineas_dashboard import render_lineas_dashboard as _render_lineas
except ImportError:
    from UTILS.lineas_dashboard import mostrar_dashboard_lineas as _render_lineas

try:
    from UTILS.coples_dashboard import render_coples_dashboard as _render_coples
except ImportError:
    from UTILS.coples_dashboard import mostrar_dashboard_coples as _render_coples

# Mapa para el selector de la UI
AREA_MAP = {
    "Líneas": DOM_LINEAS,
    "Coples": DOM_COPLES,
}


def _render_lineas_with_cache(datos_rechazos_cache):
    """
    Wrapper para renderizar dashboard de líneas con datos de rechazos cacheados.
    """
    try:
        _render_lineas(datos_rechazos_cache)
    except Exception as e:
        # En lugar de llamar el fallback que causaría claves duplicadas,
        # mostramos un mensaje de error y evitamos el renderizado duplicado
        st.error(f"❌ Error al renderizar dashboard de líneas: {e}")
        st.info("🔄 Intenta recargar la página para resolver el problema.")
        return


def _render_coples_with_cache(datos_rechazos_cache):
    """
    Wrapper para renderizar dashboard de coples con datos de rechazos cacheados.
    """
    try:
        _render_coples(datos_rechazos_cache)
    except Exception as e:
        # En lugar de llamar el fallback que causaría claves duplicadas,
        # mostramos un mensaje de error y evitamos el renderizado duplicado
        st.error(f"❌ Error al renderizar dashboard de coples: {e}")
        st.info("🔄 Intenta recargar la página para resolver el problema.")
        return


def _render_quick_mode():
    """Modo rápido embebido: analiza los Excel presentes en DATOS/ y muestra Top productos."""
    import streamlit as _st
    import pandas as _pd
    from UTILS.insights import prepare_df_for_analysis, compute_producto_critico, build_producto_critico_figure, export_df_to_csv_bytes, apply_filters

    _st.header("Modo Rápido — Top Productos desde DATOS/")

    domain_opt = _st.selectbox("Dominio a analizar", ["Líneas", "Coples", "Ambos"], key="quick_domain")
    top_n = _st.slider("Top N", min_value=5, max_value=50, value=10, key="quick_top_n")

    try:
        if domain_opt == "Líneas":
            df_raw, _ = cargar_lineas_con_manifiesto(recursive=False)
            df_combined = df_raw
        elif domain_opt == "Coples":
            df_raw, _ = cargar_coples_con_manifiesto(recursive=False)
            df_combined = df_raw
        else:
            df_l, _ = cargar_lineas_con_manifiesto(recursive=False)
            df_c, _ = cargar_coples_con_manifiesto(recursive=False)
            df_combined = _pd.concat([df_l, df_c], ignore_index=True) if (not df_l.empty or not df_c.empty) else _pd.DataFrame()
    except Exception as _e:
        _st.error("Error leyendo datos desde DATOS/. Revisa que los archivos estén en la estructura esperada.")
        _st.exception(_e)
        return

    if df_combined.empty:
        _st.warning("No se encontraron filas en los Excel leídos. Verifica el manifiesto o el formato (header en fila 2).")
        return

    try:
        df_prepared = prepare_df_for_analysis(df_combined)
    except Exception as _e:
        _st.error("Error al preparar los datos. Es posible que falten columnas esenciales en los Excel.")
        _st.exception(_e)
        return

    # Rango de fechas opcional
    try:
        s_fecha = _pd.to_datetime(df_prepared["Fecha"], format="%d/%m/%Y", errors="coerce")
        min_date = s_fecha.min().date() if _pd.notna(s_fecha.min()) else None
        max_date = s_fecha.max().date() if _pd.notna(s_fecha.max()) else None
        sel_fechas = _st.date_input("Rango de fechas (opcional)", value=(min_date, max_date) if min_date and max_date else (), key="quick_dates")
        fechas_tuple = sel_fechas if isinstance(sel_fechas, tuple) and len(sel_fechas) == 2 else None
        df_filtered = apply_filters(df_prepared, fechas=fechas_tuple, dominio=None)
    except Exception:
        df_filtered = df_prepared

    if df_filtered.empty:
        _st.info("No hay datos con los filtros aplicados.")
        return

    df_critico = compute_producto_critico(df_filtered, top_n=top_n)
    if df_critico.empty:
        _st.info("No se detectaron productos críticos con los filtros actuales.")
        return

    fig = build_producto_critico_figure(df_critico, title=f"Top {top_n} Productos Críticos (Modo Rápido)")
    _st.plotly_chart(fig, use_container_width=True)

    with _st.expander("Tabla: productos críticos"):
        _st.dataframe(df_critico)
        csv = export_df_to_csv_bytes(df_critico)
        _st.download_button("Exportar CSV", data=csv, file_name=f"top_productos_{top_n}.csv", mime="text/csv")


def main():
    """
    Orquesta la aplicación Streamlit con sistema de precarga optimizado.

    Este archivo principal implementa un sistema de precarga que mejora
    significativamente la experiencia de alternancia entre dashboards.
    """
    # 2. Configuración base
    # ======================
    st.set_page_config(
        page_title="Proyecto Cpk's - Dashboard Optimizado",
        layout="wide",
        page_icon="�"
    )

    # ===========================================
    # PRECARGA OPTIMIZADA DE DATOS
    # ===========================================
    st.title("🚀 Proyecto Cpk's - Dashboard de Producción")

    # Precargar datos de ambos dominios para mejor UX
    with st.spinner("🔄 Inicializando aplicación y precargando datos..."):
        datos_cache = precargar_datos_produccion()
        datos_rechazos_cache = precargar_datos_rechazos()

    # Mostrar estado de precarga
    col1, col2 = st.columns(2)
    with col1:
        lineas_count = len(datos_cache.get('lineas', (pd.DataFrame(), pd.DataFrame()))[0])
        rechazos_lineas_count = len(datos_rechazos_cache.get('lineas', pd.DataFrame()))
        if lineas_count > 0:
            st.success(f"✅ LÍNEAS: {lineas_count:,} registros + {rechazos_lineas_count:,} rechazos")
        else:
            st.warning("⚠️ LÍNEAS: No se encontraron datos")

    with col2:
        coples_count = len(datos_cache.get('coples', (pd.DataFrame(), pd.DataFrame()))[0])
        rechazos_coples_count = len(datos_rechazos_cache.get('coples', pd.DataFrame()))
        if coples_count > 0:
            st.success(f"✅ COPLES: {coples_count:,} registros + {rechazos_coples_count:,} rechazos")
        else:
            st.warning("⚠️ COPLES: No se encontraron datos")

    st.divider()

    # 3. Sidebar Optimizado
    # ======================
    st.sidebar.title("⚙️ Controles")

    # Botón de recarga global
    if st.sidebar.button(
        "🔄 Recargar todos los datos",
        key="app_reload",
        help="Limpia toda la caché y recarga los datos de ambos dominios."
    ):
        st.cache_data.clear()
        st.success("✅ Caché limpiada. Los datos se recargarán automáticamente.")
        st.info("🔄 Recargando aplicación...")
        time.sleep(1)  # Pequeña pausa para que el usuario vea el mensaje
        st.rerun()

    # Selector de área con mejor feedback
    selected_area_label = st.sidebar.radio(
        "📍 Área de Producción",
        options=list(AREA_MAP.keys()),
        key="app_area",
        help="Selecciona el área que deseas analizar. Los datos ya están precargados para una experiencia fluida."
    )
    selected_area_domain = AREA_MAP[selected_area_label]

    # 4. Enrutamiento Optimizado
    # ==========================
    st.divider()

    # Limpiar cualquier estado residual antes de renderizar
    if 'last_rendered_area' not in st.session_state:
        st.session_state.last_rendered_area = None

    try:
        if selected_area_domain == DOM_LINEAS:
            # Usar datos precargados para LÍNEAS
            df_raw, manifest_df = obtener_datos_cacheados('lineas', datos_cache)
            if df_raw.empty:
                st.error("❌ No se pudieron cargar los datos de LÍNEAS")
                return

            st.sidebar.success("📊 Mostrando: **LÍNEAS**")
            # Los datos ya están en caché, la función se ejecutará instantáneamente
            _render_lineas_with_cache(datos_rechazos_cache)
            st.session_state.last_rendered_area = DOM_LINEAS

        elif selected_area_domain == DOM_COPLES:
            # Usar datos precargados para COPLES
            df_raw, manifest_df = obtener_datos_cacheados('coples', datos_cache)
            if df_raw.empty:
                st.error("❌ No se pudieron cargar los datos de COPLES")
                return

            st.sidebar.success("📊 Mostrando: **COPLES**")
            # Los datos ya están en caché, la función se ejecutará instantáneamente
            _render_coples_with_cache(datos_rechazos_cache)
            st.session_state.last_rendered_area = DOM_COPLES

    except Exception as e:
        st.error(f"❌ Error al renderizar el dashboard de '{selected_area_label}':")
        st.exception(e)
        st.info("💡 **Solución**: Usa el botón 'Recargar todos los datos' para limpiar la caché.")

    # 5. Footer con información útil
    # ===============================
    st.sidebar.divider()
    total_registros = lineas_count + coples_count
    st.sidebar.caption(
        f"📈 **Total de registros**: {total_registros:,}\n\n"
        f"📍 **Área actual**: {selected_area_label}\n\n"
        f"⚡ **Optimización**: Datos precargados para alternancia instantánea"
    )

# 7. Punto de entrada
# ===================
if __name__ == "__main__":
    main()

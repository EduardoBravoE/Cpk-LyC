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

# 1. Imports mínimos y explícitos con compatibilidad
# ===================================================
from UTILS.common import DOM_LINEAS, DOM_COPLES

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


def main():
    """
    Orquesta la aplicación Streamlit.

    Este archivo principal no contiene lógica de negocio ni de carga de datos.
    Su única responsabilidad es enrutar a la función de renderizado del
    dashboard apropiado basándose en la selección del usuario.
    """
    # 2. Configuración base
    # ======================
    st.set_page_config(
        page_title="Proyecto Cpk's",
        layout="wide",
        page_icon="📊"
    )
    st.title("Proyecto Cpk's — Producto Crítico")

    # 3. Sidebar
    # ==========
    st.sidebar.title("⚙️ Controles")

    # Botón de recarga
    def on_reload_click():
        st.cache_data.clear()
        st.rerun()

    st.sidebar.button(
        "🔄 Recargar datos (limpiar caché)",
        on_click=on_reload_click,
        key="app_reload",
        help="Limpia la caché de datos para forzar la relectura de los archivos Excel."
    )

    # Selector de área
    selected_area_label = st.sidebar.radio(
        "Área",
        options=list(AREA_MAP.keys()),
        key="app_area",
    )
    selected_area_domain = AREA_MAP[selected_area_label]

    # 4. Enrutamiento
    # ===============
    st.divider()
    try:
        if selected_area_domain == DOM_LINEAS:
            _render_lineas()
        elif selected_area_domain == DOM_COPLES:
            _render_coples()
    except Exception as e:
        st.error(f"Ocurrió un error al renderizar el dashboard de '{selected_area_label}':")
        st.exception(e)
        st.info("💡 Tip: Intenta usar el botón 'Recargar datos' o verifica los archivos fuente en las carpetas 'datos/lineas' y 'datos/coples'.")

    # 5. Footer opcional
    # ==================
    st.sidebar.divider()
    st.sidebar.caption(
        f"Viendo: **{selected_area_label}**. "
        "El botón 'Recargar' limpia la caché de datos sin perder los filtros."
    )

# 7. Punto de entrada
# ===================
if __name__ == "__main__":
    main()

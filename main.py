import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import os

# Función para mostrar fechas en formato dd-mes-aaaa en español
def formato_fecha_es(fecha):
    meses = {
        1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
        5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
        9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
    }
    return f"{fecha.day:02d}-{meses[fecha.month]}-{fecha.year}"

# Archivos Excel a consolidar
archivos_excel = [
    'BDD 2024 Líneas.xlsx',
    'BDD ENERO 2025.xlsx',
    'BDD FEBRERO 2025.xlsx',
    'BDD MARZO 2025.xlsx',
    'BDD ABRIL 2025.xlsx',
    'BDD MAYO 2025.xlsx',
    'BDD JUNIO 2025.xlsx'
]

# Cargar claves de rechazo válidas desde archivo auxiliar
claves_rechazo_df = pd.read_excel('Claves de Rechazo - GPT.xlsx', engine='openpyxl')
claves_validas = claves_rechazo_df[claves_rechazo_df['Origen'].isin(['Líneas', 'Ambas'])]
claves_rechazo_validas = set(claves_validas['Clave'].astype(str))
descripciones_rechazo_validas = dict(zip(claves_validas['Clave'].astype(str), claves_validas['Descripcion']))

# Inicializar listas
datos_combinados = []
descripciones_totales = {}

# Procesar cada archivo
for archivo in archivos_excel:
    if os.path.exists(archivo):
        hojas = pd.read_excel(archivo, sheet_name=None, engine='openpyxl', header=1)
        descripciones_raw = pd.read_excel(archivo, sheet_name=None, engine='openpyxl', header=None, nrows=1)

        for nombre_hoja in ['L-A', 'L-B', 'L-C']:
            if nombre_hoja in hojas:
                df = hojas[nombre_hoja]
                desc_hoja = descripciones_raw[nombre_hoja].iloc[0]

                columnas_fijas = df.columns[:40]
                columnas_claves = df.columns[40:]

                descripciones_claves = {
                    clave: desc_hoja[i] for i, clave in enumerate(df.columns)
                    if clave in columnas_claves
                }
                descripciones_totales.update(descripciones_claves)

                df['Linea'] = nombre_hoja
                df_fijo = df[columnas_fijas.tolist() + ['Linea']]
                df_claves = df[['Fecha', 'Turno', 'Maquina'] + columnas_claves.tolist()]
                df_claves = df_claves.drop(columns=[col for col in ['Fecha', 'Turno', 'Maquina'] if col in df_fijo.columns])
                df_final = pd.concat([df_fijo, df_claves], axis=1)
                datos_combinados.append(df_final)

# Consolidar DataFrame
if datos_combinados:
    df_unificado = pd.concat(datos_combinados, ignore_index=True).copy()
    df_unificado['Fecha'] = pd.to_datetime(df_unificado['Fecha'], errors='coerce', dayfirst=True)

    # Streamlit config
    st.set_page_config(page_title="Dashboard CPK", layout="wide")
    st.title("🔍 Análisis de Producción y Rechazos para CPK")

    # 🎛️ Filtros interactivos
    st.sidebar.header("🎛️ Filtros Interactivos")

    fecha_min = df_unificado['Fecha'].min()
    fecha_max = df_unificado['Fecha'].max()
    rango_fechas = st.sidebar.date_input("📅 Rango de fechas", [fecha_min, fecha_max], min_value=fecha_min, max_value=fecha_max)

    lineas = df_unificado['Linea'].dropna().unique().tolist()
    lineas_seleccionadas = st.sidebar.multiselect("🏭 Línea", options=lineas, default=lineas)

    maquinas = df_unificado['Maquina'].dropna().unique().tolist()
    maquinas_seleccionadas = st.sidebar.multiselect("⚙️ Máquina", options=maquinas, default=maquinas)

    turnos = df_unificado['Turno'].dropna().unique().tolist()
    turnos_seleccionados = st.sidebar.multiselect("🕒 Turno", options=turnos, default=turnos)

    # Aplicar filtros
    df_unificado = df_unificado[
        (df_unificado['Fecha'] >= pd.to_datetime(rango_fechas[0])) &
        (df_unificado['Fecha'] <= pd.to_datetime(rango_fechas[1])) &
        (df_unificado['Linea'].isin(lineas_seleccionadas)) &
        (df_unificado['Maquina'].isin(maquinas_seleccionadas)) &
        (df_unificado['Turno'].isin(turnos_seleccionados))
    ]

    # 📅 Periodo observado
    fecha_min_filtrada = df_unificado['Fecha'].min()
    fecha_max_filtrada = df_unificado['Fecha'].max()
    fecha_min_str = formato_fecha_es(fecha_min_filtrada)
    fecha_max_str = formato_fecha_es(fecha_max_filtrada)

    with st.container():
        st.markdown(
            f"""
            <div style="background-color:#f0f2f6;padding:10px 20px;border-radius:10px;margin-bottom:20px">
                <h4 style="color:#333333">📅 Periodo observado</h4>
                <p style="font-size:16px;color:#555555">{fecha_min_str} &nbsp; ➡️ &nbsp; {fecha_max_str}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # 🔧 Producto Crítico: Top 10 combinaciones más productivas
    st.subheader("📦 Producto Crítico: Top 10 combinaciones más productivas")

    df_filtrado = df_unificado.dropna(subset=['Rosca', 'Diametro', 'Acero', 'Libraje']).copy()
    df_filtrado['Combinacion'] = df_filtrado[['Rosca', 'Diametro', 'Acero', 'Libraje']].astype(str).agg(' - '.join, axis=1)

    resumen = df_filtrado.groupby('Combinacion').agg({
        'TotalPiezas': 'sum',
        'PzasRech': 'sum'
    }).reset_index()

    resumen['PzasOK'] = resumen['TotalPiezas'] - resumen['PzasRech']
    resumen['%Rechazo'] = (resumen['PzasRech'] / resumen['PzasOK'].replace(0, pd.NA)) * 100

    top10 = resumen.sort_values(by='TotalPiezas', ascending=False).head(10)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top10['Combinacion'],
        y=top10['TotalPiezas'],
        name='Total Piezas',
        marker_color='steelblue',
        hovertemplate='<b>%{x}</b><br>Total Piezas: %{y:,}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=top10['Combinacion'],
        y=top10['%Rechazo'],
        name='% Rechazo',
        mode='lines+markers',
        marker=dict(color='darkred', symbol='circle'),
        line=dict(dash='dash'),
        yaxis='y2',
        hovertemplate='<b>%{x}</b><br>% Rechazo: %{y:.2f}%<extra></extra>'
    ))
    fig.update_layout(
        title='Top 10 combinaciones más productivas con índice de rechazo',
        xaxis=dict(title='Combinación', tickangle=-45),
        yaxis=dict(title='Total Piezas'),
        yaxis2=dict(title='% Rechazo', overlaying='y', side='right'),
        legend=dict(x=0.85, y=1.15),
        margin=dict(t=60, b=100)
    )
    st.plotly_chart(fig)

    # ❌ Top 10 claves de rechazo más frecuentes
    st.subheader("❌ Top 10 claves de rechazo más frecuentes")

    claves_dinamicas = df_unificado.columns[41:]
    claves_rechazo = [c for c in claves_dinamicas if str(c) in claves_rechazo_validas]

    df_unificado[claves_rechazo] = df_unificado[claves_rechazo].apply(pd.to_numeric, errors='coerce')
    conteo_claves = df_unificado[claves_rechazo].sum().sort_values(ascending=False).head(10).astype(int)

    df_claves_top = pd.DataFrame({
        'Clave': conteo_claves.index,
        'Ocurrencias': conteo_claves.values,
        'Descripción': [descripciones_rechazo_validas.get(str(k), '') for k in conteo_claves.index]
    })

    fig2 = px.bar(
        df_claves_top,
        x='Ocurrencias',
        y='Descripción',
        orientation='h',
        title='Top 10 claves de rechazo más frecuentes',
        labels={'Ocurrencias': 'Ocurrencias', 'Descripción': 'Descripción'},
        color='Ocurrencias'
    )
    fig2.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig2)

else:
    st.error("No se encontraron datos en los archivos especificados.")
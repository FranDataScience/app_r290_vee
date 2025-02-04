


import gdown
import streamlit as st
import pandas as pd
import numpy as np
import CoolProp as cp
from CoolProp.CoolProp import PropsSI
import pickle
import sys
import os
from streamlit_autorefresh import st_autorefresh
import pymysql
from collections import Counter
import math
import plotly.graph_objects as go

# Directorio donde se almacenarán los modelos descargados
MODEL_DIR = "models_vee"
os.makedirs(MODEL_DIR, exist_ok=True)

# Diccionario con los nombres de los modelos y sus respectivos IDs de Google Drive
MODELOS_DRIVE = {
    'model_t_ev.pkl': '18NG91i8EyCr8TkaxIFYh1RS5ujtqIwCx', 
    'model_t_cd.pkl': '18vZzV90q8Vt628cVVv5tCGQWYOzEBtMf',
    'model_rec.pkl': '1KJJLNqysxW1PR2Ptl0KZ1grawJCxcqmW',
    'model_t_des.pkl': '1r58Ab80yNa1A9ejRN4YXYHD0yL7ktZiQ',
    'model_subf.pkl': '11Gp7cnknk-qsao51VLPq0lvgMh3UVSvc',
    'model_dta_evap.pkl': '1dr0DHRch8aeabjL57CbZREI8H8xujthx',
    'model_dta_cond.pkl': '1P-CCsm_bL_k-kdqPw34iskYUPmUIp9uY',
    'model_pot_abs.pkl': '1mRv5Fiw7K7j8DLUvEnqJ2q54vcR0Cq9i',
    'model_cop.pkl': '1aJuLDtz5bcZK7r2L2r34UUYyBLuq9EC2',
    'model_ef_comp.pkl': '148EVPmYV5xK8Hp7jYHQijsSdcuCfS9Ia'
}

def descargar_modelo(nombre_archivo, file_id):
    """
    Descarga un modelo desde Google Drive si no está presente localmente.
    """
    ruta_completa = os.path.join(MODEL_DIR, nombre_archivo)
    if not os.path.exists(ruta_completa):
        url = f"https://drive.google.com/uc?id={file_id}"
        st.info(f"Descargando {nombre_archivo}...")
        try:
            gdown.download(url, ruta_completa, quiet=False)
            st.success(f"{nombre_archivo} descargado exitosamente.")
        except Exception as e:
            st.error(f"Error al descargar {nombre_archivo}: {e}")
            raise e
    #else:
        #st.info(f"{nombre_archivo} ya está presente.")
    return ruta_completa

# Función para cargar modelos preentrenados 
def cargar_modelo(nombre_archivo):
    # Descargar el modelo si no existe
    ruta_completa = descargar_modelo(nombre_archivo, MODELOS_DRIVE[nombre_archivo])
    
    # Cargar el modelo
    try:
        with open(ruta_completa, 'rb') as archivo:
            return pickle.load(archivo)
    except Exception as e:
        st.error(f"Error al cargar {nombre_archivo}: {e}")
        raise e

# Configuramos la página en modo "wide" (una única vez).
st.set_page_config(page_title="Detector fallos R290", layout="wide")

# 2. Configuración del auto-refresh cada 30 segundos
count = st_autorefresh(interval=30000, limit=None, key="fizzbuzzcounter")

st.markdown("""
    <style>
    /* Quita o reduce el padding superior en la parte central (modo wide) */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 1rem !important;
    }

    /* Quita o reduce el padding en el sidebar */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin: 0rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Función para calcular la temperatura (ºC) conocida la presión relativa (bar) - R290
def convert_PT_R290(presion):
    temperatura = np.round(PropsSI('T', 'P', (presion+1)*100000, 'Q', 1, 'R290')-273.15, 2)
    return temperatura

# Función para calcular la entalpía (kJ/kg) fuera de la campana de saturación, conocidas la presión relativa (bar) y la temperatura (ºC) - R290
def entalpia_R290_PT(presion, temperatura):
    entalpia = np.round(PropsSI('H', 'P', (presion+1)*100000, 'T', temperatura+273.15, 'R290')/1000, 2)
    return entalpia

# Función para calcular el COP - R290
def cop(pb, pa, t_des, t_liq, t_asp):
    h_liq = entalpia_R290_PT(pa, t_liq)
    h_asp = entalpia_R290_PT(pb, t_asp)
    h_des = entalpia_R290_PT(pa, t_des)
    cop = np.round((h_asp-h_liq)/(h_des-h_asp), 2)
    return cop

# Función para calcular la eficiencia isoentrópica de compresión - R290
def ef_comp(pb, pa, t_des, t_asp):
    h_asp = entalpia_R290_PT(pb, t_asp)
    s_asp = PropsSI('S', 'P', (pb+1)*100000, 'T', t_asp+273.15, 'R290')/1000
    h_des_iso = PropsSI('H', 'P', (pa+1)*100000, 'S', s_asp*1000, 'R290')/1000
    h_des = entalpia_R290_PT(pa, t_des)
    ef_comp = np.round((h_des_iso-h_asp)/(h_des-h_asp), 3)
    return ef_comp

# Cargar modelos
with st.spinner("Cargando modelos..."):
        models = {
            't_ev': cargar_modelo('model_t_ev.pkl'),
            't_cd': cargar_modelo('model_t_cd.pkl'),
            'rec': cargar_modelo('model_rec.pkl'),
            't_des': cargar_modelo('model_t_des.pkl'),
            'subf': cargar_modelo('model_subf.pkl'),
            'dta_evap': cargar_modelo('model_dta_evap.pkl'),
            'dta_cond': cargar_modelo('model_dta_cond.pkl'),
            'pot_abs': cargar_modelo('model_pot_abs.pkl'),
            'cop': cargar_modelo('model_cop.pkl'),
            'ef_comp': cargar_modelo('model_ef_comp.pkl')
        }
#st.success("Todos los modelos han sido cargados correctamente.")


stats = {
    't_ev': {'mean': -18.92, 'std': 2},
    't_cd': {'mean': 37.98, 'std': 2},
    'rec': {'mean': 10.21, 'std': 2},
    't_des': {'mean': 61.46, 'std': 5},
    'subf': {'mean': 4.48, 'std': 2},
    'dta_evap': {'mean': 2.33, 'std': 1},
    'dta_cond': {'mean': 5.18, 'std': 1},
    'pot_abs': {'mean': 360.00, 'std': 20},
    'cop': {'mean': 3.85, 'std': 0.40},
    'ef_comp': {'mean': 0.92, 'std': 0.05},
    'pot_frig': {'mean': 1000, 'std': 60},
}

# Configuración de la app
st.markdown(
    """
    <h1 style='text-align: center; margin-top: 0px;'>
        Detector de fallos Equipo R290
    </h1>
    """,
    unsafe_allow_html=True
)

# Captura de datos de entrada
import pymysql
import streamlit as st

# Función para establecer la conexión con la base de datos
def get_connection():
    return pymysql.connect(
        host="5.134.116.201",
        port=3306,
        user="juandeeu_digital",
        password="ContraseN.4",
        database="juandeeu_db"
    )

# Función para obtener todas las fechas disponibles en la tabla
def get_all_dates():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT fecha
        FROM incalab
        ORDER BY fecha DESC
    """)
    dates = cursor.fetchall()
    cursor.close()
    conn.close()
    return [date[0] for date in dates]  # Convertimos las fechas a una lista

# Función para obtener los datos según la fecha seleccionada
def get_data_by_date(selected_date):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fecha, pa, pb, t_asp, t_des, t_liq, ta_in_cond, ta_in_evap,
               ta_out_cond, ta_out_evap, pot_abs
        FROM incalab
        WHERE fecha = %s
    """, (selected_date,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    # Cambiar nombres de variables ta_in_cond -> t_amb y ta_in_evap -> t_cam
    if row:
        row = list(row)
        row[6] = row[6]  # ta_in_cond -> t_amb
        row[7] = row[7]  # ta_in_evap -> t_cam
    return row

# =============================
# USO EN STREAMLIT
# =============================

# Muestra el logotipo en la parte superior de la barra lateral
st.sidebar.image("Logo2.png", width=300)  # Ajusta el width que más te convenga

st.sidebar.header("Datos recibidos en tiempo real")

# Obtener todas las fechas disponibles
fechas_disponibles = get_all_dates()

# Selector de fecha en el sidebar
#selected_date = st.sidebar.selectbox(
    #"Seleccione la fecha del registro:",
    #fechas_disponibles

selected_date = fechas_disponibles[0]


# Obtener los datos correspondientes a la fecha seleccionada
datos = get_data_by_date(selected_date)

if datos:
    # Desempaquetamos la tupla en las variables
    (fecha, pa, pb, t_asp, t_des, t_liq,
     t_amb, t_cam, ta_out_cond, ta_out_evap,
     pot_abs) = datos
   
    # Intentar convertir pot_abs a número
    try:
        pot_abs_num = float(pot_abs)  # Convertir a flotante
        pot_abs_formateado = f"{pot_abs_num:.0f}"  # Redondear sin decimales
    except ValueError:
        pot_abs_formateado = "Dato inválido"

    # Mostramos en el sidebar
    st.sidebar.write(f"**Fecha último registro:** {fecha}")
    st.sidebar.write(f"**Tª ambiente (°C):** {t_amb}")
    st.sidebar.write(f"**Tª cámara (°C):** {t_cam}")
    st.sidebar.write(f"**Presión alta (bar):** {pa}")
    st.sidebar.write(f"**Presión baja (bar):** {pb}")
    st.sidebar.write(f"**Tª aspiración (°C):** {t_asp}")
    st.sidebar.write(f"**Tª descarga (°C):** {t_des}")
    st.sidebar.write(f"**Tª líquido (°C):** {t_liq}")
    st.sidebar.write(f"**Tª aire salida condensador (°C):** {ta_out_cond}")
    st.sidebar.write(f"**Tª aire salida evaporador (°C):** {ta_out_evap}")
    st.sidebar.write(f"**Potencia absorbida (W):** {pot_abs_formateado}")
else:
    st.sidebar.error("No se encontraron registros para la fecha seleccionada.")

    
# En el sidebar: Celda para modificar el parámetro umbral
umbral = st.sidebar.number_input(
     "Nº desviaciones típicas umbral para fallo",
      min_value=0.0,  # Valor mínimo permitido
      max_value=10.0,  # Valor máximo permitido
      value=2.0,  # Valor por defecto
      step=0.1  # Incrementos de 0.1
)
    


# Crear DataFrame de entrada
datos = {
    'pa': [pa],
    'pb': [pb],
    't_asp': [t_asp],
    't_des': [t_des],
    't_liq': [t_liq],
    't_amb': [t_amb],
    't_cam': [t_cam],
    'ta_out_cond': [ta_out_cond],
    'ta_out_evap': [ta_out_evap],
    'pot_abs': [pot_abs]
}

df = pd.DataFrame(datos)

# Convertir columnas relevantes a numéricas
columnas_numericas = ['pa', 'pb', 't_asp', 't_des', 't_liq', 't_amb', 't_cam', 'ta_out_cond', 'ta_out_evap', 'pot_abs']
for col in columnas_numericas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con NaN en las columnas clave
df = df.dropna(subset=columnas_numericas)

# Calcular variables reales
#df['t_ev'] = np.round(df['pb'].apply(convert_PT_R290), 1)
df['t_ev'] = np.round(convert_PT_R290(df['pb']),1)
#df['t_cd'] = np.round(df['pa'].apply(convert_PT_R290), 1)
df['t_cd'] = np.round(convert_PT_R290(df['pa']),1)
df['t_asp'] = np.maximum(df['t_asp'], df['t_ev'])
df['t_liq'] = np.minimum(df['t_liq'], df['t_cd'])
df['rec'] = df['t_asp'] - df['t_ev']
df['subf'] = df['t_cd'] - df['t_liq']
df['dt_ev'] = df['t_cam'] - df['t_ev']
df['dt_cd'] = df['t_cd'] - df['t_amb']
df['dta_evap'] = df['t_cam'] - df['ta_out_evap']
df['dta_cond'] = df['ta_out_cond'] - df['t_amb']
df['cop'] = np.round(cop(df['pb'].iloc[0], df['pa'].iloc[0], df['t_des'].iloc[0], df['t_liq'].iloc[0], df['t_asp'].iloc[0]),1)
df['ef_comp'] = ef_comp(df['pb'].iloc[0], df['pa'].iloc[0], df['t_des'].iloc[0], df['t_asp'].iloc[0])
df['pot_frig'] = np.round(df['pot_abs'] * df['cop'])

# Eliminar columnas innecesarias
df.drop(columns=['t_asp', 't_liq','ta_out_cond','ta_out_evap'], inplace=True)

# Ordenar columnas
orden = ['t_cam','t_amb','t_ev','t_cd','t_des','rec','subf','dt_ev','dt_cd',
         'dta_evap','dta_cond','pot_abs','pot_frig','cop','ef_comp','pb','pa']
df = df[orden]

# Calculadora final basada en los modelos en cascada
def calcular_variables(df):
    resultados = {}

    # Predicción de t_ev y t_cd
    resultados['t_ev'] = models['t_ev'].predict(df[['t_cam', 't_amb']])[0]
    resultados['t_cd'] = models['t_cd'].predict(df[['t_cam', 't_amb']])[0]

    # Predicción de rec
    #df['t_ev'] = resultados['t_ev']
    #df['t_cd'] = resultados['t_cd']
    resultados['rec'] = models['rec'].predict(df[['t_cam', 't_amb', 't_ev', 't_cd']])[0]

    # Predicción de t_des
    #df['rec'] = resultados['rec']
    resultados['t_des'] = models['t_des'].predict(df[['t_ev', 't_cd', 'rec', 't_amb']])[0]

    # Predicción de subf
    #df['t_des'] = resultados['t_des']
    resultados['subf'] = models['subf'].predict(df[['t_amb', 't_cd', 't_des']])[0]

    # Predicción de dta_evap
    #df['subf'] = resultados['subf']
    resultados['dta_evap'] = models['dta_evap'].predict(df[['t_cam', 't_ev']])[0]

    # Predicción de dta_cond
    resultados['dta_cond'] = models['dta_cond'].predict(df[['t_cd', 't_amb']])[0]

    # Predicción de pot_abs usando las columnas esperadas por el modelo
    resultados['pot_abs'] = np.round(models['pot_abs'].predict(df[['t_des', 't_ev', 't_cd', 't_amb', 't_cam']])[0])

    # Predicción de cop
    resultados['cop'] = np.round(models['cop'].predict(df[['t_ev', 't_cd', 't_des', 'rec']])[0],1)

    # Predicción de ef_comp
    resultados['ef_comp'] = models['ef_comp'].predict(df[['t_ev', 't_cd', 't_des', 't_amb', 't_cam']])[0]
    
    # Cálculo directo de dt_ev, dt_cd y pot_frig
    resultados['dt_ev'] = round(df['t_cam'].iloc[0] - resultados['t_ev'], 1)
    resultados['dt_cd'] = round(resultados['t_cd'] - df['t_amb'].iloc[0], 1)
    resultados['pot_frig'] = round(resultados['pot_abs'] * resultados['cop'], 0)

    return resultados

# Crear un DataFrame inicial con los valores reales
df['registro'] = 'real'

# Calcular los valores esperados usando los modelos
valores_esperados = calcular_variables(df.iloc[0:1])
df_esperado = pd.DataFrame([valores_esperados])
df_esperado['registro'] = 'esperado'

# Calcular la desviación
df_desviacion = df.select_dtypes(include=[np.number]).iloc[0] - df_esperado.select_dtypes(include=[np.number]).iloc[0]
df_desviacion['registro'] = 'desviación'

# Calcular el número de desviaciones típicas
n_sd = {}
for var in stats:
    n_sd[var] = (df_desviacion[var] / stats[var]['std']) if var in df_desviacion else 0

# Convertir a DataFrame y añadir al registro
df_n_sd = pd.DataFrame([n_sd])
df_n_sd['registro'] = 'n_sd'

# Concatenar los DataFrames
final_df = pd.concat([df, df_esperado, df_desviacion.to_frame().T, df_n_sd], ignore_index=True)

# Ordenar columnas
orden = ['registro','t_cam','t_amb','t_ev','t_cd','t_des','rec','subf','dt_ev','dt_cd',
         'dta_evap','dta_cond','pot_abs','pot_frig','cop','ef_comp','pb','pa']
final_df = final_df[orden]

# Redondear los valores en el DataFrame final, asegurando que no haya NaN
final_df['cop'] = np.round(final_df['cop'].fillna(0), 1)
final_df['ef_comp'] = np.round(final_df['ef_comp'].fillna(0), 2)
final_df['t_ev'] = np.round(final_df['t_ev'].fillna(0), 1)
final_df['t_cd'] = np.round(final_df['t_cd'].fillna(0), 1)
final_df['t_des'] = np.round(final_df['t_des'].fillna(0), 1)
final_df['rec'] = np.round(final_df['rec'].fillna(0), 1)
final_df['subf'] = np.round(final_df['subf'].fillna(0), 1)
final_df['dta_evap'] = np.round(final_df['dta_evap'].fillna(0), 1)
final_df['dta_cond'] = np.round(final_df['dta_cond'].fillna(0), 1)
final_df['dt_ev'] = np.round(final_df['dt_ev'].fillna(0), 1)
final_df['dt_cd'] = np.round(final_df['dt_cd'].fillna(0), 1)
final_df['pot_abs'] = np.round(final_df['pot_abs'].fillna(0))
final_df['pot_frig'] = np.round(final_df['pot_frig'].fillna(0))

# ---------------------------------
# DETECCIÓN DE FALLOS
# ---------------------------------

fallos = []

# Fallo 1: Obstrucción en línea de líquido / expansión insuficiente / filtro sucio
# Si t_ev supera el umbral en negativo y rec en positivo, y subf es normal
if n_sd['t_ev'] < -2*umbral and n_sd['rec'] > 1.5*umbral and n_sd['subf'] > -umbral:
    fallos.append("Fallo 1: Obstrucción en línea de líquido / expansión insuficiente / filtro sucio")
    
# Fallo 2: Falta de refrigerante
# Si t_ev y subf superan el umbral en negativo y rec en positivo
if n_sd['t_ev'] < -0.4*umbral and n_sd['subf'] < -0.5*umbral and n_sd['rec'] > umbral:
    fallos.append("Fallo 2: Falta de refrigerante")

# Fallo 3: Caudal de aire insuficiente en el evaporador / fallo ventilador
# Si t_ev supera el umbral en negativo y tda_evap en positivo
if n_sd['t_ev'] < -0.5*umbral and n_sd['dta_evap'] > umbral:
    fallos.append("Fallo 3: Caudal de aire insuficiente en el evaporador / fallo ventilador")
    
# Fallo 3 y 4: Caudal de aire insuficiente en el evaporador / fallo ventilador + Transmisión insuficiente en el evaporador / suciedad / escarcha 
# Si t_ev supera el umbral en negativo
if n_sd['t_ev'] < -umbral and n_sd['dta_evap'] <= umbral and n_sd['dta_evap'] >= -umbral:
    fallos.append("Fallo 3 y 4: Caudal de aire insuficiente en el evaporador / fallo ventilador + Transmisión insuficiente en el evaporador / suciedad / escarcha ")

# Fallo 4: Transmisión insuficiente en el evaporador / suciedad / escarcha
# Si t_ev y tda_evap superan el umbral en negativo
if n_sd['t_ev'] < -umbral and n_sd['dta_evap'] < -umbral:
    fallos.append("Fallo 4: Transmisión insuficiente en el evaporador / suciedad / escarcha")

# Fallo 5: Válvula de expansión demasiado abierta
# Si t_ev supera el umbral en positivo y rec en negativo
if n_sd['t_ev'] > 0*umbral and n_sd['rec'] < -1.2*umbral:
    fallos.append("Fallo 5: Válvula de expansión demasiado abierta")

# Fallo 6: Falta de capacidad o by-pass en compresor
# Si t_ev supera el umbral en positivo y t_cd en negativo
if n_sd['t_ev'] > 0.5*umbral and n_sd['t_cd'] < -0.2*umbral:
    fallos.append("Fallo 6: Falta de capacidad o by-pass en compresor")
    
# Fallo 7: Caudal de aire insuficiente en el condensador / fallo ventilador
# Si t_cd y dta_cond superan el umbral en positivo 
if n_sd['t_cd'] > umbral and n_sd['dta_cond'] > umbral:
    fallos.append("Fallo 7: Caudal de aire insuficiente en el condensador / fallo ventilador")

# Fallo 7 y 8: Caudal de aire insuficiente en el condensador / fallo ventilador + Transmisión insuficiente o suciedad en el condensador
# Si t_cd supera el umbral en positivo 
if n_sd['t_cd'] > umbral and n_sd['dta_cond'] <= umbral and n_sd['dta_cond'] >= -umbral:
    fallos.append("Fallo 7 y 8: Caudal de aire insuficiente en el condensador / fallo ventilador + Transmisión insuficiente o suciedad en el condensador")
    
# Fallo 8: Transmisión insuficiente o suciedad en el condensador
# Si t_cd supera el umbral en positivo y dta_cond en negativo
if n_sd['t_cd'] > umbral and n_sd['dta_cond'] < -umbral:
    fallos.append("Fallo 8: Transmisión insuficiente en el condensador / suciedad")
    
# Fallo 9: Exceso de refrigerante
# Si t_cd y subf supera el umbral en positivo
if n_sd['t_cd'] > umbral and n_sd['subf'] > umbral:
    fallos.append("Fallo 9: Exceso de refrigerante")

# Mostrar el DataFrame completo
print("Resultados:")
print(final_df)

if fallos:
    print("--- DETECCIÓN DE FALLOS ---")
    for i, fallo in enumerate(fallos, start=1):
        print(f"• {fallo}")
else:
    print("Sistema funcionando correctamente")
    
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import streamlit as st

# =============================
# FUNCIÓN PARA OBTENER TODAS LAS FECHAS DISPONIBLES
# =============================
def get_all_dates():

    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT DISTINCT fecha
        FROM incalab
        ORDER BY fecha ASC
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Convertir a lista de fechas
    return [pd.to_datetime(row[0], format="%d-%m-%Y (%H:%M:%S)") for row in rows]

# =============================
# FUNCIÓN PARA OBTENER LOS ÚLTIMOS 300 REGISTROS
# =============================
def get_last_300_records():

    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, fecha, pa, pb, t_asp, t_des, t_liq, ta_in_cond, ta_in_evap,
               ta_out_cond, ta_out_evap, pot_abs
        FROM incalab
        ORDER BY id DESC
        LIMIT 300
    """)
    
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Convertir a DataFrame y ordenar por fecha ascendente
    columns = ["fecha", "pa", "pb", "t_asp", "t_des", "t_liq", "t_amb", "t_cam", "ta_out_cond", "ta_out_evap", "pot_abs"]
    df = pd.DataFrame(rows, columns=columns)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d-%m-%Y (%H:%M:%S)")  # Convertir a datetime
    return df.sort_values(by="fecha")  # Ordenar por fecha en orden ascendente

# =============================
# FUNCIÓN PARA PROCESAR LOS DATOS CRUDOS
# =============================
def process_raw_data(df_raw):

    df_raw["fecha"] = pd.to_datetime(df_raw["fecha"])  # Asegurar que la fecha sea datetime

    # Convertir columnas relevantes a numéricas
    columnas_numericas = ['pa', 'pb', 't_asp', 't_des', 't_liq', 't_amb', 't_cam', 'ta_out_cond', 'ta_out_evap', 'pot_abs']
    for col in columnas_numericas:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # Eliminar filas con NaN en las columnas clave
    df_raw = df_raw.dropna(subset=columnas_numericas)

    # Calcular variables reales
    df_raw['t_ev'] = np.round(df_raw['pb'].apply(convert_PT_R290), 1)
    df_raw['t_cd'] = np.round(df_raw['pa'].apply(convert_PT_R290), 1)
    df_raw['t_asp'] = np.maximum(df_raw['t_asp'], df_raw['t_ev'])
    df_raw['t_liq'] = np.minimum(df_raw['t_liq'], df_raw['t_cd'])
    df_raw['rec'] = df_raw['t_asp'] - df_raw['t_ev']
    df_raw['subf'] = df_raw['t_cd'] - df_raw['t_liq']
    df_raw['dt_ev'] = df_raw['t_cam'] - df_raw['t_ev']
    df_raw['dt_cd'] = df_raw['t_cd'] - df_raw['t_amb']
    df_raw['dta_evap'] = df_raw['t_cam'] - df_raw['ta_out_evap']
    df_raw['dta_cond'] = df_raw['ta_out_cond'] - df_raw['t_amb']
    df_raw['cop'] = df_raw.apply(lambda row: cop(row['pb'], row['pa'], row['t_des'], row['t_liq'], row['t_asp']), axis=1)
    df_raw['ef_comp'] = df_raw.apply(lambda row: ef_comp(row['pb'], row['pa'], row['t_des'], row['t_asp']), axis=1)
    df_raw['pot_frig'] = np.round(df_raw['pot_abs'] * df_raw['cop'], 1)

    # Eliminar columnas innecesarias
    df_raw.drop(columns=['t_asp', 't_liq', 'ta_out_cond', 'ta_out_evap'], inplace=True)

    # Ordenar columnas
    orden = ['fecha', 't_cam', 't_amb', 't_ev', 't_cd', 't_des', 'rec', 'subf', 'dt_ev', 'dt_cd',
             'dta_evap', 'dta_cond', 'pot_abs', 'pot_frig', 'cop', 'ef_comp', 'pb', 'pa']
    return df_raw[orden]

# =============================
# USO DE LA FUNCIÓN EN STREAMLIT
# =============================


# Obtener todos los registros desde la base de datos
def get_all_records():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id fecha, pa, pb, t_asp, t_des, t_liq, ta_in_cond, ta_in_evap,
               ta_out_cond, ta_out_evap, pot_abs
        FROM incalab
        ORDER BY id ASC
        """
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convertir a DataFrame
    columns = ["id","fecha", "pa", "pb", "t_asp", "t_des", "t_liq", "t_amb", "t_cam", "ta_out_cond", "ta_out_evap", "pot_abs"]
    df = pd.DataFrame(rows, columns=columns)

    # Ajustar el formato de las fechas al formato real de la base de datos
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d-%m-%Y (%H:%M:%S)", errors="coerce")
    
    # Eliminar registros con fechas inválidas
    df = df.dropna(subset=["fecha"])

    return df

# Cargar todos los registros
df_raw = get_all_records()

if df_raw.empty:
    st.warning("No se encontraron registros en la base de datos.")
else:
    # Procesar los datos crudos
    df_processed = process_raw_data(df_raw)

    # Fechas por defecto basadas en los últimos 300 registros
    df_last_300 = df_processed.tail(300)
    default_start_datetime = df_last_300["fecha"].min()
    default_end_datetime = df_last_300["fecha"].max()

    # Generar las fechas con el formato "2025-01-21 (17:59:29)" y ordenarlas de más reciente a menos reciente
    formatted_dates = [d.strftime("%Y-%m-%d (%H:%M:%S)") for d in df_processed["fecha"]]
    formatted_dates_desc = list(reversed(formatted_dates))  # Ordenar de más reciente a menos reciente

    # Seleccionar rango de tiempo personalizado
    col1, col2 = st.columns(2)
    with col1:
        start_datetime_str = st.selectbox(
            "Fecha y hora de inicio",
            formatted_dates_desc,
            index=formatted_dates_desc.index(default_start_datetime.strftime("%Y-%m-%d (%H:%M:%S)"))
        )
    with col2:
        end_datetime_str = st.selectbox(
            "Fecha y hora de final",
            formatted_dates_desc,
            index=formatted_dates_desc.index(default_end_datetime.strftime("%Y-%m-%d (%H:%M:%S)"))
        )

    # Convertir las fechas seleccionadas de vuelta a datetime
    start_datetime = pd.to_datetime(start_datetime_str, format="%Y-%m-%d (%H:%M:%S)")
    end_datetime = pd.to_datetime(end_datetime_str, format="%Y-%m-%d (%H:%M:%S)")

    # Filtrar datos según el rango seleccionado
    mask = (df_processed["fecha"] >= start_datetime) & (df_processed["fecha"] <= end_datetime)
    df_filtered = df_processed.loc[mask]

    if df_filtered.empty:
        st.warning("No hay registros para el rango de tiempo seleccionado.")
    else:
        # Variables principales por defecto y variables adicionales opcionales
        default_variables = ["t_cam", "t_amb", "t_ev", "t_cd", "t_des", "rec", "subf"]
        additional_variables = ['dt_ev', 'dt_cd', 'dta_evap', 'dta_cond', 'pot_abs', 'pot_frig', 'cop', 'ef_comp']

        # Mostrar todas las variables en el multiselect, con las predeterminadas seleccionadas por defecto
        selected_variables = st.multiselect(
            "Seleccione las variables para graficar:",
            default_variables + additional_variables,  # Combina las listas
            default=default_variables  # Seleccionadas por defecto
        )

        if selected_variables:
            # Crear la gráfica
            fig, ax = plt.subplots(figsize=(12, 5))  # Tamaño ajustado de la gráfica
            
            # Diccionario para personalizar los colores de las líneas
            line_colors = {
                "t_cam": "#33e6ff",
                "t_amb": "orange",
                "t_ev": "#3c33ff",
                "t_cd": "red",
                "t_des": "purple",
                "rec": "#33ff5b",
                "subf": "magenta",
                "dt_ev": "brown",
                "dt_cd": "pink",
                "dta_evap": "lime",
                "dta_cond": "teal",
                "pot_abs": "black",
                "pot_frig": "gold",
                "cop": "darkred",
                "ef_comp": "darkgreen"
        }

            # Diccionario para personalizar el grosor de las líneas
            line_widths = {
                "t_cam": 1,
                "t_amb": 1,
                "t_ev": 2,
                "t_cd": 2,
                "t_des": 1,
                "rec": 1,
                "subf": 1,
                "dt_ev": 1,
                "dt_cd": 1,
                "dta_evap": 1,
                "dta_cond": 1,
                "pot_abs": 1,
                "pot_frig": 1,
                "cop": 1,
                "ef_comp": 1
            }
            # Generar líneas con grosores y colores personalizados
            for var in selected_variables:
                ax.plot(
                    df_filtered["fecha"], 
                    df_filtered[var], 
                    label=var, 
                    linewidth=line_widths.get(var, 1),  # Grosor de la línea
                    color=line_colors.get(var, "gray")  # Color de la línea (por defecto: "gray" si no está en el diccionario)
            )

            # Configuración de la gráfica
            ax.set_title("Monitorización de variables", fontsize=10)  # Tamaño del título
            ax.set_xlabel("Fecha", fontsize=8)  # Tamaño del texto del eje X
            ax.set_ylabel("Tª (ºC), Pot (W)", fontsize=9)  # Tamaño del texto del eje Y
            ax.legend(fontsize=7)  # Reducir tamaño de la leyenda
            ax.tick_params(axis="both", labelsize=8)  # Tamaño de los números en los ejes
            ax.grid(True)  # Activar cuadrícula

            # Mostrar la gráfica
            st.pyplot(fig)



import math
import plotly.graph_objects as go

# (Opcional) Para que la app use todo el ancho de la ventana:
# st.set_page_config(layout="wide")

# =============================
# CONFIGURACIÓN VISUAL DE LOS INDICADORES
# =============================

# Configuración para indicadores principales (2x2)
config_principales = {
    "height": 250,  # Altura del indicador
    "width": 200,   # Ancho del indicador
    "titulo_tamano": 20,  # Tamaño del título
    "numero_tamano": 45,  # Tamaño del número central
    "tick_tamano": 20,    # Tamaño de los ticks
    "margenes": {"t": 30, "b": 10, "l": 50, "r": 50}# Márgenes
}

# Configuración para indicadores secundarios (3x3)
config_secundarios = {
    "height": 250,  # Altura del indicador
    "width": 200,   # Ancho del indicador
    "titulo_tamano": 20,  # Tamaño del título
    "numero_tamano": 45,  # Tamaño del número central
    "tick_tamano": 20,    # Tamaño de los ticks
    "margenes": {"t": 30, "b": 10, "l": 50, "r": 50}  # Márgenes
}

# =============================
# FUNCIÓN PARA CREAR INDICADORES
# =============================
def crear_indicador_aguja(variable, valor_real, texto, unidad,
                          config, valor_esperado=None, std=None, rangos=None):
    # 1. Determinar el rango de la escala
    if variable == "t_amb":
        rango_min, rango_max = 0, 60
    elif variable == "t_cam":
        rango_min, rango_max = -30, 0
    elif std and (valor_esperado is not None):
        rango_min = valor_esperado - 3 * std
        rango_max = valor_esperado + 3 * std
    else:
        if valor_real == 0:
            valor_real = 0.1
        rango_min = valor_real - abs(valor_real) * 0.5
        rango_max = valor_real + abs(valor_real) * 0.5

    # 2. Configurar ticks para respetar valor esperado y extremos
    tickmode = "array"
    tickvals = None
    ticktext = None

    if std and (valor_esperado is not None):
        # Ticks centrados en el valor esperado ± 3*std
        min_tick = rango_min
        mid_tick = valor_esperado
        max_tick = rango_max
        tickvals = [min_tick, mid_tick, max_tick]
        if variable in ["pot_abs", "pot_frig"]:
            ticktext = [f"{min_tick:.0f}", f"{mid_tick:.0f}", f"{max_tick:.0f}"]
        else:
            ticktext = [f"{min_tick:.1f}", f"{mid_tick:.1f}", f"{max_tick:.1f}"]

    # 3. Configurar formato sin decimales para "pot_abs" y "pot_frig"
    tickformat = None
    if variable in ["pot_abs", "pot_frig"]:
        tickformat = ".0f"  # Mostrar solo enteros

    # 4. Elegir color de aguja y número en función de rangos o std
    color_aguja = "gray"
    if rangos:
        if rangos["verde"][0] <= valor_real < rangos["verde"][1]:
            color_aguja = "green"
        elif rangos["naranja"][0] <= valor_real < rangos["naranja"][1]:
            color_aguja = "orange"
        else:
            color_aguja = "red"
    elif std and (valor_esperado is not None):
        diff = abs(valor_real - valor_esperado)
        if diff <= std:
            color_aguja = "green"
        elif diff <= 2 * std:
            color_aguja = "orange"
        else:
            color_aguja = "red"

    # 5. Redondear valor si es "pot_abs" o "pot_frig"
    if variable in ["pot_abs", "pot_frig"]:
        valor_real = round(valor_real)

    # 6. Crear la figura del indicador
    indicador = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor_real,  # Número central redondeado
        title={"text": f"{texto} ({unidad})", "font": {"size": config["titulo_tamano"]}},  # Tamaño del título
        number={"font": {"color": color_aguja, "size": config["numero_tamano"]}},  # Tamaño del número central
        gauge={
            "axis": {
                "range": [rango_min, rango_max],
                "tickmode": tickmode,
                "tickvals": tickvals,
                "ticktext": ticktext,
                "tickformat": tickformat,
                "tickfont": {"size": config["tick_tamano"]}  # Tamaño de los ticks
            },
            "bar": {"color": "white"},   # Sin barra rellena
            "threshold": {
                "line": {"color": color_aguja, "width": 6},
                "value": valor_real
            },
            "shape": "angular"
        }
    ))

    # 7. Añadir marca negra para el valor esperado (si lo hay)
    if valor_esperado is not None:
        frac_esp = (valor_esperado - rango_min) / (rango_max - rango_min)
        frac_esp = max(0, min(1, frac_esp))

        # 0° = derecha, 180° = izquierda
        angle_deg = 180 * (1 - frac_esp)
        angle_rad = math.radians(angle_deg)

        center_x, center_y = 0.5, 0.5
        inner_radius = 0.7
        outer_radius = 0.75

        x0 = center_x + inner_radius * math.cos(angle_rad)
        y0 = center_y + inner_radius * math.sin(angle_rad)
        x1 = center_x + outer_radius * math.cos(angle_rad)
        y1 = center_y + outer_radius * math.sin(angle_rad)

        indicador.add_shape(
            type="line",
            x0=x0, y0=y0,
            x1=x1, y1=y1,
            xref="paper", yref="paper",
            line=dict(color="black", width=3)
        )

    # 8. Ajustar layout
    indicador.update_layout(
        height=config["height"],  # Tamaño del indicador
        width=config["width"],   # Ancho del indicador
        margin=config["margenes"],  # Márgenes
        paper_bgcolor="white"
    )
    return indicador

st.markdown(
    "<h2 style='text-align: center;'>Variables en tiempo real</h2>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# CREAR 4 COLUMNAS
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4, gap="large")

# --------------------------------------------------
# UNIFICAR TODOS LOS INDICADORES EN UNA LISTA
# --------------------------------------------------
all_indicators = [
    # --- 4 indicadores principales ---
    (
        "t_cam", 
        df["t_cam"].iloc[0], 
        "Tª cámara", 
        "ºC", 
        None,  # valores_esperados.get("t_cam") si lo deseas
        None,  # stats.get("t_cam", {}).get("std") si lo deseas
        {"verde": (-30, -15), "naranja": (-15, -5), "rojo": (-5, float("inf"))},
        config_principales  # <-- Usa la config principal
    ),
    (
        "t_amb", 
        df["t_amb"].iloc[0], 
        "Tª ambiente", 
        "ºC", 
        None,  
        None,  
        {"verde": (0, 45), "naranja": (45, 55), "rojo": (55, float("inf"))},
        config_principales
    ),
    (
        "t_ev",  
        df["t_ev"].iloc[0],  
        "Tª evaporación", 
        "ºC",
        valores_esperados.get("t_ev"), 
        stats["t_ev"]["std"], 
        None,
        config_principales
    ),
    (
        "t_cd",  
        df["t_cd"].iloc[0],  
        "Tª condensación", 
        "ºC",
        valores_esperados.get("t_cd"), 
        stats["t_cd"]["std"], 
        None,
        config_principales
    ),

    # --- 6 indicadores secundarios ---
    (
        "t_des",  
        df["t_des"].iloc[0],  
        "Tª descarga", 
        "ºC",
        valores_esperados.get("t_des"), 
        stats["t_des"]["std"], 
        None,
        config_secundarios  # <-- Usa la config secundaria
    ),
    (
        "rec",    
        df["rec"].iloc[0],    
        "Recalentamiento", 
        "K",
        valores_esperados.get("rec"), 
        stats["rec"]["std"], 
        None,
        config_secundarios
    ),
    (
        "subf",   
        df["subf"].iloc[0],   
        "Subenfriamiento", 
        "K",
        valores_esperados.get("subf"), 
        stats["subf"]["std"], 
        None,
        config_secundarios
    ),
    (
        "pot_abs", 
        df["pot_abs"].iloc[0], 
        "Potencia absorbida", 
        "W",
        valores_esperados.get("pot_abs"), 
        stats["pot_abs"]["std"], 
        None,
        config_secundarios
    ),
    (
        "pot_frig", 
        df["pot_frig"].iloc[0], 
        "Potencia frigorífica", 
        "W",
        valores_esperados.get("pot_frig"), 
        stats["pot_frig"]["std"], 
        None,
        config_secundarios
    ),
    (
        "cop",    
        df["cop"].iloc[0],    
        "COP", 
        "",
        valores_esperados.get("cop"), 
        stats["cop"]["std"], 
        None,
        config_secundarios
    )
]


# --------------------------------------------------
# VERIFICAR CONDICIÓN DE pb
# --------------------------------------------------
if (
    (
        df["pb"].iloc[0] < 0.5
        and n_sd['subf'] > -umbral
        and df["t_cam"].iloc[0] < -19
    )
    or (df["pot_abs"].iloc[0] < 200)
):
    # Si se cumple la condición de recogida de gas, mostramos SOLO los 4 primeros indicadores (principales)
    indicators_to_show = all_indicators[:4]
else:
    # Si pb >= 0.5, mostramos TODOS
    indicators_to_show = all_indicators

# --------------------------------------------------
# MOSTRAR INDICADORES EN 4 COLUMNAS
# --------------------------------------------------


for i, (var, val_real, texto, unidad, val_esp, std, rangos, cfg) in enumerate(indicators_to_show):
    
    # Si cop <= 0, saltamos pot_frig y cop (si lo deseas)
    if var in ["pot_frig", "cop"] and df["cop"].iloc[0] <= 0:
        continue
    
    # Generar la figura
    fig = crear_indicador_aguja(
        variable=var,
        valor_real=val_real,
        texto=texto,
        unidad=unidad,
        config=cfg,  # <- Se aplica config_principales o config_secundarios
        valor_esperado=val_esp,
        std=std,
        rangos=rangos
    )
    
    # Ubicar en la columna correcta
    if i % 4 == 0:
        col1.plotly_chart(fig, use_container_width=True)
    elif i % 4 == 1:
        col2.plotly_chart(fig, use_container_width=True)
    elif i % 4 == 2:
        col3.plotly_chart(fig, use_container_width=True)
    else:
        col4.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# MENSAJE SI HAY RECOGIDA CDE GAS
# --------------------------------------------------

if (
    (
        df["pb"].iloc[0] < 0.5
        and n_sd['subf'] > -umbral
        and df["t_cam"].iloc[0] < -19
    )
    or (df["pot_abs"].iloc[0] < 200)
):
    st.markdown(
        """
        <div style="
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            color: white;
            background-color: #5B9BD5;
            padding: 10px;
            animation: blinker 2s linear infinite;
            margin-bottom: 50px;
        ">
            Equipo en parada por recogida de gas
        </div>
        <style>
            @keyframes blinker {
                50% { opacity: 0; }
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
      
# ========================================
# FUNCIÓN que detecta fallos en UNA fila
# ========================================
def detectar_fallos_en_fila(df_fila: pd.DataFrame, umbral: float) -> list:

    # 1) Crear df_esperado para esa fila
    valores_esperados_local = calcular_variables(df_fila.iloc[0:1])
    df_esperado_local = pd.DataFrame([valores_esperados_local])

    # 2) Calcular desviaciones
    df_desv = df_fila.select_dtypes(include=[np.number]).iloc[0] - df_esperado_local.select_dtypes(include=[np.number]).iloc[0]

    # 3) Calcular n_sd local
    n_sd_local = {}
    for var in stats:
        if var in df_desv:
            n_sd_local[var] = df_desv[var] / stats[var]['std']
        else:
            n_sd_local[var] = 0

    # 4) Aplicar la misma lógica de fallos
    fallos_local = []

    # == Mismo set de condiciones que en tu código ==
    if n_sd_local['t_ev'] < -2*umbral and n_sd_local['rec'] > 1.5*umbral and n_sd_local['subf'] > -umbral:
        fallos_local.append("Fallo 1: Obstrucción en línea de líquido / expansión insuficiente / filtro sucio")
    if n_sd_local['t_ev'] < -0.4*umbral and n_sd_local['subf'] < -0.5*umbral and n_sd_local['rec'] > umbral:
        fallos_local.append("Fallo 2: Falta de refrigerante")
    if n_sd_local['t_ev'] < -0.5*umbral and n_sd_local['dta_evap'] > umbral:
        fallos_local.append("Fallo 3: Caudal de aire insuficiente en el evaporador / fallo ventilador")
    if n_sd_local['t_ev'] < -umbral and n_sd_local['dta_evap'] <= umbral and n_sd_local['dta_evap'] >= -umbral:
        fallos_local.append("Fallo 3 y 4: Caudal de aire insuficiente en el evaporador / fallo ventilador + Transmisión insuficiente")
    if n_sd_local['t_ev'] < -umbral and n_sd_local['dta_evap'] < -umbral:
        fallos_local.append("Fallo 4: Transmisión insuficiente en el evaporador / suciedad / escarcha")
    if n_sd_local['t_ev'] > 0*umbral and n_sd_local['rec'] < -1.2*umbral:
        fallos_local.append("Fallo 5: Válvula de expansión demasiado abierta")
    if n_sd_local['t_ev'] > 0.5*umbral and n_sd_local['t_cd'] < -0.2*umbral:
        fallos_local.append("Fallo 6: Falta de capacidad o by-pass en compresor")
    if n_sd_local['t_cd'] > umbral and n_sd_local['dta_cond'] > umbral:
        fallos_local.append("Fallo 7: Caudal de aire insuficiente en el condensador / fallo ventilador")
    if n_sd_local['t_cd'] > umbral and -umbral <= n_sd_local['dta_cond'] <= umbral:
        fallos_local.append("Fallo 7 y 8: Caudal de aire insuficiente en el condensador + Transmisión insuficiente o suciedad")
    if n_sd_local['t_cd'] > umbral and n_sd_local['dta_cond'] < -umbral:
        fallos_local.append("Fallo 8: Transmisión insuficiente o suciedad en el condensador")
    if n_sd_local['t_cd'] > umbral and n_sd_local['subf'] > umbral:
        fallos_local.append("Fallo 9: Exceso de refrigerante")

    return fallos_local

# ========================================
# DETECCIÓN DE FALLOS EN LOS N ÚLTIMOS REGISTROS
# ========================================
# Suponiendo que df_processed es el DataFrame
# con todos los registros que usas para la gráfica,
# ordenado cronológicamente.

# En el sidebar: Selector para definir el número de registros N
n = st.sidebar.number_input(
    label="Nº registros consecutivos para detección de fallos",
    min_value=1,           # Valor mínimo permitido
    max_value=100,         # Valor máximo permitido
    value=6,               # Valor por defecto
    step=1,                # Incrementos de 1
)
    
df_ultimosN = df_processed.tail(n)

if len(df_ultimosN) < n:
    st.warning("No hay al menos N registros en el rango seleccionado.")
else:
    from collections import Counter
    
    fallos_counter = Counter()
    for i in range(len(df_ultimosN)):
        df_fila = df_ultimosN.iloc[i:i+1].copy()
        
        # Aplicas "detectar_fallos_en_fila"
        fallos_fila = detectar_fallos_en_fila(df_fila, umbral)
        
        for f in set(fallos_fila):
            fallos_counter[f] += 1

    fallos_en_N = [f for f, c in fallos_counter.items() if c == n]

    bloqueo_cumplido = (
        (
            df["pb"].iloc[0] < 0.5
            and n_sd['subf'] > -umbral
            and df["t_cam"].iloc[0] < -19
        )
        or (df["pot_abs"].iloc[0] < 200)
    )


    if fallos_en_N and not bloqueo_cumplido:

        # Mensaje de fallos, sin parpadeo, con fondo rojo claro
        for ff in sorted(fallos_en_N):
            st.markdown(
                f""" 
                <div style="
                    text-align: center;
                    background-color: #f08080;
                    color: white;
                    font-size: 24px;
                    font-weight: bold;
                    padding: 10px;
                    border-radius: 0px;
                    margin-bottom: 50;
                ">
                    {ff}
                </div>

                """,
                unsafe_allow_html=True
            )

    else:
        # Mensaje verde, idéntico, sin animación
        st.markdown(
            """
            <div style="
                text-align: center;
                background-color: #5cb85c;
                color: white;
                font-size: 24px;
                font-weight: bold;
                padding: 10px;
                border-radius: 0px;
            ">
                Sistema funcionando correctamente
            </div>

            """,
            unsafe_allow_html=True
        )


# Un bloque con 40px de margen vertical
st.markdown(
    """
    <div style="margin-top:50px;"></div>
    """,
    unsafe_allow_html=True
)



# ========================================
# Comparativa de valores Reales vs Esperados
# ========================================
st.markdown(
    "<h3 style='text-align: center;'>Comparativa valores Reales vs Esperados (Último registro)</h3>",
    unsafe_allow_html=True
)
col1, col2, col3 = st.columns([0.5, 4, 0.5])  # La columna central es más ancha
with col2:
    st.dataframe(final_df, width=1200, height=178)

# Mostrar detección de fallos
#if fallos and not bloqueo_cumplido:
    #for fallo in fallos:
        #col1, col2, col3 = st.columns([1, 2, 1])
        #with col2:
            #st.error(fallo)
#else:
    #col1, col2, col3 = st.columns([1, 2, 1])
    #with col2:
        #st.success("Sistema funcionando correctamente")
        


"""IO Solver 4.1 — seis módulos de Investigación de Operaciones y chatbot de IA.

Ejecución: python -m streamlit run appTuki.py
Autoprueba sin Streamlit: python appTuki.py --self-test
Dependencias: streamlit, scipy, matplotlib, numpy, google-genai
"""
from __future__ import annotations

import math
import random
import sys
import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from scipy.optimize import linprog, linear_sum_assignment

# Datos académicos fijos: se muestran a los visitantes, no son editables.
UNIVERSIDAD = "Universidad Mariano Gálvez de Guatemala"
ASIGNATURA = "Investigación de Operaciones"
INTEGRANTE_1 = "José Luis Turcios"
CARNE_1 = "2990-24-3012"
INTEGRANTE_2 = "Kevin Ernesto Mata Lemus"
CARNE_2 = "2990-24-13756"
# Sección confirmada por los integrantes.
SECCION = "A"


DISEÑO_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');
:root { color-scheme: dark; --ios-bg: #080d1d; --ios-panel: #121b34; --ios-panel2: #161f3c; --ios-border: rgba(164,185,249,.13); --ios-text: #eff2ff; --ios-muted: #a0add1; --ios-violet: #8d66ff; --ios-blue: #4da7ff; }
html, body, [class*="css"], .stApp { font-family: 'DM Sans', system-ui, sans-serif; }
.stApp, [data-testid="stAppViewContainer"] { color: var(--ios-text); background: radial-gradient(ellipse 75% 60% at 70% -25%, rgba(89,91,212,.22), transparent 72%), radial-gradient(ellipse 38% 34% at 95% 70%, rgba(24,71,167,.12), transparent 70%), var(--ios-bg); }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { max-width: 1350px; padding-top: 1.25rem !important; padding-bottom: 2.5rem !important; }
.stMainBlockContainer { padding-left: 2.9rem; padding-right: 2.9rem; }
h1, h2, h3, h4 { font-family: 'Outfit', 'DM Sans',sans-serif !important; letter-spacing: -.035em; }
h1 { font-size: clamp(2.1rem, 4vw, 3rem) !important; font-weight: 800 !important; }
h2 { font-weight: 700 !important; }
p, li, label { line-height: 1.6; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--ios-muted) !important; }
[data-testid="stSidebar"] { background: linear-gradient(174deg,#121b38 0%,#0d1530 53%, #0a1025 100%) !important; border-right: 1px solid var(--ios-border); }
[data-testid="stSidebar"] > div:first-child { padding-top: 1.55rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { margin-bottom: .2rem; }
[data-testid="stSidebar"] [role="radiogroup"] { gap: .35rem; }
[data-testid="stSidebar"] [role="radiogroup"] label { padding: .62rem .8rem; border-radius: 12px; transition: background .14s ease, border .14s ease; border: 1px solid transparent; }
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(131,118,239,.14); }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background: linear-gradient(100deg,#596aff,#9362ff) !important; border:1px solid rgba(211,198,255,.42); box-shadow:0 8px 22px rgba(80,77,207,.23); color:#fff !important; }
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p { color:#fff !important; font-weight:700; }
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stWidgetLabel"] {display:none;}
[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stMarkdownContainer"] p {font-size:.9rem;}
[data-testid="stSidebar"] [data-testid="stExpander"] {border-radius:14px; overflow:hidden;}
[data-testid="stSidebar"] hr { border-color:var(--ios-border); }
/* Main form controls */
[data-testid="stForm"], [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stAlert"] {border-radius:16px !important;}
[data-testid="stForm"] { border:1px solid var(--ios-border) !important; background:rgba(20,30,56,.65) !important; padding:1.2rem 1.3rem !important; }
[data-testid="stMetric"] { background:linear-gradient(160deg,rgba(32,45,83,.95),rgba(15,24,46,.90)); border:1px solid var(--ios-border); border-radius:16px; padding:1.1rem 1.25rem; box-shadow:0 8px 25px rgba(0,0,0,.12); }
[data-testid="stMetricLabel"] { color:#aab9e5; }
[data-testid="stMetricValue"] { color:#fff; font-weight:800; }
[data-testid="stNumberInput"] input, [data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea, [data-testid="stSelectbox"] div[data-baseweb="select"] > div { border-radius:10px !important; background:#0e1830 !important; border-color:#2c395c !important; }
[data-testid="stNumberInput"] input:focus, [data-testid="stTextInput"] input:focus { border-color:var(--ios-violet) !important; }
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] { background:linear-gradient(105deg,#466ff5,#8c5bee) !important; border:none !important; color:white !important; box-shadow:0 6px 16px rgba(98,88,220,.25); }
div.stButton > button, div.stFormSubmitButton > button { border-radius:11px !important; font-weight:700; min-height:2.65rem; transition:filter .15s, transform .15s; }
div.stButton > button:hover, div.stFormSubmitButton > button:hover {filter:brightness(1.1); transform:translateY(-1px);}
[data-baseweb="tab-list"] { border-bottom:1px solid var(--ios-border); }
button[data-baseweb="tab"] { font-weight:700; }
hr {border-color:var(--ios-border) !important;}
a {color:#98b3ff;}
/* Custom HTML UI */
.ios-topline {display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin:0 0 24px;padding:8px 0 18px;border-bottom:1px solid var(--ios-border);}
.ios-breadcrumb {font-size:.77rem;letter-spacing:.12em;text-transform:uppercase;color:#91a3cf;font-weight:700;}
.ios-status {display:inline-flex;align-items:center;gap:8px;padding:8px 13px;border:1px solid rgba(65,216,160,.16);color:#8ceac0;background:rgba(37,144,108,.13);border-radius:999px;font-size:.73rem;font-weight:800;letter-spacing:.025em;}
.ios-dot {height:7px;width:7px;display:inline-block;background:#51dda5;border-radius:50%;box-shadow:0 0 14px rgba(76,238,168,.5);}
.ios-brand {display:flex;align-items:center;gap:13px;margin:6px 0 25px;}
.ios-brandmark {flex:0 0 49px;width:49px;height:49px;background:linear-gradient(140deg,#496dff,#ac5fff);border-radius:15px;display:grid;place-items:center;box-shadow:0 8px 24px rgba(101,80,231,.34);font-weight:800;color:#fff;font-size:25px;}
.ios-brandname {color:#f4f5ff;font-family:'Outfit',sans-serif;font-size:1.42rem;font-weight:800;line-height:1.05;letter-spacing:-.035em;}
.ios-brandname span {color:#a58aff;}
.ios-brandsub {font-size:.68rem;color:#99a8d3;line-height:1.35;margin-top:5px;}
.ios-side-note {color:#8496c1;font-size:.72rem;margin:8px 0 0;line-height:1.5;}
.ios-hero {position:relative;min-height:340px;display:grid;grid-template-columns:minmax(0,1.46fr) minmax(250px,.9fr);gap:22px;align-items:center;overflow:hidden;border:1px solid rgba(123,149,255,.2);border-radius:23px;background:radial-gradient(circle at 71% 30%, rgba(75,98,245,.28), transparent 45%),linear-gradient(118deg,#182550 0%,#101a38 62%,#111b3c 100%);box-shadow:0 16px 54px rgba(2,5,20,.23);padding:42px 44px;}
.ios-hero::before {content:'';position:absolute;inset:0;background:linear-gradient(145deg,rgba(255,255,255,.035),transparent 40%);pointer-events:none;}
.ios-hero-copy {position:relative;z-index:2;}
.ios-eyebrow {color:#b9a8ff;font-weight:800;font-size:.69rem;letter-spacing:.14em;text-transform:uppercase;display:inline-block;margin-bottom:14px;}
.ios-hero h1 {font-size:clamp(2rem,3.4vw,3.35rem) !important;font-weight:800;line-height:1.06;letter-spacing:-.05em;margin:0 0 16px;max-width:630px;color:#fff;}
.ios-hero h1 span {background:linear-gradient(90deg,#77c2ff,#b79bff);-webkit-background-clip:text;background-clip:text;color:transparent;}
.ios-hero p {font-size:.96rem;line-height:1.6;color:#b7c5e7;max-width:555px;margin:0;}
.ios-pills {display:flex;gap:8px;flex-wrap:wrap;margin-top:24px;}
.ios-pills span {font-weight:700;font-size:.72rem;color:#d3dcff;background:rgba(147,170,255,.1);border:1px solid rgba(174,191,255,.18);border-radius:99px;padding:6px 11px;}
.ios-art {position:relative;height:260px;min-width:230px;}
.ios-orb {position:absolute;top:19px;left:17%;height:210px;width:210px;border-radius:50%;background:radial-gradient(circle at 32% 27%,#87bbff 0%,#4478ed 28%,#3340b9 63%,#131d4c 100%);box-shadow:inset -18px -13px 32px rgba(6,12,50,.6),0 0 62px rgba(84,103,252,.37);}
.ios-orb::after {content:'';position:absolute;inset:-35px -32px;border:1px solid rgba(156,185,255,.42);border-radius:50%;transform:rotate(-25deg) scaleY(.45);box-shadow:0 0 12px rgba(105,127,248,.25);}
.ios-ring {position:absolute;width:300px;height:90px;border:2px solid rgba(160,191,255,.43);border-radius:50%;top:79px;left:1%;transform:rotate(-24deg);box-shadow:0 0 16px rgba(85,140,255,.14);}
.ios-mini {position:absolute;z-index:2;background:rgba(19,32,69,.78);border:1px solid rgba(174,187,255,.23);border-radius:11px;color:#d7e0ff;backdrop-filter:blur(7px);font-size:.72rem;padding:10px 12px;font-weight:800;box-shadow:0 7px 20px rgba(3,7,24,.18);}
.ios-mini.top {right:0;top:0;}.ios-mini.bottom {left:0;bottom:5px;}
.ios-paloma-frame {position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:0 6px;}
.ios-paloma-frame img {width:min(100%,335px);max-height:290px;object-fit:contain;filter:drop-shadow(0 0 28px rgba(116,184,255,.26)) drop-shadow(0 0 54px rgba(168,105,255,.22));}
.ios-section-line {display:flex;justify-content:space-between;align-items:flex-end;margin:28px 0 15px;gap:14px;}
.ios-section-line h2 {color:#f0f3ff;font-size:1.4rem !important;margin:0;font-weight:800;}
.ios-section-line p {color:#9daed7;font-size:.83rem;margin:4px 0 0;}
.ios-info-card {padding:22px 22px 16px;min-height:166px;border:1px solid var(--ios-border);border-radius:17px;background:linear-gradient(160deg,#18233e,#121b32);margin-bottom:9px;}
.ios-icon {display:inline-flex;height:39px;width:39px;align-items:center;justify-content:center;border-radius:11px;background:rgba(107,123,255,.18);color:#bad0ff;font-size:21px;margin-bottom:10px;}
.ios-info-card strong {display:block;font-family:'Outfit',sans-serif;font-size:1.04rem;color:#f2f4ff;margin:0 0 5px;}
.ios-info-card p {font-size:.78rem;color:#aab9db;line-height:1.45;margin:0;}
.ios-smallbadge {color:#b7c5ed;font-size:.72rem;}
.ios-footer {font-size:.76rem;color:#8fa1cc;border-top:1px solid var(--ios-border);margin-top:25px;padding-top:17px;text-align:center;line-height:1.7;}
.ios-privacy {font-size:.72rem;color:#96a7cd;margin:0 0 15px;}
@media(max-width:900px){.ios-hero{grid-template-columns:1fr;min-height:unset;padding:30px 26px}.ios-art{display:none}.ios-hero h1{font-size:2.35rem!important}.stMainBlockContainer{padding-left:1rem;padding-right:1rem}.block-container{padding-top:1rem!important}}
@media(max-width:550px){.ios-topline{align-items:flex-start}.ios-status{font-size:.63rem}.ios-hero h1{font-size:2rem!important}.ios-info-card{min-height:0;padding:18px}}
</style>
"""



# ===========================================================================
# MOTOR DE CÁLCULO — independiente de Streamlit
# ===========================================================================

def calcular_eoq(demanda: float, pedido: float, mantenimiento: float) -> float:
    if not all(math.isfinite(v) for v in (demanda, pedido, mantenimiento)):
        raise ValueError("Ingresa valores numéricos finitos.")
    if min(demanda, pedido, mantenimiento) <= 0:
        raise ValueError("La demanda y los costos deben ser mayores que cero.")
    return math.sqrt(2 * demanda * pedido / mantenimiento)


def calcular_reorden(demanda: float, dias: float, entrega: float, seguridad: float) -> tuple[float, int]:
    if not all(math.isfinite(v) for v in (demanda, dias, entrega, seguridad)):
        raise ValueError("Ingresa valores numéricos finitos.")
    if demanda <= 0 or dias <= 0 or entrega < 0 or seguridad < 0:
        raise ValueError("Demanda y días deben ser positivos; entrega y seguridad no negativos.")
    valor = (demanda / dias) * entrega + seguridad
    return valor, math.ceil(valor)


def resolver_lp(tipo: str, coef: list[float], restricciones: list[tuple[float, float, str, float]]) -> dict:
    """LP con x,y continuas >=0; signos ≤, ≥, =."""
    if tipo not in ("Maximizar", "Minimizar") or len(coef) != 2 or not restricciones:
        raise ValueError("El modelo lineal está incompleto.")
    if not all(math.isfinite(float(v)) for v in coef):
        raise ValueError("Los coeficientes deben ser finitos.")
    Au, bu, Ae, be = [], [], [], []
    for a, b, op, rhs in restricciones:
        if not all(math.isfinite(float(v)) for v in (a, b, rhs)):
            raise ValueError("Las restricciones necesitan valores finitos.")
        if op == "≤":
            Au.append([a, b]); bu.append(rhs)
        elif op == "≥":
            Au.append([-a, -b]); bu.append(-rhs)
        elif op == "=":
            Ae.append([a, b]); be.append(rhs)
        else:
            raise ValueError("Solo se admiten ≤, ≥ o =.")
    c = [-float(v) for v in coef] if tipo == "Maximizar" else [float(v) for v in coef]
    res = linprog(c=c, A_ub=Au or None, b_ub=bu or None,
                  A_eq=Ae or None, b_eq=be or None,
                  bounds=[(0, None), (0, None)], method="highs")
    if res.status == 2:
        return {"estado": "inviable", "detalle": "Las restricciones son incompatibles."}
    if res.status == 3:
        return {"estado": "no_acotado", "detalle": "No existe un óptimo finito."}
    if not res.success:
        return {"estado": "error", "detalle": str(res.message)}
    x, y = [float(v) for v in res.x]
    z = float(coef[0] * x + coef[1] * y)
    return {"estado": "optimo", "x": x, "y": y, "z": z}


def resolver_simplex(coef: list[float], restricciones: list[tuple[float, float, float]], max_iter: int = 100) -> dict:
    """Tableaux simplex primales. Maximización, Ax<=b con b>=0 y x,y>=0.

    Usa regla de Bland para reducir riesgos de ciclaje por degeneración.
    """
    m = len(restricciones)
    if m < 1:
        raise ValueError("Incluye al menos una restricción.")
    if len(coef) != 2 or not all(math.isfinite(float(v)) for v in coef):
        raise ValueError("Coeficientes de objetivo inválidos.")
    for a, b, rhs in restricciones:
        if not all(math.isfinite(float(v)) for v in (a, b, rhs)) or rhs < 0:
            raise ValueError("En este simplex los límites deben ser no negativos.")
    nombres = ["x", "y"] + [f"s{i+1}" for i in range(m)]
    n = m + 2
    tabla: list[list[float]] = []
    for i, (a, b, rhs) in enumerate(restricciones):
        fila = [float(a), float(b)] + [0.0] * m + [float(rhs)]
        fila[2+i] = 1.0
        tabla.append(fila)
    tabla.append([-float(coef[0]), -float(coef[1])] + [0.0] * m + [0.0])
    base = list(range(2, n))
    pasos = [{"tabla": [f.copy() for f in tabla], "base": base.copy(), "detalle": "Tabla inicial"}]
    tol = 1e-9

    for _ in range(max_iter):
        entrantes = [j for j in range(n) if tabla[-1][j] < -tol]
        if not entrantes:
            sol = [0.0] * n
            for i, j in enumerate(base):
                sol[j] = tabla[i][-1]
            return {"estado": "optimo", "x": sol[0], "y": sol[1],
                    "z": tabla[-1][-1], "pasos": pasos, "nombres": nombres}
        entrada = min(entrantes)
        candidatos = [(tabla[i][-1] / tabla[i][entrada], base[i], i)
                      for i in range(m) if tabla[i][entrada] > tol]
        if not candidatos:
            return {"estado": "no_acotado", "pasos": pasos, "nombres": nombres}
        _, _, salida = min(candidatos)
        saliente = nombres[base[salida]]
        razones = [(nombres[base[i]], tabla[i][-1], tabla[i][entrada])
                   for i in range(m) if tabla[i][entrada] > tol]
        pivote = tabla[salida][entrada]
        tabla[salida] = [v / pivote for v in tabla[salida]]
        for i in range(m + 1):
            if i == salida:
                continue
            factor = tabla[i][entrada]
            tabla[i] = [tabla[i][j] - factor * tabla[salida][j] for j in range(n+1)]
        base[salida] = entrada
        for fila in tabla:
            for j in range(len(fila)):
                if abs(fila[j]) < tol:
                    fila[j] = 0.0
        pasos.append({"tabla": [f.copy() for f in tabla], "base": base.copy(),
                      "detalle": f"Entra {nombres[entrada]}, sale {saliente}; pivote = {pivote:.4g}.",
                      "razones": razones})
    return {"estado": "limite", "pasos": pasos, "nombres": nombres}


def resolver_transporte(ofertas: list[float], demandas: list[float], costos: list[list[float]]) -> dict:
    no, nd = len(ofertas), len(demandas)
    if not no or not nd or len(costos) != no or any(len(f) != nd for f in costos):
        raise ValueError("Las dimensiones de la matriz de transporte son incorrectas.")
    valores = ofertas + demandas + [v for f in costos for v in f]
    if not all(math.isfinite(float(v)) and v >= 0 for v in valores):
        raise ValueError("Ofertas, demandas y costos deben ser finitos y no negativos.")
    if sum(demandas) <= 0:
        raise ValueError("Debe haber demanda positiva.")
    if sum(demandas) > sum(ofertas) + 1e-8:
        raise ValueError("La oferta total es insuficiente para cubrir la demanda.")
    n = no * nd
    A_o = [[1 if k // nd == i else 0 for k in range(n)] for i in range(no)]
    A_d = [[1 if k % nd == j else 0 for k in range(n)] for j in range(nd)]
    res = linprog(c=[costos[i][j] for i in range(no) for j in range(nd)],
                  A_ub=A_o, b_ub=ofertas, A_eq=A_d, b_eq=demandas,
                  bounds=(0, None), method="highs")
    if not res.success:
        raise ValueError("No se encontró un plan de transporte factible.")
    matriz = [[float(res.x[i*nd+j]) if abs(res.x[i*nd+j]) > 1e-8 else 0.0
               for j in range(nd)] for i in range(no)]
    return {"costo": float(res.fun), "envios": matriz,
            "sobrante": float(sum(ofertas)-sum(demandas))}


def resolver_asignacion(matriz: list[list[float]], maximizar: bool = False) -> dict:
    n = len(matriz)
    if n < 2 or any(len(f) != n for f in matriz):
        raise ValueError("Usa la misma cantidad de trabajadores y tareas (al menos 2).")
    if not all(math.isfinite(float(v)) for f in matriz for v in f):
        raise ValueError("La matriz debe contener números finitos.")
    fi, co = linear_sum_assignment(matriz, maximize=maximizar)
    parejas = [(int(i), int(j), float(matriz[i][j])) for i, j in zip(fi, co)]
    return {"valor": sum(t[2] for t in parejas), "parejas": parejas}


# ===========================================================================
# CONTROLES Y PRESENTACIÓN
# ===========================================================================

def lp_form(st, clave: str, con_descripciones: bool = False):
    n = st.selectbox("Cantidad de restricciones", [2, 3, 4, 5, 6], key=f"{clave}_n")
    with st.form(f"form_{clave}"):
        texto = ""
        nombre_x, nombre_y = "Variable X", "Variable Y"
        if con_descripciones:
            texto = st.text_area("Enunciado (referencia opcional)", key=f"{clave}_enunciado")
            cx1, cx2 = st.columns(2)
            nombre_x = cx1.text_input("¿Qué representa X?", "Producto A", key=f"{clave}_nombre_x")
            nombre_y = cx2.text_input("¿Qué representa Y?", "Producto B", key=f"{clave}_nombre_y")
        tipo = st.radio("Objetivo", ["Maximizar", "Minimizar"], horizontal=True, key=f"{clave}_tipo")
        st.markdown("**Función objetivo**")
        c1, c2 = st.columns(2)
        a = c1.number_input("Coeficiente de x", value=40.0, key=f"{clave}_cx")
        b = c2.number_input("Coeficiente de y", value=30.0, key=f"{clave}_cy")
        st.markdown("**Restricciones** (se supone x ≥ 0, y ≥ 0)")
        filas = []
        for i in range(n):
            cc = st.columns([1, 1, 1, 1.3])
            d_a = float(2 if i == 0 else 1 if i == 1 else 0)
            d_b = float(1 if i == 0 else 2 if i == 1 else 0)
            d_rhs = float(100 if i == 0 else 80 if i == 1 else 0)
            r_a = cc[0].number_input(f"x · R{i+1}", value=d_a, key=f"{clave}_a{i}")
            r_b = cc[1].number_input(f"y · R{i+1}", value=d_b, key=f"{clave}_b{i}")
            r_op = cc[2].selectbox(f"Signo R{i+1}", ["≤", "≥", "="], key=f"{clave}_op{i}")
            r_rhs = cc[3].number_input(f"Límite R{i+1}", value=d_rhs, key=f"{clave}_rhs{i}")
            filas.append((r_a, r_b, r_op, r_rhs))
        ok = st.form_submit_button("Resolver ejercicio", use_container_width=True, type="primary")
    return ok, tipo, [a, b], filas, (nombre_x, nombre_y, texto)


def mostrar_modelo(st, tipo, coef, restricciones):
    st.subheader("Modelo matemático")
    st.write(f"Objetivo: **{tipo}**")
    st.latex(rf"Z = ({coef[0]:g})x + ({coef[1]:g})y")
    signos = {"≤": r"\leq", "≥": r"\geq", "=": "="}
    for i, (a, b, op, rhs) in enumerate(restricciones, 1):
        st.latex(rf"R_{{{i}}}: ({a:g})x + ({b:g})y {signos[op]} {rhs:g}")
    st.latex(r"x\geq 0,\quad y\geq 0")


def mostrar_solucion_lp(st, tipo, coef, restricciones, res, nombres=None):
    if res["estado"] != "optimo":
        if res["estado"] == "inviable":
            st.error("Problema sin solución factible: las restricciones son incompatibles.")
        elif res["estado"] == "no_acotado":
            st.warning("La función objetivo no tiene un óptimo finito.")
        else:
            st.error(f"No se pudo resolver: {res.get('detalle', '')}")
        return
    x, y, z = res["x"], res["y"], res["z"]
    c1, c2, c3 = st.columns(3)
    c1.metric("X óptima", f"{x:,.4f}")
    c2.metric("Y óptima", f"{y:,.4f}")
    c3.metric("Valor óptimo Z", f"{z:,.4f}")
    st.success("Solución óptima encontrada.")
    st.subheader("Sustitución y comprobación")
    st.latex(rf"Z = ({coef[0]:g})({x:.4f}) + ({coef[1]:g})({y:.4f}) = {z:.4f}")
    for i, (a, b, op, rhs) in enumerate(restricciones, 1):
        lhs = a*x+b*y
        tol = 1e-6 * max(1, abs(lhs), abs(rhs))
        pasa = lhs <= rhs+tol if op == "≤" else lhs >= rhs-tol if op == "≥" else abs(lhs-rhs) <= tol
        st.write(f"R{i}: {a:g}({x:.4f}) + {b:g}({y:.4f}) = {lhs:.4f} {op} {rhs:g} — {'✅ Cumple' if pasa else '⚠️ Revisar'}")
    if nombres:
        st.info(f"Interpretación: una solución óptima consiste en {x:.2f} de {nombres[0]} y {y:.2f} de {nombres[1]}, con Z = {z:.2f}.")
    else:
        st.info(f"Interpretación: para {tipo.lower()} Z, una solución óptima es x = {x:.2f}, y = {y:.2f}, Z = {z:.2f}.")
    st.caption("Las variables son continuas y no negativas; no se exige que las cantidades sean enteras.")


def recortar_poligono(poly, a, b, c):
    """Recorta un polígono con el semiplano ax+by<=c."""
    if not poly:
        return []
    nuevo = []
    anterior = poly[-1]
    for actual in poly:
        f_prev = a*anterior[0]+b*anterior[1]-c
        f_act = a*actual[0]+b*actual[1]-c
        d_prev, d_act = f_prev <= 1e-8, f_act <= 1e-8
        if d_prev != d_act:
            t = f_prev/(f_prev-f_act)
            nuevo.append((anterior[0]+t*(actual[0]-anterior[0]),
                          anterior[1]+t*(actual[1]-anterior[1])))
        if d_act:
            nuevo.append(actual)
        anterior = actual
    return nuevo


def graficar_lp(st, coef, restricciones, res):
    import numpy as np
    import matplotlib.pyplot as plt

    x, y, z = res["x"], res["y"], res["z"]
    posibles_x, posibles_y = [10.0, x*1.3+2], [10.0, y*1.3+2]
    for a, b, _, rhs in restricciones:
        if abs(a) > 1e-10 and rhs/a > 0:
            posibles_x.append((rhs/a)*1.2)
        if abs(b) > 1e-10 and rhs/b > 0:
            posibles_y.append((rhs/b)*1.2)
    max_x, max_y = min(max(posibles_x), 1e7), min(max(posibles_y), 1e7)
    poly = [(0, 0), (max_x, 0), (max_x, max_y), (0, max_y)]
    for a, b, op, rhs in restricciones:
        if op in ("≤", "="):
            poly = recortar_poligono(poly, a, b, rhs)
        if op in ("≥", "="):
            poly = recortar_poligono(poly, -a, -b, -rhs)
    fig, ax = plt.subplots(figsize=(9, 5.7))
    fig.patch.set_facecolor("#111a31")
    ax.set_facecolor("#121e38")
    ax.set_prop_cycle(color=["#67b7ff", "#d196ff", "#ffcd86", "#71dfc2", "#ee84b7", "#8fa9ff"])
    ax.tick_params(colors="#c6d2f1")
    ax.xaxis.label.set_color("#e3e9ff")
    ax.yaxis.label.set_color("#e3e9ff")
    ax.title.set_color("#f1f4ff")
    for spine in ax.spines.values():
        spine.set_color("#45577a")
    if len(poly) >= 3:
        ax.fill([p[0] for p in poly], [p[1] for p in poly], alpha=0.25, color="#4ce1bb", label="Región factible visible")
    elif len(poly) == 2:
        ax.plot([p[0] for p in poly], [p[1] for p in poly], linewidth=4, label="Conjunto factible visible")
    xv = np.linspace(0, max_x, 400)
    for i, (a, b, op, rhs) in enumerate(restricciones, 1):
        etiqueta = f"R{i}: {a:g}x {b:+g}y {op} {rhs:g}"
        if abs(b) > 1e-10:
            ax.plot(xv, (rhs-a*xv)/b, linewidth=1.7, label=etiqueta)
        elif abs(a) > 1e-10:
            ax.axvline(rhs/a, linewidth=1.7, label=etiqueta)
    if abs(coef[1]) > 1e-10:
        ax.plot(xv, (z-coef[0]*xv)/coef[1], linestyle="--", color="#ffd18a", alpha=0.85, label="Objetivo óptimo")
    elif abs(coef[0]) > 1e-10:
        ax.axvline(z/coef[0], linestyle="--", color="#ffd18a", alpha=0.85, label="Objetivo óptimo")
    ax.scatter([x], [y], s=80, color="#ff769b", zorder=5, label=f"Óptimo ({x:.2f}, {y:.2f})")
    ax.annotate(f"({x:.2f}, {y:.2f})", (x, y), xytext=(8, 9), textcoords="offset points", fontsize=9, color="#f3f5ff")
    ax.set(xlim=(0,max_x), ylim=(0,max_y), xlabel="Variable X", ylabel="Variable Y", title="Región factible y solución óptima")
    ax.grid(alpha=0.16, color="#9fb6ee")
    leg = ax.legend(fontsize=8, loc="best", facecolor="#1b2947", edgecolor="#374b78")
    for t in leg.get_texts():
        t.set_color("#f2f5ff")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Se representa una ventana del plano. Una región no acotada puede continuar fuera de estos límites.")


def mostrar_tablas_simplex(st, res):
    st.subheader("Tablas e iteraciones")
    nombres = res["nombres"]
    for i, paso in enumerate(res["pasos"]):
        st.markdown(f"**{'Tabla inicial' if i == 0 else f'Iteración {i}'}**")
        st.write(paso["detalle"])
        if i > 0:
            st.write("Razones consideradas: " + "; ".join(
                f"{nom}: {rhs:.4f}/{coef:.4f} = {rhs/coef:.4f}"
                for nom, rhs, coef in paso.get("razones", [])))
            st.caption("Se normaliza la fila pivote y se eliminan los demás valores de su columna.")
        filas = []
        for k, fila in enumerate(paso["tabla"]):
            f = {"Base": nombres[paso["base"][k]] if k < len(paso["base"]) else "Z"}
            for j, nombre in enumerate(nombres):
                f[nombre] = round(fila[j], 4)
            f["LD"] = round(fila[-1], 4)
            filas.append(f)
        st.dataframe(filas, use_container_width=True, hide_index=True)


def ir_a(pagina):
    import streamlit as st
    st.session_state["navegacion_io"] = pagina




def _archivo_recurso(nombre: str) -> Path | None:
    candidatos = [Path(__file__).resolve().parent / nombre, Path('/mnt/data') / nombre]
    for ruta in candidatos:
        if ruta.exists():
            return ruta
    return None


def _hero_art_markup() -> str:
    ruta = _archivo_recurso('paloma_paraclito.png') or _archivo_recurso('paloma_celestial_en_un_portal_de_luz.png')
    if ruta is not None:
        b64 = base64.b64encode(ruta.read_bytes()).decode('ascii')
        return f'<div class="ios-paloma-frame"><img src="data:image/png;base64,{b64}" alt="Paloma luminosa"></div>'
    return '<div class="ios-orb"></div><div class="ios-ring"></div><div class="ios-mini top">↗ Optimización real</div><div class="ios-mini bottom">✦ Matemática aplicada</div>'

def pantalla_inicio(st):
    hero_art = _hero_art_markup()
    st.markdown(f"""
    <div class="ios-hero">
      <div class="ios-hero-copy">
        <span class="ios-eyebrow">✦ BIENVENIDO A IO SOLVER</span>
        <h1>Convierte desafíos en <span>decisiones inteligentes.</span></h1>
        <p>Tu espacio académico para modelar, optimizar y comprender problemas de Investigación de Operaciones. Calcula con datos reales y aprende cada procedimiento.</p>
        <div class="ios-pills"><span>✓ 6 módulos</span><span>✓ Soluciones paso a paso</span><span>✓ Sin registro</span></div>
      </div>
      <div class="ios-art" aria-hidden="true">
        {hero_art}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    empezar, practicar, _ = st.columns([1.3, 1.15, 2])
    with empezar:
        st.button("🤖 Preguntar a la IA", key="inicio_resolver", type="primary", use_container_width=True,
                  on_click=ir_a, args=("🤖 Asistente IA",))
    with practicar:
        st.button("📚 Ver prácticas", key="inicio_practicar", use_container_width=True,
                  on_click=ir_a, args=("📚 Centro de aprendizaje",))
    st.write("")

    c1, c2, c3 = st.columns(3, gap="medium")
    c1.metric("Herramientas académicas", "6 módulos + IA")
    c2.metric("Ejercicios resueltos", "11 ejemplos")
    c3.metric("Acceso a la plataforma", "Sin registro")

    st.markdown('<div class="ios-section-line"><div><h2>Explora las herramientas</h2><p>Selecciona un área para comenzar a trabajar.</p></div><span class="ios-smallbadge">INVESTIGACIÓN DE OPERACIONES</span></div>', unsafe_allow_html=True)
    herramientas = [
        ("🤖", "Asistente IA", "Describe tu problema y deja que la IA escoja el método.", "🤖 Asistente IA"),
        ("📝", "Modelación", "Identifica variables, objetivos y restricciones.", "📝 Modelación"),
        ("📈", "Programación lineal", "Optimiza funciones con restricciones.", "📈 Programación lineal"),
        ("📊", "Método gráfico", "Visualiza regiones factibles y puntos óptimos.", "📊 Método gráfico"),
        ("🧮", "Método simplex", "Aprende pivoteo, tablas e iteraciones.", "🧮 Método simplex"),
        ("🚚", "Transporte y asignación", "Distribuye recursos con el menor costo.", "🚚 Transporte y asignación"),
        ("📦", "Inventarios", "Calcula EOQ y punto de reorden.", "📦 Inventarios"),
    ]
    for offset in (0, 3, 6):
        columnas = st.columns(3, gap="medium")
        for idx, columna in enumerate(columnas):
            if offset + idx >= len(herramientas):
                continue
            icono, nombre, texto, pagina = herramientas[offset + idx]
            with columna:
                st.markdown(f'<div class="ios-info-card"><span class="ios-icon">{icono}</span><strong>{nombre}</strong><p>{texto}</p></div>', unsafe_allow_html=True)
                st.button(f"Abrir {nombre} →", key=f"home_go_{offset+idx}", use_container_width=True,
                          on_click=ir_a, args=(pagina,))
    st.write("")
    st.info("**¿Estás aprendiendo?** Visita el Centro de aprendizaje para practicar con ejercicios nuevos y recibir retroalimentación.")

def pantalla_modelacion(st):
    st.title("📝 Modelación matemática")
    st.caption("Modelación guiada: el enunciado es de referencia; introduce manualmente los coeficientes y restricciones.")
    ok, tipo, c, rs, info = lp_form(st, "modelo", con_descripciones=True)
    if ok:
        nx, ny, texto = info
        if not nx.strip() or not ny.strip() or nx.strip().casefold() == ny.strip().casefold():
            st.error("Describe dos variables de decisión diferentes."); return
        st.subheader("Variables de decisión")
        st.write(f"**x:** {nx}  |  **y:** {ny}")
        if texto.strip():
            with st.expander("Enunciado del usuario"):
                st.write(texto)
        mostrar_modelo(st, tipo, c, rs)
        try:
            res = resolver_lp(tipo, c, rs)
            mostrar_solucion_lp(st, tipo, c, rs, res, (nx, ny))
        except ValueError as e:
            st.error(str(e))


def pantalla_lineal(st, grafica: bool = False):
    if grafica:
        st.title("📊 Método gráfico")
        st.write("Introduce un modelo de dos variables. IO Solver dibujará las restricciones, la región factible visible y la solución óptima.")
        clave = "grafico"
    else:
        st.title("📈 Programación lineal")
        st.write("Resuelve maximización y minimización con dos variables y restricciones ≤, ≥ o =.")
        clave = "lineal"
    ok, tipo, c, rs, _ = lp_form(st, clave)
    if ok:
        try:
            mostrar_modelo(st, tipo, c, rs)
            res = resolver_lp(tipo, c, rs)
            mostrar_solucion_lp(st, tipo, c, rs, res)
            if res.get("estado") == "optimo":
                if grafica:
                    st.divider()
                    st.subheader("Región factible y solución gráfica")
                    graficar_lp(st, c, rs, res)
                else:
                    st.divider()
                    st.subheader("Vista gráfica del resultado")
                    st.caption("Además del resultado numérico, aquí se muestra la representación del método gráfico para el mismo problema de dos variables.")
                    graficar_lp(st, c, rs, res)
                    st.info("Si deseas trabajar solamente la representación visual, también puedes abrir el módulo independiente **Método gráfico** desde el menú lateral.")
        except ValueError as e:
            st.error(str(e))


def pantalla_simplex(st):
    st.title("🧮 Método simplex")
    st.info("Versión básica: maximización, dos variables no negativas, restricciones ≤ y límites no negativos.")
    n = st.selectbox("Cantidad de restricciones", [2,3,4,5,6], key="simp_n")
    with st.form("form_simplex2"):
        a, b = st.columns(2)
        c = [a.number_input("Coeficiente de x", value=40.0, key="simp_cx"),
             b.number_input("Coeficiente de y", value=30.0, key="simp_cy")]
        rs = []
        for i in range(n):
            a, b, d = st.columns(3)
            aa = a.number_input(f"x · R{i+1}", value=float(2 if i == 0 else 1 if i == 1 else 0), key=f"simp_a{i}")
            bb = b.number_input(f"y · R{i+1}", value=float(1 if i == 0 else 2 if i == 1 else 0), key=f"simp_b{i}")
            rhs = d.number_input(f"Límite · R{i+1}", min_value=0.0, value=float(100 if i == 0 else 80 if i == 1 else 0), key=f"simp_rhs{i}")
            rs.append((aa, bb, rhs))
        ok = st.form_submit_button("Resolver mediante tablas simplex", use_container_width=True, type="primary")
    if ok:
        try:
            st.subheader("Transformación a forma estándar")
            st.latex(rf"\max\ Z = ({c[0]:g})x+({c[1]:g})y")
            for i, (aa, bb, rhs) in enumerate(rs, 1):
                st.latex(rf"({aa:g})x+({bb:g})y+s_{{{i}}}={rhs:g}")
            st.latex(r"x,y,s_i\geq0")
            res = resolver_simplex(c, rs)
            mostrar_tablas_simplex(st, res)
            if res["estado"] == "optimo":
                a, b, z = st.columns(3)
                a.metric("X", f"{res['x']:.4f}")
                b.metric("Y", f"{res['y']:.4f}")
                z.metric("Máximo Z", f"{res['z']:.4f}")
                st.latex(rf"Z=({c[0]:g})({res['x']:.4f})+({c[1]:g})({res['y']:.4f})={res['z']:.4f}")
                st.success(f"Óptimo encontrado tras {len(res['pasos'])-1} iteraciones.")
            elif res["estado"] == "no_acotado":
                st.warning("No existe máximo finito (problema no acotado).")
            else:
                st.warning("Se alcanzó el límite de iteraciones; no se confirma un óptimo.")
        except ValueError as e:
            st.error(str(e))


def pantalla_transporte_asignacion(st):
    st.title("🚚 Transporte y asignación")
    tr, asg = st.tabs(["Transporte", "Asignación"])
    with tr:
        st.write("Calcula envíos entre orígenes y destinos, con demanda satisfecha y costo mínimo.")
        no = st.selectbox("Orígenes", [2,3,4], key="t_no")
        nd = st.selectbox("Destinos", [2,3,4], key="t_nd")
        with st.form("form_transport_v2"):
            col_a, col_b = st.columns(2)
            with col_a:
                ofertas = [st.number_input(f"Oferta O{i+1}", min_value=0.0, value=float(30 if i==0 else 40 if i==1 else 0), key=f"t_o{i}") for i in range(no)]
            with col_b:
                demandas = [st.number_input(f"Demanda D{j+1}", min_value=0.0, value=float(30 if j==0 else 40 if j==1 else 0), key=f"t_d{j}") for j in range(nd)]
            st.markdown("**Costos por unidad enviada**")
            base = [[4,6],[5,3]]
            costos = []
            for i in range(no):
                columnas = st.columns(nd)
                fila = []
                for j in range(nd):
                    valor = float(base[i][j] if i<2 and j<2 else 5)
                    fila.append(columnas[j].number_input(f"O{i+1} a D{j+1}", min_value=0.0, value=valor, key=f"t_c{i}_{j}"))
                costos.append(fila)
            ok = st.form_submit_button("Optimizar transporte", type="primary", use_container_width=True)
        if ok:
            try:
                res = resolver_transporte(ofertas, demandas, costos)
                st.latex(r"\min Z=\sum_i\sum_j c_{ij}x_{ij}")
                st.write("Oferta por origen (≤ capacidad) y demanda por destino (= necesidad).")
                datos = []
                for i, fila in enumerate(res["envios"]):
                    datos.append({"Origen":f"O{i+1}", **{f"D{j+1}":round(v,4) for j,v in enumerate(fila)}})
                st.dataframe(datos, hide_index=True, use_container_width=True)
                st.subheader("Procedimiento y costo por ruta")
                for i, fila in enumerate(res["envios"]):
                    for j, cantidad in enumerate(fila):
                        if cantidad > 1e-8:
                            st.write(f"O{i+1} → D{j+1}: {cantidad:.4f} × {costos[i][j]:.4f} = {cantidad*costos[i][j]:.4f}")
                st.metric("Costo mínimo total", f"Q {res['costo']:,.2f}")
                if res["sobrante"] > 1e-7:
                    st.info(f"Quedan {res['sobrante']:.2f} unidades de oferta sin utilizar.")
            except ValueError as e:
                st.error(str(e))
    with asg:
        st.write("Asignación uno a uno: igual número de trabajadores y tareas.")
        n = st.selectbox("Número de trabajadores y tareas", [2,3,4,5], index=1, key="as_n")
        with st.form("form_assignment_v2"):
            tipo = st.radio("Objetivo", ["Minimizar costo", "Maximizar beneficio"], horizontal=True, key="as_tipo")
            valores = [[9,2,7],[6,4,3],[5,8,1]]
            matriz = []
            for i in range(n):
                columnas = st.columns(n)
                fila = [columnas[j].number_input(f"Trabajador {i+1}, tarea {j+1}", min_value=0.0,
                                                   value=float(valores[i][j] if i<3 and j<3 else 5), key=f"as_c{i}_{j}") for j in range(n)]
                matriz.append(fila)
            ok = st.form_submit_button("Resolver asignación", type="primary", use_container_width=True)
        if ok:
            try:
                res = resolver_asignacion(matriz, tipo.startswith("Maximizar"))
                st.latex(r"Z=\sum_i\sum_j c_{ij}x_{ij},\quad x_{ij}\in\{0,1\}")
                st.write("Cada trabajador realiza una sola tarea y cada tarea tiene un trabajador.")
                st.dataframe([{"Trabajador":f"T{i+1}","Tarea":f"Tarea {j+1}","Valor":v} for i,j,v in res["parejas"]],
                             hide_index=True, use_container_width=True)
                st.write("Sustitución: " + " + ".join(f"{v:g}" for _,_,v in res["parejas"]) + f" = {res['valor']:.2f}")
                st.metric("Beneficio máximo" if tipo.startswith("Maximizar") else "Costo mínimo", f"{res['valor']:,.2f}")
                st.caption("Optimización combinatoria exacta; no se representan iteraciones del método húngaro.")
            except ValueError as e:
                st.error(str(e))


def pantalla_inventarios(st):
    st.title("📦 Inventarios")
    eoq, rop = st.tabs(["Cantidad económica de pedido (EOQ)", "Punto de reorden"])
    with eoq:
        with st.form("form_eoq2"):
            D = st.number_input("Demanda anual (D)", min_value=0.01, value=1200.0, key="eoq_d")
            S = st.number_input("Costo por pedido (S)", min_value=0.01, value=50.0, key="eoq_s")
            H = st.number_input("Mantenimiento por unidad al año (H)", min_value=0.01, value=2.0, key="eoq_h")
            ok = st.form_submit_button("Calcular EOQ", type="primary", use_container_width=True)
        if ok:
            try:
                resultado = calcular_eoq(D,S,H)
                st.latex(r"EOQ=\sqrt{\frac{2DS}{H}}")
                st.latex(rf"EOQ=\sqrt{{\frac{{2({D:g})({S:g})}}{{{H:g}}}}}={resultado:.4f}")
                st.metric("Cantidad económica óptima", f"{resultado:.2f} unidades")
                st.info(f"En el modelo clásico, la cantidad económica es {resultado:.2f} unidades por pedido. Si se requieren unidades enteras, evalúa redondear a {round(resultado)}.")
            except ValueError as e:
                st.error(str(e))
    with rop:
        with st.form("form_rop2"):
            D = st.number_input("Demanda anual", min_value=0.01, value=1200.0, key="rop_d")
            N = st.number_input("Días de operación por año", min_value=1, max_value=366, value=300, key="rop_n")
            L = st.number_input("Tiempo de entrega en días", min_value=0.0, value=5.0, key="rop_l")
            SS = st.number_input("Stock de seguridad", min_value=0.0, value=0.0, key="rop_ss")
            ok = st.form_submit_button("Calcular punto de reorden", type="primary", use_container_width=True)
        if ok:
            try:
                exacto, entero = calcular_reorden(D,N,L,SS)
                st.latex(r"PR=dL+SS,\qquad d=\frac{D}{N}")
                st.latex(rf"d=\frac{{{D:g}}}{{{N:g}}}={D/N:.4f}")
                st.latex(rf"PR=({D/N:.4f})({L:g})+{SS:g}\approx{exacto:.4f}")
                st.metric("Punto de reorden", f"{entero} unidades")
                st.info(f"Realiza un pedido cuando la posición del inventario alcance {entero} unidades aproximadamente (redondeado hacia arriba).")
                st.caption("Modelo con demanda y tiempo de entrega constantes.")
            except ValueError as e:
                st.error(str(e))


# ===========================================================================
# CENTRO DE APRENDIZAJE — 11 ejemplos y prácticas dinámicas
# ===========================================================================

def catalogo():
    # Tipo, título, enunciado, datos. Los resultados NO están precalculados.
    return [
        ("eoq", "01. EOQ — Papelería", "D=1200 unidades/año, S=Q50 por pedido, H=Q2/unidad/año.", (1200,50,2)),
        ("eoq", "02. EOQ — Almacén", "D=3200, S=Q75 y H=Q3.", (3200,75,3)),
        ("reorden", "03. Punto de reorden — Producto A", "D=1200, N=300 días, L=5 días, SS=0.", (1200,300,5,0)),
        ("reorden", "04. Punto de reorden — Producto B", "D=2400, N=300 días, L=7 días, SS=15.", (2400,300,7,15)),
        ("lp", "05. Programación lineal — Máximo", "Max Z=40x+30y; 2x+y≤100, x+2y≤80.", ("Maximizar",[40,30],[(2,1,"≤",100),(1,2,"≤",80)])),
        ("lp", "06. Programación lineal — Mínimo", "Min Z=3x+5y; x+y≥8, x+3y≥12.", ("Minimizar",[3,5],[(1,1,"≥",8),(1,3,"≥",12)])),
        ("grafico", "07. Método gráfico", "Max Z=60x+50y; 2x+y≤80, x+2y≤70.", ("Maximizar",[60,50],[(2,1,"≤",80),(1,2,"≤",70)])),
        ("simplex", "08. Método simplex", "Max Z=3x+5y; x≤4, 2y≤12, 3x+2y≤18.", ([3,5],[(1,0,4),(0,2,12),(3,2,18)])),
        ("transporte", "09. Transporte", "Ofertas=[30,40], demandas=[30,40]; costos=[[4,6],[5,3]].", ([30,40],[30,40],[[4,6],[5,3]])),
        ("asignacion", "10. Asignación", "Minimiza costos de tres trabajadores y tres tareas: [[9,2,7],[6,4,3],[5,8,1]].", ([[9,2,7],[6,4,3],[5,8,1]],False)),
        ("modelacion", "11. Modelación — Fábrica", "x=producto A, y=producto B. Max Z=25x+40y; 2x+y≤100, x+2y≤80.", ("Maximizar",[25,40],[(2,1,"≤",100),(1,2,"≤",80)])),
    ]


def generar_practica():
    tipo = random.choice(["eoq","reorden","lp","grafico","simplex","transporte","asignacion","modelacion"])
    if tipo == "eoq":
        D,S,H = random.randrange(600,4801,100), random.randrange(20,101,10), random.randint(2,10)
        return (tipo,"Práctica EOQ",f"Demanda anual D={D}, costo de pedido S={S}, mantenimiento H={H}.",(D,S,H))
    if tipo == "reorden":
        D,L,SS = random.randrange(600,3601,100),random.randint(2,12),random.randint(0,30)
        return (tipo,"Práctica de reorden",f"D={D}, N=300 días, L={L} días, SS={SS}.",(D,300,L,SS))
    if tipo in ("lp","grafico","modelacion"):
        c1,c2 = random.randint(10,60), random.randint(10,60)
        b1,b2 = random.randint(60,120), random.randint(60,120)
        descr = f"Maximizar Z={c1}x+{c2}y, con 2x+y≤{b1}, x+2y≤{b2}, x,y≥0."
        return (tipo, f"Práctica de { {'lp':'programación lineal','grafico':'método gráfico','modelacion':'modelación'}[tipo]}",
                descr,("Maximizar",[c1,c2],[(2,1,"≤",b1),(1,2,"≤",b2)]))
    if tipo == "simplex":
        c1,c2 = random.randint(5,45), random.randint(5,45)
        b1,b2 = random.randint(40,100), random.randint(40,100)
        return (tipo,"Práctica de simplex",f"Maximizar Z={c1}x+{c2}y, con 2x+y≤{b1}, x+2y≤{b2}, x,y≥0.",
                ([c1,c2],[(2,1,b1),(1,2,b2)]))
    if tipo == "transporte":
        a,b = random.randint(15,45), random.randint(15,45)
        costos = [[random.randint(2,12) for _ in range(2)] for _ in range(2)]
        return (tipo,"Práctica de transporte",f"Oferta={[a,b]}, demanda={[a,b]}, costos={costos}.",([a,b],[a,b],costos))
    costos = [[random.randint(1,15) for _ in range(3)] for _ in range(3)]
    return (tipo,"Práctica de asignación",f"Minimiza el costo con matriz={costos}.",(costos,False))


def resolver_ejercicio(ej):
    tipo, _, _, datos = ej
    if tipo == "eoq":
        D,S,H = datos
        val = calcular_eoq(D,S,H)
        pasos = ["EOQ = √(2DS/H)",f"EOQ = √(2·{D}·{S}/{H}) = {val:.4f}"]
        return val, pasos, None
    if tipo == "reorden":
        D,N,L,SS = datos
        exacto, val = calcular_reorden(D,N,L,SS)
        return float(val), ["PR=(D/N)·L+SS", f"PR=({D}/{N})·{L}+{SS}={exacto:.4f}",f"Punto de reorden redondeado hacia arriba: {val}"], None
    if tipo in ("lp","grafico","modelacion"):
        modo, c, rs = datos
        res = resolver_lp(modo,c,rs)
        if res["estado"] != "optimo":
            raise ValueError("Esta práctica no tiene un óptimo finito.")
        pasos = [f"{modo} Z={c[0]}x+{c[1]}y"] + [f"{a}x+{b}y {op} {rhs}" for a,b,op,rhs in rs]
        pasos += [f"x={res['x']:.4f}, y={res['y']:.4f}",
                  f"Z={c[0]}({res['x']:.4f})+{c[1]}({res['y']:.4f})={res['z']:.4f}"]
        return res["z"], pasos, res
    if tipo == "simplex":
        c,rs = datos
        res = resolver_simplex(c,rs)
        if res["estado"] != "optimo":
            raise ValueError("No se encontró solución óptima finita.")
        return res["z"], [f"Max Z={c[0]}x+{c[1]}y",f"x={res['x']:.4f}, y={res['y']:.4f}, Z={res['z']:.4f}",
                                    f"Iteraciones simplex: {len(res['pasos'])-1}"], res
    if tipo == "transporte":
        ofertas,demandas,costos = datos
        res = resolver_transporte(ofertas,demandas,costos)
        pasos = [f"Oferta total={sum(ofertas)}, demanda total={sum(demandas)}"]
        for i,fila in enumerate(res["envios"]):
            for j,v in enumerate(fila):
                if v>1e-8:
                    pasos.append(f"O{i+1}→D{j+1}: {v:.2f}×{costos[i][j]:.2f}={v*costos[i][j]:.2f}")
        pasos.append(f"Costo total mínimo={res['costo']:.2f}")
        return res["costo"],pasos,res
    if tipo == "asignacion":
        matriz,maximizar = datos
        res = resolver_asignacion(matriz,maximizar)
        pasos = [f"Trabajador {i+1}→Tarea {j+1}: {v:.2f}" for i,j,v in res["parejas"]]
        pasos.append(f"Resultado óptimo={res['valor']:.2f}")
        return res["valor"],pasos,res
    raise ValueError("Ejercicio no reconocido.")


def pantalla_aprendizaje(st):
    st.title("📚 Centro de aprendizaje")
    st.write("11 ejemplos resueltos con cálculo real y prácticas aleatorias con retroalimentación.")
    ej_tab, practica_tab = st.tabs(["11 ejemplos", "Generar práctica"])
    with ej_tab:
        catalog = catalogo()
        n = st.selectbox("Elige un ejercicio", range(len(catalog)), format_func=lambda i: catalog[i][1], key="ej_indice")
        ej = catalog[n]
        st.write(ej[2])
        if st.button("Mostrar solución y procedimiento", key="mostrar_ejemplo"):
            try:
                ans,pasos,res = resolver_ejercicio(ej)
                st.success(f"Resultado calculado: {ans:,.2f}")
                for p in pasos:
                    st.write("• " + p)
                if ej[0] == "grafico" and res and res["estado"] == "optimo":
                    graficar_lp(st,ej[3][1],ej[3][2],res)
                elif ej[0] == "simplex" and res:
                    mostrar_tablas_simplex(st,res)
            except ValueError as e:
                st.error(str(e))
    with practica_tab:
        if "practica_actual_v2" not in st.session_state:
            st.session_state.practica_actual_v2 = generar_practica()
            st.session_state.practica_n_v2 = 0
            st.session_state.feedback_v2 = None
        if st.button("🎲 Generar nueva práctica", key="nuevo_problema"):
            st.session_state.practica_actual_v2 = generar_practica()
            st.session_state.practica_n_v2 += 1
            st.session_state.feedback_v2 = None
        ej = st.session_state.practica_actual_v2
        st.subheader(ej[1]); st.write(ej[2])
        st.caption("Escribe el valor óptimo final. Se admiten pequeñas diferencias por redondeo.")
        with st.form(f"form_practica_{st.session_state.practica_n_v2}"):
            valor = st.number_input("Tu respuesta", min_value=0.0, value=0.0, step=0.01, format="%.2f",
                                    key=f"respuesta_v2_{st.session_state.practica_n_v2}")
            comprobar = st.form_submit_button("Comprobar respuesta", type="primary", use_container_width=True)
        if comprobar:
            try:
                ans, pasos, extra = resolver_ejercicio(ej)
                acierto = abs(valor-ans) <= max(0.05, abs(ans)*0.0001)
                st.session_state.feedback_v2 = (acierto,ans,pasos,extra)
            except ValueError as e:
                st.error(str(e))
        if st.session_state.feedback_v2 is not None:
            acierto,ans,pasos,extra = st.session_state.feedback_v2
            if acierto:
                st.success("¡Correcto! Tu resultado es válido.")
            else:
                st.error(f"Aún no es correcto. El resultado es {ans:,.2f}.")
            with st.expander("Ver fórmula, procedimiento y retroalimentación", expanded=not acierto):
                for p in pasos:
                    st.write("• " + p)
                st.write("Consejo: identifica las variables, aplica la fórmula o modelo y revisa el redondeo.")
                if ej[0] == "grafico" and extra:
                    graficar_lp(st,ej[3][1],ej[3][2],extra)
                if ej[0] == "simplex" and extra:
                    mostrar_tablas_simplex(st,extra)




# ===========================================================================
# CHATBOT DE IA: Gemini interpreta el lenguaje humano; SciPy realiza el cálculo
# ===========================================================================

METODOS_IA = {
    "eoq": "Inventarios · Cantidad económica de pedido (EOQ)",
    "reorden": "Inventarios · Punto de reorden",
    "programacion_lineal": "Programación lineal",
    "grafico": "Método gráfico",
    "simplex": "Método simplex",
    "transporte": "Transporte y asignación · Transporte",
    "asignacion": "Transporte y asignación · Asignación",
    "modelacion": "Modelación matemática",
    "teoria": "Explicación conceptual"
}

INSTRUCCIONES_IA = """Eres la parte lingüística de IO Solver, un asistente académico en español.
Tu única tarea es ENTENDER la pregunta, ELEGIR el método más pertinente de los seis módulos
(Modelación, Programación lineal, Método gráfico, Método simplex, Transporte y asignación, Inventarios)
y extraer los datos existentes. NO HAGAS LOS CÁLCULOS; los hace Python.
Debes devolver SOLO un objeto JSON válido, sin markdown ni otros textos:
{
  "metodo":"eoq|reorden|programacion_lineal|grafico|simplex|transporte|asignacion|modelacion|teoria",
  "razon":"razón breve de escoger ese método",
  "mensaje":"explicación didáctica breve o pregunta por datos faltantes",
  "faltantes":["nombre del dato que falta"],
  "tipo":"Maximizar|Minimizar",
  "coeficientes":[40,30],
  "restricciones":[[2,1,"≤",100],[1,2,"≤",80]],
  "variables":["producto A","producto B"],
  "demanda":1200,"costo_pedido":50,"mantenimiento":2,
  "dias_operacion":300,"tiempo_entrega":5,"stock_seguridad":0,
  "ofertas":[30,40],"demandas":[30,40],"costos":[[4,6],[5,3]],
  "matriz":[[9,2,7],[6,4,3],[5,8,1]],"maximizar_asignacion":false
}
Los números del ejemplo de formato JSON son SOLO ILUSTRATIVOS: nunca los copies como valores del usuario.
NO RELLENES con datos inventados los campos que no están expresamente disponibles en el enunciado o diálogo; omite esos campos y colócalos en faltantes cuando son obligatorios.
Distingue claramente DEMANDA ANUAL, COSTO DE PEDIDO y MANTENIMIENTO ANUAL para EOQ.
Para punto de reorden se requiere D, días de operación, plazo de entrega y stock de seguridad, si no se menciona stock de seguridad pregunta cuánto es (puede ser 0).
Para programación lineal 2 variables continuas no negativas: extrae exactamente 2 coeficientes y TODAS las restricciones (a,b,signo,limite). Si no se sabe si maximizar o minimizar, pregunta.
Para simplex, SOLO maximizar con restricciones ≤ y lado derecho >=0. Si el ejercicio no satisface esto, elige programacion_lineal y explica por qué.
Para grafico: dos variables y restricciones; permite maximizar/minimizar.
Para transporte: costos de una matriz, ofertas por origen y demandas por destino. Para asignación: matriz cuadrada y objetivo min/max.
Si piden SOLO explicación teórica responde con metodo=teoria y mensaje didáctico claro, sin cifras calculadas.
Si el usuario pide "simplex", "método gráfico" o "modelación" explícitamente y se cumplen sus condiciones, prefiere ese método.
El usuario puede aportar datos faltantes en la siguiente pregunta; usa la conversación anterior para completarlos, nunca inventes.
No infieras restricciones que el enunciado no menciona. Responde respetuosamente en español de Guatemala.
"""


def clave_gemini(st):
    """Lee la clave únicamente del servidor; nunca del navegador o código público."""
    try:
        llave = st.secrets.get("GEMINI_API_KEY", "")
    except (FileNotFoundError, KeyError, AttributeError):
        llave = ""
    return llave or os.getenv("GEMINI_API_KEY", "")


def _json_ia(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError("Formato de IA inválido.")
    if obj.get("metodo") not in METODOS_IA:
        raise ValueError("El método propuesto por la IA no es reconocido.")
    return obj


def consultar_gemini(llave: str, pregunta: str, mensajes: list[dict], modelo: str = "gemini-3.5-flash-lite") -> dict:
    """Solo clasifica/extracta; las operaciones las calcula el motor existente."""
    from google import genai
    from google.genai import types

    contexto = []
    for item in mensajes[-8:]:
        if item.get("rol") == "usuario":
            contexto.append("USUARIO: " + item.get("texto", "")[:1400])
        elif item.get("rol") == "asistente":
            # Preservar los datos extraídos para responder a aclaraciones posteriores.
            anterior = item.get("datos")
            if anterior:
                contexto.append("ANÁLISIS ANTERIOR: " + json.dumps(anterior, ensure_ascii=False)[:2800])
            else:
                contexto.append("ASISTENTE: " + item.get("texto", "")[:500])
    entrada = "CONVERSACIÓN RECIENTE:\n" + "\n".join(contexto) + "\n\nCONSULTA ACTUAL:\n" + pregunta
    client = genai.Client(api_key=llave, http_options=types.HttpOptions(timeout=60000))
    try:
        respuesta = client.models.generate_content(
            model=modelo,
            contents=entrada,
            config=types.GenerateContentConfig(
                system_instruction=INSTRUCCIONES_IA,
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=4096
            )
        )
        if not (respuesta.text or "").strip():
            raise ValueError("Gemini no devolvió texto JSON; revisa el límite de tokens o el estado del modelo.")
        return _json_ia(respuesta.text)
    finally:
        client.close()


def _num_ia(valor, nombre):
    if isinstance(valor, bool) or valor is None:
        raise ValueError(f"Falta indicar {nombre}.")
    try:
        v = float(valor)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(f"Falta un número válido para {nombre}.") from None
    if not math.isfinite(v):
        raise ValueError(f"{nombre} debe ser un valor finito.")
    return v


def _lista_ia(datos, nombre, minimo=1, maximo=6):
    if not isinstance(datos, list) or not (minimo <= len(datos) <= maximo):
        raise ValueError(f"Faltan datos en {nombre}: se admiten entre {minimo} y {maximo} elementos.")
    return [_num_ia(v, f"{nombre} {i+1}") for i, v in enumerate(datos)]


def _restricciones_ia(datos):
    if not isinstance(datos, list) or not (1 <= len(datos) <= 6):
        raise ValueError("Indica las restricciones matemáticas (máximo seis).")
    rs = []
    for i, fila in enumerate(datos):
        if not isinstance(fila, list) or len(fila) != 4:
            raise ValueError(f"Completa la restricción {i+1}: [coef. x, coef. y, signo, límite].")
        a = _num_ia(fila[0], "coeficiente x")
        b = _num_ia(fila[1], "coeficiente y")
        op = str(fila[2]).strip().replace("<=", "≤").replace(">=", "≥").replace("==", "=")
        if op not in ("≤", "≥", "="):
            raise ValueError(f"Indica un signo válido en la restricción {i+1} (≤, ≥ o =).")
        rhs = _num_ia(fila[3], "límite")
        rs.append((a,b,op,rhs))
    return rs


def resolver_consulta_ia(datos: dict) -> dict:
    """Valida TODAS las entradas del modelo externo antes de usar los solvers."""
    metodo = datos.get("metodo")
    if metodo not in METODOS_IA:
        raise ValueError("La IA no reconoció un método compatible.")
    salida = {"metodo": metodo, "titulo": METODOS_IA[metodo],
              "razon": str(datos.get("razon") or "")[:350],
              "mensaje": str(datos.get("mensaje") or "")[:1000]}
    if metodo == "teoria":
        salida["estado"] = "teoria"
        return salida
    faltantes = datos.get("faltantes")
    if isinstance(faltantes, list) and faltantes:
        salida.update(estado="faltantes", faltantes=[str(v)[:80] for v in faltantes[:8]])
        return salida

    try:
        if metodo == "eoq":
            D = _num_ia(datos.get("demanda"), "la demanda anual")
            S = _num_ia(datos.get("costo_pedido"), "el costo por pedido")
            H = _num_ia(datos.get("mantenimiento"), "el costo de mantenimiento anual")
            r = calcular_eoq(D,S,H)
            salida.update(estado="optimo", resultado=r, pasos=[
                "Método EOQ: cantidad económica de pedido.",
                "EOQ = √(2 × D × S / H)",
                f"EOQ = √(2 × {D:g} × {S:g} / {H:g}) = {r:.4f} unidades."
            ])
        elif metodo == "reorden":
            D = _num_ia(datos.get("demanda"), "la demanda anual")
            N = _num_ia(datos.get("dias_operacion"), "los días de operación")
            L = _num_ia(datos.get("tiempo_entrega"), "el plazo de entrega")
            SS = _num_ia(datos.get("stock_seguridad"), "el stock de seguridad (0 si no hay)")
            exacto, entero = calcular_reorden(D,N,L,SS)
            salida.update(estado="optimo", resultado=entero, pasos=[
                "Demanda diaria d = D / N; Punto de reorden PR = d × L + SS.",
                f"d = {D:g} / {N:g} = {D/N:.4f} unidades por día.",
                f"PR = ({D/N:.4f} × {L:g}) + {SS:g} = {exacto:.4f}; redondeado hacia arriba: {entero}."
            ])
        elif metodo in ("programacion_lineal","grafico","modelacion","simplex"):
            modo = datos.get("tipo")
            if modo not in ("Maximizar", "Minimizar"):
                raise ValueError("Indica si el objetivo es maximizar o minimizar.")
            c = _lista_ia(datos.get("coeficientes"), "función objetivo", 2, 2)
            rs = _restricciones_ia(datos.get("restricciones"))
            mostrar_grafica = metodo in ("programacion_lineal","grafico","modelacion")
            if metodo == "simplex":
                if modo != "Maximizar" or any(op != "≤" or rhs < 0 for _,_,op,rhs in rs):
                    salida["aviso"] = "Este simplex básico solo acepta maximización con restricciones ≤ y límites no negativos; se resolvió por programación lineal general."
                    lp = resolver_lp(modo,c,rs)
                    salida.update(estado=lp["estado"], solucion=lp, tipo=modo, coef=c, restricciones=rs)
                    mostrar_grafica = True
                else:
                    res = resolver_simplex(c, [(a,b,rhs) for a,b,_,rhs in rs])
                    salida.update(estado=res["estado"], solucion=res, tipo=modo, coef=c, restricciones=rs, simplex=True)
            else:
                lp = resolver_lp(modo,c,rs)
                salida.update(estado=lp["estado"], solucion=lp, tipo=modo, coef=c, restricciones=rs)
            if salida["estado"] == "optimo":
                r = salida["solucion"]
                salida["resultado"] = r["z"]
                salida["pasos"] = [
                    f"{modo} Z = {c[0]:g}x + ({c[1]:g})y",
                    *[f"R{i}: {a:g}x + ({b:g})y {op} {rhs:g}" for i,(a,b,op,rhs) in enumerate(rs,1)],
                    f"x = {r['x']:.4f}; y = {r['y']:.4f}",
                    f"Sustitución: Z = ({c[0]:g})({r['x']:.4f}) + ({c[1]:g})({r['y']:.4f}) = {r['z']:.4f}."
                ]
                salida["grafica"] = mostrar_grafica
                nombres = datos.get("variables")
                if isinstance(nombres,list) and len(nombres)==2:
                    salida["variables"] = [str(s)[:70] for s in nombres]
        elif metodo == "transporte":
            ofertas = _lista_ia(datos.get("ofertas"), "oferta", 2, 4)
            demandas = _lista_ia(datos.get("demandas"), "demanda", 2, 4)
            costos = datos.get("costos")
            if not isinstance(costos, list) or len(costos) != len(ofertas):
                raise ValueError("Indica una matriz de costos con una fila por origen.")
            matriz = [_lista_ia(f, "costos por destino", len(demandas), len(demandas)) for f in costos]
            res = resolver_transporte(ofertas,demandas,matriz)
            pasos = [f"Oferta total = {sum(ofertas):g}; demanda total = {sum(demandas):g}."]
            for i,f in enumerate(res["envios"]):
                for j,cantidad in enumerate(f):
                    if cantidad > 1e-7:
                        pasos.append(f"Origen {i+1} → Destino {j+1}: {cantidad:.4f} × {matriz[i][j]:g} = {cantidad*matriz[i][j]:.4f}")
            salida.update(estado="optimo", resultado=res["costo"], pasos=pasos, tabla=[
                {"Origen":f"O{i+1}", **{f"D{j+1}":round(v,4) for j,v in enumerate(f)}}
                for i,f in enumerate(res["envios"])
            ])
        elif metodo == "asignacion":
            matriz = datos.get("matriz")
            if not isinstance(matriz,list) or not (2<=len(matriz)<=5):
                raise ValueError("La matriz de asignación debe tener de 2 a 5 trabajadores y tareas.")
            costos = [_lista_ia(f, "costos de tarea",len(matriz),len(matriz)) for f in matriz]
            maximizar = datos.get("maximizar_asignacion")
            if not isinstance(maximizar,bool):
                raise ValueError("Indica si deseas minimizar costos o maximizar beneficios.")
            res = resolver_asignacion(costos,maximizar)
            pasos = [f"Trabajador {i+1} → Tarea {j+1}: {v:g}" for i,j,v in res["parejas"]]
            pasos.append("Total: " + " + ".join(f"{v:g}" for _,_,v in res["parejas"]) + f" = {res['valor']:.4f}")
            salida.update(estado="optimo", resultado=res["valor"], pasos=pasos, tabla=[
                {"Trabajador":f"T{i+1}", "Tarea":f"Tarea {j+1}", "Valor":v}
                for i,j,v in res["parejas"]
            ])
    except ValueError as e:
        salida["estado"] = "faltantes"
        salida["faltantes"] = [str(e)]
    return salida


def presentar_respuesta_ia(st, salida: dict):
    st.markdown(f"**Método elegido:** {salida['titulo']}")
    if salida.get("razon"):
        st.caption(salida["razon"])
    estado = salida.get("estado")
    if estado in ("faltantes", "teoria"):
        if salida.get("mensaje"):
            st.write(salida["mensaje"])
        if estado == "faltantes":
            st.info("Para resolver correctamente necesito: " + "; ".join(salida.get("faltantes", [])))
        return
    if salida.get("aviso"):
        st.warning(salida["aviso"])
    if estado != "optimo":
        st.warning("El problema no presenta un óptimo finito con las restricciones proporcionadas, o el cálculo no pudo completarse.")
        return
    st.success(f"**Resultado calculado por Python: {salida['resultado']:,.4f}**")
    for p in salida.get("pasos", []):
        st.write("• " + p)
    if salida.get("variables"):
        st.caption(f"x = {salida['variables'][0]}; y = {salida['variables'][1]}")
    if salida.get("tabla"):
        st.dataframe(salida["tabla"], hide_index=True, use_container_width=True)
    if salida.get("simplex"):
        with st.expander("Ver tablas simplex e iteraciones", expanded=False):
            mostrar_tablas_simplex(st, salida["solucion"])
    if salida.get("grafica"):
        with st.expander("📊 Ver región factible y método gráfico", expanded=True):
            graficar_lp(st, salida["coef"], salida["restricciones"], salida["solucion"])
    st.caption("La IA interpretó el enunciado; el resultado y las gráficas fueron calculados por los algoritmos de IO Solver.")


def pantalla_ia(st):
    st.title("🤖 Asistente de Inteligencia Artificial")
    st.write("Escribe tu problema **en lenguaje natural**. La IA elegirá el método y Python realizará las operaciones para que puedas estudiar el procedimiento.")
    llave = clave_gemini(st)
    if not llave:
        st.warning("El chatbot aún no tiene una clave Gemini configurada en el servidor. Los seis módulos matemáticos siguen funcionando sin IA.")
        st.info("Administrador: crea una clave en Google AI Studio y colócala en **Streamlit Community Cloud → App settings → Secrets**, bajo el nombre `GEMINI_API_KEY`. Nunca la subas a GitHub.")
        return
    st.caption("Motor: Google Gemini 3.5 Flash-Lite · No incluyas contraseñas ni información personal. El servicio de IA tiene límites de uso.")
    if "io_ia_mensajes" not in st.session_state:
        st.session_state.io_ia_mensajes = []
    if "io_ia_cantidad" not in st.session_state:
        st.session_state.io_ia_cantidad = 0
    if st.button("🧹 Borrar conversación", key="io_borrar_chat"):
        st.session_state.io_ia_mensajes = []
        st.session_state.io_ia_cantidad = 0
        st.rerun()
    if not st.session_state.io_ia_mensajes:
        st.info("Ejemplo: «Maximiza Z = 40x + 30y con 2x + y ≤ 100, x + 2y ≤ 80, x,y ≥ 0. ¿Qué método utilizas y cuál es la solución?»")
    for item in st.session_state.io_ia_mensajes:
        with st.chat_message("user" if item["rol"] == "usuario" else "assistant"):
            if item["rol"] == "usuario":
                st.markdown(item["texto"])
            elif item.get("salida"):
                presentar_respuesta_ia(st,item["salida"])
            else:
                st.write(item.get("texto", ""))
    texto = st.chat_input("Describe tu ejercicio o pregunta…", max_chars=1400)
    if texto:
        if st.session_state.io_ia_cantidad >= 20:
            st.warning("Has alcanzado el máximo de 20 consultas por sesión. Puedes iniciar otra sesión más tarde.")
            return
        historial = st.session_state.io_ia_mensajes.copy()
        st.session_state.io_ia_mensajes.append({"rol":"usuario", "texto":texto})
        st.session_state.io_ia_cantidad += 1
        with st.spinner("Interpretando el problema y calculando la solución..."):
            try:
                modelo = str(os.getenv("IO_GEMINI_MODEL", "gemini-3.5-flash-lite"))
                interpretacion = consultar_gemini(llave,texto,historial,modelo)
                resultado = resolver_consulta_ia(interpretacion)
                st.session_state.io_ia_mensajes.append({"rol":"asistente", "datos":interpretacion, "salida":resultado})
            except Exception as e:
                # No imprimimos la clave, la solicitud ni el texto completo del proveedor.
                codigo_raw = getattr(e, "code", None) or getattr(e, "status_code", None)
                codigo_texto = str(codigo_raw or "")
                encontrado = re.search(r"\b(400|401|402|403|404|408|429|500|502|503|504)\b", codigo_texto)
                codigo = encontrado.group(1) if encontrado else ""
                clase = type(e).__name__
                logging.warning("IO Solver: Gemini falló. Clase=%s; codigo_http=%s; modelo=%s", clase, codigo or "sin código", modelo)
                if codigo == "429" or "RESOURCE_EXHAUSTED" in clase:
                    mensaje = "Gemini alcanzó una cuota de uso (HTTP 429). Revisa los límites de frecuencia en Google AI Studio y vuelve a probar cuando haya cuota."
                elif codigo in ("401", "403"):
                    mensaje = "Gemini rechazó el acceso (HTTP " + codigo + "). Revisa que la clave sea válida y que tu proyecto tenga acceso al modelo configurado."
                elif codigo == "404":
                    mensaje = "El modelo de Gemini solicitado no está disponible para esta clave (HTTP 404). Revisa el modelo habilitado en Google AI Studio."
                elif codigo == "400":
                    mensaje = "Gemini rechazó el formato de la consulta (HTTP 400). Hay que ajustar la configuración del modelo."
                elif codigo in ("408", "500", "502", "503", "504") or "Timeout" in clase:
                    mensaje = "El servicio de Gemini no respondió correctamente (HTTP " + (codigo or "sin código") + "). Puedes intentar de nuevo."
                elif isinstance(e, (json.JSONDecodeError, ValueError)):
                    mensaje = "Gemini respondió, pero la información no llegó en el formato JSON esperado. Repite la consulta o revisa el modelo y los registros de la aplicación."
                else:
                    mensaje = "El chatbot no pudo completar la consulta. Revisa los registros del servidor para conocer el tipo de error."
                # Solo mostramos clase técnica y código HTTP; nunca se expone la clave ni el error completo.
                mensaje += f" (Diagnóstico: {clase}" + (f", HTTP {codigo}" if codigo else "") + ")"
                st.session_state.io_ia_mensajes.append({"rol":"asistente", "texto":mensaje})
        st.rerun()


def main():
    import streamlit as st
    st.set_page_config(page_title="IO Solver | Investigación de Operaciones", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")
    st.markdown(DISEÑO_CSS, unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div class="ios-brand">
      <div class="ios-brandmark">✦</div>
      <div><div class="ios-brandname">IO <span>Solver</span></div>
      <div class="ios-brandsub">Asistente de<br>Investigación de Operaciones</div></div>
    </div>
    """, unsafe_allow_html=True)
    paginas = ["🏠 Inicio", "🤖 Asistente IA", "📝 Modelación", "📈 Programación lineal", "📊 Método gráfico",
               "🧮 Método simplex", "🚚 Transporte y asignación", "📦 Inventarios", "📚 Centro de aprendizaje"]
    pagina = st.sidebar.radio("Navegación", paginas, label_visibility="collapsed", key="navegacion_io")
    st.sidebar.divider()
    with st.sidebar.expander("📋 Ficha académica", expanded=False):
        st.caption(UNIVERSIDAD)
        st.markdown(f"**Asignatura:** {ASIGNATURA}")
        st.markdown(f"**Sección:** {SECCION}")
        st.markdown(f"**Integrante 1:** {INTEGRANTE_1}")
        st.caption(f"Carné: {CARNE_1}")
        st.markdown(f"**Integrante 2:** {INTEGRANTE_2}")
        st.caption(f"Carné: {CARNE_2}")
    st.sidebar.markdown('<p class="ios-side-note">✓ Aplicación académica gratuita<br>✓ No solicita cuenta al visitante</p>', unsafe_allow_html=True)
    nombre_pagina = pagina.split(" ", 1)[1]
    st.markdown(f'<div class="ios-topline"><span class="ios-breadcrumb">IO SOLVER / {nombre_pagina.upper()}</span><span class="ios-status"><span class="ios-dot"></span> LISTO PARA CALCULAR</span></div>', unsafe_allow_html=True)

    if pagina == paginas[0]: pantalla_inicio(st)
    elif pagina == paginas[1]: pantalla_ia(st)
    elif pagina == paginas[2]: pantalla_modelacion(st)
    elif pagina == paginas[3]: pantalla_lineal(st, grafica=False)
    elif pagina == paginas[4]: pantalla_lineal(st, grafica=True)
    elif pagina == paginas[5]: pantalla_simplex(st)
    elif pagina == paginas[6]: pantalla_transporte_asignacion(st)
    elif pagina == paginas[7]: pantalla_inventarios(st)
    else: pantalla_aprendizaje(st)

    st.markdown(f'<div class="ios-footer">IO Solver · {ASIGNATURA} · Sección {SECCION}<br>{INTEGRANTE_1} &amp; {INTEGRANTE_2} · Uso académico</div>', unsafe_allow_html=True)

def autoprueba():
    pruebas = []
    def validar(nombre, obtenido, esperado, tol=1e-5):
        assert math.isclose(obtenido, esperado, abs_tol=tol, rel_tol=tol), (nombre, obtenido, esperado)
        pruebas.append((nombre, obtenido))
    validar("EOQ", calcular_eoq(1200,50,2), math.sqrt(60000))
    validar("Reorden", calcular_reorden(1200,300,5,0)[1], 20)
    lp = resolver_lp("Maximizar",[40,30],[(2,1,"≤",100),(1,2,"≤",80)])
    validar("LP máximo", lp["z"], 2200)
    validar("LP X", lp["x"], 40)
    validar("LP Y", lp["y"], 20)
    validar("Gráfico (óptimo)",resolver_lp("Maximizar",[60,50],[(2,1,"≤",80),(1,2,"≤",70)])["z"],2800)
    simplex = resolver_simplex([40,30],[(2,1,100),(1,2,80)])
    validar("Simplex", simplex["z"], 2200)
    validar("Iteraciones simplex",len(simplex["pasos"])-1,2)
    validar("Transporte",resolver_transporte([30,40],[30,40],[[4,6],[5,3]])["costo"],240)
    validar("Asignación",resolver_asignacion([[9,2,7],[6,4,3],[5,8,1]])["valor"],9)
    for ej in catalogo():
        res, pasos, _ = resolver_ejercicio(ej)
        assert math.isfinite(res) and pasos,ej[1]
        pruebas.append((ej[1], res))
    muestras_ia = [
        ({"metodo":"eoq","demanda":1200,"costo_pedido":50,"mantenimiento":2},math.sqrt(60000)),
        ({"metodo":"reorden","demanda":1200,"dias_operacion":300,"tiempo_entrega":5,"stock_seguridad":0},20),
        ({"metodo":"programacion_lineal","tipo":"Maximizar","coeficientes":[40,30],"restricciones":[[2,1,"≤",100],[1,2,"≤",80]]},2200),
        ({"metodo":"grafico","tipo":"Maximizar","coeficientes":[60,50],"restricciones":[[2,1,"≤",80],[1,2,"≤",70]]},2800),
        ({"metodo":"simplex","tipo":"Maximizar","coeficientes":[40,30],"restricciones":[[2,1,"≤",100],[1,2,"≤",80]]},2200),
        ({"metodo":"transporte","ofertas":[30,40],"demandas":[30,40],"costos":[[4,6],[5,3]]},240),
        ({"metodo":"asignacion","matriz":[[9,2,7],[6,4,3],[5,8,1]],"maximizar_asignacion":False},9),
    ]
    for payload,esperado in muestras_ia:
        r = resolver_consulta_ia(payload)
        assert r['estado']=="optimo", (payload,r)
        validar('IA → ' + payload['metodo'],r['resultado'],esperado)
    assert resolver_consulta_ia({"metodo":"eoq"})['estado']=="faltantes"
    assert resolver_consulta_ia({"metodo":"teoria","mensaje":"Explicación"})['estado']=="teoria"
    assert _json_ia('{"metodo":"eoq"}')['metodo']=="eoq"
    for nombre, obtenido in pruebas:
        print(f"OK | {nombre}: {obtenido:.4f}")
    print(f"\n{len(pruebas)} pruebas correctas.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        autoprueba()
    else:
        main()

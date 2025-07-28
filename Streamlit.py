import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import requests
from sklearn.linear_model import LinearRegression
from supabase import create_client

# —————— CONFIGURAÇÃO SUPABASE ——————
SUPABASE_URL = "https://etoduafzrmgvbdzapiwb.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImV0b2R1YWZ6cm1ndmJkemFwaXdiIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI4NjUyMjcsImV4cCI6MjA2ODQ0MTIyN30.1wMvRBNFm_exW3reNHzfN1tdoNmhHkzyQDicdo-TCVE"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
# ————————————————————————————————————

st.set_page_config(page_title="App de Investimentos", layout="wide")
BRAPI_KEY = "49bBsco9TxEEEPuscv2zRZ"


def autenticar_usuario(login, senha):
    resp = (
        supabase
        .table("usuarios")
        .select("*")
        .eq("login", login)
        .eq("senha", senha)
        .execute()
    )
    if resp.data and len(resp.data) == 1:
        return resp.data[0]
    else:
        return None


def pagina_login():
    st.title("Login")
    login = st.text_input("Login")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        usuario = autenticar_usuario(login, senha)
        if usuario:
            st.session_state["usuario"] = usuario
            st.session_state.page = "home"

        else:
            st.error("Login ou senha incorretos.")


# EXIGIR LOGIN
if "usuario" not in st.session_state:
    pagina_login()
    st.stop()

# ———————————————————————————— ESTILO E MENU
st.markdown("""
<style>
section[data-testid="stSidebar"] { width: 200px; }
.sidebar-button { display: block; padding: 8px; margin: 4px 0; width: 100%; text-align: left; border: none; background: #f0f0f0; font-size: 18px; cursor: pointer; }
.sidebar-button.active { background-color: #ddd; }
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "home"

with st.sidebar:
    st.markdown(f"**Usuário:** {st.session_state['usuario']['nome']}")
    for key, label in [("home", "🏠 Início"), ("search", "🔍 Buscar Ação"), ("calc", "📊 Calculadora")]:
        if st.button(label, key=key):
            st.session_state.page = key
    if st.button("Sair"):
        del st.session_state["usuario"]


# ———————————————————————————— FUNÇÕES

@st.cache_data(ttl=360000)
def forecast_trend(df, days_ahead=30):
    df2 = df.copy()
    df2["ts"] = df2["date"].map(datetime.datetime.toordinal)
    X = df2["ts"].values.reshape(-1, 1)
    y = df2["close"].values
    model = LinearRegression().fit(X, y)
    last = df2["date"].max()
    future = [last + datetime.timedelta(days=i) for i in range(1, days_ahead + 1)]
    Xf = np.array([d.toordinal() for d in future]).reshape(-1, 1)
    yf = model.predict(Xf)
    return pd.DataFrame({"date": future, "close": yf})


@st.cache_data(ttl=36000)
def fetch_history_brapi(ticker, range_code):
    url = f"https://brapi.dev/api/quote/{ticker}"
    params = {"range": range_code, "interval": "1d", "token": BRAPI_KEY}
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json().get("results") or []
    if not data:
        return None
    hist = data[0].get("historicalDataPrice") or []
    if not hist:
        return None
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"], unit="s")
    return df[["date", "close"]]


@st.cache_data(ttl=36000)
def fetch_bcb_rate(series_code):
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados/ultimos/1?formato=json"
    r = requests.get(url, timeout=5)
    r.raise_for_status()
    d = r.json() or []
    return float(d[0]["valor"].replace(",", ".")) if d else None


def calcular_valor_futuro(PV, A, n, taxa):
    if taxa is None or n <= 0:
        return PV
    r = taxa / 100
    rm = (1 + r) ** (1 / 12) - 1
    vf0 = PV * (1 + rm) ** n
    vf_ap = A * (((1 + rm) ** n - 1) / rm) if rm != 0 else A * n
    return vf0 + vf_ap

# ———————————————————————————— PÁGINAS
def page_home():
    st.title("App de Investimentos")
    selic = fetch_bcb_rate(11)
    cdi = fetch_bcb_rate(12)
    st.markdown("### Indicadores atuais")
    st.write(f"**SELIC:** {selic:.4f}%") if selic else st.write("**SELIC:** indisponível")
    st.write(f"**CDI:** {cdi:.4f}%") if cdi else st.write("**CDI:** indisponível")
    st.markdown("---")
    st.markdown("#### Desenvolvido por Igor")

def page_buscar_ativo():
    st.title("Buscar Ação + Projeção")
    t = st.text_input("Ticker:", "").upper().strip().replace(".SA", "")
    p = st.selectbox("Período:", ["1 mês", "3 meses"])
    cmap = {"1 mês": "1mo", "3 meses": "3mo"}
    if st.button("Buscar e projetar"):
        if not t:
            st.error("Digite um ticker.")
            return
        df = fetch_history_brapi(t, cmap[p])
        if df is None or df.empty:
            st.error("Não foi possível obter dados.")
            return
        df2 = forecast_trend(df)
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["close"], ".-", label="Histórico")
        ax.plot(df2["date"], df2["close"], "--", label="Projeção")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

def page_calculadora():
    st.title("Calculadora")
    PV = st.number_input("Inv. inicial:", 1000.0)
    A = st.number_input("Aporte:", 100.0)
    n = st.number_input("Meses:", 1, 12)
    selic = fetch_bcb_rate(11)
    cdi = fetch_bcb_rate(12)
    cdi_anual = ((1 + cdi / 100) ** 252 - 1) * 100 if cdi else None
    st.write(f"SELIC: {selic:.4f}%")
    st.write(f"CDI an.: {cdi_anual:.2f}%")
    taxa_p = st.number_input("Poupança (%):", 4.5)
    perc_c = st.number_input("CDB (% do CDI):", 100.0) if cdi_anual else None
    perc_l = st.number_input("LCI (% do CDI):", 90.0) if cdi_anual else None

    def ir(d): return 0.225 if d <= 180 else 0.20 if d <= 360 else 0.175 if d <= 720 else 0.15

    if st.button("Calcular"):
        inv = PV + A * n
        prod = {"Poupança": taxa_p}
        if cdi_anual:
            prod["CDB"] = cdi_anual * perc_c / 100
            prod["LCI"] = cdi_anual * perc_l / 100
            prod["LCA"] = cdi_anual * perc_l / 100
        res = {k: calcular_valor_futuro(PV, A, n, t) for k, t in prod.items()}
        df = pd.DataFrame(res, index=["Valor Futuro"]).T
        df["Valor Investido"] = inv
        df["Lucro Bruto"] = df["Valor Futuro"] - inv
        d = n * 30
        df["Imposto IR"] = [df.loc[k, "Lucro Bruto"] * ir(d) if k == "CDB" else 0 for k in df.index]
        df["Lucro Líquido"] = df["Lucro Bruto"] - df["Imposto IR"]
        fmt = lambda x: f"R$ {x:,.2f}"
        df = df.map(fmt)
        st.table(df)
        xs = np.arange(1, n + 1)
        fig, ax = plt.subplots()
        for k in df.index:
            ys = [calcular_valor_futuro(PV, A, m, prod[k]) for m in xs]
            ax.plot(xs, ys, label=k)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

# ———————————————————————————— EXIBIR PÁGINA ATUAL
if st.session_state.page == "home":
    page_home()
elif st.session_state.page == "search":
    page_buscar_ativo()
else:
    page_calculadora()

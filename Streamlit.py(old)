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
            st.success("Login com Sucesso! Redirecionando...")
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
        st.error("Deslogando")
        del st.session_state["usuario"]


# ———————————————————————————— FUNÇÕES

@st.cache_data(ttl=3600)
def forecast_trend(df: pd.DataFrame, days_ahead: int = 30):
    # Converte datas para ordinais
    df2 = df.copy()
    df2["ts"] = df2["date"].map(datetime.datetime.toordinal)
    X = df2["ts"].values.reshape(-1, 1)
    y = df2["close"].values
    # Ajusta modelo linear
    model = LinearRegression().fit(X, y)
    # Gera datas futuras
    last_date = df2["date"].max()
    future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days_ahead + 1)]
    Xf = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    yf = model.predict(Xf)
    return pd.DataFrame({"date": future_dates, "close": yf})


@st.cache_data(ttl=36000)
def fetch_history_brapi(ticker: str, range_code: str):
    url = f"https://brapi.dev/api/quote/{ticker}"
    params = {"range": range_code, "interval": "1d", "token": BRAPI_KEY}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("results") or []
    if not data:
        return None
    hist = data[0].get("historicalDataPrice")
    if not hist:
        return None
    df = pd.DataFrame(hist)
    # histórico vem com timestamp em segundos
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
    st.write(f"**SELIC (diária):** {selic:.4f}%" if selic else "**SELIC:** indisponível")
    st.write(f"**CDI (diária):** {cdi:.4f}%" if cdi else "**CDI:** indisponível")
    st.write("-------------------------------------------------------------------------------")
    st.markdown("#### Desenvolvido pelo Elon Musk da Petrobras. Igor Zuckerberg")

def page_buscar_ativo():
    st.title("Buscar Ação + Projeção (brapi.dev)")
    ticker = st.text_input("Ticker (ex: PETR4):", "").upper().strip().replace(".SA", "")
    period = st.selectbox("Período:", ["1 mês","3 meses"])
    code_map = {"1 mês":"1mo","3 meses":"3mo"}
    if st.button("Buscar e projetar"):
        if not ticker:
            st.error("Digite um ticker.")
            return
        df = fetch_history_brapi(ticker, code_map[period])
        if df is None or df.empty:
            st.error("Não foi possível obter dados.")
            return

        df = df.sort_values("date")
        df_future = forecast_trend(df, days_ahead=30)

        # Plot histórico + projeção
        st.subheader("Histórico e Projeção (próximos 30 dias)")
        fig, ax = plt.subplots()
        ax.plot(df["date"], df["close"], label="Histórico", marker=".", linewidth=1)
        ax.plot(df_future["date"], df_future["close"], label="Projeção", linestyle="--")
        ax.set_xlabel("Data")
        ax.set_ylabel("Preço (R$)")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

def page_calculadora():
    st.title("Calculadora de Investimentos")

    PV = st.number_input("Investimento inicial (R$):", value=1000.0)
    A = st.number_input("Aporte mensal (R$):", value=100.0)
    n = st.number_input("Meses:", min_value=1, value=12)

    # Busca taxas de referência
    selic = fetch_bcb_rate(11)
    cdi = fetch_bcb_rate(12)
    cdi_anual = ((1 + cdi/100)**252 - 1) * 100 if cdi else None

    st.markdown("### Taxas de referência")
    st.write(f"SELIC diária: {selic:.4f}%")
    st.write(f"CDI anual aprox.: {cdi_anual:.2f}%")

    st.markdown("### Configuração de produtos")
    taxa_poup = st.number_input("Poupança anual (%):", value=4.5)
    perc_cdb = st.number_input("CDB (% do CDI):", min_value=0.0, value=100.0) if cdi_anual else None
    perc_lci = st.number_input("LCI (% do CDI):", min_value=0.0, value=90.0) if cdi_anual else None
    perc_lca = st.number_input("LCA (% do CDI):", min_value=0.0, value=90.0) if cdi_anual else None

    def ir_rate(days: int) -> float:
        if days <= 180:
            return 0.225
        elif days <= 360:
            return 0.20
        elif days <= 720:
            return 0.175
        else:
            return 0.15

    if st.button("Calcular"):
        # Valor investido total
        inv_total = PV + A * n
        # Mapear cada produto em sua taxa anual
        produtos = {
            "Poupança": taxa_poup,
        }
        if cdi_anual:
            produtos["CDB"] = cdi_anual * perc_cdb / 100
            produtos["LCI"] = cdi_anual * perc_lci / 100
            produtos["LCA"] = cdi_anual * perc_lca / 100

        # Calcular valor futuro
        resultados = {k: calcular_valor_futuro(PV, A, n, taxa) for k, taxa in produtos.items()}
        # Montar DataFrame
        df = pd.DataFrame.from_dict(resultados, orient="index", columns=["Valor Futuro"])
        df["Valor Investido"] = inv_total
        df["Lucro Bruto"] = df["Valor Futuro"] - df["Valor Investido"]

        # Imposto e lucro líquido
        # converte meses em dias para IR
        days = n * 30
        impostos = []
        for produto, bruto in df["Lucro Bruto"].items():
            if produto == "CDB":
                rate = ir_rate(days)
                impostos.append(bruto * rate)
            else:
                impostos.append(0.0)
        df["Imposto IR"] = impostos
        df["Lucro Líquido"] = df["Lucro Bruto"] - df["Imposto IR"]

        # Formatar em reais
        fmt = lambda x: f"R$ {x:,.2f}"
        for col in ["Valor Futuro", "Valor Investido", "Lucro Bruto", "Imposto IR", "Lucro Líquido"]:
            df[col] = df[col].map(fmt)

        st.table(df)

        # Gráfico comparativo
        xs = np.arange(1, n + 1)
        fig, ax = plt.subplots()
        for produto, taxa in produtos.items():
            # mesma lógica de IR para a linha de Net: aqui só plotamos bruto
            ys = [calcular_valor_futuro(PV, A, m, taxa) for m in xs]
            ax.plot(xs, ys, label=produto)
        ax.set(xlabel="Meses", ylabel="Valor (R$)")
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

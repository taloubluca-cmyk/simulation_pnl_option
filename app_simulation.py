import pandas as pd 
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy.random as nrd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
import plotly.graph_objects as go
import streamlit as st

@st.cache_data
def get_sabr_params_from_excel(expiry: str, excel_file: str = "P3000.xlsx", beta: float = 0.5):
    """
    Calibre SABR à partir des données IV les plus récentes (dernière ligne non-NaN)
    pour l'échéance choisie. Le résultat est mis en cache => calibrage fait 1 seule fois
    tant que les données/expiry ne changent pas.
    """
    if expiry == "Sep26":
        sheet_iv = "IV Sep26"
    else:
        sheet_iv = "IV Jul26"

    df_iv = pd.read_excel(excel_file, sheet_name=sheet_iv, index_col=0).dropna()

    # Dernière ligne complète = données les plus récentes
    last_row = df_iv.iloc[-1]

    F0_smile = float(last_row.iloc[0])
    T_smile = float(last_row.iloc[1] / 365.0)

    K_list = np.array(df_iv.columns[2:]).astype(float)
    iv_market = np.array(last_row.iloc[2:]).astype(float) / 100.0

    params_sabr = calibrate_sabr_from_iv(F0_smile, T_smile, K_list, iv_market, beta=beta)
    return params_sabr


def price_put_on_future(Ft,K,t,T,r,sigma):
  if t == T:
    return max(K - Ft,0)
  else:
    d1 = (np.log(Ft/K) + 0.5*(sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
    d2 = d1 - sigma*np.sqrt(T-t)
    return np.exp(-r*(T-t))*(K*norm.cdf(-d2) - Ft*norm.cdf(-d1))

def delta_put_on_future(Ft,K,t,T,r,sigma):
  d1 = (np.log(Ft/K) + 0.5*(sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
  return -np.exp(-r*(T-t))*norm.cdf(-d1)

def gamma_on_future(Ft,K,t,T,r,sigma):
  d1 = (np.log(Ft/K) + 0.5*(sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
  return np.exp(-r*(T-t))*norm.pdf(d1)/(Ft*sigma*np.sqrt(T-t))

def vega_on_future(Ft,K,t,T,r,sigma):
  d1 = (np.log(Ft/K) + 0.5*(sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
  return np.exp(-r*(T-t))*Ft*np.sqrt(T-t)*norm.pdf(d1)

def theta_put_on_future(Ft,K,t,T,r,sigma):
  d1 = (np.log(Ft/K) + 0.5*(sigma**2)*(T-t))/(sigma*np.sqrt(T-t))
  return -r*price_put_on_future(Ft,K,t,T,r,sigma) + np.exp(-r*(T-t))*Ft*norm.pdf(d1)*sigma/(2*np.sqrt(T-t))



def sabr_implied_vol(F0, K, T, alpha, beta, rho, nu):
    """
    Vol implicite Black (approx Hagan) du modèle SABR pour un put/call sur future.
    """
    if F0 <= 0 or K <= 0 or T <= 0:
        return 0.0

    if F0 == K:
        FK_beta = F0**(1 - beta)
        term1 = alpha / (FK_beta)
        term2 = ((1 - beta)**2 / 24 * (alpha**2) / (FK_beta**2)
                 + 0.25 * rho * beta * nu * alpha / FK_beta
                 + (2 - 3 * rho**2) * (nu**2) / 24) * T
        return term1 * (1 + term2)

    lnFK = np.log(F0 / K)
    FK_beta = (F0 * K)**((1 - beta) / 2.0)
    z = (nu / alpha) * FK_beta * lnFK
    x_z = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho))

    A = alpha / (FK_beta)
    B1 = ((1 - beta)**2 / 24) * (alpha**2) / (FK_beta**2)
    B2 = 0.25 * rho * beta * nu * alpha / FK_beta
    B3 = (2 - 3 * rho**2) * (nu**2) / 24
    B = 1 + (B1 + B2 + B3) * T

    if abs(z) < 1e-8:
        return A * B

    return A * (z / x_z) * B


def calibrate_sabr_from_iv(F0, T, K_list, iv_market, beta=0.5):
    K_array = np.asarray(K_list, dtype=float)
    iv_mkt  = np.asarray(iv_market, dtype=float)

    # Strike ATM ~ le plus proche de F0
    idx_atm = np.argmin(np.abs(K_array - F0))
    sigma_atm = iv_mkt[idx_atm]

    Fbeta = F0**(1 - beta)
    alpha0 = sigma_atm * Fbeta
    rho0   = -0.3
    nu0    = 0.5

    alpha_min = 1e-6
    alpha_max = max(5 * alpha0, 1.0)
    nu_min    = 1e-6
    nu_max    = 5.0

    lb = np.array([alpha_min, -0.999, nu_min])
    ub = np.array([alpha_max,  0.999, nu_max])

    x0 = np.array([alpha0, rho0, nu0])
    x0 = np.minimum(np.maximum(x0, lb), ub)

    def residuals(x):
        alpha, rho, nu = x
        if alpha <= 0 or nu <= 0 or not (-0.999 < rho < 0.999):
            return 1e6 * np.ones_like(iv_mkt)

        iv_model = np.array([
            sabr_implied_vol(F0, K_array[i], T, alpha, beta, rho, nu)
            for i in range(len(K_array))
        ])
        return iv_model - iv_mkt

    res = least_squares(
        residuals,
        x0=x0,
        bounds=(lb, ub),
        method="trf",
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        max_nfev=100
    )

    alpha_opt, rho_opt, nu_opt = res.x
    return {
        "alpha": float(alpha_opt),
        "beta": float(beta),
        "rho": float(rho_opt),
        "nu": float(nu_opt),
        "success": bool(res.success),
        "cost": float(res.cost),
        "nfev": int(res.nfev),
        "alpha_bounds": (alpha_min, alpha_max),
        "sigma_atm": float(sigma_atm)
    }


def simulate_sabr_path(F0, alpha0, beta, rho, nu, T, N, seed=None):
    """
    Simule un chemin SABR (F_t, alpha_t).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / N
    t_grid = np.linspace(0.0, T, N + 1)

    F_path = np.zeros(N + 1, dtype=float)
    alpha_path = np.zeros(N + 1, dtype=float)

    F_path[0] = F0
    alpha_path[0] = alpha0

    for i in range(N):
        Z1 = np.random.normal()
        Z2 = np.random.normal()
        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1.0 - rho**2) * Z2)

        F_t = F_path[i]
        alpha_t = alpha_path[i]

        dF = alpha_t * (F_t ** beta) * dW1
        dalpha = nu * alpha_t * dW2

        F_new = F_t + dF
        alpha_new = alpha_t + dalpha

        F_path[i + 1] = max(F_new, 1e-12)
        alpha_path[i + 1] = max(alpha_new, 1e-12)

    return t_grid, F_path, alpha_path



def build_curve_by_strike(K, IV, S_const, T,K_target):
    w = IV**2*T
    k = np.log(K / float(S_const))
    idx = np.argsort(k)
    k, w, K = k[idx], w[idx], K[idx]
    interp = PchipInterpolator(k, w, extrapolate=False)
    dcore = interp.derivative()
    kmin, kmax = float(k[0]), float(k[-1])
    wmin, wmax = float(w[0]), float(w[-1])
    mL = float(dcore(kmin))
    mR = float(dcore(kmax))
    kq = np.log(float(K_target) / float(S_const))
    wq = float(interp(kq))
    if kq < kmin:
        wq = wmin + mL * (kq - kmin)
    elif kq > kmax:
        wq = wmax + mR * (kq - kmax)
    iv = np.sqrt(max(wq, 0.0) / T)
    return iv

def build_curve_from_columns(IV, K, T, S, xq):
    delta_col = [0.0] * len(IV)
    for i in range(len(IV)):
        delta_col[i] = delta_put_on_future(S, K[i], 0, T, 0, IV[i])
    x = np.asarray(delta_col, dtype=float)     
    y = np.asarray(IV**2 * T, dtype=float)       
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    interp = PchipInterpolator(x, y, extrapolate=False)
    dcore = interp.derivative()
    xmin, xmax = float(x[0]),  float(x[-1])
    ymin, ymax = float(y[0]),  float(y[-1])
    mL = float(dcore(xmin))    
    mR = float(dcore(xmax))   
    yq = float(interp(xq))
    if xq < xmin:
        yq = ymin + mL * (xq - xmin)
    elif xq > xmax:
        yq = ymax + mR * (xq - xmax)
    iv_curve = np.sqrt(max(yq, 0.0) / T)
    return iv_curve


def simulation_future_const(F0, T, sigma, N):
  delta_t = T/N
  F = [F0]
  for i in range(N):
    F.append(round(F[i]*np.exp((-sigma**2/2)*delta_t +sigma*np.sqrt(delta_t)*nrd.randn())))
  return np.array(F[1:])

def simulation_future_sticky_delta(
    F0, T_sim, sigma0, N,
    IV_smile, K_smile, T_smile, S_smile,
    K_target, T_maturity_path
):
    delta_t = T_sim / N
    F = [F0]
    sigmas = [sigma0]

    for i in range(N):
        T_remain = T_maturity_path[i] 
        delta_target = delta_put_on_future(
            F[i],
            K_target,
            0,
            T_remain,
            0,
            sigmas[-1]
        )
        sigma_next = build_curve_from_columns(
            IV_smile,
            K_smile,
            T_smile,
            S_smile,
            delta_target
        )
        dW = nrd.randn()
        F_next = F[i] * np.exp((-sigma_next**2 / 2) * delta_t + sigma_next * np.sqrt(delta_t) * dW)
        F.append(round(F_next))
        sigmas.append(sigma_next)

    return np.array(F[1:]), np.array(sigmas[1:])

def simulation_future_sabr(
    F0,
    K_target,
    T_maturity_path,
    N,
    r,
    params_sabr,
    sim_seed=None
):

    beta = params_sabr["beta"]
    alpha0 = params_sabr["alpha"]
    rho = params_sabr["rho"]
    nu = params_sabr["nu"]

    T_sim = float(T_maturity_path[0])
    t_grid, F_path, alpha_path = simulate_sabr_path(
        F0=F0,
        alpha0=alpha0,
        beta=beta,
        rho=rho,
        nu=nu,
        T=T_sim,
        N=N,
        seed=sim_seed
    )

    F_path = F_path[1:]
    alpha_path = alpha_path[1:]

    iv_path = np.zeros(N, dtype=float)

    for i in range(N):
        tau = T_maturity_path[i]
        if tau <= 0:
            iv_path[i] =  iv_path[i-1]
        else:
            iv_path[i] = sabr_implied_vol(
                F0=F_path[i],
                K=K_target,
                T=tau,
                alpha=alpha_path[i],
                beta=beta,
                rho=rho,
                nu=nu
            )

    return np.round(F_path), iv_path, alpha_path


def prepare_dataframe(expiry: str, sim_seed: int, vol_mode: str,
                      excel_file: str = "P3000.xlsx", K_strike: float = 3000.0):

    # -----------------------------
    # 1. Lecture des données marché
    # -----------------------------
    if expiry == "Sep26":
        sheet_price = "Sep26"
        sheet_iv = "IV Sep26"
    else:
        sheet_price = "Jul26"
        sheet_iv = "IV Jul26"

    # Prix/IV historiques
    df = pd.read_excel(excel_file, sheet_name=sheet_price, index_col=0)
    df = df[df["IV"] != 0]
    df = df.sort_index()

    # Table IV pour le smile (utilisé pour vol_impl, sticky, SABR calibration)
    df_iv = pd.read_excel(excel_file, sheet_name=sheet_iv, index_col=0)

    # On enlève les lignes complètement vides
    df_iv = df_iv.dropna(how="all")

    # On suppose que la 1ère ligne non-NaN sert pour le smile "référence"
    first_row = df_iv.dropna().iloc[0]

    K = np.array(df_iv.columns[2:]).astype(float)
    IV = np.array(first_row.iloc[2:]).astype(float) / 100.0
    T = float(first_row.iloc[1] / 365.0)
    F0 = float(first_row.iloc[0])

    # Vol implicite "atm" (smile converti via build_curve_by_strike)
    vol_impl = build_curve_by_strike(K, IV, F0, T, F0)

    # Ajout d'une colonne pour la vol instantanée SABR (remplie seulement si SABR)
    if "Alpha SABR" not in df.columns:
        df["Alpha SABR"] = np.nan

    # -----------------------------
    # 2. Extension du DataFrame dans le futur
    # -----------------------------
    maturity_last = int(df["Maturity"].iloc[-1])

    all_dates = pd.date_range(start=df.index[-1], periods=maturity_last + 1)[1:]
    business_dates = all_dates[all_dates.weekday < 5]
    df = pd.concat([df, pd.DataFrame(index=business_dates)])

    # On propage la dernière IV connue vers le futur (pour init / fallback)
    df["IV"] = df["IV"].ffill()

    # FIX : on se base sur la dernière date avec Underlying non-NaN
    last_non_na_idx = df["Underlying"].dropna().index[-1]
    mask_future = df.index > last_non_na_idx

    # -----------------------------
    # 3. Simulation du futur selon le mode de vol
    # -----------------------------
    nrd.seed(sim_seed)

    if mask_future.sum() > 0:
        n_steps = len(df.loc[mask_future])

        # T_sim sert pour les modèles "classiques" (constante / sticky)
        T_sim = n_steps / 252.0

        F_start = df["Underlying"].dropna().iloc[-1]

        # Recalcule Maturity (jours restants jusqu'à la dernière date)
        df["Maturity"] = (df.index[-1] - df.index).days
        T_maturity_path = df.loc[mask_future, "Maturity"].values / 365.0

        if vol_mode == "Vol sticky delta":
            # Sticky delta : vol smile bouge en fonction du delta
            F_path, sigma_path = simulation_future_sticky_delta(
                F_start,
                T_sim,
                vol_impl,
                n_steps,
                IV,
                K,
                T,
                F0,
                K_strike,
                T_maturity_path
            )
            df.loc[mask_future, "Underlying"] = F_path
            df.loc[mask_future, "IV"] = sigma_path * 100.0

        elif vol_mode == "Vol SABR (stochastique)":
            # SABR : calibration sur les données IV les plus récentes
            params_sabr = get_sabr_params_from_excel(
                expiry,
                excel_file=excel_file,
                beta=0.5
            )

            # Simulation SABR : F_t, IV SABR(t), alpha_t
            F_path, iv_sabr_path, alpha_path = simulation_future_sabr(
                F0=F_start,
                K_target=K_strike,
                T_maturity_path=T_maturity_path,
                N=n_steps,
                r=0.0,
                params_sabr=params_sabr,
                sim_seed=sim_seed
            )

            df.loc[mask_future, "Underlying"] = F_path
            df.loc[mask_future, "IV"] = iv_sabr_path * 100.0
            df.loc[mask_future, "Alpha SABR"] = alpha_path

        else:
            # Vol constante : diffusion lognormale simple
            F_path = simulation_future_const(
                F_start,
                T_sim,
                vol_impl,
                n_steps,
            )
            df.loc[mask_future, "Underlying"] = F_path

    # -----------------------------
    # 4. Prix & grecs (utilisent toujours df["IV"])
    # -----------------------------
    df["Price"] = df.apply(
        lambda x: price_put_on_future(
            x["Underlying"],
            K_strike,
            0,
            x["Maturity"] / 365.0,
            0.0,
            x["IV"] / 100.0,
        ),
        axis=1,
    )

    df["Delta"] = df.apply(
        lambda x: delta_put_on_future(
            x["Underlying"], K_strike, 0, x["Maturity"] / 365.0, 0.0, x["IV"] / 100.0
        ),
        axis=1,
    )
    df["Gamma"] = df.apply(
        lambda x: gamma_on_future(
            x["Underlying"], K_strike, 0, x["Maturity"] / 365.0, 0.0, x["IV"] / 100.0
        ),
        axis=1,
    )
    df["Vega"] = df.apply(
        lambda x: vega_on_future(
            x["Underlying"], K_strike, 0, x["Maturity"] / 365.0, 0.0, x["IV"] / 100.0
        ),
        axis=1,
    )
    df["Theta"] = df.apply(
        lambda x: theta_put_on_future(
            x["Underlying"], K_strike, 0, x["Maturity"] / 365.0, 0.0, x["IV"] / 100.0
        ),
        axis=1,
    )

    # -----------------------------
    # 5. PnL & décomposition
    # -----------------------------
    df["Var. Underlying"] = df["Underlying"].diff()
    df["Var. IV"] = df["IV"].diff()
    df["Var. Time"] = df["Maturity"].diff()
    df["PnL Real"] = df["Price"].diff()

    df["PnL Delta"] = df["Delta"].shift(1) * df["Var. Underlying"]
    df["PnL Gamma"] = 0.5 * df["Gamma"].shift(1) * df["Var. Underlying"] ** 2
    df["PnL Vega"] = df["Vega"].shift(1) * df["Var. IV"] / 100.0
    df["PnL Theta"] = df["Theta"].shift(1) * df["Var. Time"] / 365.0

    df["PnL Unexplained"] = (
        df["PnL Real"] - df["PnL Delta"] - df["PnL Gamma"] - df["PnL Vega"] - df["PnL Theta"]
    )

    # Cumuls
    for col in ["PnL Real", "PnL Delta", "PnL Gamma", "PnL Vega", "PnL Theta", "PnL Unexplained"]:
        df[col] = df[col].cumsum()

    return df, vol_impl

# ===============================
#      APP STREAMLIT
# ===============================

st.set_page_config(page_title="Analyse Option Futur", layout="wide")

st.title("Analyse PnL Option – Sep26 / Jul26")

# Session state pour contrôler la simulation
if "sim_seed" not in st.session_state:
    st.session_state.sim_seed = 0

# Barre latérale
with st.sidebar:
    st.header("Paramètres")

    expiry = st.selectbox("Échéance", ["Sep26", "Jul26"])

    K_strike = st.number_input("Strike K", value=3000.0, step=50.0)

    vol_mode = st.radio(
        "Modèle de volatilité",
        ["Vol constante", "Vol sticky delta","Vol SABR (stochastique)"],
        index=0
    )

    st.markdown("---")
    resimulate_clicked = st.button("🔁 Re-simuler le sous-jacent")

    if resimulate_clicked:
        st.session_state.sim_seed += 1  # change la graine => nouvelle simu

# Calcul des données (les mêmes si aucune re-simu, nouvelles si bouton cliqué)
df, vol_impl = prepare_dataframe(expiry, st.session_state.sim_seed,vol_mode, excel_file="P3000.xlsx", K_strike=K_strike)

# Petits KPIs
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Dernier Underlying", f"{df['Underlying'].iloc[-1]:,.2f}")
with col2:
    st.metric("Dernier Price", f"{df['Price'].iloc[-1]:,.2f}")
with col3:
    st.metric("Vol implicite utilisée (smile ATM)", f"{vol_impl*100:.2f} %")

st.markdown("---")

# ===============================
#   FIGURE 1 : 3D Price / Underlying / IV
# ===============================
st.subheader("Chart of price option, future & implied volatility")

fig_axes = go.Figure()

# Underlying : axe Y principal (gauche)
fig_axes.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Underlying"],
        name="Underlying",
        yaxis="y1",
        mode="lines",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Underlying: %{y:.2f}<extra></extra>",
    )
)

# Price : deuxième axe Y (droite)
fig_axes.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Price"],
        name="Price",
        yaxis="y2",
        mode="lines",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Price: %{y:.2f}<extra></extra>",
    )
)

# IV : troisième axe Y (droite, décalé)
fig_axes.add_trace(
    go.Scatter(
        x=df.index,
        y=df["IV"],
        name="IV (%)",
        yaxis="y3",
        mode="lines",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>IV: %{y:.2f} %<extra></extra>",
    )
)

# Vol instantanée SABR (si présente et si on est en mode SABR)
if vol_mode == "Vol SABR (stochastique)" and "Alpha SABR" in df.columns:
    fig_axes.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Alpha SABR"],
            name="Vol instantanée SABR (%)",
            yaxis="y3",   # on la met sur le même axe que l'IV
            mode="lines",
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Alpha SABR: %{y:.2f} %<extra></extra>",
        )
    )


fig_axes.update_layout(
    xaxis=dict(title="Date"),

    # Axe Y1 : Underlying
    yaxis=dict(
        title="Underlying",
        side="left",
    ),

    # Axe Y2 : Price (superposé à y, à droite)
    yaxis2=dict(
        title="Price",
        overlaying="y",
        side="right",
    ),

    # Axe Y3 : IV (superposé aussi à y, à droite mais un peu décalé)
    yaxis3=dict(
        title="IV (%)",
        overlaying="y",
        side="right",   # décale un peu l’axe pour qu’il ne soit pas pile sur y2
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),
    height=500,
    margin=dict(l=40, r=80, t=40, b=40),
)

st.plotly_chart(fig_axes, use_container_width=True)



st.markdown("---")

# ===============================
#   FIGURE 2 : PnL & Greeks dans le temps
# ===============================

st.subheader("PnL of greeks")

# Choix des séries à afficher
pnl_cols = ["PnL Delta", "PnL Gamma", "PnL Vega", "PnL Theta"]
selected_series = st.multiselect(
    "Séries à afficher",
    options=pnl_cols,
    default=["PnL Delta", "PnL Gamma", "PnL Vega", "PnL Theta"],
)

fig_pnl = go.Figure()

for col in selected_series:
    fig_pnl.add_trace(
        go.Scatter(
            x=df.index,
            y=df[col],
            mode="lines",
            name=col,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>" + col + ": %{y:.2f}<extra></extra>",
        )
    )

fig_pnl.update_layout(
    xaxis_title="Date",
    yaxis_title="PnL (cumulé)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=500,
    margin=dict(l=40, r=20, t=40, b=40),
)

st.plotly_chart(fig_pnl, use_container_width=True)

# ===============================
#   TABLEAU (optionnel)
# ===============================

with st.expander("Voir les dernières lignes du dataframe (debug / contrôle)"):

    st.dataframe(df.tail(20))

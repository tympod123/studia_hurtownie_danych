import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Wine + Food Pairings — Analiza i Integracja",
    page_icon="🍷",
    layout="wide"
)

DATA_DIR = "data"
WINE_QUALITY_PATH = os.path.join(DATA_DIR, "winequality-red.csv")
PAIRINGS_PATH = os.path.join(DATA_DIR, "wine_food_pairings.csv")

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_data():
    wine = pd.read_csv(WINE_QUALITY_PATH)
    pairings = pd.read_csv(PAIRINGS_PATH)
    return wine, pairings

@st.cache_data
def basic_profile(df: pd.DataFrame):
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing_total": int(df.isna().sum().sum()),
        "missing_by_col": df.isna().sum().sort_values(ascending=False),
        "dtypes": df.dtypes
    }

@st.cache_data
def correlation_df(wine_df: pd.DataFrame):
    corr = wine_df.corr(numeric_only=True)
    return corr

def quality_tier(q: float):
    # tiers for integration (editable)
    if q <= 5.0:
        return "Low"
    elif q <= 6.0:
        return "Mid"
    else:
        return "High"

def tier_to_pairing_threshold(tier: str):
    # how strict to be with pairing_quality (1..5)
    if tier == "Low":
        return 3
    if tier == "Mid":
        return 4
    return 4

@st.cache_resource
def train_quality_model(wine_df: pd.DataFrame):
    # Model: predict "quality" from chem features
    target = "quality"
    X = wine_df.drop(columns=[target])
    y = wine_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=400,
            random_state=42,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds))
    }

    # Permutation importance would be heavier; here use RF feature importance
    rf = model.named_steps["rf"]
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    return model, metrics, importances, X.columns.tolist()

def make_corr_heatmap(corr: pd.DataFrame):
    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        title="Macierz korelacji (winequality-red)"
    )
    fig.update_layout(height=650)
    return fig

def safe_value(v, default):
    try:
        return float(v)
    except Exception:
        return default

# -----------------------------
# Load data
# -----------------------------
st.title("🍷 Wine Quality + 🍽️ Food Pairings — Aplikacja analityczna (Streamlit)")
st.caption("Analiza dwóch datasetów + integracja: chemia wina → przewidywana jakość → rekomendacje pairingów.")

if not (os.path.exists(WINE_QUALITY_PATH) and os.path.exists(PAIRINGS_PATH)):
    st.error(
        "Brakuje plików CSV w folderze `data/`.\n\n"
        "Upewnij się, że w repo są:\n"
        "- data/winequality-red.csv\n"
        "- data/wine_food_pairings.csv"
    )
    st.stop()

wine_df, pairings_df = load_data()

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("⚙️ Sterowanie")
page = st.sidebar.radio(
    "Wybierz moduł",
    ["Dashboard", "Wine Quality — EDA", "Food Pairings — EDA", "Integracja + Rekomendacje", "Eksplorator danych"]
)

st.sidebar.divider()
st.sidebar.subheader("Filtry (pairings)")
wine_type_filter = st.sidebar.multiselect(
    "wine_type",
    options=sorted(pairings_df["wine_type"].unique()),
    default=[pairings_df["wine_type"].mode().iloc[0]] if "wine_type" in pairings_df.columns else None
)

food_cat_filter = st.sidebar.multiselect(
    "food_category",
    options=sorted(pairings_df["food_category"].unique()),
    default=[]
)

cuisine_filter = st.sidebar.multiselect(
    "cuisine",
    options=sorted(pairings_df["cuisine"].unique()),
    default=[]
)

min_pairing_quality = st.sidebar.slider("Min pairing_quality", 1, 5, 3)

def filter_pairings(df: pd.DataFrame):
    out = df.copy()
    if wine_type_filter:
        out = out[out["wine_type"].isin(wine_type_filter)]
    if food_cat_filter:
        out = out[out["food_category"].isin(food_cat_filter)]
    if cuisine_filter:
        out = out[out["cuisine"].isin(cuisine_filter)]
    out = out[out["pairing_quality"] >= min_pairing_quality]
    return out

pairings_f = filter_pairings(pairings_df)

# -----------------------------
# Dashboard
# -----------------------------
if page == "Dashboard":
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("📌 Szybki profil datasetów")

        w_prof = basic_profile(wine_df)
        p_prof = basic_profile(pairings_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("winequality-red: wiersze", w_prof["rows"])
        c2.metric("winequality-red: kolumny", w_prof["cols"])
        c3.metric("winequality-red: braki", w_prof["missing_total"])

        c1, c2, c3 = st.columns(3)
        c1.metric("pairings: wiersze", p_prof["rows"])
        c2.metric("pairings: kolumny", p_prof["cols"])
        c3.metric("pairings: braki", p_prof["missing_total"])

        st.markdown("**Najczęstsze wine_type (po filtrach):**")
        if len(pairings_f) > 0:
            st.dataframe(pairings_f["wine_type"].value_counts().head(10))
        else:
            st.info("Brak wyników po filtrach.")

    with right:
        st.subheader("📊 Rozkład jakości wina i jakości pairingów")

        fig1 = px.histogram(wine_df, x="quality", nbins=12, title="Rozkład quality (winequality-red)")
        fig1.update_layout(height=350)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(pairings_f, x="pairing_quality", nbins=10, title="Rozkład pairing_quality (po filtrach)")
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("🔥 Top relacje w pairings (po filtrach)")
    if len(pairings_f) > 0:
        top_food = pairings_f.groupby(["food_category", "food_item"])["pairing_quality"].mean().sort_values(ascending=False).head(20)
        st.dataframe(top_food.reset_index().rename(columns={"pairing_quality": "avg_pairing_quality"}))
    else:
        st.info("Brak rekordów po filtrach.")

# -----------------------------
# Wine Quality — EDA
# -----------------------------
elif page == "Wine Quality — EDA":
    st.subheader("🍷 Wine Quality — EDA i wizualizacje")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Dane")
        st.dataframe(wine_df.head(30), use_container_width=True)

    with c2:
        st.markdown("### Statystyki opisowe")
        st.dataframe(wine_df.describe().T, use_container_width=True)

    st.divider()

    # Correlation heatmap
    corr = correlation_df(wine_df)
    st.plotly_chart(make_corr_heatmap(corr), use_container_width=True)

    st.divider()

    st.markdown("### Interaktywny scatter: cechy vs jakość")
    num_cols = [c for c in wine_df.columns if c != "quality"]
    xcol = st.selectbox("Oś X", options=num_cols, index=num_cols.index("alcohol") if "alcohol" in num_cols else 0)
    ycol = st.selectbox("Oś Y", options=num_cols, index=num_cols.index("volatile acidity") if "volatile acidity" in num_cols else 1)
    color_by = st.selectbox("Kolor", options=["quality"] + num_cols, index=0)

    fig_sc = px.scatter(
        wine_df, x=xcol, y=ycol, color=color_by,
        title=f"{xcol} vs {ycol} (kolor: {color_by})",
        opacity=0.7
    )
    fig_sc.update_layout(height=550)
    st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    st.markdown("### Rozkłady cech chemicznych (wybór)")
    selected = st.multiselect(
        "Wybierz cechy",
        options=num_cols,
        default=["alcohol", "volatile acidity", "sulphates"] if set(["alcohol","volatile acidity","sulphates"]).issubset(num_cols) else num_cols[:3]
    )
    if selected:
        df_melt = wine_df[selected].melt(var_name="feature", value_name="value")
        fig_hist = px.histogram(df_melt, x="value", facet_col="feature", facet_col_wrap=3, title="Histogramy cech")
        fig_hist.update_layout(height=600)
        st.plotly_chart(fig_hist, use_container_width=True)

# -----------------------------
# Food Pairings — EDA
# -----------------------------
elif page == "Food Pairings — EDA":
    st.subheader("🍽️ Food Pairings — EDA i wizualizacje (z filtrami z sidebaru)")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("### Dane (po filtrach)")
        st.dataframe(pairings_f.head(50), use_container_width=True)

    with c2:
        st.markdown("### Statystyki")
        st.write("Liczba rekordów po filtrach:", len(pairings_f))
        st.dataframe(pairings_f.describe(include="all").T.head(30), use_container_width=True)

    st.divider()

    if len(pairings_f) == 0:
        st.warning("Brak danych po filtrach — poluzuj filtry w sidebarze.")
        st.stop()

    # Quality label distribution
    fig_q = px.bar(
        pairings_f["quality_label"].value_counts().reset_index(),
        x="index", y="quality_label",
        title="Rozkład quality_label (po filtrach)",
        labels={"index": "quality_label", "quality_label": "count"}
    )
    fig_q.update_layout(height=400)
    st.plotly_chart(fig_q, use_container_width=True)

    # Heatmap: cuisine vs food_category (count)
    pivot = pd.pivot_table(
        pairings_f,
        index="cuisine",
        columns="food_category",
        values="pairing_quality",
        aggfunc="mean"
    ).fillna(np.nan)

    fig_hm = px.imshow(
        pivot,
        aspect="auto",
        title="Średnia pairing_quality: cuisine × food_category"
    )
    fig_hm.update_layout(height=650)
    st.plotly_chart(fig_hm, use_container_width=True)

    st.divider()

    # Top foods
    top_n = st.slider("Top N food_item", 5, 50, 20)
    top_food = (
        pairings_f.groupby(["food_category", "food_item"])["pairing_quality"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig_top = px.bar(
        top_food,
        x="pairing_quality",
        y="food_item",
        color="food_category",
        orientation="h",
        title=f"Top {top_n} food_item wg średniego pairing_quality"
    )
    fig_top.update_layout(height=800)
    st.plotly_chart(fig_top, use_container_width=True)

# -----------------------------
# Integration + Recommendations
# -----------------------------
elif page == "Integracja + Rekomendacje":
    st.subheader("🔗 Integracja datasetów: chemia → przewidywana jakość → rekomendacje jedzenia")

    st.markdown("""
Ten moduł robi 3 rzeczy:
1) trenuje model predykcji `quality` na `winequality-red.csv`  
2) pozwala wybrać wino z datasetu **lub** wprowadzić parametry ręcznie  
3) mapuje przewidywaną jakość na „tier” i pokazuje **najlepsze pairingi** z `wine_food_pairings.csv`
""")

    model, metrics, importances, feature_cols = train_quality_model(wine_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Model MAE (holdout)", f"{metrics['MAE']:.3f}")
    c2.metric("Model R² (holdout)", f"{metrics['R2']:.3f}")
    c3.metric("Cech", len(feature_cols))

    st.divider()

    left, right = st.columns([1, 1.2])

    with left:
        st.markdown("### 1) Wybierz sposób wejścia")
        mode = st.radio("Źródło parametrów", ["Wybierz rekord z datasetu", "Wprowadź ręcznie"], horizontal=True)

        if mode == "Wybierz rekord z datasetu":
            idx = st.slider("Indeks rekordu (winequality-red)", 0, len(wine_df) - 1, 0)
            row = wine_df.iloc[idx]
            x = row.drop(labels=["quality"]).to_frame().T
            st.write("Wybrany rekord (cechy):")
            st.dataframe(x, use_container_width=True)
            st.write("Prawdziwa quality:", int(row["quality"]))
        else:
            # Manual input with sensible defaults from mean
            defaults = wine_df.drop(columns=["quality"]).mean(numeric_only=True)
            inputs = {}
            for col in feature_cols:
                val = st.number_input(col, value=float(defaults[col]), format="%.5f")
                inputs[col] = val
            x = pd.DataFrame([inputs])

    with right:
        st.markdown("### 2) Predykcja jakości + interpretacja")
        pred = float(model.predict(x)[0])
        tier = quality_tier(pred)

        st.metric("Przewidywana quality", f"{pred:.2f}")
        st.metric("Tier (do rekomendacji)", tier)

        # Feature importance chart
        imp_df = importances.head(12).reset_index()
        imp_df.columns = ["feature", "importance"]
        fig_imp = px.bar(imp_df, x="importance", y="feature", orientation="h", title="Top 12 feature importance (RF)")
        fig_imp.update_layout(height=520)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.divider()

    st.markdown("### 3) Rekomendacje pairingów na podstawie tieru i filtrów")

    # choose mapping to wine type/category for pairings
    # since winequality-red is red wine, we try to prioritize wine_type containing "Red"
    pair_df = pairings_df.copy()
    # soften: match any case with 'red' in wine_type if possible
    if "wine_type" in pair_df.columns:
        mask_red = pair_df["wine_type"].astype(str).str.contains("red", case=False, na=False)
        if mask_red.any():
            pair_df = pair_df[mask_red]

    # apply sidebar filters too, but starting from "red subset"
    # reapply same filter function but on pair_df
    def filter_pairings_local(df):
        out = df.copy()
        if wine_type_filter:
            out = out[out["wine_type"].isin(wine_type_filter)]
        if food_cat_filter:
            out = out[out["food_category"].isin(food_cat_filter)]
        if cuisine_filter:
            out = out[out["cuisine"].isin(cuisine_filter)]
        return out

    pair_df = filter_pairings_local(pair_df)

    threshold = tier_to_pairing_threshold(tier)
    st.write(f"Minimalny próg pairing_quality dla tieru **{tier}**: **{threshold}** (możesz też użyć suwaka w sidebarze)")

    # combine thresholds: max(sidebar_min, tier_threshold)
    final_min = max(min_pairing_quality, threshold)
    pair_df = pair_df[pair_df["pairing_quality"] >= final_min]

    if len(pair_df) == 0:
        st.warning("Brak pairingów spełniających warunki. Zmień filtry lub obniż minimalny próg.")
        st.stop()

    # rank recommendations
    recs = (
        pair_df
        .groupby(["food_category", "food_item", "cuisine"], as_index=False)
        .agg(avg_pairing_quality=("pairing_quality", "mean"), n=("pairing_quality", "size"))
        .sort_values(["avg_pairing_quality", "n"], ascending=[False, False])
    )

    topk = st.slider("Ile rekomendacji pokazać?", 5, 50, 20)

    st.dataframe(recs.head(topk), use_container_width=True)

    # Visualizations of recommendations
    fig_rec = px.scatter(
        recs.head(200),
        x="avg_pairing_quality",
        y="n",
        color="food_category",
        hover_data=["food_item", "cuisine"],
        title="Rekomendacje: avg_pairing_quality vs liczba wystąpień (top 200)"
    )
    fig_rec.update_layout(height=550)
    st.plotly_chart(fig_rec, use_container_width=True)

    # Show distribution of cuisines in top recs
    top_recs = recs.head(200)
    fig_cui = px.bar(
        top_recs["cuisine"].value_counts().head(15).reset_index(),
        x="index", y="cuisine",
        title="Top kuchnie w rekomendacjach (top 200)",
        labels={"index": "cuisine", "cuisine": "count"}
    )
    fig_cui.update_layout(height=450)
    st.plotly_chart(fig_cui, use_container_width=True)

# -----------------------------
# Explorer
# -----------------------------
elif page == "Eksplorator danych":
    st.subheader("🔎 Eksplorator danych — własne przekroje i wykresy")
    dataset = st.selectbox("Wybierz dataset", ["winequality-red", "wine_food_pairings (po filtrach)"])
    df = wine_df if dataset == "winequality-red" else pairings_f

    st.write("Podgląd:")
    st.dataframe(df.head(200), use_container_width=True)

    st.divider()
    st.markdown("### Wykres (Plotly Express)")
    cols = df.columns.tolist()

    chart_type = st.selectbox("Typ wykresu", ["Histogram", "Scatter", "Box", "Bar (count)"])

    if chart_type == "Histogram":
        x = st.selectbox("X", cols)
        fig = px.histogram(df, x=x, title=f"Histogram: {x}")
        fig.update_layout(height=550)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter":
        x = st.selectbox("X", cols, index=0)
        y = st.selectbox("Y", cols, index=min(1, len(cols)-1))
        color = st.selectbox("Kolor (opcjonalnie)", ["(none)"] + cols)
        fig = px.scatter(
            df, x=x, y=y,
            color=None if color == "(none)" else color,
            title=f"Scatter: {x} vs {y}"
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box":
        y = st.selectbox("Y", cols)
        x = st.selectbox("Grupuj po (opcjonalnie)", ["(none)"] + cols)
        fig = px.box(df, y=y, x=None if x == "(none)" else x, title=f"Boxplot: {y}")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

    else:  # Bar count
        x = st.selectbox("Kategoria", cols)
        vc = df[x].astype(str).value_counts().head(30).reset_index()
        vc.columns = [x, "count"]
        fig = px.bar(vc, x=x, y="count", title=f"Top wartości: {x} (count)")
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

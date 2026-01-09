import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# Przyjazne nazwy kolumn (UI)
# ---------------------------------------------------------
FRIENDLY_WINE_COLS = {
    "fixed acidity": "Kwasowość stała",
    "volatile acidity": "Kwasowość lotna",
    "citric acid": "Kwas cytrynowy",
    "residual sugar": "Cukier resztkowy",
    "chlorides": "Chlorki",
    "free sulfur dioxide": "Wolny SO₂",
    "total sulfur dioxide": "Całkowity SO₂",
    "density": "Gęstość",
    "pH": "pH",
    "sulphates": "Siarczany",
    "alcohol": "Alkohol (%)",
    "quality": "Jakość (ocena)",
}

FRIENDLY_PAIR_COLS = {
    "wine_type": "Typ wina",
    "wine_category": "Kategoria wina",
    "food_item": "Danie / produkt",
    "food_category": "Kategoria jedzenia",
    "cuisine": "Kuchnia",
    "pairing_quality": "Jakość parowania (1–5)",
    "quality_label": "Opis jakości",
    "description": "Opis (skąd ocena)",
}

def label_wine(col: str) -> str:
    return FRIENDLY_WINE_COLS.get(col, col)

def label_pair(col: str) -> str:
    return FRIENDLY_PAIR_COLS.get(col, col)

def options_with_labels(cols, label_fn):
    """Zwraca listę etykiet do selectboxa oraz mapę etykieta->kolumna."""
    labels = [label_fn(c) for c in cols]
    seen = {}
    fixed_labels = []
    for c, l in zip(cols, labels):
        if l in seen:
            seen[l] += 1
            fixed_labels.append(f"{l} ({seen[l]})")
        else:
            seen[l] = 1
            fixed_labels.append(l)
    mapping = {l: c for l, c in zip(fixed_labels, cols)}
    return fixed_labels, mapping

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(page_title="Wine Analytics & Food Pairings", layout="wide")

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown(
    "Analiza `winequality-red.csv` + `wine_food_pairings.csv` oraz moduł doradczy "
    "„wino → predykcja jakości → rekomendacje parowań”."
)

# ---------------------------------------------------------
# Wczytywanie danych (cache)
# ---------------------------------------------------------
@st.cache_data
def load_wine_quality(path="winequality-red.csv"):
    return pd.read_csv(path)

@st.cache_data
def load_wine_food_pairings(path="wine_food_pairings.csv"):
    return pd.read_csv(path)

def dataset_profile(df: pd.DataFrame) -> dict:
    missing_by_col = df.isna().sum()
    missing_total = int(missing_by_col.sum())
    dup_count = int(df.duplicated().sum())
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "dtypes": df.dtypes,
        "missing_total": missing_total,
        "missing_by_col": missing_by_col[missing_by_col > 0].sort_values(ascending=False),
        "duplicates": dup_count,
    }

def quick_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols or len(df) == 0:
        return pd.DataFrame()
    out = df[cols].describe().T[["mean", "50%", "min", "max"]].rename(columns={"50%": "median"})
    return out

def quality_tier(q_pred: float) -> str:
    if q_pred <= 5.0:
        return "Low"
    elif q_pred <= 6.0:
        return "Mid"
    else:
        return "High"

def tier_min_pairing_quality(tier: str) -> int:
    if tier == "Low":
        return 3
    if tier == "Mid":
        return 4
    return 4

# ---------------------------------------------------------
# Próba wczytania danych (z obsługą błędów)
# ---------------------------------------------------------
wine_quality_df = None
wine_food_pairings_df = None
wine_quality_error = None
pairings_error = None

try:
    wine_quality_df = load_wine_quality()
except Exception as e:
    wine_quality_error = str(e)

try:
    wine_food_pairings_df = load_wine_food_pairings()
except Exception as e:
    pairings_error = str(e)

# ---------------------------------------------------------
# Sidebar – wybór modułu
# ---------------------------------------------------------
st.sidebar.header("⚙️ Moduły")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=[
        "Analiza jakości wina",
        "Parowanie wina z jedzeniem",
        "Doradca parowania",
    ],
    key="module_radio"
)

# =========================================================
# 1) ANALIZA JAKOŚCI WINA
# =========================================================
if module == "Analiza jakości wina":
    st.subheader("📊 Analiza jakości czerwonych win (winequality-red.csv)")

    with st.expander("ℹ️ Jak czytać ten moduł? (dla początkujących)", expanded=True):
        st.write(
            "Tu analizujesz parametry chemiczne czerwonych win i ich ocenę jakości.\n\n"
            "• **Filtrowanie** zawęża dane do interesującego Cię zakresu.\n"
            "• **Rozkłady** (histogram/boxplot) pokazują częstość i wartości odstające.\n"
            "• **Porównanie grup** pokazuje różnice między grupami jakości.\n"
            "• **Wykres 3D** pokazuje zależności między trzema parametrami.\n"
            "• **Model ML** to edukacyjna predykcja jakości na podstawie parametrów."
        )

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co aplikacja."
        )
        st.stop()

    df = wine_quality_df.copy()
    prof = dataset_profile(df)

    st.markdown("## Podstawowa eksploracja danych (EDA)")

    st.markdown("### Podgląd danych")
    st.dataframe(df.rename(columns=FRIENDLY_WINE_COLS).head(20), use_container_width=True)

    st.markdown("### Profil")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Wiersze", f"{prof['rows']}")
    m2.metric("Kolumny", f"{prof['cols']}")
    m3.metric("Braki (razem)", f"{prof['missing_total']}")
    m4.metric("Duplikaty", f"{prof['duplicates']}")

    st.markdown("#### Typy danych")
    dtypes_df = pd.DataFrame({
        "Kolumna": [label_wine(c) for c in df.columns],
        "Typ": [str(df[c].dtype) for c in df.columns],
    })
    st.dataframe(dtypes_df, use_container_width=True)

    with st.expander("Braki w danych (ile i gdzie)"):
        if len(prof["missing_by_col"]) == 0:
            st.info("Brak brakujących wartości.")
        else:
            miss = prof["missing_by_col"].copy()
            miss.index = miss.index.map(label_wine)
            st.dataframe(miss.rename("Liczba braków"), use_container_width=True)

    st.divider()

    st.markdown("## Filtrowanie i szybkie wnioski")
    st.caption("Wybierz zakres jakości oraz dodatkowy parametr, aby zawęzić wyniki.")

    min_q, max_q = int(df["quality"].min()), int(df["quality"].max())
    quality_range = st.slider(
        "Zakres jakości (quality)",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q),
        key="wq_quality_range"
    )

    feature_cols = [c for c in df.columns if c != "quality"]
    feat_labels, feat_map = options_with_labels(feature_cols, label_wine)

    chosen_feature_label = st.selectbox(
        "Wybierz parametr do filtrowania (zakres)",
        options=feat_labels,
        index=0,
        key="wq_feat_filter_select"
    )
    chosen_feature = feat_map[chosen_feature_label]

    f_min = float(df[chosen_feature].min())
    f_max = float(df[chosen_feature].max())
    feature_range = st.slider(
        f"Zakres dla: {chosen_feature_label}",
        min_value=f_min,
        max_value=f_max,
        value=(f_min, f_max),
        key="wq_feat_filter_range"
    )

    filtered = df[
        (df["quality"].between(quality_range[0], quality_range[1])) &
        (df[chosen_feature].between(feature_range[0], feature_range[1]))
    ]

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.write(f"✅ Rekordów po filtrach: **{len(filtered)}** / {len(df)}")
        st.dataframe(filtered.rename(columns=FRIENDLY_WINE_COLS).head(50), use_container_width=True)

    with c2:
        st.markdown("### Proste statystyki (po filtrach)")
        stats = quick_stats(filtered, ["quality", chosen_feature, "alcohol", "volatile acidity"])
        if stats.empty:
            st.info("Brak statystyk do pokazania.")
        else:
            stats.index = stats.index.map(label_wine)
            st.dataframe(stats, use_container_width=True)

    st.divider()

    st.markdown("## Rozkłady i porównania")
    default_feat_label = label_wine("alcohol") if "alcohol" in feature_cols else feat_labels[0]
    feat_label = st.selectbox(
        "Wybierz parametr do rozkładów (histogram + boxplot)",
        options=feat_labels,
        index=feat_labels.index(default_feat_label) if default_feat_label in feat_labels else 0,
        key="wq_feat_dist_select"
    )
    feat = feat_map[feat_label]

    colA, colB = st.columns(2)
    with colA:
        fig_hist = px.histogram(df, x=feat, nbins=40, title=f"Histogram: {feat_label}")
        fig_hist.update_layout(height=420)
        st.plotly_chart(fig_hist, use_container_width=True)
    with colB:
        fig_box = px.box(df, y=feat, points="outliers", title=f"Boxplot: {feat_label}")
        fig_box.update_layout(height=420)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Porównanie rozkładów dla dwóch grup jakości")
    compare_mode = st.radio(
        "Tryb porównania",
        options=["quality ≤ X vs quality > X", "quality = A vs quality = B"],
        horizontal=True,
        key="wq_compare_mode"
    )

    if compare_mode == "quality ≤ X vs quality > X":
        x_thr = st.slider("Wybierz próg X", min_value=min_q, max_value=max_q, value=5, key="wq_thr")
        g1 = df[df["quality"] <= x_thr].copy()
        g2 = df[df["quality"] > x_thr].copy()
        g1["group"] = f"Jakość ≤ {x_thr}"
        g2["group"] = f"Jakość > {x_thr}"
        comp = pd.concat([g1, g2], ignore_index=True)
    else:
        q_vals = sorted(df["quality"].unique())
        qa = st.selectbox("A", options=q_vals, index=0, key="wq_qa")
        qb = st.selectbox("B", options=q_vals, index=min(1, len(q_vals) - 1), key="wq_qb")
        g1 = df[df["quality"] == qa].copy()
        g2 = df[df["quality"] == qb].copy()
        g1["group"] = f"Jakość = {qa}"
        g2["group"] = f"Jakość = {qb}"
        comp = pd.concat([g1, g2], ignore_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_cmp_hist = px.histogram(comp, x=feat, color="group", barmode="overlay", nbins=40, title="Porównanie histogramów")
        fig_cmp_hist.update_layout(height=420)
        st.plotly_chart(fig_cmp_hist, use_container_width=True)
    with c2:
        fig_cmp_box = px.box(comp, y=feat, color="group", points="outliers", title="Porównanie boxplotów")
        fig_cmp_box.update_layout(height=420)
        st.plotly_chart(fig_cmp_box, use_container_width=True)

    st.divider()

    st.markdown("## Wykresy 3D")
    def idx_or_0(label):
        return feat_labels.index(label) if label in feat_labels else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        x3_label = st.selectbox("Oś X", options=feat_labels, index=idx_or_0(label_wine("alcohol")), key="wq_3d_x")
    with c2:
        y3_label = st.selectbox("Oś Y", options=feat_labels, index=idx_or_0(label_wine("volatile acidity")), key="wq_3d_y")
    with c3:
        z3_label = st.selectbox("Oś Z", options=feat_labels, index=idx_or_0(label_wine("sulphates")), key="wq_3d_z")

    x3 = feat_map[x3_label]
    y3 = feat_map[y3_label]
    z3 = feat_map[z3_label]

    df3 = filtered if len(filtered) > 0 else df
    fig3d = px.scatter_3d(df3, x=x3, y=y3, z=z3, color="quality",
                          title=f"3D: {x3_label} vs {y3_label} vs {z3_label}", opacity=0.7)
    fig3d.update_layout(height=700)
    st.plotly_chart(fig3d, use_container_width=True)

    st.divider()

    st.markdown("## Model ML: przewidywanie jakości (RandomForest)")
    with st.expander("⚙️ Ustawienia i trening modelu"):
        test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05, key="ml_test_size")
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50, key="ml_n_estimators")
        random_state = st.number_input("random_state", value=42, step=1, key="ml_random_state")

        X = df.drop(columns=["quality"])
        y = df["quality"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))
        model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=-1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        st.success("Model wytrenowany ✅")
        c1, c2 = st.columns(2)
        c1.metric("R²", f"{r2_score(y_test, preds):.3f}")
        c2.metric("MAE", f"{mean_absolute_error(y_test, preds):.3f}")

    st.markdown("### 🔮 Predykcja jakości na podstawie suwaków")
    with st.form("prediction_form"):
        inputs = {}
        for col in feature_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            step = (col_max - col_min) / 100 if col_max > col_min else 0.01
            inputs[col] = st.slider(label_wine(col), col_min, col_max, col_mean, step=step, key=f"pred_{col}")
        submitted = st.form_submit_button("Oblicz predykcję")

    if submitted:
        if "model" not in locals():
            st.warning("Najpierw wytrenuj model w sekcji powyżej (expander).")
        else:
            pred_quality = float(model.predict(pd.DataFrame([inputs]))[0])
            st.success(f"Przewidywana jakość (quality): **{pred_quality:.2f}**")

# =========================================================
# 2) PAROWANIE WINA Z JEDZENIEM
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem (wine_food_pairings.csv)")

    if wine_food_pairings_df is None:
        st.error(f"Nie udało się wczytać `wine_food_pairings.csv`: `{pairings_error}`")
        st.stop()

    dfp = wine_food_pairings_df.copy()
    prof = dataset_profile(dfp)

    st.markdown("## Podstawowa eksploracja danych (EDA)")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Podgląd danych")
        st.dataframe(dfp.rename(columns=FRIENDLY_PAIR_COLS).head(20), use_container_width=True)

    with c2:
        st.markdown("### Profil")
        st.write(f"**Wiersze:** {prof['rows']}")
        st.write(f"**Kolumny:** {prof['cols']}")
        st.write(f"**Braki (razem):** {prof['missing_total']}")
        st.write(f"**Duplikaty:** {prof['duplicates']}")

    st.divider()

    st.markdown("## Filtrowanie i szybkie wnioski")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wine_type_sel = st.multiselect(label_pair("wine_type"), options=sorted(dfp["wine_type"].dropna().unique()), default=[], key="pair_wine_type")
    with col2:
        food_category_sel = st.multiselect(label_pair("food_category"), options=sorted(dfp["food_category"].dropna().unique()), default=[], key="pair_food_cat")
    with col3:
        cuisine_sel = st.multiselect(label_pair("cuisine"), options=sorted(dfp["cuisine"].dropna().unique()), default=[], key="pair_cuisine")
    with col4:
        min_pairing_quality_sel = st.slider("Minimalna jakość parowania", 1, 5, 3, key="pair_min_quality")

    filtered = dfp.copy()
    if wine_type_sel:
        filtered = filtered[filtered["wine_type"].isin(wine_type_sel)]
    if food_category_sel:
        filtered = filtered[filtered["food_category"].isin(food_category_sel)]
    if cuisine_sel:
        filtered = filtered[filtered["cuisine"].isin(cuisine_sel)]
    filtered = filtered[filtered["pairing_quality"] >= min_pairing_quality_sel]

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.write(f"✅ Rekordów po filtrach: **{len(filtered)}** / {len(dfp)}")
        show_cols = ["food_item", "cuisine", "wine_type", "wine_category", "pairing_quality", "quality_label", "description"]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols].sort_values("pairing_quality", ascending=False).head(200).rename(columns=FRIENDLY_PAIR_COLS),
                     use_container_width=True)

    with c2:
        st.markdown("### Proste statystyki (po filtrach)")
        stats = quick_stats(filtered, ["pairing_quality"])
        if not stats.empty:
            st.dataframe(stats.rename(index=label_pair), use_container_width=True)

    st.divider()

    st.markdown("## Wizualizacje")
    if len(filtered) == 0:
        st.warning("Brak danych po filtrach — nie da się narysować wykresów.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            vc = filtered["quality_label"].astype(str).value_counts(dropna=False)
            vc_df = vc.reset_index()
            vc_df.columns = ["quality_label", "count"]
            st.plotly_chart(px.bar(vc_df, x="quality_label", y="count", title="Rozkład quality_label (po filtrach)"),
                            use_container_width=True)
        with c2:
            wt_mean = filtered.groupby("wine_type", dropna=False)["pairing_quality"].mean().sort_values(ascending=False).head(20).reset_index()
            wt_mean.columns = ["wine_type", "avg_pairing_quality"]
            st.plotly_chart(px.bar(wt_mean, x="wine_type", y="avg_pairing_quality",
                                   title="Średnia pairing_quality per wine_type (top 20)"),
                            use_container_width=True)

# =========================================================
# 3) DORADCA PAROWANIA
# =========================================================
else:
    st.subheader("🧑‍🍳🍷 Doradca parowania")

    if wine_quality_df is None:
        st.error(f"Nie udało się wczytać `winequality-red.csv`: `{wine_quality_error}`")
        st.stop()
    if wine_food_pairings_df is None:
        st.error(f"Nie udało się wczytać `wine_food_pairings.csv`: `{pairings_error}`")
        st.stop()

    df = wine_quality_df.copy()
    dfp = wine_food_pairings_df.copy()

    with st.expander("ℹ️ Co robi ten moduł? (prosto)", expanded=True):
        st.write(
            "1) opisujesz wino → 2) model przewiduje jakość → 3) dobieramy parowania → "
            "4) możesz też wyszukać najlepsze typy win do wybranego jedzenia."
        )

    # 1) Model jakości
    st.markdown("## 1) Model jakości (RandomForest)")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Ile danych na test? (test_size)", 0.1, 0.5, 0.2, 0.05, key="adv_test_size")
    with c2:
        n_estimators = st.slider("Liczba drzew (n_estimators)", 50, 600, 300, 50, key="adv_n_estimators")
    with c3:
        random_state = st.number_input("Losowość (random_state)", value=42, step=1, key="adv_random_state")

    X = df.drop(columns=["quality"])
    y = df["quality"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=int(random_state))

    model = RandomForestRegressor(n_estimators=int(n_estimators), random_state=int(random_state), n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    c1, c2 = st.columns(2)
    c1.metric("R² (holdout)", f"{r2_score(y_test, preds):.3f}")
    c2.metric("MAE (holdout)", f"{mean_absolute_error(y_test, preds):.3f}")

    st.divider()

    # 2) Opis wina
    st.markdown("## 2) Wybierz / opisz wino")
    mode = st.radio(
        "Źródło parametrów",
        options=["Wybierz rekord z danych", "Wprowadź parametry ręcznie"],
        horizontal=True,
        key="adv_mode"
    )

    feature_cols = [c for c in df.columns if c != "quality"]

    if mode == "Wybierz rekord z danych":
        idx = st.slider("Indeks rekordu (wino z datasetu)", 0, len(df) - 1, 0, key="adv_row_idx")
        row = df.iloc[idx]
        input_df = row.drop(labels=["quality"]).to_frame().T
        st.caption(f"Prawdziwa jakość dla tego rekordu: **{int(row['quality'])}**")
        st.dataframe(input_df.rename(columns=FRIENDLY_WINE_COLS), use_container_width=True)
    else:
        defaults = df.drop(columns=["quality"]).mean(numeric_only=True)
        inputs = {}
        with st.form("adv_wine_form"):
            for col in feature_cols:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                col_def = float(defaults[col])
                step = (col_max - col_min) / 100 if col_max > col_min else 0.01
                inputs[col] = st.slider(label_wine(col), col_min, col_max, col_def, step=step, key=f"adv_{col}")
            submitted = st.form_submit_button("Użyj parametrów")
        if not submitted:
            st.info("Ustaw parametry i kliknij „Użyj parametrów”.")
            st.stop()
        input_df = pd.DataFrame([inputs])

    pred_quality = float(model.predict(input_df)[0])
    tier = quality_tier(pred_quality)
    st.success(f"Przewidywana jakość: **{pred_quality:.2f}**  → poziom: **{tier}**")

    st.divider()

    # 3) Rekomendacje parowań (jak wcześniej)
    st.markdown("## 3) Rekomendacje parowań")
    wine_types = sorted(dfp["wine_type"].dropna().unique())
    red_like = [w for w in wine_types if "red" in str(w).lower()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sel_wine_type = st.multiselect(label_pair("wine_type"), options=wine_types, default=red_like[:1] if red_like else [],
                                       key="adv_rec_wine_type")
    with c2:
        food_category_sel = st.multiselect(label_pair("food_category"), options=sorted(dfp["food_category"].dropna().unique()),
                                           default=[], key="adv_rec_food_cat")
    with c3:
        cuisine_sel = st.multiselect(label_pair("cuisine"), options=sorted(dfp["cuisine"].dropna().unique()),
                                     default=[], key="adv_rec_cuisine")
    with c4:
        min_pair = st.slider("Minimalna jakość parowania (1–5)", 1, 5, value=tier_min_pairing_quality(tier), key="adv_rec_min_pair")

    rec = dfp.copy()
    if sel_wine_type:
        rec = rec[rec["wine_type"].isin(sel_wine_type)]
    if food_category_sel:
        rec = rec[rec["food_category"].isin(food_category_sel)]
    if cuisine_sel:
        rec = rec[rec["cuisine"].isin(cuisine_sel)]
    rec = rec[rec["pairing_quality"] >= min_pair]

    st.write(f"✅ Rekordów po filtrach: **{len(rec)}** / {len(dfp)}")
    if len(rec) == 0:
        st.warning("Brak wyników. Poluzuj filtry lub obniż minimalną jakość parowania.")
    else:
        ranked = (
            rec.groupby(["food_category", "food_item", "cuisine"], as_index=False)
            .agg(avg_pairing_quality=("pairing_quality", "mean"), n=("pairing_quality", "size"))
            .sort_values(["avg_pairing_quality", "n"], ascending=[False, False])
        )
        topk = st.slider("Ile rekomendacji pokazać?", 5, 50, 20, key="adv_rec_topk")
        pretty_ranked = ranked.head(topk).rename(columns={
            "food_category": "Kategoria jedzenia",
            "food_item": "Danie / produkt",
            "cuisine": "Kuchnia",
            "avg_pairing_quality": "Średnia jakość parowania",
            "n": "Liczba wystąpień",
        })
        st.dataframe(pretty_ranked, use_container_width=True)

    st.divider()

    # 4) NOWA: reverse lookup - znajdź typy win do jedzenia
    st.markdown("## 4) Znajdź wino do jedzenia")
    st.caption("Wybierz jedzenie i zakres jakości parowania, a dostaniesz listę typów/kategorii win najlepiej pasujących.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        food_cat_pick = st.multiselect(
            "Kategoria jedzenia",
            options=sorted(dfp["food_category"].dropna().unique()),
            default=[],
            key="adv_find_food_cat"
        )
    with c2:
        cuisine_pick = st.multiselect(
            "Kuchnia",
            options=sorted(dfp["cuisine"].dropna().unique()),
            default=[],
            key="adv_find_cuisine"
        )
    with c3:
        qmin = int(dfp["pairing_quality"].min())
        qmax = int(dfp["pairing_quality"].max())
        pairing_range = st.slider(
            "Zakres jakości parowania",
            min_value=qmin,
            max_value=qmax,
            value=(max(1, qmin), qmax),
            key="adv_find_pair_range"
        )
    with c4:
        top_wines = st.slider("Ile win pokazać?", 5, 50, 15, key="adv_find_top_wines")

    base = dfp.copy()
    if food_cat_pick:
        base = base[base["food_category"].isin(food_cat_pick)]
    if cuisine_pick:
        base = base[base["cuisine"].isin(cuisine_pick)]
    base = base[base["pairing_quality"].between(pairing_range[0], pairing_range[1])]

    st.write(f"✅ Dopasowanych rekordów parowań: **{len(base)}** / {len(dfp)}")
    if len(base) == 0:
        st.warning("Brak wyników dla wybranych kryteriów. Poszerz filtry lub zakres jakości.")
    else:
        wine_rank = (
            base.groupby(["wine_type", "wine_category"], dropna=False, as_index=False)
            .agg(
                avg_pairing_quality=("pairing_quality", "mean"),
                median_pairing_quality=("pairing_quality", "median"),
                matches=("pairing_quality", "size"),
            )
            .sort_values(["avg_pairing_quality", "matches"], ascending=[False, False])
        )

        pretty_wine_rank = wine_rank.head(top_wines).rename(columns={
            "wine_type": "Typ wina",
            "wine_category": "Kategoria wina",
            "avg_pairing_quality": "Średnia jakość parowania",
            "median_pairing_quality": "Mediana jakości",
            "matches": "Liczba dopasowań",
        })
        st.dataframe(pretty_wine_rank, use_container_width=True)

        fig_wines = px.bar(
            wine_rank.head(25),
            x="wine_type",
            y="avg_pairing_quality",
            color="wine_category",
            title="Top typy win wg średniej jakości parowania (top 25)",
            labels={
                "wine_type": "Typ wina",
                "avg_pairing_quality": "Średnia jakość parowania",
                "wine_category": "Kategoria wina",
            }
        )
        fig_wines.update_layout(height=500)
        st.plotly_chart(fig_wines, use_container_width=True)

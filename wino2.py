import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide",
)

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown(
    "Analiza `winequality-red.csv` + `wine_food_pairings.csv` oraz moduł integracyjny "
    "„Wybierz/opisz wino → predykcja quality → rekomendacje parowań”."
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
        "duplicates": dup_count
    }

def quick_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    out = df[cols].describe().T[["mean", "50%", "min", "max"]].rename(columns={"50%": "median"})
    return out

def quality_tier(q_pred: float) -> str:
    # proste progi - łatwe do zmiany
    if q_pred <= 5.0:
        return "Low"
    elif q_pred <= 6.0:
        return "Mid"
    else:
        return "High"

def tier_min_pairing_quality(tier: str) -> int:
    # im wyższa przewidywana jakość, tym bardziej rygorystycznie dobieramy pairingi
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
st.sidebar.header("⚙️ Ustawienia")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=[
        "Analiza jakości wina",
        "Parowanie wina z jedzeniem",
        "Integracja: opisz wino → predykcja → rekomendacje"
    ]
)

# =========================================================
# 1) ANALIZA JAKOŚCI WINA
# =========================================================
if module == "Analiza jakości wina":
    st.subheader("📊 Analiza jakości czerwonych win (winequality-red.csv)")

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co aplikacja."
        )
        st.stop()

    df = wine_quality_df.copy()

    # --- Podstawowa eksploracja danych (wymagania) ---
    st.markdown("## Podstawowa eksploracja danych (EDA)")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Podgląd danych")
        st.dataframe(df.head(20), use_container_width=True)

    prof = dataset_profile(df)
    with c2:
        st.markdown("### Profil")
        st.write(f"**Wiersze:** {prof['rows']}")
        st.write(f"**Kolumny:** {prof['cols']}")
        st.write(f"**Braki (razem):** {prof['missing_total']}")
        st.write(f"**Duplikaty:** {prof['duplicates']}")
        st.markdown("**Typy danych:**")
        st.dataframe(prof["dtypes"].astype(str), use_container_width=True)

        st.markdown("**Braki per kolumna (tylko > 0):**")
        if len(prof["missing_by_col"]) == 0:
            st.info("Brak brakujących wartości.")
        else:
            st.dataframe(prof["missing_by_col"], use_container_width=True)

    st.divider()

    # --- Filtrowanie + szybkie wnioski ---
    st.markdown("## Filtrowanie i szybkie wnioski")

    min_q, max_q = int(df["quality"].min()), int(df["quality"].max())
    quality_range = st.slider(
        "Zakres quality",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q)
    )

    feature_cols = [c for c in df.columns if c != "quality"]
    chosen_feature = st.selectbox("Wybierz cechę do filtrowania (zakres)", options=feature_cols, index=0)

    f_min = float(df[chosen_feature].min())
    f_max = float(df[chosen_feature].max())
    feature_range = st.slider(
        f"Zakres dla: {chosen_feature}",
        min_value=f_min,
        max_value=f_max,
        value=(f_min, f_max)
    )

    filtered = df[
        (df["quality"].between(quality_range[0], quality_range[1])) &
        (df[chosen_feature].between(feature_range[0], feature_range[1]))
    ]

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.write(f"✅ Rekordów po filtrach: **{len(filtered)}** / {len(df)}")
        st.dataframe(filtered.head(50), use_container_width=True)
    with c2:
        st.markdown("### Proste statystyki (po filtrach)")
        stats = quick_stats(filtered, ["quality", chosen_feature, "alcohol", "volatile acidity"])
        if stats.empty:
            st.info("Brak statystyk do pokazania.")
        else:
            st.dataframe(stats, use_container_width=True)

    st.divider()

    # --- Rozkłady i porównania ---
    st.markdown("## Rozkłady i porównania (winequality-red)")

    feat = st.selectbox(
        "Wybierz cechę do rozkładów (histogram + boxplot)",
        options=feature_cols,
        index=feature_cols.index("alcohol") if "alcohol" in feature_cols else 0
    )

    colA, colB = st.columns(2)
    with colA:
        fig_hist = px.histogram(df, x=feat, nbins=40, title=f"Histogram cechy: {feat}")
        fig_hist.update_layout(height=420)
        st.plotly_chart(fig_hist, use_container_width=True)
    with colB:
        fig_box = px.box(df, y=feat, points="outliers", title=f"Boxplot cechy: {feat}")
        fig_box.update_layout(height=420)
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### Porównanie rozkładów dla dwóch grup jakości")

    compare_mode = st.radio(
        "Tryb porównania",
        options=["quality ≤ X vs quality > X", "quality = A vs quality = B"],
        horizontal=True
    )

    if compare_mode == "quality ≤ X vs quality > X":
        x_thr = st.slider("Wybierz próg X", min_value=min_q, max_value=max_q, value=5)
        g1 = df[df["quality"] <= x_thr].copy()
        g2 = df[df["quality"] > x_thr].copy()
        g1["group"] = f"quality ≤ {x_thr}"
        g2["group"] = f"quality > {x_thr}"
        comp = pd.concat([g1, g2], ignore_index=True)
    else:
        q_vals = sorted(df["quality"].unique())
        qa = st.selectbox("A", options=q_vals, index=0)
        qb = st.selectbox("B", options=q_vals, index=min(1, len(q_vals) - 1))
        g1 = df[df["quality"] == qa].copy()
        g2 = df[df["quality"] == qb].copy()
        g1["group"] = f"quality = {qa}"
        g2["group"] = f"quality = {qb}"
        comp = pd.concat([g1, g2], ignore_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_cmp_hist = px.histogram(comp, x=feat, color="group", barmode="overlay", nbins=40,
                                   title="Porównanie histogramów (overlay)")
        fig_cmp_hist.update_layout(height=420)
        st.plotly_chart(fig_cmp_hist, use_container_width=True)
    with c2:
        fig_cmp_box = px.box(comp, y=feat, color="group", points="outliers", title="Porównanie boxplotów")
        fig_cmp_box.update_layout(height=420)
        st.plotly_chart(fig_cmp_box, use_container_width=True)

    st.divider()

    # --- Wykresy 3D ---
    st.markdown("## Wykresy 3D (Plotly)")

    c1, c2, c3 = st.columns(3)
    with c1:
        x3 = st.selectbox("Oś X", options=feature_cols, index=feature_cols.index("alcohol") if "alcohol" in feature_cols else 0)
    with c2:
        y3 = st.selectbox("Oś Y", options=feature_cols, index=feature_cols.index("volatile acidity") if "volatile acidity" in feature_cols else 1)
    with c3:
        z3 = st.selectbox("Oś Z", options=feature_cols, index=feature_cols.index("sulphates") if "sulphates" in feature_cols else 2)

    df3 = filtered if len(filtered) > 0 else df
    fig3d = px.scatter_3d(
        df3,
        x=x3, y=y3, z=z3,
        color="quality",
        title=f"3D scatter: {x3} vs {y3} vs {z3} (kolor: quality)",
        opacity=0.7
    )
    fig3d.update_layout(height=700)
    st.plotly_chart(fig3d, use_container_width=True)

    st.divider()

    # --- Model ML (jak w oryginale, ale trochę czytelniej) ---
    st.markdown("## Model ML: RandomForestRegressor")

    with st.expander("⚙️ Ustawienia i trening modelu"):
        test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05)
        n_estimators = st.slider("n_estimators", 50, 500, 200, 50)
        random_state = st.number_input("random_state", value=42, step=1)

        X = df.drop(columns=["quality"])
        y = df["quality"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)

        st.success("Model wytrenowany ✅")
        c1, c2 = st.columns(2)
        c1.metric("R²", f"{r2:.3f}")
        c2.metric("MAE", f"{mae:.3f}")

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.markdown("### Feature importance")
        st.bar_chart(importances)

    # Predykcja interaktywna (wymaga, żeby model został utworzony)
    st.markdown("### Predykcja quality na podstawie suwaków")

    with st.form("prediction_form"):
        inputs = {}
        for col in feature_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            step = (col_max - col_min) / 100 if col_max > col_min else 0.01
            inputs[col] = st.slider(col, col_min, col_max, col_mean, step=step)

        submitted = st.form_submit_button("🔮 Predykcja")

    if submitted:
        if "model" not in locals():
            st.warning("Najpierw wytrenuj model w sekcji powyżej (expander).")
        else:
            input_df = pd.DataFrame([inputs])
            pred_quality = model.predict(input_df)[0]
            st.success(f"Przewidywana jakość (quality): **{pred_quality:.2f}**")

# =========================================================
# 2) PAROWANIE WINA Z JEDZENIEM
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem (wine_food_pairings.csv)")

    if wine_food_pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co aplikacja."
        )
        st.stop()

    dfp = wine_food_pairings_df.copy()

    # --- Podstawowa eksploracja danych (wymagania) ---
    st.markdown("## Podstawowa eksploracja danych (EDA)")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Podgląd danych")
        st.dataframe(dfp.head(20), use_container_width=True)

    prof = dataset_profile(dfp)
    with c2:
        st.markdown("### Profil")
        st.write(f"**Wiersze:** {prof['rows']}")
        st.write(f"**Kolumny:** {prof['cols']}")
        st.write(f"**Braki (razem):** {prof['missing_total']}")
        st.write(f"**Duplikaty:** {prof['duplicates']}")
        st.markdown("**Typy danych:**")
        st.dataframe(prof["dtypes"].astype(str), use_container_width=True)

        st.markdown("**Braki per kolumna (tylko > 0):**")
        if len(prof["missing_by_col"]) == 0:
            st.info("Brak brakujących wartości.")
        else:
            st.dataframe(prof["missing_by_col"], use_container_width=True)

    st.divider()

    # --- Filtrowanie + szybkie wnioski ---
    st.markdown("## Filtrowanie i szybkie wnioski")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        wine_type_sel = st.multiselect(
            "wine_type",
            options=sorted(dfp["wine_type"].dropna().unique()),
            default=[]
        )
    with col2:
        food_category_sel = st.multiselect(
            "food_category",
            options=sorted(dfp["food_category"].dropna().unique()),
            default=[]
        )
    with col3:
        cuisine_sel = st.multiselect(
            "cuisine",
            options=sorted(dfp["cuisine"].dropna().unique()),
            default=[]
        )
    with col4:
        min_pairing_quality_sel = st.slider(
            "Minimalna pairing_quality",
            min_value=int(dfp["pairing_quality"].min()),
            max_value=int(dfp["pairing_quality"].max()),
            value=3
        )

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
        show_cols = [
            "food_item", "cuisine", "wine_type", "wine_category",
            "pairing_quality", "quality_label", "description"
        ]
        show_cols = [c for c in show_cols if c in filtered.columns]
        st.dataframe(filtered[show_cols].sort_values("pairing_quality", ascending=False).head(200), use_container_width=True)

    with c2:
        st.markdown("### Proste statystyki (po filtrach)")
        # 2–3 proste statystyki: pairing_quality oraz liczność per label
        stats = quick_stats(filtered, ["pairing_quality"])
        if not stats.empty:
            st.dataframe(stats, use_container_width=True)
        st.markdown("**Top quality_label (count):**")
        if "quality_label" in filtered.columns and len(filtered) > 0:
            st.dataframe(filtered["quality_label"].value_counts().head(10), use_container_width=True)
        else:
            st.info("Brak danych po filtrach.")

    st.divider()

    # --- Wizualizacje ---
    st.markdown("## Wizualizacje")

    c1, c2 = st.columns(2)
    with c1:
        fig_lbl = px.bar(
            filtered["quality_label"].value_counts().reset_index(),
            x="index", y="quality_label",
            labels={"index": "quality_label", "quality_label": "count"},
            title="Rozkład quality_label (po filtrach)"
        )
        fig_lbl.update_layout(height=450)
        st.plotly_chart(fig_lbl, use_container_width=True)

    with c2:
        fig_wt = px.bar(
            filtered.groupby("wine_type")["pairing_quality"].mean().sort_values(ascending=False).head(20).reset_index(),
            x="wine_type", y="pairing_quality",
            title="Średnia pairing_quality per wine_type (top 20)"
        )
        fig_wt.update_layout(height=450)
        st.plotly_chart(fig_wt, use_container_width=True)

    st.divider()

    # --- Rekomendacja dla dania (jak w oryginale) ---
    st.markdown("## 🔎 Rekomendacje na podstawie nazwy dania")
    chosen_food = st.text_input("Wpisz nazwę dania (fragment):", "")

    if chosen_food.strip():
        tmp = dfp[dfp["food_item"].astype(str).str.contains(chosen_food, case=False, na=False)].copy()

        if len(tmp) == 0:
            st.warning("Nie znaleziono pasujących dań.")
        else:
            cuisine_opt = ["(dowolna)"] + sorted(tmp["cuisine"].dropna().unique())
            chosen_cuisine = st.selectbox("Doprecyzuj kuchnię (opcjonalnie)", options=cuisine_opt)

            if chosen_cuisine != "(dowolna)":
                tmp = tmp[tmp["cuisine"] == chosen_cuisine]

            tmp = tmp.sort_values("pairing_quality", ascending=False)

            st.dataframe(
                tmp[
                    [
                        "food_item",
                        "cuisine",
                        "wine_type",
                        "wine_category",
                        "pairing_quality",
                        "quality_label",
                        "description"
                    ]
                ].head(20),
                use_container_width=True
            )
    else:
        st.info("Wpisz fragment nazwy dania, aby zobaczyć rekomendacje.")

# =========================================================
# 3) INTEGRACJA: opisz wino → predykcja → rekomendacje
# =========================================================
else:
    st.subheader("🔗 Integracja: opisz wino → przewidź quality → rekomenduj parowania")

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`"
        )
        st.stop()

    if wine_food_pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`"
        )
        st.stop()

    df = wine_quality_df.copy()
    dfp = wine_food_pairings_df.copy()

    st.markdown(
        "Ten moduł łączy oba datasety: "
        "parametry wina → model ML (RandomForest) → przewidywana jakość → filtrowanie i ranking pairingów."
    )

    st.divider()

    # --- Trening modelu (ustawienia) ---
    st.markdown("## 1) Model jakości (RandomForest)")
    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("test_size", 0.1, 0.5, 0.2, 0.05)
    with c2:
        n_estimators = st.slider("n_estimators", 50, 600, 300, 50)
    with c3:
        random_state = st.number_input("random_state", value=42, step=1)

    X = df.drop(columns=["quality"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=int(random_state)
    )

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    c1, c2 = st.columns(2)
    c1.metric("R² (holdout)", f"{r2:.3f}")
    c2.metric("MAE (holdout)", f"{mae:.3f}")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    with st.expander("Feature importance"):
        st.dataframe(importances.reset_index().rename(columns={"index": "feature", 0: "importance"}), use_container_width=True)

    st.divider()

    # --- Wybór: rekord z datasetu vs ręcznie ---
    st.markdown("## 2) Wybierz / opisz wino")

    mode = st.radio(
        "Źródło parametrów",
        options=["Wybierz rekord z winequality-red", "Wprowadź parametry ręcznie"],
        horizontal=True
    )

    feature_cols = [c for c in df.columns if c != "quality"]

    if mode == "Wybierz rekord z winequality-red":
        idx = st.slider("Indeks rekordu", 0, len(df) - 1, 0)
        row = df.iloc[idx]
        input_df = row.drop(labels=["quality"]).to_frame().T
        st.caption(f"Prawdziwe quality dla tego rekordu: {int(row['quality'])}")
        st.dataframe(input_df, use_container_width=True)
    else:
        defaults = df.drop(columns=["quality"]).mean(numeric_only=True)
        inputs = {}
        with st.form("integrated_wine_form"):
            for col in feature_cols:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                col_def = float(defaults[col])
                step = (col_max - col_min) / 100 if col_max > col_min else 0.01
                inputs[col] = st.slider(col, col_min, col_max, col_def, step=step)
            submitted = st.form_submit_button("Użyj parametrów")
        if not submitted:
            st.info("Ustaw parametry i kliknij „Użyj parametrów”.")
            st.stop()
        input_df = pd.DataFrame([inputs])

    # --- Predykcja jakości ---
    pred_quality = float(model.predict(input_df)[0])
    tier = quality_tier(pred_quality)

    st.success(f"Przewidywana jakość (quality): **{pred_quality:.2f}**  → tier: **{tier}**")

    st.divider()

    # --- Rekomendacje pairingów ---
    st.markdown("## 3) Rekomendacje parowań (wine_food_pairings)")

    # preferuj "red" jeśli występuje w wine_type (ale użytkownik może nadpisać)
    wine_types = sorted(dfp["wine_type"].dropna().unique())
    red_like = [w for w in wine_types if "red" in str(w).lower()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        default_wine_types = red_like[:1] if red_like else []
        sel_wine_type = st.multiselect("wine_type", options=wine_types, default=default_wine_types)

    with c2:
        food_category_sel = st.multiselect("food_category", options=sorted(dfp["food_category"].dropna().unique()), default=[])

    with c3:
        cuisine_sel = st.multiselect("cuisine", options=sorted(dfp["cuisine"].dropna().unique()), default=[])

    with c4:
        tier_min = tier_min_pairing_quality(tier)
        min_pair = st.slider("Minimalna pairing_quality", 1, 5, value=tier_min)

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
        st.warning("Brak wyników. Poluzuj filtry lub obniż minimalną pairing_quality.")
        st.stop()

    # ranking: pairing_quality desc, a przy remisach stabilizuj liczbą wystąpień
    ranked = (
        rec.groupby(["food_category", "food_item", "cuisine"], as_index=False)
        .agg(avg_pairing_quality=("pairing_quality", "mean"), n=("pairing_quality", "size"))
        .sort_values(["avg_pairing_quality", "n"], ascending=[False, False])
    )

    topk = st.slider("Ile rekomendacji pokazać?", 5, 50, 20)
    st.dataframe(ranked.head(topk), use_container_width=True)

    # szybkie statystyki rekomendacji
    with st.expander("Szybkie statystyki rekomendacji"):
        st.dataframe(quick_stats(rec, ["pairing_quality"]), use_container_width=True)
        if "quality_label" in rec.columns:
            st.dataframe(rec["quality_label"].value_counts().head(10), use_container_width=True)

    # wykresy rekomendacji
    fig_sc = px.scatter(
        ranked.head(300),
        x="avg_pairing_quality",
        y="n",
        color="food_category",
        hover_data=["food_item", "cuisine"],
        title="Rekomendacje: średnia jakość vs liczność (top 300)"
    )
    fig_sc.update_layout(height=550)
    st.plotly_chart(fig_sc, use_container_width=True)

    fig_cui = px.bar(
        ranked.head(300)["cuisine"].value_counts().head(15).reset_index(),
        x="index", y="cuisine",
        labels={"index": "cuisine", "cuisine": "count"},
        title="Top kuchnie w rekomendacjach (top 300)"
    )
    fig_cui.update_layout(height=450)
    st.plotly_chart(fig_cui, use_container_width=True)

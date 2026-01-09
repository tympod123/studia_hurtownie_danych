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
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide",
)

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown(
    "Analiza `winequality-red.csv` + `wine_food_pairings.csv` oraz moduł integracyjny "
    "„Wybierz/opisz wino → predykcja jakości → rekomendacje parowań”."
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
    if not cols or len(df) == 0:
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

    with st.expander("ℹ️ Jak czytać ten moduł? (dla początkujących)", expanded=True):
        st.write(
            "Tu analizujesz parametry chemiczne czerwonych win i ich ocenę jakości.\n\n"
            "• **Filtrowanie** zawęża dane do interesującego Cię zakresu (np. jakość i poziom alkoholu).\n"
            "• **Rozkłady** (histogram/boxplot) pokazują, jak często występują wartości i czy są odstające.\n"
            "• **Porównanie grup** pokazuje różnice np. między winami słabszymi i lepszymi.\n"
            "• **Wykres 3D** pozwala zobaczyć zależności między trzema cechami naraz.\n"
            "• **Model ML** to „próba zgadnięcia jakości” na podstawie parametrów."
        )

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co aplikacja."
        )
        st.stop()

    df = wine_quality_df.copy()

    # --- Podstawowa eksploracja danych (EDA) ---
    st.markdown("## Podstawowa eksploracja danych (EDA)")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Podgląd danych")
        st.dataframe(df.rename(columns=FRIENDLY_WINE_COLS).head(20), use_container_width=True)

    prof = dataset_profile(df)
    with c2:
        st.markdown("### Profil")
        st.write(f"**Wiersze:** {prof['rows']}")
        st.write(f"**Kolumny:** {prof['cols']}")
        st.write(f"**Braki (razem):** {prof['missing_total']}")
        st.write(f"**Duplikaty:** {prof['duplicates']}")
        st.markdown("**Typy danych:**")
        st.dataframe(prof["dtypes"].astype(str).rename(index=label_wine), use_container_width=True)

        st.markdown("**Braki per kolumna (tylko > 0):**")
        if len(prof["missing_by_col"]) == 0:
            st.info("Brak brakujących wartości.")
        else:
            miss = prof["missing_by_col"].copy()
            miss.index = miss.index.map(label_wine)
            st.dataframe(miss, use_container_width=True)

    st.divider()

    # --- Filtrowanie + szybkie wnioski ---
    st.markdown("## Filtrowanie i szybkie wnioski")
    st.caption("Wybierz zakres jakości i dodatkowy parametr, żeby zawęzić wyniki do interesującego podzbioru.")

    min_q, max_q = int(df["quality"].min()), int(df["quality"].max())
    quality_range = st.slider(
        "Zakres jakości (quality)",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q)
    )

    feature_cols = [c for c in df.columns if c != "quality"]
    feat_labels, feat_map = options_with_labels(feature_cols, label_wine)

    chosen_feature_label = st.selectbox(
        "Wybierz parametr do filtrowania (zakres)",
        options=feat_labels,
        index=0
    )
    chosen_feature = feat_map[chosen_feature_label]

    f_min = float(df[chosen_feature].min())
    f_max = float(df[chosen_feature].max())
    feature_range = st.slider(
        f"Zakres dla: {chosen_feature_label}",
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

    # --- Rozkłady i porównania ---
    st.markdown("## Rozkłady i porównania (winequality-red)")
    st.caption("Histogram pokazuje częstość wartości, a boxplot pomaga zobaczyć medianę i wartości odstające.")

    default_feat_label = label_wine("alcohol") if "alcohol" in feature_cols else feat_labels[0]
    feat_label = st.selectbox(
        "Wybierz parametr do rozkładów (histogram + boxplot)",
        options=feat_labels,
        index=feat_labels.index(default_feat_label) if default_feat_label in feat_labels else 0
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
    st.caption("Porównaj np. „gorsze” vs „lepsze” wina i zobacz, jak zmienia się rozkład parametru.")

    compare_mode = st.radio(
        "Tryb porównania",
        options=["quality ≤ X vs quality > X", "quality = A vs quality = B"],
        horizontal=True
    )

    if compare_mode == "quality ≤ X vs quality > X":
        x_thr = st.slider("Wybierz próg X", min_value=min_q, max_value=max_q, value=5)
        g1 = df[df["quality"] <= x_thr].copy()
        g2 = df[df["quality"] > x_thr].copy()
        g1["group"] = f"Jakość ≤ {x_thr}"
        g2["group"] = f"Jakość > {x_thr}"
        comp = pd.concat([g1, g2], ignore_index=True)
    else:
        q_vals = sorted(df["quality"].unique())
        qa = st.selectbox("A", options=q_vals, index=0)
        qb = st.selectbox("B", options=q_vals, index=min(1, len(q_vals) - 1))
        g1 = df[df["quality"] == qa].copy()
        g2 = df[df["quality"] == qb].copy()
        g1["group"] = f"Jakość = {qa}"
        g2["group"] = f"Jakość = {qb}"
        comp = pd.concat([g1, g2], ignore_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_cmp_hist = px.histogram(
            comp, x=feat, color="group", barmode="overlay", nbins=40,
            title="Porównanie histogramów (overlay)"
        )
        fig_cmp_hist.update_layout(height=420)
        st.plotly_chart(fig_cmp_hist, use_container_width=True)
    with c2:
        fig_cmp_box = px.box(comp, y=feat, color="group", points="outliers", title="Porównanie boxplotów")
        fig_cmp_box.update_layout(height=420)
        st.plotly_chart(fig_cmp_box, use_container_width=True)

    st.divider()

    # --- Wykresy 3D ---
    st.markdown("## Wykresy 3D (Plotly)")
    st.caption("Wykres 3D pokazuje zależność między trzema parametrami naraz (kolor = ocena jakości).")

    # osie jako etykiety
    def idx_or_0(label):
        return feat_labels.index(label) if label in feat_labels else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        x3_label = st.selectbox("Oś X", options=feat_labels, index=idx_or_0(label_wine("alcohol")))
    with c2:
        y3_label = st.selectbox("Oś Y", options=feat_labels, index=idx_or_0(label_wine("volatile acidity")))
    with c3:
        z3_label = st.selectbox("Oś Z", options=feat_labels, index=idx_or_0(label_wine("sulphates")))

    x3 = feat_map[x3_label]
    y3 = feat_map[y3_label]
    z3 = feat_map[z3_label]

    df3 = filtered if len(filtered) > 0 else df
    fig3d = px.scatter_3d(
        df3,
        x=x3, y=y3, z=z3,
        color="quality",
        title=f"3D: {x3_label} vs {y3_label} vs {z3_label} (kolor: {label_wine('quality')})",
        opacity=0.7
    )
    fig3d.update_layout(height=700)
    st.plotly_chart(fig3d, use_container_width=True)

    st.divider()

    # --- Model ML ---
    st.markdown("## Model ML: przewidywanie jakości (RandomForest)")
    st.caption("Model próbuje przewidzieć ocenę jakości na podstawie parametrów. To narzędzie edukacyjne/eksploracyjne.")

    with st.expander("⚙️ Ustawienia i trening modelu"):
        test_size = st.slider("Ile danych na test? (test_size)", 0.1, 0.5, 0.2, 0.05)
        n_estimators = st.slider("Liczba drzew (n_estimators)", 50, 500, 200, 50)
        random_state = st.number_input("Losowość (random_state)", value=42, step=1)

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
        st.caption("R²: im bliżej 1, tym lepiej. MAE: średni błąd (im mniej, tym lepiej).")

        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        pretty_imp = importances.copy()
        pretty_imp.index = pretty_imp.index.map(label_wine)

        st.markdown("### Co najbardziej wpływa na wynik modelu?")
        st.caption("„Ważność” to przybliżona informacja, które parametry model uznaje za najbardziej istotne.")
        st.bar_chart(pretty_imp)

    # Predykcja interaktywna
    st.markdown("### 🔮 Predykcja jakości na podstawie suwaków")
    st.caption("Ustaw parametry wina i zobacz przewidywaną ocenę jakości.")

    # przyjazne etykiety w sliderach, ale nadal zapisujemy pod oryginalnymi nazwami
    with st.form("prediction_form"):
        inputs = {}
        for col in feature_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_mean = float(df[col].mean())
            step = (col_max - col_min) / 100 if col_max > col_min else 0.01

            inputs[col] = st.slider(
                label_wine(col),
                col_min, col_max, col_mean,
                step=step
            )

        submitted = st.form_submit_button("Oblicz predykcję")

    if submitted:
        if "model" not in locals():
            st.warning("Najpierw wytrenuj model w sekcji powyżej (expander).")
        else:
            input_df = pd.DataFrame([inputs])
            pred_quality = float(model.predict(input_df)[0])
            st.success(f"Przewidywana jakość (quality): **{pred_quality:.2f}**")

# =========================================================
# 2) PAROWANIE WINA Z JEDZENIEM
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem (wine_food_pairings.csv)")

    with st.expander("ℹ️ Jak korzystać z parowań? (dla początkujących)", expanded=True):
        st.write(
            "Tu wybierasz filtry (typ wina, kuchnia, kategoria dania) i dostajesz listę rekomendacji.\n\n"
            "• **Jakość parowania (1–5)**: im wyżej, tym lepsze dopasowanie.\n"
            "• Filtry pozwalają szybko ograniczyć wyniki do tego, co Cię interesuje.\n"
            "• Na dole możesz wpisać nazwę dania i znaleźć pasujące wina."
        )

    if wine_food_pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co aplikacja."
        )
        st.stop()

    dfp = wine_food_pairings_df.copy()

    # --- Podstawowa eksploracja danych (EDA) ---
    st.markdown("## Podstawowa eksploracja danych (EDA)")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Podgląd danych")
        st.dataframe(dfp.rename(columns=FRIENDLY_PAIR_COLS).head(20), use_container_width=True)

    prof = dataset_profile(dfp)
    with c2:
        st.markdown("### Profil")
        st.write(f"**Wiersze:** {prof['rows']}")
        st.write(f"**Kolumny:** {prof['cols']}")
        st.write(f"**Braki (razem):** {prof['missing_total']}")
        st.write(f"**Duplikaty:** {prof['duplicates']}")
        st.markdown("**Typy danych:**")
        st.dataframe(prof["dtypes"].astype(str).rename(index=label_pair), use_container_width=True)

        st.markdown("**Braki per kolumna (tylko > 0):**")
        if len(prof["missing_by_col"]) == 0:
            st.info("Brak brakujących wartości.")
        else:
            miss = prof["missing_by_col"].copy()
            miss.index = miss.index.map(label_pair)
            st.dataframe(miss, use_container_width=True)

    st.divider()

    # --- Filtrowanie + szybkie wnioski ---
    st.markdown("## Filtrowanie i szybkie wnioski")
    st.caption("Ustaw filtry, żeby szybko znaleźć dopasowania. Im wyższa minimalna ocena, tym mniej wyników.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        wine_type_sel = st.multiselect(
            label_pair("wine_type"),
            options=sorted(dfp["wine_type"].dropna().unique()),
            default=[]
        )
    with col2:
        food_category_sel = st.multiselect(
            label_pair("food_category"),
            options=sorted(dfp["food_category"].dropna().unique()),
            default=[]
        )
    with col3:
        cuisine_sel = st.multiselect(
            label_pair("cuisine"),
            options=sorted(dfp["cuisine"].dropna().unique()),
            default=[]
        )
    with col4:
        min_pairing_quality_sel = st.slider(
            label_pair("pairing_quality"),
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

        pretty = filtered[show_cols].copy().rename(columns=FRIENDLY_PAIR_COLS)
        # sortujemy po oryginalnej kolumnie, potem pokazujemy już przyjazną tabelę
        st.dataframe(
            filtered[show_cols].sort_values("pairing_quality", ascending=False)
            .head(200).rename(columns=FRIENDLY_PAIR_COLS),
            use_container_width=True
        )

    with c2:
        st.markdown("### Proste statystyki (po filtrach)")
        stats = quick_stats(filtered, ["pairing_quality"])
        if not stats.empty:
            stats.index = stats.index.map(label_pair)
            st.dataframe(stats, use_container_width=True)

        st.markdown("**Najczęstsze etykiety jakości (count):**")
        if "quality_label" in filtered.columns and len(filtered) > 0:
            st.dataframe(filtered["quality_label"].value_counts().head(10), use_container_width=True)
        else:
            st.info("Brak danych po filtrach.")

    st.divider()

    # --- Wizualizacje (POPRAWKA + opis) ---
    st.markdown("## Wizualizacje")
    st.caption("Te wykresy pomagają szybko ocenić, jak rozkładają się wyniki po zastosowaniu filtrów.")

    if len(filtered) == 0:
        st.warning("Brak danych po filtrach — nie da się narysować wykresów.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            vc = filtered["quality_label"].astype(str).value_counts(dropna=False)
            vc_df = vc.reset_index()
            vc_df.columns = ["quality_label", "count"]

            fig_lbl = px.bar(
                vc_df,
                x="quality_label",
                y="count",
                title="Rozkład: Opis jakości (po filtrach)",
                labels={"quality_label": "Opis jakości", "count": "Liczba rekordów"}
            )
            fig_lbl.update_layout(height=450)
            st.plotly_chart(fig_lbl, use_container_width=True)

        with c2:
            wt_mean = (
                filtered.groupby("wine_type", dropna=False)["pairing_quality"]
                .mean()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )
            wt_mean.columns = ["wine_type", "avg_pairing_quality"]

            fig_wt = px.bar(
                wt_mean,
                x="wine_type",
                y="avg_pairing_quality",
                title="Średnia jakość parowania wg typu wina (top 20)",
                labels={"wine_type": "Typ wina", "avg_pairing_quality": "Średnia jakość (1–5)"}
            )
            fig_wt.update_layout(height=450)
            st.plotly_chart(fig_wt, use_container_width=True)

    st.divider()

    # --- Rekomendacja dla dania ---
    st.markdown("## 🔎 Rekomendacje na podstawie nazwy dania")
    st.caption("Wpisz fragment nazwy dania, aby znaleźć pasujące wina (np. „pasta”, „steak”, „salmon”).")

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
                        "description",
                    ]
                ].head(20).rename(columns=FRIENDLY_PAIR_COLS),
                use_container_width=True
            )
    else:
        st.info("Wpisz fragment nazwy dania, aby zobaczyć rekomendacje.")

# =========================================================
# 3) INTEGRACJA: opisz wino → predykcja → rekomendacje
# =========================================================
else:
    st.subheader("🔗 Integracja: opisz wino → przewidź jakość → rekomenduj parowania")

    with st.expander("ℹ️ Co robi integracja? (prosto)", expanded=True):
        st.write(
            "Ten moduł działa jak mini-doradca:\n"
            "1) opisujesz wino (z listy lub ręcznie)\n"
            "2) model przewiduje jego jakość\n"
            "3) na tej podstawie dobieramy rekomendacje parowania z jedzeniem."
        )

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
        "Ten moduł łączy oba datasety: parametry wina → model ML → przewidywana jakość → rekomendacje parowań."
    )

    st.divider()

    # --- Trening modelu ---
    st.markdown("## 1) Model jakości (RandomForest)")
    st.caption("Ustawienia pozwalają zmienić, jak model się uczy. Jeśli nie wiesz co wybrać — zostaw domyślne.")

    c1, c2, c3 = st.columns(3)
    with c1:
        test_size = st.slider("Ile danych na test? (test_size)", 0.1, 0.5, 0.2, 0.05)
    with c2:
        n_estimators = st.slider("Liczba drzew (n_estimators)", 50, 600, 300, 50)
    with c3:
        random_state = st.number_input("Losowość (random_state)", value=42, step=1)

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
    st.caption("R²: im bliżej 1, tym lepiej. MAE: średni błąd (im mniej, tym lepiej).")

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    with st.expander("Co najbardziej wpływa na predykcję? (feature importance)"):
        tmp = importances.reset_index()
        tmp.columns = ["Parametr", "Ważność"]
        tmp["Parametr"] = tmp["Parametr"].map(label_wine)
        st.dataframe(tmp, use_container_width=True)

    st.divider()

    # --- Wybór: rekord vs ręcznie ---
    st.markdown("## 2) Wybierz / opisz wino")
    st.caption("Możesz wybrać istniejące wino z danych albo ustawić parametry ręcznie suwakami.")

    mode = st.radio(
        "Źródło parametrów",
        options=["Wybierz rekord z danych", "Wprowadź parametry ręcznie"],
        horizontal=True
    )

    feature_cols = [c for c in df.columns if c != "quality"]

    if mode == "Wybierz rekord z danych":
        idx = st.slider("Indeks rekordu (wino z datasetu)", 0, len(df) - 1, 0)
        row = df.iloc[idx]
        input_df = row.drop(labels=["quality"]).to_frame().T
        st.caption(f"Prawdziwa jakość dla tego rekordu: **{int(row['quality'])}**")
        st.dataframe(input_df.rename(columns=FRIENDLY_WINE_COLS), use_container_width=True)
    else:
        defaults = df.drop(columns=["quality"]).mean(numeric_only=True)
        inputs = {}
        with st.form("integrated_wine_form"):
            for col in feature_cols:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                col_def = float(defaults[col])
                step = (col_max - col_min) / 100 if col_max > col_min else 0.01
                inputs[col] = st.slider(label_wine(col), col_min, col_max, col_def, step=step)
            submitted = st.form_submit_button("Użyj parametrów")
        if not submitted:
            st.info("Ustaw parametry i kliknij „Użyj parametrów”.")
            st.stop()
        input_df = pd.DataFrame([inputs])

    # --- Predykcja jakości ---
    pred_quality = float(model.predict(input_df)[0])
    tier = quality_tier(pred_quality)

    st.success(f"Przewidywana jakość: **{pred_quality:.2f}**  → poziom (tier): **{tier}**")
    st.caption("Tier jest prostą „etykietą” (Low/Mid/High), która pomaga dobrać minimalną jakość parowania.")

    st.divider()

    # --- Rekomendacje pairingów ---
    st.markdown("## 3) Rekomendacje parowań (wine_food_pairings)")
    st.caption("Ustaw filtry i zobacz propozycje dań, które najlepiej pasują do wybranego stylu wina.")

    wine_types = sorted(dfp["wine_type"].dropna().unique())
    red_like = [w for w in wine_types if "red" in str(w).lower()]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        default_wine_types = red_like[:1] if red_like else []
        sel_wine_type = st.multiselect(label_pair("wine_type"), options=wine_types, default=default_wine_types)

    with c2:
        food_category_sel = st.multiselect(
            label_pair("food_category"),
            options=sorted(dfp["food_category"].dropna().unique()),
            default=[]
        )

    with c3:
        cuisine_sel = st.multiselect(
            label_pair("cuisine"),
            options=sorted(dfp["cuisine"].dropna().unique()),
            default=[]
        )

    with c4:
        tier_min = tier_min_pairing_quality(tier)
        min_pair = st.slider("Minimalna jakość parowania (1–5)", 1, 5, value=tier_min)

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
        st.stop()

    ranked = (
        rec.groupby(["food_category", "food_item", "cuisine"], as_index=False)
        .agg(avg_pairing_quality=("pairing_quality", "mean"), n=("pairing_quality", "size"))
        .sort_values(["avg_pairing_quality", "n"], ascending=[False, False])
    )

    topk = st.slider("Ile rekomendacji pokazać?", 5, 50, 20)
    pretty_ranked = ranked.head(topk).copy()
    pretty_ranked = pretty_ranked.rename(columns={
        "food_category": "Kategoria jedzenia",
        "food_item": "Danie / produkt",
        "cuisine": "Kuchnia",
        "avg_pairing_quality": "Średnia jakość parowania",
        "n": "Liczba wystąpień",
})
    st.dataframe(pretty_ranked, use_container_width=True)

    with st.expander("Szybkie statystyki rekomendacji"):
        st.dataframe(quick_stats(rec, ["pairing_quality"]).rename(index=label_pair), use_container_width=True)
        if "quality_label" in rec.columns:
            st.dataframe(rec["quality_label"].value_counts().head(10), use_container_width=True)

    fig_sc = px.scatter(
        ranked.head(300),
        x="avg_pairing_quality",
        y="n",
        color="food_category",
        hover_data=["food_item", "cuisine"],
        title="Rekomendacje: średnia jakość vs liczność (top 300)",
        labels={
            "avg_pairing_quality": "Średnia jakość parowania",
            "n": "Liczb

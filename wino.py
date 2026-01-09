import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# Konfiguracja strony
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Analytics & Food Pairings",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🍷 Wine Analytics & Food Pairings")
st.markdown(
    "Aplikacja do eksploracji jakości czerwonych win oraz "
    "parowania win z jedzeniem."
)

# ---------------------------------------------------------
# Funkcje wczytywania danych
# ---------------------------------------------------------
@st.cache_data
def load_wine_quality(path: str = "winequality-red.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_wine_food_pairings(path: str = "wine_food_pairings.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

# ---------------------------------------------------------
# Próba wczytania danych + komunikaty błędów
# ---------------------------------------------------------
wine_quality_df, pairings_df = None, None
wine_quality_error, pairings_error = None, None

try:
    wine_quality_df = load_wine_quality()
except Exception as e:
    wine_quality_error = str(e)

try:
    pairings_df = load_wine_food_pairings()
except Exception as e:
    pairings_error = str(e)

# ---------------------------------------------------------
# Sidebar – wybór modułu
# ---------------------------------------------------------
st.sidebar.header("⚙️ Ustawienia")
module = st.sidebar.radio(
    "Wybierz moduł:",
    options=["Analiza jakości wina", "Parowanie wina z jedzeniem"]
)

# =========================================================
# 1. ANALIZA JAKOŚCI WINA (winequality-red.csv)
# =========================================================
if module == "Analiza jakości wina":
    st.subheader("📊 Analiza jakości czerwonych win")

    if wine_quality_df is None:
        st.error(
            "Nie udało się wczytać `winequality-red.csv`.\n\n"
            f"Komunikat błędu:\n`{wine_quality_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py`."
        )
        st.stop()

    df = wine_quality_df.copy()

    # -------------------------
    # Podstawowe informacje
    # -------------------------
    st.markdown("### Podgląd danych")
    st.write("Pierwsze wiersze datasetu:")
    st.dataframe(df.head())

    with st.expander("Informacje o datasetcie"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Kształt (liczba rekordów, liczba kolumn):**")
            st.write(df.shape)
            st.write("**Typy danych:**")
            st.write(df.dtypes)
        with col2:
            st.write("**Podstawowe statystyki opisowe:**")
            st.write(df.describe().T)

    # -------------------------
    # Filtrowanie po jakości
    # -------------------------
    st.markdown("### Filtrowanie po ocenie jakości")
    min_q = int(df["quality"].min())
    max_q = int(df["quality"].max())

    quality_range = st.slider(
        "Zakres jakości (kolumna `quality`):",
        min_value=min_q,
        max_value=max_q,
        value=(min_q, max_q),
        step=1
    )

    filtered = df[(df["quality"] >= quality_range[0]) & (df["quality"] <= quality_range[1])]

    st.write(f"Liczba rekordów po filtrze: **{filtered.shape[0]}**")
    st.dataframe(filtered.head())

    # -------------------------
    # Rozkład jakości
    # -------------------------
    st.markdown("### Rozkład jakości wina")

    fig, ax = plt.subplots()
    ax.hist(df["quality"], bins=range(min_q, max_q + 2), edgecolor="black")
    ax.set_xlabel("Jakość (quality)")
    ax.set_ylabel("Liczba próbek")
    ax.set_title("Histogram ocen jakości wina")
    st.pyplot(fig)

    # -------------------------
    # Korelacja cech
    # -------------------------
    st.markdown("### Korelacje między cechami")

    corr = df.corr(numeric_only=True)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax_corr)
    ax_corr.set_title("Macierz korelacji")
    st.pyplot(fig_corr)

    # -------------------------
    # Scatter: wybrana cecha vs jakość
    # -------------------------
    st.markdown("### Zależność cechy od jakości")

    feature_cols = [c for c in df.columns if c != "quality"]
    x_feature = st.selectbox("Wybierz cechę (oś X):", feature_cols, index=feature_cols.index("alcohol"))

    fig_scatter, ax_scatter = plt.subplots()
    ax_scatter.scatter(df[x_feature], df["quality"], alpha=0.6)
    ax_scatter.set_xlabel(x_feature)
    ax_scatter.set_ylabel("quality")
    ax_scatter.set_title(f"{x_feature} vs quality")
    st.pyplot(fig_scatter)

    # -------------------------
    # Prosty model predykcji jakości
    # -------------------------
    st.markdown("### 🤖 Prosty model predykcji jakości (RandomForest)")

    with st.expander("Parametry modelu i metryki"):
        test_size = st.slider(
            "Udział danych testowych",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05
        )
        n_estimators = st.slider(
            "Liczba drzew (n_estimators)",
            min_value=50,
            max_value=500,
            value=200,
            step=50
        )
        random_state = st.number_input(
            "Random state",
            min_value=0,
            value=42,
            step=1
        )

        # Przygotowanie danych
        X = df.drop("quality", axis=1)
        y = df["quality"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=random_state
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("R² na zbiorze testowym", f"{r2:.3f}")
        with col_m2:
            st.metric("MAE na zbiorze testowym", f"{mae:.3f}")

        # Ważność cech
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        st.markdown("**Ważność cech (feature importance):**")
        st.bar_chart(importances)

    # -------------------------
    # Interaktywna predykcja dla użytkownika
    # -------------------------
    st.markdown("### 🔮 Predykcja jakości dla podanych parametrów")

    with st.form("prediction_form"):
        cols = st.columns(3)
        user_input = {}

        for i, col_name in enumerate(X.columns):
            col = cols[i % 3]
            min_val = float(df[col_name].min())
            max_val = float(df[col_name].max())
            mean_val = float(df[col_name].mean())
            user_input[col_name] = col.slider(
                col_name,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                step=(max_val - min_val) / 100 if max_val > min_val else 0.01
            )

        submitted = st.form_submit_button("Oblicz przewidywaną jakość")

    if "model" in locals() and submitted:
        input_df = pd.DataFrame([user_input])
        pred_quality = model.predict(input_df)[0]
        st.success(f"Przewidywana jakość wina: **{pred_quality:.2f}** (w skali jak w kolumnie `quality`)")

# =========================================================
# 2. PAROWANIE WINA Z JEDZENIEM (wine_food_pairings.csv)
# =========================================================
elif module == "Parowanie wina z jedzeniem":
    st.subheader("🍽️ Parowanie wina z jedzeniem")

    if pairings_df is None:
        st.error(
            "Nie udało się wczytać `wine_food_pairings.csv`.\n\n"
            f"Komunikat błędu:\n`{pairings_error}`\n\n"
            "Upewnij się, że plik znajduje się w tym samym katalogu co `app.py`."
        )
        st.stop()

    dfp = pairings_df.copy()

    # -------------------------
    # Podgląd danych
    # -------------------------
    st.markdown("### Podgląd danych")
    st.dataframe(dfp.head())

    with st.expander("Informacje o datasetcie"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Kształt:**", dfp.shape)
            st.write("**Kolumny:**")
            st.write(list(dfp.columns))
        with col2:
            st.write("**Przykładowe wartości kategorii:**")
            st.write("wine_type:", dfp["wine_type"].unique()[:10])
            st.write("food_category:", dfp["food_category"].unique()[:10])
            st.write("cuisine:", dfp["cuisine"].unique()[:10])
            st.write("quality_label:", dfp["quality_label"].unique())

    # -------------------------
    # Filtrowanie parowań
    # -------------------------
    st.markdown("### Filtrowanie rekomendacji")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    with col_f1:
        wine_type_sel = st.multiselect(
            "Typ wina (`wine_type`):",
            options=sorted(dfp["wine_type"].unique()),
            default=None
        )

    with col_f2:
        food_cat_sel = st.multiselect(
            "Kategoria jedzenia (`food_category`):",
            options=sorted(dfp["food_category"].unique()),
            default=None
        )

    with col_f3:
        cuisine_sel = st.multiselect(
            "Kuchnia (`cuisine`):",
            options=sorted(dfp["cuisine"].unique()),
            default=None
        )

    with col_f4:
        min_pair_quality = int(dfp["pairing_quality"].min())
        max_pair_quality = int(dfp["pairing_quality"].max())
        pairing_quality_sel = st.slider(
            "Minimalna ocena parowania (`pairing_quality`):",
            min_value=min_pair_quality,
            max_value=max_pair_quality,
            value=min_pair_quality,
            step=1
        )

    filt = dfp.copy()
    if wine_type_sel:
        filt = filt[filt["wine_type"].isin(wine_type_sel)]
    if food_cat_sel:
        filt = filt[filt["food_category"].isin(food_cat_sel)]
    if cuisine_sel:
        filt = filt[filt["cuisine"].isin(cuisine_sel)]

    filt = filt[filt["pairing_quality"] >= pairing_quality_sel]

    st.markdown(f"Znaleziono **{filt.shape[0]}** dopasowań.")
    st.dataframe(
        filt[
            [
                "wine_type",
                "wine_category",
                "food_item",
                "food_category",
                "cuisine",
                "pairing_quality",
                "quality_label",
                "description"
            ]
        ].sort_values(by="pairing_quality", ascending=False)
    )

    # -------------------------
    # Statystyki jakości parowań
    # -------------------------
    st.markdown("### Podsumowanie jakości parowań")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.write("**Liczba parowań per etykieta jakości (`quality_label`):**")
        st.bar_chart(dfp["quality_label"].value_counts())

    with col_s2:
        st.write("**Średnia ocena parowania per typ wina (`wine_type`):**")
        mean_quality_by_wine = (
            dfp.groupby("wine_type")["pairing_quality"]
            .mean()
            .sort_values(ascending=False)
        )
        st.bar_chart(mean_quality_by_wine)

    # -------------------------
    # Prosta „rekomendacja” na podstawie wyboru użytkownika
    # -------------------------
    st.markdown("### 🔍 Znajdź rekomendacje dla konkretnego dania")

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        chosen_food = st.text_input(
            "Podaj nazwę dania (część nazwy z kolumny `food_item`):",
            value=""
        )
    with col_r2:
        chosen_cuisine = st.selectbox(
            "Wybierz kuchnię (opcjonalnie):",
            options=["(dowolna)"] + sorted(dfp["cuisine"].unique().tolist())
        )

    if chosen_food.strip():
        rec = dfp[dfp["food_item"].str.contains(chosen_food, case=False, na=False)]
        if chosen_cuisine != "(dowolna)":
            rec = rec[rec["cuisine"] == chosen_cuisine]

        rec = rec.sort_values(by="pairing_quality", ascending=False)

        if rec.empty:
            st.warning("Brak rekomendacji spełniających kryteria.")
        else:
            st.success(f"Znaleziono **{rec.shape[0]}** rekomendacji.")
            st.dataframe(
                rec[
                    [
                        "food_item",
                        "cuisine",
                        "wine_type",
                        "wine_category",
                        "pairing_quality",
                        "quality_label",
                        "description"
                    ]
                ].head(20)
            )
    else:
        st.info("Wpisz fragment nazwy dania, aby zobaczyć rekomendacje.")


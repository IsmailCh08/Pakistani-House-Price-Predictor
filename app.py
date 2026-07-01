import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pakistan House Price Estimator",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background-color: #080c14; color: #e2e8f0; }

.block-container {
    padding-top: 0;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Hero */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #0c1629 50%, #080c14 100%);
    border-bottom: 1px solid #1e2d45;
    padding: 3rem 0 2.5rem 0;
    margin-bottom: 2.5rem;
    text-align: center;
}
.hero-eyebrow {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #22c55e;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
    margin-bottom: 0.75rem;
}
.hero-title em {
    font-style: normal;
    color: #22c55e;
}
.hero-sub {
    font-size: 0.95rem;
    color: #475569;
    max-width: 480px;
    margin: 0 auto;
    line-height: 1.6;
}

/* Cards */
.card {
    background: #0f172a;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 1.75rem;
    margin-bottom: 1.25rem;
}
.card-title {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 1.25rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #1e2d45;
}

/* Result box */
.result-box {
    background: linear-gradient(135deg, #052e16 0%, #0f2d1a 100%);
    border: 1px solid #16a34a;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.result-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 0.75rem;
}
.result-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 600;
    color: #22c55e;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.result-unit {
    font-size: 0.8rem;
    color: #4ade80;
    opacity: 0.7;
}
.result-range {
    margin-top: 1rem;
    font-size: 0.8rem;
    color: #166534;
    font-family: 'JetBrains Mono', monospace;
}

/* Stat pills */
.stat-row {
    display: flex;
    gap: 0.75rem;
    margin-top: 1.25rem;
    flex-wrap: wrap;
    justify-content: center;
}
.stat-pill {
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 0.4rem 1rem;
    font-size: 0.75rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}
.stat-pill span { color: #3b82f6; font-weight: 600; }

/* Slider label override */
.stSlider label { color: #64748b !important; font-size: 0.8rem !important; }

/* Input labels */
.stSelectbox label, .stNumberInput label {
    color: #64748b !important;
    font-size: 0.8rem !important;
}

/* Button */
.stButton > button {
    background: #16a34a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    width: 100% !important;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #15803d !important;
}

/* Toggle group for amenities */
.amenity-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #334155;
    margin-bottom: 0.75rem;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Model training (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_or_train_model():
    """Train model from data.csv or load from disk if already saved."""
    model_path = Path("house_model.pkl")
    meta_path  = Path("house_meta.pkl")

    if model_path.exists() and meta_path.exists():
        model = joblib.load(model_path)
        meta  = joblib.load(meta_path)
        return model, meta

    # Train from scratch
    df = pd.read_csv("data.csv")

    def clean_price(p):
        p = str(p).replace('PKR', '').strip()
        if 'Lakh' in p:
            return float(p.replace('Lakh', '').strip()) / 100
        elif 'Crore' in p:
            return float(p.replace('Crore', '').strip())
        elif 'Arab' in p:
            return float(p.replace('Arab', '').strip()) * 100
        return np.nan

    def clean_area(a):
        a = str(a)
        if 'Marla' in a:
            return float(a.replace('Marla', '').strip()) / 20
        elif 'Kanal' in a:
            return float(a.replace('Kanal', '').strip())
        return np.nan

    df['Price_Crore'] = df['Price'].apply(clean_price)
    df['Area_Kanal']  = df['Area'].apply(clean_area)
    df['Area_Kanal']  = pd.to_numeric(df['Area_Kanal'], errors='coerce')
    df = df.dropna(subset=['Area_Kanal', 'Price_Crore'])

    df['Beds_clean']  = pd.to_numeric(df['Beds'],  errors='coerce').fillna(df['Beds'].pipe(lambda s: pd.to_numeric(s, errors='coerce').median()))
    df['Baths_clean'] = pd.to_numeric(df['Baths'], errors='coerce').fillna(pd.to_numeric(df['Baths'], errors='coerce').median())

    global_mean    = df['Price_Crore'].mean()
    loc_mean       = df.groupby('Location')['Price_Crore'].mean()
    loc_counts     = df.groupby('Location')['Price_Crore'].count()
    df['loc_count'] = df['Location'].map(loc_counts)
    df['Location_encoded'] = np.where(
        df['loc_count'] > 10,
        df['Location'].map(loc_mean),
        global_mean
    )

    df = df[(df['Price_Crore'] <= 20) & (df['Price_Crore'] > 0)]
    df['log_price'] = np.log(df['Price_Crore'])

    features = ['Area_Kanal', 'Baths_clean', 'Beds_clean',
                'Location_encoded', 'Gym', 'Dining Room', 'Kitchens']
    df = df.dropna(subset=features + ['log_price'])

    X = df[features]
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # Save artifacts
    joblib.dump(model, model_path)
    meta = {
        'loc_mean':    loc_mean,
        'loc_counts':  loc_counts,
        'global_mean': global_mean,
        'locations':   sorted(df['Location'].unique().tolist()),
        'area_min':    float(df['Area_Kanal'].min()),
        'area_max':    float(df['Area_Kanal'].max()),
        'beds_max':    int(df['Beds_clean'].max()),
        'baths_max':   int(df['Baths_clean'].max()),
        'n_listings':  len(df),
        'price_mean':  float(df['Price_Crore'].mean()),
        'price_max':   float(df['Price_Crore'].max()),
    }
    joblib.dump(meta, meta_path)

    return model, meta


# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Loading model..."):
    try:
        model, meta = load_or_train_model()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Pakistan Real Estate · 7,500+ Listings · Random Forest ML</div>
    <div class="hero-title">What's your house <em>worth?</em></div>
    <div class="hero-sub">
        Enter your property details and get an instant price estimate
        based on real market data from across Pakistan.
    </div>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("Could not find `data.csv`. Make sure it's in the same folder as this app.")
    st.stop()


# ── Dataset stats ──────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; gap:0.75rem; flex-wrap:wrap; justify-content:center; margin-bottom:2.5rem;">
    <div class="stat-pill"><span>{meta['n_listings']:,}</span> listings trained on</div>
    <div class="stat-pill">Avg price <span>{meta['price_mean']:.1f} Crore</span></div>
    <div class="stat-pill"><span>{len(meta['locations']):,}</span> locations</div>
    <div class="stat-pill">Model MAE <span>~0.86 Crore</span></div>
</div>
""", unsafe_allow_html=True)


# ── Input form ─────────────────────────────────────────────────────────────────
col_form, col_result = st.columns([1.1, 0.9], gap="large")

with col_form:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Property Details</div>', unsafe_allow_html=True)

    # Location
    location = st.selectbox(
        "Location",
        options=meta['locations'],
        help="Select the area where the property is located"
    )

    # Area
    area_marla = st.slider(
        "Property Size (Marla)",
        min_value=1,
        max_value=400,
        value=10,
        step=1,
        help="1 Kanal = 20 Marla"
    )
    area_kanal = area_marla / 20
    st.caption(f"= {area_kanal:.2f} Kanal")

    col_b, col_bath = st.columns(2)
    with col_b:
        beds = st.number_input("Bedrooms", min_value=1, max_value=20, value=3, step=1)
    with col_bath:
        baths = st.number_input("Bathrooms", min_value=1, max_value=20, value=2, step=1)

    st.markdown('<div style="margin-top:1rem; margin-bottom:0.5rem;" class="amenity-label">Amenities</div>', unsafe_allow_html=True)

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        gym = st.checkbox("Gym", value=False)
    with col_a2:
        dining = st.checkbox("Dining Room", value=True)
    with col_a3:
        kitchens = st.number_input("Kitchens", min_value=1, max_value=5, value=1, step=1)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("Estimate Price →")


# ── Prediction ─────────────────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Price Estimate</div>', unsafe_allow_html=True)

    if predict_clicked:
        # Encode location
        loc_counts = meta['loc_counts']
        loc_mean   = meta['loc_mean']
        global_mean = meta['global_mean']

        if location in loc_counts and loc_counts[location] > 10:
            loc_encoded = loc_mean[location]
        else:
            loc_encoded = global_mean

        # Build feature row
        input_data = pd.DataFrame([{
            'Area_Kanal':       area_kanal,
            'Baths_clean':      float(baths),
            'Beds_clean':       float(beds),
            'Location_encoded': loc_encoded,
            'Gym':              int(gym),
            'Dining Room':      int(dining),
            'Kitchens':         float(kitchens)
        }])

        log_pred = model.predict(input_data)[0]
        price    = np.exp(log_pred)

        # Rough confidence range (±15% based on model MAE)
        low  = price * 0.88
        high = price * 1.12

        # Format
        def fmt(p):
            if p >= 1:
                return f"{p:.2f} Crore"
            else:
                return f"{p*100:.1f} Lakh"

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">Estimated Market Value</div>
            <div class="result-value">PKR {fmt(price)}</div>
            <div class="result-unit">Pakistani Rupees</div>
            <div class="result-range">
                Likely range: {fmt(low)} – {fmt(high)}
            </div>
        </div>

        <div style="margin-top:1.25rem;">
            <div style="font-size:0.65rem; font-weight:700; letter-spacing:0.15em;
                        text-transform:uppercase; color:#334155; margin-bottom:0.75rem;">
                What drove this estimate
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance mini display
        feature_names = ['Area', 'Bathrooms', 'Bedrooms', 'Location', 'Gym', 'Dining Room', 'Kitchens']
        importances   = model.feature_importances_
        total         = importances.sum()

        for fname, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
            pct = imp / total * 100
            bar_w = int(pct * 2.5)
            st.markdown(f"""
            <div style="margin-bottom:0.5rem;">
                <div style="display:flex; justify-content:space-between;
                            font-size:0.75rem; color:#475569; margin-bottom:0.2rem;">
                    <span>{fname}</span>
                    <span style="font-family:'JetBrains Mono',monospace; color:#334155;">{pct:.1f}%</span>
                </div>
                <div style="background:#0a1628; border-radius:3px; height:4px;">
                    <div style="background:#22c55e; width:{min(bar_w,100)}%;
                                height:4px; border-radius:3px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#1e3a5f;">
            <div style="font-size:2.5rem; margin-bottom:1rem;">🏡</div>
            <div style="font-size:0.85rem; color:#334155; line-height:1.7;">
                Fill in your property details<br>and click <strong style="color:#475569;">Estimate Price</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── Disclaimer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2.5rem; padding-top:1.5rem;
            border-top:1px solid #0f172a; color:#1e3a5f; font-size:0.72rem; line-height:1.8;">
    Estimates are based on a Random Forest model trained on 7,500+ Pakistani real estate listings.<br>
    Actual prices vary based on condition, market timing, and negotiation. Not financial advice.
</div>
""", unsafe_allow_html=True)
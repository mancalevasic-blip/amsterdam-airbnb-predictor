"""
Airbnb Amsterdam Stay Value Predictor — Streamlit app.
Fast path: widget keys let Streamlit manage values natively; URL lookup just
sets those keys once, then the sidebar is fully independent on every rerun.
"""

import os, re, json

# macOS: XGBoost needs libomp; use the one bundled with scikit-learn if brew's is missing
try:
    import sklearn
    _omp_dir = os.path.join(os.path.dirname(sklearn.__file__), ".dylibs")
    if os.path.isdir(_omp_dir):
        os.environ.setdefault("DYLD_LIBRARY_PATH", _omp_dir)
except Exception:
    pass

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Amsterdam Airbnb Predictor", page_icon="🏠",
                   layout="wide", initial_sidebar_state="expanded")

# Auto-download data if missing (runs on Streamlit Cloud first boot)
_root     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_lookup   = os.path.join(_root, "data", "lookup.csv")
_raw_csv  = os.path.join(_root, "data", "raw", "listings.csv")
if not os.path.exists(_lookup) or not os.path.exists(_raw_csv):
    with st.spinner("First-time setup: downloading dataset (~6 MB) …"):
        import subprocess, sys
        subprocess.run(
            [sys.executable, os.path.join(_root, "scripts", "download_data.py")],
            check=True,
        )

# ── Load (cached) ──────────────────────────────────────────────────────────────
_root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(_root, "models")
DATA_DIR   = os.path.join(_root, "data")

@st.cache_resource
def load_models():
    pm = joblib.load(os.path.join(MODELS_DIR, "price_model.pkl"))
    am = joblib.load(os.path.join(MODELS_DIR, "avail_model.pkl"))
    return pm, am

@st.cache_data
def load_lookup():
    return pd.read_csv(os.path.join(DATA_DIR, "lookup.csv"), low_memory=False)

@st.cache_data
def build_neigh_stats(df):
    s = df.groupby("neighbourhood_cleansed").agg(
        lat=("latitude","mean"), lon=("longitude","mean"),
        count=("id","count"), median_price=("price_numeric","median"),
        low_avail_pct=("low_availability","mean"),
    ).reset_index()
    s["median_price"] = s["median_price"].fillna(0)
    return s

# Map figure is expensive — cache by selected neighbourhood
@st.cache_data
def make_map(selected: str, _stats):
    sizes  = (_stats["count"] / _stats["count"].max() * 35 + 10).tolist()
    hover  = [
        f"<b>{r['neighbourhood_cleansed']}</b><br>"
        f"Listings: {r['count']}<br>Median: €{r['median_price']:.0f}/night<br>"
        f"Heavily booked: {r['low_avail_pct']:.0%}"
        for _, r in _stats.iterrows()
    ]
    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=_stats["lat"].tolist(), lon=_stats["lon"].tolist(), mode="markers",
        marker=dict(size=sizes, color=_stats["median_price"].tolist(),
                    colorscale="RdYlGn_r", colorbar=dict(title="€/night", thickness=12),
                    opacity=0.85),
        text=_stats["neighbourhood_cleansed"].tolist(),
        hovertext=hover, hoverinfo="text",
        customdata=_stats["neighbourhood_cleansed"].tolist(),
        selected=dict(marker=dict(opacity=1.0)),
        unselected=dict(marker=dict(opacity=0.4)),
    ))
    if selected and selected in _stats["neighbourhood_cleansed"].values:
        r = _stats[_stats["neighbourhood_cleansed"] == selected].iloc[0]
        fig.add_trace(go.Scattermap(
            lat=[r["lat"]], lon=[r["lon"]], mode="markers",
            marker=dict(size=22, color="white", opacity=1.0),
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scattermap(
            lat=[r["lat"]], lon=[r["lon"]], mode="markers+text",
            marker=dict(size=16, color="#4a6cf7", opacity=1.0),
            text=[selected], textposition="top center",
            textfont=dict(size=11, color="#1a1a2e"),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=52.366, lon=4.896), zoom=11),
        margin=dict(l=0, r=0, t=0, b=0), height=420, clickmode="event+select",
    )
    return fig

@st.cache_data
def get_feature_importance(_model):
    try:
        xgb_step  = list(_model.named_steps.values())[-1]
        prep_step = list(_model.named_steps.values())[-2]
        imp = xgb_step.feature_importances_
        try:
            names = prep_step.get_feature_names_out()
            names = [n.split("__", 1)[-1] for n in names]
        except Exception:
            names = [f"feature_{i}" for i in range(len(imp))]
        df = pd.DataFrame({"feature": names, "importance": imp})
        df = df.groupby("feature", as_index=False)["importance"].sum()
        return df.sort_values("importance", ascending=False).head(15).reset_index(drop=True)
    except Exception:
        return pd.DataFrame({"feature": [], "importance": []})

@st.cache_data
def dataset_stats(df):
    prices = df["price_numeric"].dropna()
    return dict(
        n_listings   = len(df),
        n_neigh      = df["neighbourhood_cleansed"].nunique(),
        price_median = prices.median(),
        price_min    = prices.quantile(0.05),
        price_max    = prices.quantile(0.95),
        pct_booked   = df["low_availability"].mean(),
    )

price_model, avail_model = load_models()
lookup_df   = load_lookup()
neigh_stats = build_neigh_stats(lookup_df)

NEIGHBOURHOODS = sorted(lookup_df["neighbourhood_cleansed"].dropna().unique())
ROOM_TYPES     = sorted(lookup_df["room_type"].dropna().unique())
PROP_TYPES     = sorted(lookup_df["property_type"].dropna().unique())
RESP_TIMES     = ["within an hour", "within a few hours", "within a day", "a few days or more"]

# ── Session state defaults (only set once on first load) ───────────────────────
DEFAULTS = dict(
    url_input="",
    nb=None, rt=None, pt=None,
    accommodates=2, bedrooms=1, beds=1, bathrooms=1.0, amenity_count=30,
    minimum_nights=2, maximum_nights=365, instant_bookable=False,
    number_of_reviews=20, number_of_reviews_ltm=5, reviews_per_month=0.5,
    review_scores_rating=4.5, review_scores_cleanliness=4.5, review_scores_location=4.8,
    no_ratings=False, unknown_rates=False,
    host_is_superhost=False, host_identity_verified=True, host_years=3.0,
    host_response_rate=90, host_acceptance_rate=80, resp_time=RESP_TIMES[0],
    host_listings=1,
    listed_price=None,
    listing_name=None,
    prefill_snap=None,
    lookup_error=None,
    map_nb=None,
)
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Apply pending actions before any widget renders (Streamlit requires this)
if st.session_state.get("map_nb_pending"):
    st.session_state.nb = st.session_state.pop("map_nb_pending")

if st.session_state.get("_lookup_prefill"):
    for k, v in st.session_state.pop("_lookup_prefill").items():
        st.session_state[k] = v
    st.session_state.prefill_snap = dict(
        nb=st.session_state.nb, rt=st.session_state.rt,
        accommodates=st.session_state.accommodates,
        bedrooms=st.session_state.bedrooms, beds=st.session_state.beds,
        minimum_nights=st.session_state.minimum_nights,
    )

if st.session_state.pop("clear_pending", False):
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

if st.session_state.pop("reset_pending", False):
    WIDGET_KEYS = ["nb","rt","pt","accommodates","bedrooms","beds","bathrooms",
                   "amenity_count","minimum_nights","maximum_nights","instant_bookable",
                   "number_of_reviews","number_of_reviews_ltm","reviews_per_month",
                   "review_scores_rating","review_scores_cleanliness","review_scores_location",
                   "host_is_superhost","host_identity_verified","host_years",
                   "host_response_rate","host_acceptance_rate","resp_time","host_listings"]
    for k in WIDGET_KEYS:
        st.session_state[k] = DEFAULTS[k]
    st.session_state.listed_price  = None
    st.session_state.listing_name  = None
    st.session_state.prefill_snap  = None
    st.session_state.lookup_error  = None

# ── URL lookup helpers ─────────────────────────────────────────────────────────
def _on_key_field_change():
    if st.session_state.get("prefill_snap") is not None:
        st.session_state.listed_price = None
        st.session_state.prefill_snap = None
        st.session_state.listing_name = None

def _safe(row, col, default):
    v = row.get(col, default)
    return default if pd.isna(v) else v

def do_lookup(url: str):
    m = re.search(r"airbnb\.[a-z.]+/rooms/(\d+)", url)
    if not m:
        st.session_state.lookup_error = "Could not parse a listing ID from that URL."
        st.session_state.listed_price = None
        st.session_state.listing_name = None
        st.session_state.prefill_snap = None
        return
    lid = int(m.group(1))
    row = lookup_df[lookup_df["id"] == lid]
    if row.empty:
        st.session_state.lookup_error = (
            f"Listing #{lid} is not in our dataset (scraped Sept 2025). "
            "Fill in details manually below."
        )
        st.session_state.listed_price = None
        st.session_state.listing_name = None
        st.session_state.prefill_snap = None
        return

    r = row.iloc[0]
    nb = _safe(r, "neighbourhood_cleansed", NEIGHBOURHOODS[0])
    rt = _safe(r, "room_type", ROOM_TYPES[0])
    pt = _safe(r, "property_type", PROP_TYPES[0])
    rtime = _safe(r, "host_response_time", RESP_TIMES[0])

    # Widget-backed keys cannot be set after widgets render — store in pending dict
    # and apply at the top of the next rerun before any widgets instantiate.
    st.session_state._lookup_prefill = dict(
        nb  = nb if nb in NEIGHBOURHOODS else NEIGHBOURHOODS[0],
        rt  = rt if rt in ROOM_TYPES     else ROOM_TYPES[0],
        pt  = pt if pt in PROP_TYPES     else PROP_TYPES[0],
        accommodates          = int(_safe(r, "accommodates", 2)),
        bedrooms              = int(_safe(r, "bedrooms", 1)),
        beds                  = int(_safe(r, "beds", 1)),
        bathrooms             = float(_safe(r, "bathrooms", 1.0)),
        amenity_count         = int(_safe(r, "amenity_count", 30)),
        minimum_nights        = int(_safe(r, "minimum_nights", 2)),
        maximum_nights        = int(_safe(r, "maximum_nights", 365)),
        instant_bookable      = bool(_safe(r, "instant_bookable", 0)),
        number_of_reviews     = int(_safe(r, "number_of_reviews", 20)),
        number_of_reviews_ltm = int(_safe(r, "number_of_reviews_ltm", 5)),
        reviews_per_month     = float(_safe(r, "reviews_per_month", 0.5)),
        review_scores_rating      = float(_safe(r, "review_scores_rating", 4.5)),
        review_scores_cleanliness = float(_safe(r, "review_scores_cleanliness", 4.5)),
        review_scores_location    = float(_safe(r, "review_scores_location", 4.8)),
        host_is_superhost     = bool(_safe(r, "host_is_superhost", 0)),
        host_identity_verified= bool(_safe(r, "host_identity_verified", 1)),
        host_years            = float(_safe(r, "host_years", 3.0)),
        host_response_rate    = int(_safe(r, "host_response_rate", 90)),
        host_acceptance_rate  = int(_safe(r, "host_acceptance_rate", 80)),
        resp_time             = rtime if rtime in RESP_TIMES else RESP_TIMES[0],
        host_listings         = int(_safe(r, "calculated_host_listings_count", 1)),
    )

    # These are not widget-backed — safe to set directly
    price = _safe(r, "price_numeric", None)
    st.session_state.listed_price = float(price) if price and not pd.isna(price) else None
    st.session_state.listing_name = _safe(r, "name", "Listing")
    st.session_state.lookup_error = None
    st.session_state._predict_url_pending = True
    st.rerun()

# ── Sidebar (always visible, independent of tabs) ─────────────────────────────
with st.sidebar:
    st.title("Listing Details")

    def _request_reset():
        st.session_state.reset_pending = True

    st.button("↺ Reset listing details", use_container_width=True, on_click=_request_reset)
    st.markdown("---")

    st.subheader("📍 Location & Type")
    neighbourhood = st.selectbox(
        "Neighbourhood", NEIGHBOURHOODS, key="nb",
        index=None, placeholder="Select a neighbourhood…",
        help="The area of Amsterdam the listing is in. This is one of the strongest price predictors.",
        on_change=_on_key_field_change,
    )
    with st.popover("🗺️ Pick on map", use_container_width=True):
        st.caption("Size = listing count · Colour = median price")
        map_event = st.plotly_chart(
            make_map(st.session_state.nb or "", neigh_stats),
            use_container_width=True, on_select="rerun",
            selection_mode="points", key="neigh_map",
        )
        if map_event and map_event.selection and map_event.selection.points:
            clicked = map_event.selection.points[0].get("customdata")
            if clicked and clicked in NEIGHBOURHOODS:
                st.session_state.map_nb_pending = clicked
                st.rerun()
    room_type = st.selectbox(
        "Room type", ROOM_TYPES, key="rt",
        index=None, placeholder="Select a room type…",
        help="Entire home/apt = you have the whole place to yourself. "
             "Private room = your own room, shared common areas. "
             "Shared room = you share a bedroom with others.",
        on_change=_on_key_field_change,
    )
    property_type = st.selectbox(
        "Property type", PROP_TYPES, key="pt",
        index=None, placeholder="Select a property type…",
        help="The specific type of property — e.g. apartment, house, boat. "
             "Less important than room type for pricing.",
    )

    st.subheader("🏠 Property")
    accommodates = st.slider(
        "Accommodates (guests)", 1, 16, key="accommodates",
        help="Maximum number of guests the listing fits.",
        on_change=_on_key_field_change,
    )
    bedrooms = st.number_input("Bedrooms", 0, 10, key="bedrooms", on_change=_on_key_field_change)
    beds     = st.number_input("Beds",     1, 16, key="beds",
                               help="Number of actual beds (can be more than bedrooms — e.g. sofa bed).",
                               on_change=_on_key_field_change)
    bathrooms     = st.number_input("Bathrooms", 0.5, 6.0, step=0.5, key="bathrooms")
    amenity_count = st.slider(
        "Number of amenities", 0, 100, key="amenity_count",
        help="Total count of listed amenities — things like WiFi, kitchen, washing machine, "
             "air conditioning, TV, etc. More amenities generally means a higher price.",
    )

    st.subheader("📅 Booking Rules")
    minimum_nights = st.number_input(
        "Minimum nights", 1, 365, key="minimum_nights",
        help="The shortest stay the host accepts. Many hosts set 2–3 nights. "
             "High minimums (e.g. 30 nights) reduce short-term bookings and affect availability.",
        on_change=_on_key_field_change,
    )
    maximum_nights = st.number_input(
        "Maximum nights", 1, 1125, key="maximum_nights",
        help="The longest stay allowed. Most listings set this to 365 (no limit). "
             "A low cap can mean the host is not renting long-term.",
    )
    instant_bookable = st.checkbox(
        "Instant bookable", key="instant_bookable",
        help="If ticked, guests can book immediately without waiting for host approval. "
             "Instant-bookable listings tend to get more bookings.",
    )

    st.subheader("⭐ Reviews")
    number_of_reviews = st.number_input(
        "Total reviews", 0, 2000, key="number_of_reviews",
        help="Total number of guest reviews since the listing was created. "
             "More reviews = more booking history, which the model uses as a signal.",
    )
    number_of_reviews_ltm = st.number_input(
        "Reviews (last 12 months)", 0, 500, key="number_of_reviews_ltm",
        help="Reviews in the last 12 months. A better signal of current popularity than total reviews.",
    )
    reviews_per_month = st.number_input(
        "Reviews per month", 0.0, 30.0, step=0.1, key="reviews_per_month",
        help="Average reviews per month over the listing's lifetime. Roughly 50% of stays result "
             "in a review, so multiply by ~2 to estimate actual monthly bookings.",
    )
    no_ratings = st.checkbox(
        "No ratings yet (new listing)", key="no_ratings",
        help="Tick this if the listing has no review scores yet. "
             "The model will use typical Amsterdam values instead.",
    )
    if not no_ratings:
        review_scores_rating      = st.slider("Overall rating (1–5)",  1.0, 5.0, step=0.1, key="review_scores_rating")
        review_scores_cleanliness = st.slider("Cleanliness (1–5)",     1.0, 5.0, step=0.1, key="review_scores_cleanliness")
        review_scores_location    = st.slider("Location (1–5)",        1.0, 5.0, step=0.1, key="review_scores_location")
    else:
        st.caption("Ratings will be estimated from the dataset average.")
        review_scores_rating = review_scores_cleanliness = review_scores_location = np.nan

    st.subheader("👤 Host")
    host_is_superhost = st.checkbox(
        "Superhost", key="host_is_superhost",
        help="Superhosts are experienced hosts with high ratings and fast response times. "
             "Airbnb awards this badge automatically based on performance.",
    )
    host_identity_verified = st.checkbox("Identity verified", key="host_identity_verified",
                                         help="Host has verified their ID with Airbnb.")
    host_years = st.slider(
        "Years hosting", 0.0, 20.0, step=0.5, key="host_years",
        help="How long the host has been on Airbnb. Calculated from their join date.",
    )
    unknown_rates = st.checkbox(
        "Host rates not shown / unknown", key="unknown_rates",
        help="Tick if response rate or acceptance rate aren't visible on the listing. "
             "The model will use typical host values.",
    )
    if not unknown_rates:
        host_response_rate   = st.slider("Response rate (%)",   0, 100, key="host_response_rate",
                                         help="% of enquiries the host responds to.")
        host_acceptance_rate = st.slider("Acceptance rate (%)", 0, 100, key="host_acceptance_rate",
                                         help="% of booking requests the host accepts.")
        host_response_time   = st.selectbox(
            "Response time", RESP_TIMES, key="resp_time",
            help="How quickly the host typically replies to messages.",
        )
    else:
        st.caption("Host rates will be estimated from the dataset average.")
        host_response_rate = host_acceptance_rate = np.nan
        host_response_time = "within a few hours"
    calculated_host_listings_count = st.number_input(
        "Host's total listings", 1, 200, key="host_listings",
        help="How many listings this host manages on Airbnb. "
             "Professional hosts with many listings behave differently from occasional renters.",
    )

    st.markdown("---")
    st.markdown("**Option 2 — Predict any hypothetical listing:**")
    st.caption("Uses the values above. No URL needed — no listed-price comparison.")
    predict_manual_btn = st.button(
        "🔮 Predict manual listing",
        use_container_width=True, type="primary", key="manual_predict",
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏠 Amsterdam Airbnb Stay Value Predictor")
tab_pred, tab_dash = st.tabs(["🔮 Predictor", "📊 Model Dashboard"])

# ── Tab 1: Predictor ──────────────────────────────────────────────────────────
with tab_pred:
    predict_url_btn = False
    _lookup_blocked = False
    _show_comparison = False
    _predict_url_pending = st.session_state.pop("_predict_url_pending", False)
    if _predict_url_pending:
        _show_comparison = st.session_state.listed_price is not None

    st.markdown("**Option 1 — Paste an Airbnb URL to check if it's fairly priced:**")
    with st.form("url_form", clear_on_submit=False):
        col_url, col_pred, col_clr = st.columns([5, 1.4, 1])
        with col_url:
            url_input = st.text_input(
                "URL",
                placeholder="https://www.airbnb.com/rooms/12345678",
                label_visibility="collapsed",
                key="url_input",
            )
        with col_pred:
            predict_url_btn = st.form_submit_button(
                "🔮 Predict from URL", type="primary", use_container_width=True
            )
        with col_clr:
            clear_btn = st.form_submit_button("✕ Clear", use_container_width=True)

    if clear_btn:
        st.session_state.clear_pending = True
        st.rerun()

    if predict_url_btn:
        if url_input.strip():
            # do_lookup sets _lookup_prefill + _predict_url_pending and calls st.rerun() on success,
            # or sets lookup_error directly on failure (no rerun needed).
            do_lookup(url_input.strip())
            # Reaches here only on failure (rerun was not called)
            _lookup_blocked = True
        else:
            st.info("Paste an Airbnb listing URL above, then click **Predict from URL**. "
                    "Or use **🔮 Predict manual listing** in the left sidebar.")

    if st.session_state.lookup_error:
        st.warning(st.session_state.lookup_error)

    if st.session_state.listed_price is not None:
        st.success(f"**Found:** {st.session_state.listing_name} · €{st.session_state.listed_price:.0f}/night")

    st.markdown("---")

    st.markdown(
        "Trained on **10,480 Inside Airbnb listings** from Amsterdam (Sept 2025). "
        "Paste a listing URL above to check if it is fairly priced, or fill in details manually."
    )

    with st.expander("🔬 How the models work", expanded=False):
        st.markdown("""
        ### Architecture
        Models trained offline on the raw Inside Airbnb CSV. Pipeline:
        1. **Clean** — parse price string, encode booleans, parse bathrooms, compute host seniority.
        2. **Engineer** — amenity count, beds-per-person, composite review score, multi-listing flag.
        3. **No leakage** — `availability_30/60/90/365` excluded from the availability model.
        4. **Pipeline** — `SimpleImputer → StandardScaler` for numerics; `SimpleImputer → OneHotEncoder` for categoricals — fitted on train only.
        5. **XGBoost** — tuned with 5-fold `GridSearchCV`.

        | Task | Metric | Baseline | XGBoost |
        |---|---|---|---|
        | Price regression | R² | 0.00 | **0.63** |
        | Price regression | MAE | €100 | **€63** |
        | Availability classification | ROC-AUC | 0.50 | **0.70** |
        """)

    if (predict_url_btn and not _lookup_blocked) or predict_manual_btn or _predict_url_pending:
        ss = st.session_state
        _no_ratings    = ss.get("no_ratings", False)
        _unknown_rates = ss.get("unknown_rates", False)
        _acc  = ss.get("accommodates", 2)
        _beds = ss.get("beds", 1)
        _hlc  = ss.get("host_listings", 1)
        _rsr  = np.nan if _no_ratings else ss.get("review_scores_rating", 4.5)
        _rsc  = np.nan if _no_ratings else ss.get("review_scores_cleanliness", 4.5)
        _rsl  = np.nan if _no_ratings else ss.get("review_scores_location", 4.8)
        _hrr  = np.nan if _unknown_rates else float(ss.get("host_response_rate", 90))
        _har  = np.nan if _unknown_rates else float(ss.get("host_acceptance_rate", 80))

        _r_scores = [_rsr, _rsc, _rsl]
        review_composite = (float(np.nanmean([v for v in _r_scores if not (isinstance(v, float) and np.isnan(v))]))
                            if any(not (isinstance(v, float) and np.isnan(v)) for v in _r_scores) else np.nan)

        X_input = pd.DataFrame([{
            "accommodates":   _acc,
            "bathrooms":      ss.get("bathrooms", 1.0),
            "bedrooms":       ss.get("bedrooms", 1),
            "beds":           _beds,
            "minimum_nights": ss.get("minimum_nights", 2),
            "maximum_nights": ss.get("maximum_nights", 365),
            "number_of_reviews":     ss.get("number_of_reviews", 20),
            "number_of_reviews_ltm": ss.get("number_of_reviews_ltm", 5),
            "reviews_per_month":     ss.get("reviews_per_month", 0.5),
            "review_composite":          review_composite,
            "review_scores_rating":      _rsr,
            "review_scores_cleanliness": _rsc,
            "review_scores_location":    _rsl,
            "host_years":            ss.get("host_years", 3.0),
            "host_response_rate":    _hrr,
            "host_acceptance_rate":  _har,
            "calculated_host_listings_count": _hlc,
            "amenity_count":  ss.get("amenity_count", 30),
            "beds_per_person": _beds / _acc if _acc > 0 else 1.0,
            "host_is_superhost":       int(ss.get("host_is_superhost", False)),
            "host_identity_verified":  int(ss.get("host_identity_verified", True)),
            "instant_bookable":        int(ss.get("instant_bookable", False)),
            "multi_listing_host":      int(_hlc > 1),
            "neighbourhood_cleansed":  ss.get("nb"),
            "room_type":               ss.get("rt"),
            "property_type":           ss.get("pt"),
            "host_response_time":      ss.get("resp_time"),
        }])

        price_pred  = float(np.expm1(price_model.predict(X_input)[0]))
        avail_proba = float(avail_model.predict_proba(X_input)[0, 1])

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        listed = st.session_state.listed_price if _show_comparison else None

        if listed is not None:
            diff     = listed - price_pred
            pct_diff = diff / price_pred * 100

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Model's fair price", f"€{price_pred:.0f}")
            c2.metric("🏷️ Listed price",        f"€{listed:.0f}")
            c3.metric("📊 Difference", f"€{diff:+.0f}", delta=f"{pct_diff:+.0f}%",
                      delta_color="inverse")

            if pct_diff > 15:
                verdict, color = "🔴 Overpriced", "#e53e3e"
                msg = f"Listed at <b>€{listed:.0f}</b> — <b>{pct_diff:.0f}% above</b> what similar listings charge. Fair value is around €{price_pred:.0f}."
            elif pct_diff < -15:
                verdict, color = "🟢 Great deal", "#38a169"
                msg = f"Listed at <b>€{listed:.0f}</b> — <b>{abs(pct_diff):.0f}% below</b> fair value of €{price_pred:.0f}. Looks like a good deal — book fast."
            else:
                verdict, color = "🟡 Fairly priced", "#d69e2e"
                msg = f"Listed at <b>€{listed:.0f}</b>, model says <b>€{price_pred:.0f}</b> — within 15% of fair value."

            c4.metric("Verdict", verdict)
            st.markdown(
                f"<div style='background:{color}22;border-left:4px solid {color};"
                f"padding:12px 16px;border-radius:4px;margin-top:8px'>{msg}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            a1, a2 = st.columns(2)
            a1.metric("📅 Availability",
                      "Heavily booked 🔴" if avail_proba >= 0.5 else "Likely available 🟢")
            a2.metric("🎯 Booking probability", f"{avail_proba:.0%}",
                      help="Probability this listing has ≤90 open days/year")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("💰 Estimated price", f"€{price_pred:.0f}")
            c2.metric("📅 Availability",
                      "Heavily booked 🔴" if avail_proba >= 0.5 else "Likely available 🟢")
            c3.metric("🎯 Booking probability", f"{avail_proba:.0%}")

        st.markdown("---")
        pa, pb = st.columns(2)
        with pa:
            st.markdown("#### Price context")
            st.progress(float(min(price_pred / 500, 1.0)))
            tier = ("Budget" if price_pred < 80 else "Mid-range" if price_pred < 160
                    else "Premium" if price_pred < 300 else "Luxury")
            st.caption(f"**{tier}** — Amsterdam median ≈ €130/night.")
        with pb:
            st.markdown("#### Availability context")
            st.progress(float(avail_proba))
            if avail_proba >= 0.6:
                st.caption("Typical of heavily-booked listings — book early.")
            elif avail_proba <= 0.35:
                st.caption("Plenty of open dates expected.")
            else:
                st.caption("Moderate — check the calendar directly.")

    else:
        st.info("**Option 1:** Paste an Airbnb URL above and click **Predict from URL** to check if it's fairly priced against our model.  \n"
                "**Option 2:** Fill in the sidebar and click **🔮 Predict manual listing** to estimate any hypothetical listing.")

# ── Tab 2: Model Dashboard ────────────────────────────────────────────────────
with tab_dash:
    st.markdown("## About this project")
    st.markdown(
        "This app is built on a capstone machine learning project analysing **10,480 Amsterdam Airbnb listings** "
        "scraped in September 2025 from [Inside Airbnb](https://insideairbnb.com/). "
        "Two supervised learning tasks were tackled: predicting nightly price (regression) and predicting "
        "booking demand (classification). Both use gradient-boosted decision trees (XGBoost) inside a "
        "scikit-learn pipeline."
    )

    # ── Dataset overview ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Dataset overview")
    ds = dataset_stats(lookup_df)
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Listings", f"{ds['n_listings']:,}")
    d2.metric("Neighbourhoods", ds["n_neigh"])
    d3.metric("Median price/night", f"€{ds['price_median']:.0f}")
    d4.metric("5th–95th price range", f"€{ds['price_min']:.0f} – €{ds['price_max']:.0f}")
    d5.metric("Heavily booked", f"{ds['pct_booked']:.0%}")

    st.caption("'Heavily booked' = ≤90 open days/year (availability_365 ≤ 90).")

    # Price distribution
    prices = lookup_df["price_numeric"].dropna()
    prices_clipped = prices[prices <= 600]
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=prices_clipped, nbinsx=60,
        marker_color="#4a6cf7", opacity=0.8,
        name="Listings",
    ))
    fig_hist.add_vline(
        x=float(prices.median()), line_dash="dash", line_color="#e53e3e",
        annotation_text=f"Median €{prices.median():.0f}", annotation_position="top right",
    )
    fig_hist.update_layout(
        title="Nightly price distribution (capped at €600 for readability)",
        xaxis_title="Price per night (€)", yaxis_title="Number of listings",
        margin=dict(l=0, r=0, t=40, b=0), height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Neighbourhood breakdown
    nb_summary = (
        lookup_df.groupby("neighbourhood_cleansed")
        .agg(listings=("id","count"), median_price=("price_numeric","median"),
             pct_booked=("low_availability","mean"))
        .reset_index()
        .sort_values("median_price", ascending=False)
        .rename(columns={
            "neighbourhood_cleansed": "Neighbourhood",
            "listings": "Listings",
            "median_price": "Median €/night",
            "pct_booked": "% Heavily booked",
        })
    )
    nb_summary["Median €/night"] = nb_summary["Median €/night"].round(0).astype(int)
    nb_summary["% Heavily booked"] = (nb_summary["% Heavily booked"] * 100).round(1).astype(str) + "%"
    with st.expander("Neighbourhood breakdown table", expanded=False):
        st.dataframe(nb_summary.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Model pipeline ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Model pipeline")
    st.markdown("""
Both models share the same preprocessing pipeline — only the final estimator differs.

| Step | Numerics | Categoricals |
|------|----------|--------------|
| Imputation | `SimpleImputer(median)` | `SimpleImputer(most_frequent)` |
| Encoding | `StandardScaler` | `OneHotEncoder(handle_unknown='ignore')` |
| Estimator | XGBoost | XGBoost |

**Feature engineering** applied before the pipeline:
- `amenity_count` — count of listed amenities parsed from the raw JSON array
- `beds_per_person` — beds ÷ accommodates
- `review_composite` — average of the three review sub-scores
- `multi_listing_host` — binary flag for hosts managing more than one listing
- `host_years` — years since host join date (from the scrape date)

**Train / test split:** 85% train · 15% test (stratified by neighbourhood). Hyperparameters chosen by 5-fold GridSearchCV on the training set only.
    """)

    # ── Regression: Price Prediction ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Regression — Price prediction")
    st.markdown(
        "The target is the **log-transformed nightly price** (`log1p`). Predictions are "
        "back-transformed with `expm1` before display. Log-transforming compresses the long "
        "right tail of the price distribution and makes the residuals more normally distributed."
    )

    r1, r2 = st.columns(2)
    with r1:
        st.markdown("#### Model leaderboard")
        reg_leaderboard = pd.DataFrame({
            "Model": ["Dummy (median baseline)", "XGBoost (tuned)"],
            "R²": [0.00, 0.63],
            "MAE (€)": [100, 63],
            "RMSE (€)": ["—", 107],
        })
        st.dataframe(reg_leaderboard, use_container_width=True, hide_index=True)
        st.caption(
            "R² of 0.63 means the model explains 63% of the variance in nightly price. "
            "MAE of €63 is the average absolute prediction error on held-out listings."
        )

    with r2:
        st.markdown("#### Top feature importances")
        fi_reg = get_feature_importance(price_model)
        if not fi_reg.empty:
            fig_reg = go.Figure(go.Bar(
                x=fi_reg["importance"], y=fi_reg["feature"],
                orientation="h", marker_color="#4a6cf7",
            ))
            fig_reg.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance (gain)", yaxis_title="",
                margin=dict(l=0, r=0, t=0, b=0), height=380,
            )
            st.plotly_chart(fig_reg, use_container_width=True)
        else:
            st.info("Feature importances could not be extracted from this model.")

    # ── Classification: Booking Demand ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Classification — Booking demand")
    st.markdown(
        "The target is **`low_availability`**: 1 if the listing has ≤90 open days/year "
        "(heavily booked), 0 otherwise. This binary label was derived from `availability_365`. "
        "Note: `availability_365` itself is excluded from the feature set to avoid data leakage."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Model leaderboard")
        cls_leaderboard = pd.DataFrame({
            "Model": ["Dummy (stratified baseline)", "XGBoost (tuned)"],
            "ROC-AUC": [0.50, 0.70],
            "Accuracy": ["—", "67%"],
        })
        st.dataframe(cls_leaderboard, use_container_width=True, hide_index=True)
        st.caption(
            "ROC-AUC of 0.70 means the model correctly ranks a booked listing above an "
            "available one 70% of the time. Booking demand is harder to predict than price "
            "because many relevant signals (seasonality, photos, host communication speed) "
            "aren't in the dataset."
        )

    with c2:
        st.markdown("#### Top feature importances")
        fi_cls = get_feature_importance(avail_model)
        if not fi_cls.empty:
            fig_cls = go.Figure(go.Bar(
                x=fi_cls["importance"], y=fi_cls["feature"],
                orientation="h", marker_color="#38a169",
            ))
            fig_cls.update_layout(
                yaxis=dict(autorange="reversed"),
                xaxis_title="Importance (gain)", yaxis_title="",
                margin=dict(l=0, r=0, t=0, b=0), height=380,
            )
            st.plotly_chart(fig_cls, use_container_width=True)
        else:
            st.info("Feature importances could not be extracted from this model.")

    # ── Key findings ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Key findings")
    st.markdown("""
- **Neighbourhood is the strongest price driver.** Centrum-West, Centrum-Oost, and De Pijp command the highest premiums.
- **Room type matters more than property type.** Entire homes/apartments cost roughly 2× private rooms on average.
- **Amenity count has a significant positive effect on price**, even after controlling for size.
- **Review velocity (reviews per month) is the top signal for booking demand**, suggesting that listing activity and recency drive bookings more than headline rating scores.
- **Superhost status has a modest positive effect** on both price and demand — consistent with the badge signalling reliability.
- **Log-transforming price** before training reduced MAE by ~12% compared to modelling on the raw scale.
    """)

# ── Explore Amsterdam (outside tabs) ─────────────────────────────────────────
st.markdown("---")

st.markdown("### 🌷 Explore Amsterdam")
st.markdown("Planning a trip or curious about the market? Here are some useful resources.")

_col1, _col2, _col3 = st.columns(3)
with _col1:
    st.markdown(
        "**🏠 Browse Airbnbs**\n\n"
        "Search all Amsterdam listings directly on Airbnb — filter by neighbourhood, dates, and price.\n\n"
        "[Open on Airbnb →](https://www.airbnb.com/s/Amsterdam--Netherlands/homes)"
    )
with _col2:
    st.markdown(
        "**📊 Inside Airbnb Data**\n\n"
        "The open dataset powering this app. Includes an interactive map of all Amsterdam listings.\n\n"
        "[Explore the data →](https://insideairbnb.com/amsterdam/)"
    )
with _col3:
    st.markdown(
        "**🗺️ Amsterdam Neighbourhoods**\n\n"
        "Official Amsterdam tourism guide — where to stay, what each area feels like, and tips.\n\n"
        "[Visit I Amsterdam →](https://www.iamsterdam.com/en/plan-your-trip/where-to-stay)"
    )

_col4, _col5, _col6 = st.columns(3)
with _col4:
    st.markdown(
        "**🍺 What to Do & See**\n\n"
        "Time Out's curated guide to Amsterdam — neighbourhoods, restaurants, museums, and more.\n\n"
        "[Time Out Amsterdam →](https://www.timeout.com/amsterdam)"
    )
with _col5:
    st.markdown(
        "**🚲 Getting Around**\n\n"
        "GVB is Amsterdam's public transport. Plan your route between neighbourhoods by tram, metro, or ferry.\n\n"
        "[GVB Journey Planner →](https://www.gvb.nl/en)"
    )
with _col6:
    st.markdown(
        "**🌐 Amsterdam City Info**\n\n"
        "Official City of Amsterdam portal — events, districts, regulations, and visitor information.\n\n"
        "[amsterdam.nl →](https://www.amsterdam.nl/en/)"
    )

st.markdown("---")
st.caption("Manca Levašič · MADA Capstone 2025 · "
           "Data: [Inside Airbnb](https://insideairbnb.com/) (CC-BY 4.0) · "
           "Models: scikit-learn + XGBoost")

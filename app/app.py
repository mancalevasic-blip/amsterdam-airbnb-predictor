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
    """Clear the listed-price comparison whenever the user edits a key field."""
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
    # Push values into widget keys — Streamlit will pick them up on next render
    nb = _safe(r, "neighbourhood_cleansed", NEIGHBOURHOODS[0])
    rt = _safe(r, "room_type", ROOM_TYPES[0])
    pt = _safe(r, "property_type", PROP_TYPES[0])
    st.session_state.nb  = nb if nb in NEIGHBOURHOODS else NEIGHBOURHOODS[0]
    st.session_state.rt  = rt if rt in ROOM_TYPES     else ROOM_TYPES[0]
    st.session_state.pt  = pt if pt in PROP_TYPES      else PROP_TYPES[0]
    st.session_state.accommodates         = int(_safe(r, "accommodates", 2))
    st.session_state.bedrooms             = int(_safe(r, "bedrooms", 1))
    st.session_state.beds                 = int(_safe(r, "beds", 1))
    st.session_state.bathrooms            = float(_safe(r, "bathrooms", 1.0))
    st.session_state.amenity_count        = int(_safe(r, "amenity_count", 30))
    st.session_state.minimum_nights       = int(_safe(r, "minimum_nights", 2))
    st.session_state.maximum_nights       = int(_safe(r, "maximum_nights", 365))
    st.session_state.instant_bookable     = bool(_safe(r, "instant_bookable", 0))
    st.session_state.number_of_reviews    = int(_safe(r, "number_of_reviews", 20))
    st.session_state.number_of_reviews_ltm= int(_safe(r, "number_of_reviews_ltm", 5))
    st.session_state.reviews_per_month    = float(_safe(r, "reviews_per_month", 0.5))
    st.session_state.review_scores_rating      = float(_safe(r, "review_scores_rating", 4.5))
    st.session_state.review_scores_cleanliness = float(_safe(r, "review_scores_cleanliness", 4.5))
    st.session_state.review_scores_location    = float(_safe(r, "review_scores_location", 4.8))
    st.session_state.host_is_superhost    = bool(_safe(r, "host_is_superhost", 0))
    st.session_state.host_identity_verified = bool(_safe(r, "host_identity_verified", 1))
    st.session_state.host_years           = float(_safe(r, "host_years", 3.0))
    st.session_state.host_response_rate   = int(_safe(r, "host_response_rate", 90))
    st.session_state.host_acceptance_rate = int(_safe(r, "host_acceptance_rate", 80))
    rtime = _safe(r, "host_response_time", RESP_TIMES[0])
    st.session_state.resp_time  = rtime if rtime in RESP_TIMES else RESP_TIMES[0]
    st.session_state.host_listings = int(_safe(r, "calculated_host_listings_count", 1))

    price = _safe(r, "price_numeric", None)
    st.session_state.listed_price = float(price) if price and not pd.isna(price) else None
    st.session_state.listing_name = _safe(r, "name", "Listing")
    st.session_state.lookup_error = None
    # Snapshot of key fields to detect manual edits later
    st.session_state.prefill_snap = dict(
        nb=st.session_state.nb, rt=st.session_state.rt,
        accommodates=st.session_state.accommodates,
        bedrooms=st.session_state.bedrooms, beds=st.session_state.beds,
        minimum_nights=st.session_state.minimum_nights,
    )

def do_clear():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🏠 Amsterdam Airbnb Stay Value Predictor")

col_url, col_pred, col_clr = st.columns([5, 1, 1])
with col_url:
    url_input = st.text_input(
        "Paste an Airbnb listing URL — or fill in the sidebar — then hit Predict:",
        placeholder="https://www.airbnb.com/rooms/12345678",
        label_visibility="visible",
        key="url_input",
    )
with col_pred:
    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🔮 Predict", use_container_width=True,
                                type="primary", key="main_predict")
with col_clr:
    st.markdown("<br>", unsafe_allow_html=True)
    st.button("✕ Clear", use_container_width=True, on_click=do_clear)

# If Predict was clicked with a URL, do the lookup NOW — before sidebar renders —
# so the sidebar widgets pick up the fresh session-state values this same rerun.
_lookup_blocked = False
if predict_clicked and url_input.strip():
    do_lookup(url_input.strip())
    if st.session_state.lookup_error:
        _lookup_blocked = True  # listing not found → block prediction

if st.session_state.lookup_error:
    st.warning(st.session_state.lookup_error)

# Banner — shown whenever a listing was matched (and user hasn't cleared it)
if st.session_state.listed_price is not None:
    st.success(f"**Found:** {st.session_state.listing_name} · €{st.session_state.listed_price:.0f}/night")

st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Listing Details")

    def _request_reset():
        st.session_state.reset_pending = True

    st.button("↺ Reset listing details", use_container_width=True, on_click=_request_reset)
    st.markdown("---")

    # ── Location & Type ───────────────────────────────────────────────────────
    st.subheader("📍 Location & Type")

    neighbourhood = st.selectbox(
        "Neighbourhood",
        NEIGHBOURHOODS, key="nb",
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

    # ── Property ──────────────────────────────────────────────────────────────
    st.subheader("🏠 Property")
    accommodates = st.slider(
        "Accommodates (guests)", 1, 16, key="accommodates",
        help="Maximum number of guests the listing fits.",
        on_change=_on_key_field_change,
    )
    bedrooms = st.number_input("Bedrooms",  0, 10, key="bedrooms",
                               on_change=_on_key_field_change)
    beds     = st.number_input("Beds",      1, 16, key="beds",
                               help="Number of actual beds (can be more than bedrooms — e.g. sofa bed).",
                               on_change=_on_key_field_change)
    bathrooms = st.number_input("Bathrooms", 0.5, 6.0, step=0.5, key="bathrooms")
    amenity_count = st.slider(
        "Number of amenities", 0, 100, key="amenity_count",
        help="Total count of listed amenities — things like WiFi, kitchen, washing machine, "
             "air conditioning, TV, etc. More amenities generally means a higher price.",
    )

    # ── Booking Rules ─────────────────────────────────────────────────────────
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

    # ── Reviews ───────────────────────────────────────────────────────────────
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

    # ── Host ──────────────────────────────────────────────────────────────────
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



# ── Description ───────────────────────────────────────────────────────────────
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

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked and not _lookup_blocked:
    # Always read from session state so that a same-click URL lookup is reflected
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

    listed = st.session_state.listed_price   # None if user edited / cleared

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
    st.info("Paste a listing URL above and click **Predict** to check if it's fairly priced — or fill in the sidebar manually and click **Predict** to estimate any listing.")

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

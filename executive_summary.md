# Executive Summary — Amsterdam Airbnb Stay Value Predictor

**Manca Levašič · MADA Capstone · June 2026**

---

## The Question

I am planning a trip to Amsterdam. When comparing Airbnb listings, two questions always come up: *Is this listing fairly priced?* and *Is it likely to be available when I want to book?* This project builds machine-learning models to answer both questions from publicly available listing data.

## What We Did

We used the Inside Airbnb dataset for Amsterdam — 10,480 listings scraped in September 2025 — to train two predictive models:

1. **Price model (regression):** Predicts the nightly price of a listing based on its location, size, amenities, host characteristics, and review profile.
2. **Availability model (classification):** Predicts whether a listing is likely to have low availability — defined as 90 or fewer free days in the coming year — which serves as a proxy for booking demand.

## What We Found

**Price:** Our best model (XGBoost) explains **63% of the variation in nightly price** and is on average **€63 off** per listing. The most important factors are room type (entire home vs. private room), how many guests the listing accommodates, neighbourhood, and amenity count. This is practically useful: a traveller can plug in a listing's details and see whether the listed price is above or below the model's estimate.

**Availability:** Our best model achieves a **ROC-AUC of 0.70**, meaningfully better than guessing (0.50). It identifies listings whose profile — high minimum nights, high review volume, central neighbourhoods — tends to be associated with low availability. This helps travellers prioritise listings to look at quickly.

## What We Recommend

- Travellers comparing Amsterdam listings can use the app as a sanity check: if the model predicts €90/night and the listing charges €200, that is worth investigating (or worth paying for, if the extras justify it).
- Listings in Centrum-West and Oud-West with entire-home room type and ≥ 10 amenities tend to both command a price premium and be heavily booked — book early.
- Do not treat the availability prediction as certain: hosts can manually block dates, and the model cannot distinguish blocked calendars from fully-booked listings.

## Honest Limitations

The regression model has a 37% unexplained variance — factors like specific interior style, view, or host personality are not in the data. The availability model is moderately accurate but imperfect; it should be used as a signal, not a guarantee. Both models are based on a single September 2025 snapshot and do not capture seasonal patterns (summer vs. winter demand in Amsterdam differs substantially).

---

*Full analysis report, code, and live app: see the GitHub repository linked in the submission.*

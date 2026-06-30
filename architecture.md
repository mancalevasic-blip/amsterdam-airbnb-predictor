## Architecture — Airbnb Stay Value Predictor (Amsterdam)

### Mermaid Source

```mermaid
flowchart LR
    subgraph Offline["🧪 Offline — done once on local machine"]
        direction TB
        DATA[("📦 Inside Airbnb\nAmsterdam listings.csv\n~10,480 rows · 79 cols")]
        CLEAN["🔧 Clean + Feature Engineering\n(price_numeric, low_availability,\ndrop leakage columns)"]
        TRAIN_R["📈 Train Regression Model\n(predict nightly price)"]
        TRAIN_C["📊 Train Classification Model\n(predict low_availability)"]
        MODEL_R["💾 price_model.pkl"]
        MODEL_C["💾 avail_model.pkl"]

        DATA --> CLEAN
        CLEAN --> TRAIN_R
        CLEAN --> TRAIN_C
        TRAIN_R --> MODEL_R
        TRAIN_C --> MODEL_C
    end

    subgraph Runtime["☁️ Runtime — live app"]
        direction TB
        BE["⚙️ Backend / Streamlit App\n(loads models at startup)"]
        FE["🖥️ Frontend UI\n(input form + results panel)"]
        USER(["🙋 User"])

        MODEL_R -->|loaded once at startup| BE
        MODEL_C -->|loaded once at startup| BE
        USER --> FE
        FE -->|listing features| BE
        BE -->|estimated price + availability label| FE
    end

classDef offlineNode fill:#f0f4ff,stroke:#4a6cf7,color:#1a1a2e
classDef runtimeNode fill:#f0fff4,stroke:#38a169,color:#1a2e1a
classDef modelFile fill:#fff8e1,stroke:#d69e2e,color:#2d1a00
classDef userNode fill:#fef2f2,stroke:#e53e3e,color:#2d0000

class DATA,CLEAN,TRAIN_R,TRAIN_C offlineNode
class BE,FE runtimeNode
class MODEL_R,MODEL_C modelFile
class USER userNode
```

---

### Written Explanation

Both models are trained offline, once, on my local machine using the downloaded Inside Airbnb Amsterdam listings CSV. During training, the raw data is cleaned and engineered into features — including deriving `price_numeric` and the `low_availability` binary label — and two separate models are fitted: one regression model for nightly price and one classification model for availability. Each fitted pipeline is saved to disk as a `.pkl` file (`price_model.pkl` and `avail_model.pkl`). The Streamlit app loads both model files once at startup, so no training happens at runtime. When a user fills in listing details in the frontend form and submits, the backend calls `.predict()` on the pre-loaded models and returns the estimated price and the availability prediction to the frontend — inference is fast because the models are already in memory.
```

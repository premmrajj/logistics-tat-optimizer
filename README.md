# FMCG District Logistics & Inbound Optimization Control Tower

[![Live Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://logistics-tat-optimizer-mknnbqcxxunf2vhj3vknit.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange.svg)](#)
[![Operations Research](https://img.shields.io/badge/Operations%20Research-Discrete--Event%20Simulation-purple.svg)](#)
[![Monte Carlo](https://img.shields.io/badge/Simulation-1%2C000%20Stochastic%20Trials-green.svg)](#)

> 🚀 **Live Interactive Web Application:** [**Launch FMCG Logistics Control Tower App**](https://logistics-tat-optimizer-mknnbqcxxunf2vhj3vknit.streamlit.app/)

An advanced, production-grade Industrial Engineering and Predictive Analytics system engineered for the **Graduate Engineer Trainee (GET) – Logistics & Warehousing (Level 6)** role in large-scale multi-tier distribution networks (e.g., ITC, HUL, Britannia, Amazon).

---

## 🌟 What Makes This an Advanced Project?

Rather than a simple static calculation or basic descriptive dashboard, this system implements **four advanced engineering pillars**:

1. **Predictive Machine Learning Exception Engine (`predictive_model.py`):**
   * Pre-trained Random Forest models predict continuous **highway transit delay ($\Delta TIT$)** and **Demurrage Breach Probability (ROC-AUC = 0.76)** at dispatch time before a truck even departs the CDC.
2. **Operations Research & Dynamic Dock Scheduling (`tat_simulator.py`):**
   * Discrete-event simulation comparing uncontrolled FCFS against **Staggered Slot Booking (ASN)**, **Material Handling Equipment (MHE) Palletization**, and **Earliest Due Date (EDD)** dispatching.
   * Visualized via an interactive **Plotly Gantt timeline**.
3. **Multi-Echelon Stochastic Monte Carlo Simulation (`monte_carlo_inventory.py`):**
   * Runs **1,000 stochastic trials per SKU** under joint highway lead-time disruptions (log-normal) and demand spikes to compute **Value-at-Risk (VaR 95%)** and stockout probability densities.
4. **Interactive Streamlit Web Application (`app.py`):**
   * Deployable to **Streamlit Community Cloud** with live parameter sliders, real-time ML inference, risk gauges, and transporter scorecards.

---

## 📊 Quantified Operational Results

Evaluated over **1,200 primary (K2) truck trips** into a 2-dock regional branch warehouse (WSP) across 90 days:

| Key Performance Indicator (KPI) | Baseline (Uncontrolled FCFS) | Optimized (Slotting + EDD Priority) | Impact / Savings |
| :--- | :---: | :---: | :---: |
| **Average Truck TAT (Gate-in to Gate-out)** | **15.07 Hours** | **4.11 Hours** | **▼ 72.7% Reduction** |
| **Average Yard Waiting Time** | 12.36 Hours | 2.00 Hours | ▼ 83.8% Reduction |
| **Trucks Incurring Demurrage (> 6.0 Hrs)** | 558 Trucks (46.5%) | 241 Trucks (20.1%) | ▼ 56.8% Reduction |
| **Total Demurrage Penalties Incurred** | **₹37,59,102** | **₹3,12,469** | **₹34.46 Lakhs Saved (▼ 91.7%)** |
| **TAT for Critical Stockout SKUs** | 13.66 Hours | 3.77 Hours | ▼ 72.4% Faster Dock Turn |
| **Monte Carlo Simulated Stockout Probability** | 42.9% (Static Norm) | **19.9% (Dynamic Model)** | **▼ 23.0% Risk Reduction** |
| **Highway TIT Variance Buffer Premium** | Zero (Ignored) | Captured ($+102.2\%$ Buffer) | Eliminates WD Stockout Spikes |
| **Projected Network OTIF Service Level** | 89.2% | **96.8%** | **▲ Above 95% FMCG Benchmark** |

---

## 🏭 Industry Context & Terminology (JD Mapping)

* **K1 / K2 / K3:** Primary Factory $\rightarrow$ Central DC (K1), Inter-depot CDC $\rightarrow$ Branch Warehouse (K2), Branch $\rightarrow$ Wholesale Distributor (K3).
* **WSP:** Warehouse Service Provider (3PL Godown operator).
* **WD:** Wholesale Distributor (regional distributor holding localized market buffer).
* **TIT:** Time In Transit (highway transit duration).
* **TAT:** Truck Turnaround Time (Gate-In $\rightarrow$ Weighbridge $\rightarrow$ Unloading $\rightarrow$ Gate-Out).
* **Demurrage:** Financial detention fee charged by transporters when unloading exceeds free window (6.0 hours).
* **OTIF:** On-Time In-Full customer fulfillment service level.

---

## 🚀 How to Run Locally & Deploy

### Run Locally on Streamlit:
Double-click `run_app.bat` or run:
```bash
streamlit run app.py
```
Your browser will open automatically at `http://localhost:8501`.

---

### How to Deploy to Streamlit Cloud (FREE Public URL for Your Resume):
1. **Push this repository to your GitHub account:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: FMCG Logistics Control Tower"
   git remote add origin https://github.com/<your-username>/logistics-tat-inventory-optimizer.git
   git push -u origin main
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
3. Click **"New app"**, select your repository, set the main file path to `app.py`, and click **Deploy**!
4. You will get a live public link (e.g., `https://fmcg-logistics-optimizer.streamlit.app`) to put at the top of your resume and LinkedIn profile!

---

## 📝 Advanced Resume Bullets (CAR Framework)

> **FMCG District Logistics Control Tower & Inbound Optimization System** | *Python, Streamlit, Scikit-Learn, Operations Research*
> * Engineered an end-to-end inbound logistics optimization platform for a regional branch warehouse (WSP) processing 1,200+ K2 primary shipments, modeling dock queues, MHE pallet handling, and transporter demurrage.
> * Trained Machine Learning models (Random Forest, ROC-AUC = 0.76) predicting continuous highway transit delay ($\Delta TIT$) and yard demurrage breach risk at dispatch time for proactive exception handling.
> * Formulated an Advance Shipping Notice (ASN) slot-booking and Earliest Due Date (EDD) dispatching algorithm, reducing **average Truck Turnaround Time (TAT) by 72.7% (15.1h to 4.1h)** and saving **₹34.46 Lakhs in demurrage penalties (91.7% cost reduction)**.
> * Implemented a **1,000-trial Stochastic Monte Carlo simulation** under joint highway lead-time disruptions and demand volatility, slashing Wholesale Distributor stockout risk from **42.9% to 19.9%** while achieving **96.8% projected OTIF**.
> * Deployed a full-stack **Streamlit interactive application** featuring live ML inference, Plotly Gantt dock timelines, and transporter scorecards.

"""
monte_carlo_inventory.py
Advanced Multi-Echelon Stochastic Monte Carlo Simulation.
Simulates 1,000 replenishment cycles under joint transit time & demand volatility
to calculate Value-at-Risk (VaR), Stockout Probability Density, and Freshness Decay.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

def run_monte_carlo_simulation(n_trials=1000):
    print(f"[Monte Carlo] Running {n_trials} stochastic supply-chain replenishment trials...")
    wd_file = os.path.join(DATA_DIR, "sku_inventory_wd.csv")
    df = pd.read_csv(wd_file)

    results = []

    for _, row in df.iterrows():
        sku_name = row["sku_name"]
        wd_id = row["wd_id"]
        mean_demand = row["daily_demand_mean"]
        std_demand = row["daily_demand_std"]
        mean_tit = row["replenishment_tit_mean_days"]
        std_tit = row["replenishment_tit_std_days"]
        shelf_life = row["shelf_life_days"]
        norm_days = row["norm_inventory_days"]

        # Baseline Static Safety Stock vs Dynamic Safety Stock
        ss_static = int(np.ceil(1.645 * std_demand * np.sqrt(mean_tit)))
        variance_term = (mean_tit * (std_demand ** 2)) + ((mean_demand ** 2) * (std_tit ** 2))
        ss_dynamic = int(np.ceil(1.645 * np.sqrt(variance_term)))

        # 1,000 Stochastic trials for this SKU
        # Generate Log-Normal lead times to prevent negative transit times while modeling long-tail delays
        shape = np.sqrt(np.log(1 + (std_tit / mean_tit) ** 2))
        scale = np.log(mean_tit) - 0.5 * (shape ** 2)
        simulated_lead_times = np.random.lognormal(mean=scale, sigma=shape, size=n_trials)

        # Generate total demand during each simulated lead time
        # Compound stochastic process: Demand during Lead Time = Sum of L daily demands
        simulated_ltd_demands = []
        for lt in simulated_lead_times:
            # lt days of random daily demand
            int_days = max(1, int(np.ceil(lt)))
            daily_draws = np.random.normal(loc=mean_demand, scale=std_demand, size=int_days)
            daily_draws = np.clip(daily_draws, 0, None)
            simulated_ltd_demands.append(np.sum(daily_draws))

        simulated_ltd_demands = np.array(simulated_ltd_demands)

        # Buffer level available under Static vs Dynamic
        lead_time_mean_demand = mean_demand * mean_tit
        reorder_point_static = lead_time_mean_demand + ss_static
        reorder_point_dynamic = lead_time_mean_demand + ss_dynamic

        # Stockout evaluation
        static_stockouts = simulated_ltd_demands > reorder_point_static
        dynamic_stockouts = simulated_ltd_demands > reorder_point_dynamic

        stockout_prob_static = np.mean(static_stockouts) * 100.0
        stockout_prob_dynamic = np.mean(dynamic_stockouts) * 100.0

        # Unmet Demand Quantity (Lost Sales / Backorders)
        unmet_static = np.maximum(0, simulated_ltd_demands - reorder_point_static)
        unmet_dynamic = np.maximum(0, simulated_ltd_demands - reorder_point_dynamic)

        var_95_static = np.percentile(unmet_static, 95)
        var_95_dynamic = np.percentile(unmet_dynamic, 95)

        # Freshness / Expiry risk: Days of inventory on hand
        doh_dynamic = (ss_dynamic + lead_time_mean_demand) / mean_demand
        freshness_decay_risk = max(0.0, (doh_dynamic / shelf_life) * 100.0)

        results.append({
            "wd_id": wd_id,
            "sku_id": row["sku_id"],
            "sku_name": sku_name,
            "category": row["category"],
            "ss_static": ss_static,
            "ss_dynamic": ss_dynamic,
            "stockout_prob_static_%": round(stockout_prob_static, 2),
            "stockout_prob_dynamic_%": round(stockout_prob_dynamic, 2),
            "var_95_lost_sales_static": round(var_95_static, 1),
            "var_95_lost_sales_dynamic": round(var_95_dynamic, 1),
            "doh_dynamic_days": round(doh_dynamic, 1),
            "freshness_decay_risk_%": round(freshness_decay_risk, 2)
        })

    df_mc = pd.DataFrame(results)
    out_file = os.path.join(OUTPUT_DIR, "monte_carlo_risk_profile.csv")
    df_mc.to_csv(out_file, index=False)

    avg_stockout_static = df_mc["stockout_prob_static_%"].mean()
    avg_stockout_dynamic = df_mc["stockout_prob_dynamic_%"].mean()

    print(f"[Monte Carlo] Complete! Results exported -> {out_file}")
    print(f"  -> Average Stockout Probability under Static Norm: {avg_stockout_static:.1f}%")
    print(f"  -> Average Stockout Probability under Dynamic Model: {avg_stockout_dynamic:.1f}%")
    print(f"  -> Risk Reduction: {avg_stockout_static - avg_stockout_dynamic:.1f} percentage points")

    return df_mc

if __name__ == "__main__":
    run_monte_carlo_simulation(1000)

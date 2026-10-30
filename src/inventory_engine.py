"""
inventory_engine.py
Calculates dynamic safety stocks, reorder points (ROP), and Out-of-Stock (OOS) risks
for Wholesale Distributors (WDs) taking into account Time In Transit (TIT) volatility
and demand variance.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Service level factor Z: 95% OTIF target => Z = 1.645
Z_FACTOR = 1.645

def run_inventory_optimization():
    wd_file = os.path.join(DATA_DIR, "sku_inventory_wd.csv")
    if not os.path.exists(wd_file):
        raise FileNotFoundError(f"Missing dataset: {wd_file}. Please run generate_data.py first.")

    df = pd.read_csv(wd_file)

    # 1. Traditional Static Safety Stock (Assuming constant TIT, zero transit variance)
    # SS_static = Z * std_demand * sqrt(mean_tit)
    df["ss_static_units"] = np.ceil(
        Z_FACTOR * df["daily_demand_std"] * np.sqrt(df["replenishment_tit_mean_days"])
    ).astype(int)

    # 2. Dynamic Combined Variance Safety Stock (Accounting for highway transit volatility)
    # SS_dynamic = Z * sqrt( (mean_tit * std_demand^2) + (mean_demand^2 * std_tit^2) )
    variance_term_demand = df["replenishment_tit_mean_days"] * (df["daily_demand_std"] ** 2)
    variance_term_tit = (df["daily_demand_mean"] ** 2) * (df["replenishment_tit_std_days"] ** 2)
    total_combined_std = np.sqrt(variance_term_demand + variance_term_tit)

    df["ss_dynamic_units"] = np.ceil(Z_FACTOR * total_combined_std).astype(int)

    # 3. Buffer Gap & Volatility Premium
    # How many extra units are required strictly due to transporter transit inconsistency?
    df["tit_volatility_buffer_units"] = df["ss_dynamic_units"] - df["ss_static_units"]
    df["tit_buffer_increase_pct"] = (
        (df["tit_volatility_buffer_units"] / df["ss_static_units"]) * 100
    ).round(1)

    # 4. Reorder Point (ROP) = (Mean Daily Demand * Mean TIT) + Dynamic SS
    df["lead_time_demand"] = np.ceil(df["daily_demand_mean"] * df["replenishment_tit_mean_days"]).astype(int)
    df["rop_units"] = df["lead_time_demand"] + df["ss_dynamic_units"]

    # 5. Inventory Health & OOS Vulnerability Status
    # Current Stock compared against Reorder Point and Safety Stock
    def evaluate_health(row):
        stock = row["current_stock_units"]
        rop = row["rop_units"]
        ss = row["ss_dynamic_units"]

        if stock < ss:
            return "HIGH OOS RISK (Below Safety Stock)"
        elif stock < rop:
            return "REORDER REQUIRED (Below ROP)"
        elif stock > (rop + (row["daily_demand_mean"] * row["norm_inventory_days"])):
            return "EXCESS HOLDING (Aging Risk)"
        else:
            return "HEALTHY (Within Norm)"

    df["inventory_status"] = df.apply(evaluate_health, axis=1)

    # 6. Days of Inventory on Hand (DOH)
    df["days_on_hand"] = (df["current_stock_units"] / df["daily_demand_mean"]).round(1)

    # Save recommendations
    out_file = os.path.join(OUTPUT_DIR, "safety_stock_recommendations.csv")
    df.to_csv(out_file, index=False)

    # Summary Insights
    total_skus = len(df)
    high_oos_count = len(df[df["inventory_status"] == "HIGH OOS RISK (Below Safety Stock)"])
    avg_buffer_increase = df["tit_buffer_increase_pct"].mean()

    print("=" * 65)
    print("INVENTORY OPTIMIZATION ENGINE RESULTS")
    print("=" * 65)
    print(f"Total WD-SKU Combinations Evaluated: {total_skus}")
    print(f"SKUs at Immediate Out-of-Stock (OOS) Risk: {high_oos_count} ({round(high_oos_count/total_skus*100, 1)}%)")
    print(f"Average Safety Stock Buffering Required Due to TIT Volatility: +{round(avg_buffer_increase, 1)}%")
    print(f"Output saved to: {out_file}")
    print("=" * 65)

    return df

if __name__ == "__main__":
    run_inventory_optimization()

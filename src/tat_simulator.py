"""
tat_simulator.py
Discrete-event simulation modeling warehouse inbound truck turnaround time (TAT),
dock unloading bottlenecks, and demurrage cost reduction.
Compares:
1. Baseline Scenario: Uncontrolled Clustered Arrivals + First-Come First-Served (FCFS)
2. Optimized Scenario: Staggered Inbound Slotting (ASN) + Dynamic Urgency & Demurrage Priority Dispatch
"""

import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Warehouse Operating Parameters
NUM_DOCKS = 2                  # 2 Dedicated Inbound Unloading Docks at Branch Godown (WSP)
FREE_WAITING_HOURS = 6.0       # Standard contract free hours before demurrage
DEMURRAGE_RATE_PER_HOUR = 300  # INR 300 / hour detention penalty

def simulate_dock_operations(policy="BASELINE_FCFS"):
    """
    Simulates dock unloading queue over chronological truck arrivals.
    policy: 'BASELINE_FCFS' or 'OPTIMIZED_SYSTEM'
    """
    shipments_file = os.path.join(DATA_DIR, "inbound_shipments.csv")
    df = pd.read_csv(shipments_file)
    df["arrival_dt"] = pd.to_datetime(df["arrival_timestamp"])
    df.sort_values(by="arrival_dt", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 1. Arrival leveling & MHE Handling Parameters:
    if policy == "OPTIMIZED_SYSTEM":
        # Advance Shipping Notice (ASN) & 2-Hour Appointment Slot Booking:
        # Prevents gate bunching by spreading bunched arrivals into scheduled slots
        smoothed_arrivals = []
        last_scheduled = df["arrival_dt"].min()
        for idx, row in df.iterrows():
            arr = row["arrival_dt"]
            # Enforce minimum 45-minute headway between inbound gate-ins for the 2-dock system
            if arr < last_scheduled + timedelta(minutes=45):
                arr = last_scheduled + timedelta(minutes=45)
            last_scheduled = arr
            smoothed_arrivals.append(arr)
        df["sim_arrival_dt"] = smoothed_arrivals
        # MHE efficiency boost (JD requirement): Powered Pallet Trucks (BOPT) reduce unloading time by 22%
        df["sim_unloading_hours"] = (df["unloading_est_hours"] * 0.78).round(2)
    else:
        df["sim_arrival_dt"] = df["arrival_dt"]
        df["sim_unloading_hours"] = df["unloading_est_hours"]

    # Initialize docks with earliest arrival
    first_arrival = df["sim_arrival_dt"].min()
    dock_available = [first_arrival for _ in range(NUM_DOCKS)]

    pending_trucks = df.to_dict("records")
    unprocessed = list(range(len(pending_trucks)))
    results = []

    while unprocessed:
        # Earliest time a dock becomes free
        dock_idx = int(np.argmin(dock_available))
        current_time = dock_available[dock_idx]

        # Trucks already in the yard at current_time
        arrived_indices = [i for i in unprocessed if pending_trucks[i]["sim_arrival_dt"] <= current_time]

        if arrived_indices:
            if policy == "BASELINE_FCFS":
                chosen_idx = min(arrived_indices, key=lambda i: pending_trucks[i]["sim_arrival_dt"])
            elif policy == "OPTIMIZED_SYSTEM":
                # Jackson's Sequencing Rule: Earliest Due Date (EDD) + Stockout Urgency Preemption
                def compute_priority(i):
                    t = pending_trucks[i]
                    wait_hrs = (current_time - t["sim_arrival_dt"]).total_seconds() / 3600.0
                    hours_to_demurrage = FREE_WAITING_HOURS - (wait_hrs + t["sim_unloading_hours"])

                    # If truck is within 1.5 hours of incurring demurrage, prioritize heavily (EDD rule)
                    if hours_to_demurrage <= 1.5:
                        urgency_penalty = (1.5 - hours_to_demurrage) * 120.0
                    else:
                        urgency_penalty = 0.0

                    # Stockout urgency score (High-demand WD items)
                    stockout_score = t["stockout_urgency_score"] * 15.0

                    return urgency_penalty + stockout_score + (wait_hrs * 4.0)

                chosen_idx = max(arrived_indices, key=compute_priority)
            else:
                raise ValueError(f"Unknown policy: {policy}")

            start_service = current_time
        else:
            # Idle docks; wait for next arrival
            next_arrival_idx = min(unprocessed, key=lambda i: pending_trucks[i]["sim_arrival_dt"])
            chosen_idx = next_arrival_idx
            start_service = pending_trucks[chosen_idx]["sim_arrival_dt"]

        truck = pending_trucks[chosen_idx]
        unprocessed.remove(chosen_idx)

        end_service = start_service + timedelta(hours=truck["sim_unloading_hours"])
        dock_available[dock_idx] = end_service

        yard_wait_hours = (start_service - truck["sim_arrival_dt"]).total_seconds() / 3600.0
        total_tat_hours = (end_service - truck["sim_arrival_dt"]).total_seconds() / 3600.0

        excess_hours = max(0.0, total_tat_hours - FREE_WAITING_HOURS)
        demurrage_incurred = round(excess_hours * DEMURRAGE_RATE_PER_HOUR, 2)

        results.append({
            "shipment_id": truck["shipment_id"],
            "transporter": truck["transporter"],
            "category": truck["category"],
            "stockout_urgency": truck["stockout_urgency_score"],
            "arrival_time": truck["sim_arrival_dt"].strftime("%Y-%m-%d %H:%M"),
            "dock_start_time": start_service.strftime("%Y-%m-%d %H:%M"),
            "departure_time": end_service.strftime("%Y-%m-%d %H:%M"),
            "yard_wait_hours": round(yard_wait_hours, 2),
            "total_tat_hours": round(total_tat_hours, 2),
            "demurrage_hours": round(excess_hours, 2),
            "demurrage_cost_inr": demurrage_incurred,
            "demurrage_flag": 1 if excess_hours > 0 else 0,
            "policy": policy
        })

    return pd.DataFrame(results)

def run_simulation_experiment():
    print("Running Baseline Scenario: Uncontrolled FCFS Arrivals...")
    df_fcfs = simulate_dock_operations("BASELINE_FCFS")

    print("Running Optimized Scenario: Staggered Slotting + Dynamic Priority Dispatch...")
    df_opt = simulate_dock_operations("OPTIMIZED_SYSTEM")

    # Combine results
    df_combined = pd.concat([df_fcfs, df_opt], ignore_index=True)
    out_file = os.path.join(OUTPUT_DIR, "tat_simulation_results.csv")
    df_combined.to_csv(out_file, index=False)

    # Compare key performance indicators
    metrics = {
        "Metric": [
            "Total Inbound Trucks Handled",
            "Average Yard Waiting Time (Hrs)",
            "Average Truck TAT (Gate-in to Out) (Hrs)",
            "Trucks Incurring Demurrage (>6 Hrs)",
            "% Demurrage Breach Rate",
            "Total Demurrage Penalty Incurred (INR)",
            "Avg TAT for Critical Urgency Trucks (Score >= 8) (Hrs)"
        ],
        "Baseline (FCFS)": [
            len(df_fcfs),
            round(df_fcfs["yard_wait_hours"].mean(), 2),
            round(df_fcfs["total_tat_hours"].mean(), 2),
            df_fcfs["demurrage_flag"].sum(),
            f"{round(df_fcfs['demurrage_flag'].mean() * 100, 1)}%",
            f"INR {int(df_fcfs['demurrage_cost_inr'].sum()):,}",
            round(df_fcfs[df_fcfs['stockout_urgency'] >= 8]['total_tat_hours'].mean(), 2)
        ],
        "Optimized (Slotting + Priority)": [
            len(df_opt),
            round(df_opt["yard_wait_hours"].mean(), 2),
            round(df_opt["total_tat_hours"].mean(), 2),
            df_opt["demurrage_flag"].sum(),
            f"{round(df_opt['demurrage_flag'].mean() * 100, 1)}%",
            f"INR {int(df_opt['demurrage_cost_inr'].sum()):,}",
            round(df_opt[df_opt['stockout_urgency'] >= 8]['total_tat_hours'].mean(), 2)
        ]
    }

    df_metrics = pd.DataFrame(metrics)
    print("\n" + "=" * 80)
    print("WAREHOUSE DOCK & TRUCK TAT SIMULATION BENCHMARK")
    print("=" * 80)
    print(df_metrics.to_string(index=False))
    print("=" * 80)

    # Save summary table
    summary_file = os.path.join(OUTPUT_DIR, "tat_kpi_comparison.csv")
    df_metrics.to_csv(summary_file, index=False)
    print(f"Results exported to {out_file} and {summary_file}")
    return df_combined

if __name__ == "__main__":
    run_simulation_experiment()

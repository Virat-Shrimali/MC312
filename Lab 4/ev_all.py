"""
Bass model fitting for EV adoption per country.

Outputs per country:
 - cumulative plot with fitted curve and peak year annotation
 - residuals plot
 - summary CSV with fitted parameters and Rogers category
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# -------------------------
# --- USER: csv columns ---
# -------------------------
COUNTRY_COL = "Entity"
YEAR_COL = "Year"
EV_COL = "Electric car stocks"   # cumulative EV stock
CSV_FILE = "electric-car-stocks.csv"

# -------------------------
# Bass model cumulative solution
# -------------------------
def bass_cumulative(t, p, q, C):
    p = np.maximum(p, 1e-12)  # avoid division by zero
    expo = np.exp(-(p + q) * t)
    return C * (1 - expo) / (1 + (q / p) * expo)

def bass_peak_time(p, q):
    if p <= 0 or q <= 0 or q <= p:
        return np.nan
    return np.log(q / p) / (p + q)

def rogers_category(frac):
    pct = 100 * frac
    if pct < 2.5:
        return "Innovators"
    elif pct < 16:
        return "Early adopters"
    elif pct < 50:
        return "Early majority"
    elif pct < 84:
        return "Late majority"
    else:
        return "Laggards"

# -------------------------
# Fitting routine per country
# -------------------------
def fit_country(df_country, country_name, out_dir="fits", plot=True):
    years = np.asarray(df_country[YEAR_COL], dtype=float)
    Ndata = np.asarray(df_country[EV_COL], dtype=float)
    order = np.argsort(years)
    years = years[order]
    Ndata = Ndata[order]

    # Global rebase: base year = 2010
    t = years - 2010

    if len(t) < 4:
        return None, f"Insufficient data points ({len(t)})"

    # Initial guesses and bounds
    C0 = max(Ndata) * 2.0
    p0, q0 = 0.03, 0.4
    p_lower, p_upper = 1e-6, 1.0
    q_lower, q_upper = 1e-6, 3.0
    C_lower, C_upper = max(Ndata), max(Ndata) * 1000.0

    try:
        popt, _ = curve_fit(
            bass_cumulative,
            t,
            Ndata,
            p0=[p0, q0, C0],
            bounds=([p_lower, q_lower, C_lower], [p_upper, q_upper, C_upper]),
            maxfev=10000
        )
    except Exception as e:
        return None, f"Fit failed: {e}"

    p_fit, q_fit, C_fit = popt
    N_fit = bass_cumulative(t, p_fit, q_fit, C_fit)
    residuals = Ndata - N_fit

    # Peak time and calendar year
    t_star_rel = bass_peak_time(p_fit, q_fit)
    peak_year = np.nan if np.isnan(t_star_rel) else 2010 + t_star_rel

    # Adoption fraction at latest year
    N_latest = Ndata[-1]
    frac_latest = N_latest / C_fit if C_fit > 0 else np.nan
    category = rogers_category(frac_latest)

    # Plots
    if plot:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # cumulative adoption plot
        plt.figure(figsize=(7,4.5))
        plt.scatter(years, Ndata, label="Data", color="C0")
        years_dense = np.linspace(min(years), max(years)+10, 200)
        t_dense = years_dense - 2010
        N_dense = bass_cumulative(t_dense, p_fit, q_fit, C_fit)
        plt.plot(years_dense, N_dense, label="Bass fit", color="C1")

        # Annotate peak year
        if not np.isnan(peak_year):
            N_peak = bass_cumulative(t_star_rel, p_fit, q_fit, C_fit)
            plt.scatter([peak_year], [N_peak], color="red", zorder=5)
            plt.text(peak_year, N_peak, f" Peak {peak_year:.1f}", color="red")


        plt.title(f"{country_name}: Bass fit\np={p_fit:.4f}, q={q_fit:.4f}, C={C_fit:.0f}")
        plt.xlabel("Year")
        plt.ylabel("Cumulative EVs")
        plt.legend()
        plt.grid(alpha=0.3)
        fname = os.path.join(out_dir, f"{country_name}_cumulative.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

        # residuals plot
        plt.figure(figsize=(7,3))
        plt.axhline(0, color="k", lw=0.6)
        plt.plot(years, residuals, marker="o", linestyle="-")
        plt.title(f"{country_name} — residuals (data - fit)")
        plt.xlabel("Year")
        plt.ylabel("Residual (vehicles)")
        plt.grid(alpha=0.3)
        fname = os.path.join(out_dir, f"{country_name}_residuals.png")
        plt.savefig(fname, dpi=200, bbox_inches="tight")
        plt.close()

    # Summary dictionary
    summary = {
        "country": country_name,
        "p": p_fit,
        "q": q_fit,
        "C": C_fit,
        "t_star_year": peak_year,
        "N_latest": N_latest,
        "frac_latest": frac_latest,
        "rogers_category": category,
        "fit_success": True
    }

    return summary, None

# -------------------------
# Main script
# -------------------------
def main():
    df = pd.read_csv(CSV_FILE)
    if COUNTRY_COL not in df.columns or YEAR_COL not in df.columns or EV_COL not in df.columns:
        print("CSV columns detected:", df.columns.tolist())
        raise SystemExit("Update COUNTRY_COL/YEAR_COL/EV_COL in script.")

    df = df[[COUNTRY_COL, YEAR_COL, EV_COL]].dropna()
    df[YEAR_COL] = pd.to_numeric(df[YEAR_COL], errors="coerce")
    df[EV_COL] = pd.to_numeric(df[EV_COL], errors="coerce")
    df = df.dropna()

    countries = sorted(df[COUNTRY_COL].unique())
    results, errors = [], []

    for country in countries:
        dfc = df[df[COUNTRY_COL] == country]
        if dfc[YEAR_COL].nunique() < 4:
            errors.append((country, "too few years"))
            continue

        summary, err = fit_country(dfc, country)
        if err:
            errors.append((country, err))
        else:
            results.append(summary)
            print(f"{country}: p={summary['p']:.4f}, q={summary['q']:.4f}, "
                  f"C={summary['C']:.0f}, frac={summary['frac_latest']:.3f}, "
                  f"cat={summary['rogers_category']}, peak year={summary['t_star_year']:.1f}")

    if results:
        df_res = pd.DataFrame(results)
        df_res = df_res[[
            "country","p","q","C","t_star_year",
            "N_latest","frac_latest","rogers_category"
        ]]
        df_res.to_csv("bass_fits_summary.csv", index=False)
        print("\nSaved summary to bass_fits_summary.csv")
    else:
        print("No successful fits.")

    if errors:
        print("\nSkipped/failed countries:")
        for c, e in errors:
            print(f" - {c}: {e}")

if __name__ == "__main__":
    main()

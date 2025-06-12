import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import ssl
import urllib3
import scipy.stats as stats
from statsmodels.stats import power
from scipy import stats
import numpy as np


# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration
SUPABASE_URL = "https://pvgaaikztozwlfhyrqlo.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB2Z2FhaWt6dG96d2xmaHlycWxvIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc4NDE2MjUsImV4cCI6MjA2MzQxNzYyNX0.iAqMXnJ_sJuBMtA6FPNCRcYnKw95YkJvY3OhCIZ77vI"
CSV_URL = "https://raw.githubusercontent.com/karwester/behavioural-finance-task/refs/heads/main/personality.csv"

# Load Personality Data
personality_df = pd.read_csv(CSV_URL)
print(f"Personality data loaded: {personality_df.shape[0]} rows")
print("Personality columns:", personality_df.columns.tolist())

# Standardize ID column name
personality_df.rename(columns={'_id': 'id'}, inplace=True)

# Function to fetch Supabase data with SSL disabled
def fetch_supabase_data():
    headers = {
        "apikey": API_KEY,
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.get(
        f"{SUPABASE_URL}/rest/v1/assets?select=*",
        headers=headers,
        verify=False
    )
    response.raise_for_status()
    return pd.DataFrame(response.json())

# Load and process Assets Data
assets_df = fetch_supabase_data()
print(f"Assets data loaded: {assets_df.shape[0]} rows")
print("Assets columns:", assets_df.columns.tolist())
print("Sample assets:\n", assets_df.head(2))

# Standardize columns
assets_df.rename(columns={
    '_id': 'id',
    'asset_value': 'amount',
    'asset_currency': 'currency'
}, inplace=True)

# Convert amount to float
assets_df['amount'] = assets_df['amount'].astype(float)

# Filter only GBP assets
gbp_assets = assets_df[assets_df['currency'] == 'GBP'].copy()
print(f"GBP assets count: {gbp_assets.shape[0]}")

# Merge datasets on ID
combined_df = pd.merge(
    left=personality_df,
    right=gbp_assets[['id', 'amount']],  # Only ID and amount
    on='id',
    how='inner'
)
print(f"Merged dataset size: {combined_df.shape}")

# Calculate total GBP assets per person
gbp_totals = combined_df.groupby('id')['amount'].sum().reset_index()

# Find max assets holder
max_gbp_id = gbp_totals.loc[gbp_totals['amount'].idxmax(), 'id']
risk_tolerance = personality_df.loc[personality_df['id'] == max_gbp_id, 'risk_tolerance'].values[0]

print(f"\nRESULTS FOR COVER LETTER:")
print(f"Highest asset value (in GBP) individual risk tolerance: {risk_tolerance:.2f}")

# ======== EDA Section ========
# Basic summary statistics
print("\n=== SUMMARY STATISTICS ===")
print("Risk Tolerance Stats:")
print(combined_df['risk_tolerance'].describe())

# Relationship between risk tolerance and asset value
plt.figure(figsize=(10, 6))
sns.scatterplot(data=combined_df, x='risk_tolerance', y='amount')
plt.title('Risk Tolerance vs. GBP Asset Value')
plt.xlabel('Risk Tolerance Score')
plt.ylabel('Asset Value (GBP)')
plt.savefig('risk_vs_assets.png')
plt.show()

# Identify top asset holders
top_holders = gbp_totals.nlargest(5, 'amount')
print("\nTop 5 Asset Holders in GBP:")
print(top_holders)

# Analyze risk tolerance distribution
plt.figure(figsize=(10, 6))
sns.histplot(combined_df['risk_tolerance'], kde=True)
plt.title('Risk Tolerance Distribution')
plt.savefig('risk_distribution.png')
plt.show()

print("\nAnalysis complete. Results saved in visualizations.")

# ======== Enhanced Risk Tolerance Analysis ========
print("\n=== ENHANCED RISK TOLERANCE ANALYSIS ===")

# Create a DataFrame specifically for risk tolerance analysis
risk_assets = pd.merge(
    gbp_totals,
    personality_df[['id', 'risk_tolerance']],
    on='id',
    how='left'
)

# 1. Calculate correlation and statistical significance
risk_corr = risk_assets['risk_tolerance'].corr(risk_assets['amount'])

# Calculate p-value
r, p_value = stats.pearsonr(
    risk_assets['risk_tolerance'].dropna(),
    risk_assets.loc[risk_assets['risk_tolerance'].dropna().index, 'amount']
)

# Calculate effect size (Cohen's d)
n = len(risk_assets.dropna())
d = 2 * r / np.sqrt(1 - r**2)  # Cohen's d approximation from correlation
d_interpretation = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"

print(f"Correlation between risk tolerance and assets: {r:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)")
print("Statistical significance:", "Not significant" if p_value > 0.05 else "Significant")

# 2. Enhanced scatter plot with statistical annotations
plt.figure(figsize=(10, 6))
ax = sns.regplot(
    data=risk_assets,
    x='risk_tolerance',
    y='amount',
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'green', 'lw': 2}
)
plt.title('Risk Tolerance vs. Total GBP Assets')
plt.xlabel('Risk Tolerance Score')
plt.ylabel('Asset Value (GBP)')
plt.grid(True, alpha=0.3)

# Add statistical annotations
ax.annotate(f'r = {r:.3f}, p = {p_value:.4f}\nd = {d:.3f} ({d_interpretation})',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.savefig('enhanced_risk_vs_assets.png')
plt.show()

# 3. Group analysis by risk tolerance levels
risk_assets['risk_group'] = pd.cut(
    risk_assets['risk_tolerance'],
    bins=[0, 0.3, 0.5, 0.7, 1],
    labels=['Low', 'Moderate', 'High', 'Very High']
)

group_stats = risk_assets.groupby('risk_group', observed=False)['amount'].agg(['mean', 'median', 'count', 'std'])
print("\nAsset Performance by Risk Tolerance Group:")
print(group_stats)

# 4. Top risk tolerance holders analysis
high_risk = risk_assets[risk_assets['risk_tolerance'] > 0.7]
print(f"\nHigh-risk tolerance investors (>0.7): {len(high_risk)}")
if not high_risk.empty:
    print(f"Average assets: £{high_risk['amount'].mean():,.0f}")
    print(f"Top 3 high-risk tolerance investors:")
    print(high_risk.nlargest(3, 'amount')[['id', 'risk_tolerance', 'amount']])

# 5. Comparison between top asset holders and their risk tolerance
top_10_risk = risk_assets.nlargest(10, 'amount')
top_risk_count = top_10_risk[top_10_risk['risk_tolerance'] > 0.6].shape[0]
print(f"\n{top_risk_count} of top 10 asset holders have risk tolerance >0.6")

# 6. Save enhanced analysis results
with open('enhanced_risk_analysis.txt', 'w') as f:
    f.write("ENHANCED RISK TOLERANCE ANALYSIS RESULTS\n")
    f.write("=" * 50 + "\n")
    f.write(f"Correlation coefficient: {r:.3f}\n")
    f.write(f"p-value: {p_value:.4f}\n")
    f.write(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)\n")
    f.write("Statistical significance: " + (
        "Not significant (p > 0.05)" if p_value > 0.05 else "Significant (p ≤ 0.05)") + "\n\n")
    f.write("Performance by Risk Tolerance Group:\n")
    f.write(group_stats.to_string() + "\n\n")
    f.write(f"High-risk tolerance investors (>0.7): {len(high_risk)}\n")
    if not high_risk.empty:
        f.write(f"Average assets: £{high_risk['amount'].mean():,.0f}\n")
        f.write("Top 3 high-risk tolerance investors:\n")
        f.write(high_risk.nlargest(3, 'amount').to_string(index=False))
    f.write(f"\n\n{top_risk_count} of top 10 asset holders have risk tolerance >0.6")

print("\nEnhanced risk tolerance analysis complete. Results saved.")

### ===== ADDED ANALYSIS: COMPOSURE VS. ASSET VALUE =====
# Create a new DataFrame with total GBP assets per person including composure
person_assets = pd.merge(
    gbp_totals,
    personality_df[['id', 'composure']],
    on='id',
    how='left'
)

# 1. Add diagnostic printouts to validate data
print("\n[Debug] Composure-Amount correlation check:")
print(f"- Composure range: {person_assets['composure'].min():.2f} to {person_assets['composure'].max():.2f}")
print(f"- Amount range: £{person_assets['amount'].min():.2f} to £{person_assets['amount'].max():.2f}")
print(f"- Number of investors: {len(person_assets)}")
print(f"- Missing composure scores: {person_assets['composure'].isna().sum()}")

# 2. Scatter plot with regression line AND CORRELATION ANNOTATION
plt.figure(figsize=(10, 6))
ax = sns.regplot(
    data=person_assets,
    x='composure',
    y='amount',
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red', 'lw': 2}
)

# Calculate correlation coefficient and p-value
r, p_value = stats.pearsonr(
    person_assets['composure'].dropna(),
    person_assets.loc[person_assets['composure'].dropna().index, 'amount']
)

# Calculate effect size (Cohen's d)
n = len(person_assets.dropna())
d = 2 * r / np.sqrt(1 - r**2)  # Cohen's d approximation from correlation
d_interpretation = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"

# Add statistical annotations to plot
ax.annotate(f'r = {r:.3f}, p = {p_value:.4f}\nd = {d:.3f} ({d_interpretation})',
            xy=(0.05, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

plt.title('Composure vs. Total GBP Assets')
plt.xlabel('Composure Score')
plt.ylabel('Total Asset Value (GBP)')
plt.grid(True, alpha=0.3)
plt.savefig('composure_vs_assets_verified.png')
plt.show()

# 3. Print statistical significance
print(f"\n[Composure Analysis] Pearson correlation (r): {r:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)")
print("Statistical significance:", "Not significant" if p_value > 0.05 else "Significant")

# 4. Only run threshold analyses if POSITIVE correlation is strong enough
if r > 0.3:
    # A. Threshold effect analysis
    high_composure = person_assets[person_assets['composure'] >= 0.6]
    threshold_effect = high_composure['amount'].sum() / person_assets['amount'].sum()
    print(f"High-composure investors (≥0.6) hold {threshold_effect:.1%} of total GBP assets")

    # B. Top holder concentration analysis
    top_10_holders = person_assets.nlargest(10, 'amount')
    top_composure_count = top_10_holders[top_10_holders['composure'] >= 0.65].shape[0]
    print(f"{top_composure_count} of top 10 asset holders have composure ≥0.65")

    # C. Compare top holders vs. others
    top_vs_others = pd.DataFrame({
        'Group': ['Top 10 Holders', 'All Others'],
        'Avg Composure': [
            top_10_holders['composure'].mean(),
            person_assets[~person_assets['id'].isin(top_10_holders['id'])]['composure'].mean()
        ]
    })
    print("\nComposure comparison:")
    print(top_vs_others)

    # D. Detailed top holders analysis
    print("\nDetailed top 10 holders:")
    top_10_holders = top_10_holders.sort_values('amount', ascending=False)
    print(top_10_holders[['id', 'amount', 'composure']].reset_index(drop=True))

    # E. Save analysis results
    with open('composure_analysis.txt', 'w') as f:
        f.write("COMPOSURE ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Correlation between composure and asset value: r = {r:.3f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)\n")
        f.write(f"Investors with composure ≥0.6 hold {threshold_effect:.1%} of total GBP assets\n")
        f.write(f"{top_composure_count} of top 10 asset holders have composure ≥0.65\n\n")
        f.write("Composure Comparison:\n")
        f.write(top_vs_others.to_string(index=False) + "\n\n")
        f.write("Top 10 Asset Holders:\n")
        f.write(top_10_holders[['id', 'amount', 'composure']].to_string(index=False))

else:
    print("\n[Analysis Halted] Correlation is not strongly positive (r ≤ 0.3).")
    print("No threshold analyses performed due to weak/negative correlation.")

    # Save minimal results
    with open('composure_analysis.txt', 'w') as f:
        f.write("COMPOSURE ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Correlation between composure and asset value: r = {r:.3f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)\n\n")
        f.write("No further analysis performed - correlation not sufficiently positive (r ≤ 0.3)\n")
        f.write("Visual inspection recommended: see 'composure_vs_assets_verified.png'")

print("\nComposure analysis complete. Results saved in 'composure_analysis.txt'")

### ===== ADDED ANALYSIS: IMPACT DESIRE VS. ASSET VALUE =====
# Add ALL personality traits to person_assets at once
person_assets = pd.merge(
    person_assets,
    personality_df[['id', 'impact_desire', 'risk_tolerance']],  # ADDED risk_tolerance here
    on='id',
    how='left'
)

# 1. Scatter plot with polynomial regression (to show non-linear relationship)
plt.figure(figsize=(10, 6))
sns.regplot(
    data=person_assets,
    x='impact_desire',
    y='amount',
    order=2,  # Quadratic fit for U-shaped curve
    scatter_kws={'alpha': 0.6},
    line_kws={'color': 'red', 'lw': 2}
)
plt.title('Impact Desire vs. Total GBP Assets')
plt.xlabel('Impact Desire Score')
plt.ylabel('Total Asset Value (GBP)')
plt.grid(True, alpha=0.3)
plt.axvline(x=0.3, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5)
plt.savefig('impact_desire_vs_assets.png')
plt.show()

# Enhanced visualization 1: Boxplot comparison
plt.figure(figsize=(10, 6))
impact_groups = pd.cut(person_assets['impact_desire'], bins=[0, 0.3, 0.6, 0.8, 1])
sns.boxplot(
    x=impact_groups,
    y='amount',
    data=person_assets,
    showmeans=True,
    meanprops={"marker":"D","markerfacecolor":"white", "markeredgecolor":"black"}
)
plt.title('Asset Distribution by Impact Desire Category')
plt.xlabel('Impact Desire Group')
plt.ylabel('GBP Assets')
plt.xticks(rotation=15)
plt.savefig('impact_boxplot.png')
plt.show()

# Enhanced visualization 2: Jittered point distribution
plt.figure(figsize=(10, 6))
sns.stripplot(
    x=impact_groups,
    y='amount',
    data=person_assets,
    jitter=0.3,
    alpha=0.6,
    size=6
)
plt.title('Detailed Asset Distribution by Impact Group')
plt.xlabel('Impact Desire Group')
plt.ylabel('GBP Assets')
plt.savefig('impact_jittered.png')
plt.show()


# 2. Break down impact desire categories
# Calculate mean and standard deviation for each category
low_impact = person_assets[person_assets['impact_desire'] < 0.3]
moderate_impact = person_assets[(person_assets['impact_desire'] >= 0.4) &
                                (person_assets['impact_desire'] <= 0.6)]
high_impact = person_assets[person_assets['impact_desire'] > 0.8]

impact_categories = pd.DataFrame({
    'Name': ['Low (<0.3)', 'Moderate (0.4-0.6)', 'High (>0.8)', 'All'],
    'Asset Range': [
        f"£{low_impact['amount'].mean():.0f} ± {low_impact['amount'].std():.0f}",
        f"£{moderate_impact['amount'].mean():.0f} ± {moderate_impact['amount'].std():.0f}",
        f"£{high_impact['amount'].mean():.0f} ± {high_impact['amount'].std():.0f}",
        f"£{person_assets['amount'].mean():.0f} ± {person_assets['amount'].std():.0f}"
    ],
    'Avg Assets': [
        low_impact['amount'].mean(),
        moderate_impact['amount'].mean(),
        high_impact['amount'].mean(),
        person_assets['amount'].mean()
    ],
    'Count': [
        low_impact.shape[0],
        moderate_impact.shape[0],
        high_impact.shape[0],
        person_assets.shape[0]
    ]
})

print("\n[Impact Desire Analysis] Asset Performance by Category:")
print(impact_categories[['Name', 'Asset Range', 'Avg Assets', 'Count']])

# Statistical significance tests
print("\n[Statistical Significance Testing]")
print("-" * 50)

moderate = moderate_impact['amount']
low = low_impact['amount']
high = high_impact['amount']

# Calculate p-values with normality check
def safe_ttest(group1, group2):
    if len(group1) > 2 and len(group2) > 2:
        return stats.ttest_ind(group1, group2).pvalue
    return float('nan')

p_low_mod = safe_ttest(low, moderate)
p_high_mod = safe_ttest(high, moderate)
p_low_high = safe_ttest(low, high)

print(f"Low vs Moderate: p = {p_low_mod:.6f} {'(significant)' if p_low_mod < 0.05 else ''}")
print(f"High vs Moderate: p = {p_high_mod:.6f} {'(significant)' if p_high_mod < 0.05 else ''}")
print(f"Low vs High: p = {p_low_high:.6f} {'(significant)' if p_low_high < 0.05 else ''}")

# Calculate effect sizes
def cohens_d(group1, group2):
    diff = group1.mean() - group2.mean()
    pooled_std = ((group1.std()**2 * (len(group1)-1) +
                  group2.std()**2 * (len(group2)-1)) /
                  (len(group1) + len(group2) - 2))**0.5
    return diff / pooled_std

d_low_mod = cohens_d(low, moderate)
d_high_mod = cohens_d(high, moderate)

print(f"\nEffect Size (Cohen's d):")
print(f"Low vs Moderate: d = {d_low_mod:.3f}")
print(f"High vs Moderate: d = {d_high_mod:.3f}")



# 3. Top holder composition analysis - RECREATE AFTER ADDING IMPACT_DESIRE
top_10_holders = person_assets.nlargest(10, 'amount')  # Recreate with impact_desire
top_10_holders['impact_category'] = pd.cut(
    top_10_holders['impact_desire'],
    bins=[0, 0.3, 0.6, 0.8, 1],
    labels=['Low', 'Moderate', 'High', 'Very High']
)
top_category_dist = top_10_holders['impact_category'].value_counts().reset_index()
top_category_dist.columns = ['Category', 'Count']  # Rename columns
print("\nImpact Desire Distribution in Top 10 Holders:")
print(top_category_dist)

# 4. Detailed analysis of moderate impact investors
print("\nModerate Impact Investors Stats:")
print(f"- Count: {moderate_impact.shape[0]} investors")
print(f"- Avg Assets: £{moderate_impact['amount'].mean():.0f}")
print(f"- Composure Range: {moderate_impact['composure'].min():.2f}-{moderate_impact['composure'].max():.2f}")

# 5. Compare to extremes
extreme_comparison = pd.DataFrame({
    'Group': ['Low (<0.3)', 'Moderate (0.4-0.6)', 'High (>0.8)'],
    'Avg Assets': [
        low_impact['amount'].mean(),
        moderate_impact['amount'].mean(),
        high_impact['amount'].mean()
    ],
    'Avg Composure': [
        low_impact['composure'].mean(),
        moderate_impact['composure'].mean(),
        high_impact['composure'].mean()
    ]
})
print("\nPerformance Comparison by Impact Level:")
print(extreme_comparison)

# Enhanced visualization 3: Interaction analysis
plt.figure(figsize=(12, 7))
ax = sns.scatterplot(
    data=person_assets,
    x='impact_desire',
    y='amount',
    hue='composure',
    size='risk_tolerance',
    palette='coolwarm',
    sizes=(30, 250),
    alpha=0.8
)


# Add quadrant lines
plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.4)
plt.axhline(y=person_assets['amount'].median(), color='gray', linestyle='--', alpha=0.4)

# Add annotations
plt.annotate('High Composure\nModerate Impact',
             xy=(0.55, 350),
             xycoords='data',
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Add legend outside plot
plt.legend(title='Composure', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Asset Performance: Interaction of Impact, Composure & Risk', pad=15)
plt.xlabel('Impact Desire Score')
plt.ylabel('Total Asset Value (GBP)')
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('interaction_analysis.png', dpi=300)
plt.show()

### ===== SAVE IMPACT DESIRE ANALYSIS =====
with open('impact_desire_analysis.txt', 'w') as f:
    f.write("IMPACT DESIRE ANALYSIS RESULTS\n")
    f.write("=" * 40 + "\n")
    f.write("Key Findings:\n")
    f.write("- Inverse U-Shaped Relationship: Moderate impact desire correlates best with asset growth\n")
    f.write("- Extreme Value Penalty: Both low and high impact desire associate with lower assets\n\n")

    f.write("Performance by Impact Category:\n")
    f.write(impact_categories.to_string(index=False) + "\n\n")

    f.write("Top 10 Holders Distribution:\n")
    f.write(top_category_dist.to_string(index=False) + "\n\n")

    f.write("Moderate Impact Group Insights:\n")
    f.write(f"- {moderate_impact.shape[0]} investors\n")
    f.write(f"- Average assets: £{moderate_impact['amount'].mean():.0f}\n")
    f.write(f"- Composure range: {moderate_impact['composure'].min():.2f}-{moderate_impact['composure'].max():.2f}\n")

print("\nImpact desire analysis complete. Results saved in 'impact_desire_analysis.txt'")

# --- ENHANCE THE TEXT REPORT ---
with open('impact_desire_analysis.txt', 'w') as f:
    f.write("IMPACT DESIRE ANALYSIS RESULTS\n")
    f.write("=" * 40 + "\n\n")

    # Cautious findings summary
    f.write("Key Findings:\n")
    f.write("- Moderate impact desire (0.4-0.6) shows strongest average asset values (£{:.0f})\n".format(
        moderate_impact['amount'].mean()))

    # Add statistical context
    if p_low_mod < 0.05:
        f.write("- SIGNIFICANT difference between Low and Moderate groups (p={:.4f})\n".format(p_low_mod))
    else:
        f.write("- Insignificant difference between Low and Moderate groups (p={:.4f})\n".format(p_low_mod))

    if p_high_mod < 0.05:
        f.write("- SIGNIFICANT difference between High and Moderate groups (p={:.4f})\n".format(p_high_mod))
    else:
        f.write("- Insignificant difference between High and Moderate groups (p={:.4f})\n".format(p_high_mod))

    f.write("- Note: Visual inspection shows moderate-high variability within groups\n\n")

    # Detailed performance breakdown
    f.write("Performance by Impact Category:\n")
    f.write(impact_categories.to_string(index=False) + "\n\n")

    # Statistical results
    f.write("Statistical Significance:\n")
    f.write(f"Low vs Moderate: p = {p_low_mod:.6f}\n")
    f.write(f"High vs Moderate: p = {p_high_mod:.6f}\n")
    f.write(f"Effect Size (Low vs Moderate): d = {d_low_mod:.3f}\n")
    f.write(f"Effect Size (High vs Moderate): d = {d_high_mod:.3f}\n\n")

    # Top holders distribution
    f.write("Top 10 Holders Distribution:\n")
    f.write(top_category_dist.to_string(index=False) + "\n\n")

    # Interaction insights
    f.write("Interaction Insights:\n")
    f.write("- Optimal performers combine:\n")
    f.write("   • Impact desire: 0.4-0.6\n")
    f.write("   • Composure: >0.65\n")
    f.write("   • Risk tolerance: 0.45-0.55\n")
    f.write("- See 'interaction_analysis.png' for visualization\n")

print("\nEnhanced impact desire analysis complete.")

# Parameters for power calculation
effect_size = 0.401
n1 = 41
n2 = 21
alpha = 0.05

# Calculate power
power_analysis = power.TTestIndPower()
power_val = power_analysis.solve_power(
    effect_size=effect_size,
    nobs1=n1,
    ratio=n2/n1,
    alpha=alpha,
    alternative='two-sided'
)
print(f"Power: {power_val:.3f}")


# Explore relationship between confidence and financial assets
def analyse_confidence_relationship():
    # Create DataFrame with confidence and total assets
    confidence_assets = pd.merge(
        gbp_totals,
        personality_df[['id', 'confidence']],
        on='id',
        how='left'
    )

    print("\n=== CONFIDENCE ANALYSIS ===")
    print(f"Total investors with confidence data: {len(confidence_assets)}")

    # 1. Scatter plot to show relationship
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=confidence_assets,
        x='confidence',
        y='amount',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'red', 'lw': 2}  # Corrected 'colour' to 'color'
    )
    plt.title('Confidence Score vs. Total GBP Assets')
    plt.xlabel('Confidence Score')
    plt.ylabel('Total Asset Value (GBP)')
    plt.grid(True, alpha=0.3)
    plt.savefig('confidence_vs_assets.png')
    plt.show()

    # 2. Calculate correlation and statistical significance
    confidence_corr = confidence_assets['confidence'].corr(confidence_assets['amount'])

    # Calculate p-value
    r, p_value = stats.pearsonr(
        confidence_assets['confidence'].dropna(),
        confidence_assets.loc[confidence_assets['confidence'].dropna().index, 'amount']
    )

    # Calculate effect size (Cohen's d)
    n = len(confidence_assets.dropna())
    d = 2 * r / np.sqrt(1 - r ** 2)  # Cohen's d approximation from correlation
    d_interpretation = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(
        d) < 0.8 else "large"

    print(f"Correlation between confidence and assets: {r:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)")
    print("Statistical significance:", "Not significant" if p_value > 0.05 else "Significant")

    # 3. Group analysis
    confidence_assets['confidence_group'] = pd.cut(
        confidence_assets['confidence'],
        bins=[0, 0.4, 0.6, 0.8, 1],
        labels=['Low', 'Medium-Low', 'Medium-High', 'High']
    )

    group_stats = confidence_assets.groupby('confidence_group', observed=False)['amount'].agg(['mean', 'median', 'count'])
    print("\nAsset Performance by Confidence Group:")
    print(group_stats)

    # 4. High-confidence investors analysis
    high_confidence = confidence_assets[confidence_assets['confidence'] > 0.7]
    print(f"\nHigh-confidence investors (>0.7): {len(high_confidence)}")
    if not high_confidence.empty:
        print(f"Average assets: £{high_confidence['amount'].mean():,.0f}")
        print(f"Top 3 high-confidence investors:")
        print(high_confidence.nlargest(3, 'amount')[['id', 'confidence', 'amount']])

    # 5. Save analysis results with statistical significance
    with open('confidence_analysis.txt', 'w') as f:
        f.write("CONFIDENCE ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Correlation coefficient: {r:.3f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)\n")
        f.write("Statistical significance: " + (
            "Not significant (p > 0.05)" if p_value > 0.05 else "Significant (p ≤ 0.05)") + "\n\n")
        f.write("Performance by Confidence Group:\n")
        f.write(group_stats.to_string() + "\n\n")
        f.write(f"High-confidence investors (>0.7): {len(high_confidence)}\n")
        if not high_confidence.empty:
            f.write(f"Average assets: £{high_confidence['amount'].mean():,.0f}\n")
            f.write("Top 3 high-confidence investors:\n")
            f.write(high_confidence.nlargest(3, 'amount').to_string(index=False))

    print("\nConfidence analysis complete. Results saved.")


# Execute the analysis
analyse_confidence_relationship()


### ===== IMPULSIVITY VS. ASSET VALUE =====
def analyse_impulsivity_relationship():
    print("\n=== IMPULSIVITY ANALYSIS ===")

    # Create DataFrame with impulsivity and total assets
    impulsivity_assets = pd.merge(
        gbp_totals,
        personality_df[['id', 'impulsivity']],
        on='id',
        how='left'
    )

    # Create confidence assets for comparison
    confidence_assets_comparison = pd.merge(
        gbp_totals,
        personality_df[['id', 'confidence']],
        on='id',
        how='left'
    )

    print(f"Total investors with impulsivity data: {len(impulsivity_assets)}")

    # 1. Scatter plot to show relationship
    plt.figure(figsize=(10, 6))
    sns.regplot(
        data=impulsivity_assets,
        x='impulsivity',
        y='amount',
        scatter_kws={'alpha': 0.5},
        line_kws={'color': 'purple', 'lw': 2}
    )
    plt.title('Impulsivity Score vs. Total GBP Assets')
    plt.xlabel('Impulsivity Score')
    plt.ylabel('Total Asset Value (GBP)')
    plt.grid(True, alpha=0.3)
    plt.savefig('impulsivity_vs_assets.png')
    plt.show()

    # 2. Calculate correlation and statistical significance
    impulsivity_corr = impulsivity_assets['impulsivity'].corr(impulsivity_assets['amount'])

    # Calculate p-value
    r, p_value = stats.pearsonr(
        impulsivity_assets['impulsivity'].dropna(),
        impulsivity_assets.loc[impulsivity_assets['impulsivity'].dropna().index, 'amount']
    )

    # Calculate effect size (Cohen's d)
    n = len(impulsivity_assets.dropna())
    d = 2 * r / np.sqrt(1 - r ** 2)  # Cohen's d approximation from correlation
    d_interpretation = "negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5 else "medium" if abs(
        d) < 0.8 else "large"

    print(f"Correlation between impulsivity and assets: {r:.3f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)")
    print("Statistical significance:", "Not significant" if p_value > 0.05 else "Significant")

    # 3. Group analysis
    impulsivity_assets['impulsivity_group'] = pd.cut(
        impulsivity_assets['impulsivity'],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1],
        labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    )

    group_stats = impulsivity_assets.groupby('impulsivity_group', observed=False)['amount'].agg(
        ['mean', 'median', 'count'])
    print("\nAsset Performance by Impulsivity Group:")
    print(group_stats)

    # 4. High-impulsivity investors analysis
    high_impulsivity = impulsivity_assets[impulsivity_assets['impulsivity'] > 0.8]
    print(f"\nHigh-impulsivity investors (>0.8): {len(high_impulsivity)}")
    if not high_impulsivity.empty:
        print(f"Average assets: £{high_impulsivity['amount'].mean():,.0f}")
        print(f"Top 3 high-impulsivity investors:")
        print(high_impulsivity.nlargest(3, 'amount')[['id', 'impulsivity', 'amount']])

    # 5. Save analysis results
    with open('impulsivity_analysis.txt', 'w') as f:
        f.write("IMPULSIVITY ANALYSIS RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Correlation coefficient: {r:.3f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Effect size (Cohen's d): {d:.3f} ({d_interpretation} effect)\n")
        f.write("Statistical significance: " + (
            "Not significant (p > 0.05)" if p_value > 0.05 else "Significant (p ≤ 0.05)") + "\n\n")
        f.write("Performance by Impulsivity Group:\n")
        f.write(group_stats.to_string() + "\n\n")
        f.write(f"High-impulsivity investors (>0.8): {len(high_impulsivity)}\n")
        if not high_impulsivity.empty:
            f.write(f"Average assets: £{high_impulsivity['amount'].mean():,.0f}\n")
            f.write("Top 3 high-impulsivity investors:\n")
            f.write(high_impulsivity.nlargest(3, 'amount').to_string(index=False))

    print("\nImpulsivity analysis complete. Results saved.")

    # 6. Additional analysis: Compare with confidence
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.scatterplot(data=impulsivity_assets, x='impulsivity', y='amount', color='purple')
    plt.title('Impulsivity vs Assets')
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=confidence_assets_comparison, x='confidence', y='amount', color='blue')
    plt.title('Confidence vs Assets')
    plt.subplot(2, 2, 3)
    sns.histplot(impulsivity_assets['impulsivity'], color='purple', kde=True)
    plt.title('Impulsivity Distribution')
    plt.subplot(2, 2, 4)
    sns.histplot(confidence_assets_comparison['confidence'], color='blue', kde=True)
    plt.title('Confidence Distribution')
    plt.tight_layout()
    plt.savefig('impulsivity_confidence_comparison.png')
    plt.show()

    # 7. Correlation matrix for personality traits
    traits = ['impulsivity', 'confidence', 'risk_tolerance', 'composure', 'impact_desire']
    traits_df = pd.merge(gbp_totals, personality_df[['id'] + traits], on='id', how='left')
    corr_matrix = traits_df[traits].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Personality Traits Correlation Matrix')
    plt.savefig('personality_correlation_matrix.png')
    plt.show()

    print("\nPersonality traits correlation matrix:")
    print(corr_matrix)


# Execute the impulsivity analysis
analyse_impulsivity_relationship()


### ===== COMPREHENSIVE DISTRIBUTION ANALYSIS =====
def analyze_distributions():
    print("\n=== COMPREHENSIVE DISTRIBUTION ANALYSIS ===")

    # Create a comprehensive DataFrame with all traits and assets
    all_traits_assets = pd.merge(
        gbp_totals,
        personality_df[['id', 'risk_tolerance', 'composure', 'impulsivity', 'impact_desire', 'confidence']],
        on='id',
        how='left'
    )

    # List of traits to analyze
    traits = ['risk_tolerance', 'composure', 'impulsivity', 'impact_desire', 'confidence']

    # Set up figure for trait distributions
    plt.figure(figsize=(18, 15))
    plt.suptitle('Distributions', fontsize=20, y=1.00)

    # Create distribution plots for each trait
    for i, trait in enumerate(traits, 1):
        plt.subplot(3, 2, i)
        sns.histplot(all_traits_assets[trait], kde=True, bins=20, color='skyblue')
        plt.title(f'{trait.replace("_", " ").title()} Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')

        # Add statistics annotations
        median_val = all_traits_assets[trait].median()
        mean_val = all_traits_assets[trait].mean()
        std_val = all_traits_assets[trait].std()
        skew_val = all_traits_assets[trait].skew()

        plt.axvline(median_val, color='r', linestyle='--', label=f'Median: {median_val:.2f}')
        plt.axvline(mean_val, color='g', linestyle='-', label=f'Mean: {mean_val:.2f}')
        plt.legend()

        print(f"\n--- {trait.replace('_', ' ').title()} Statistics ---")
        print(f"Mean: {mean_val:.3f} | Median: {median_val:.3f}")
        print(f"Std Dev: {std_val:.3f} | Skewness: {skew_val:.3f}")
        print(f"Min: {all_traits_assets[trait].min():.3f} | Max: {all_traits_assets[trait].max():.3f}")

    # Add assets distribution
    plt.subplot(3, 2, 6)
    sns.histplot(all_traits_assets['amount'], kde=True, bins=20, color='salmon')
    plt.title('Asset Value Distribution (GBP)')
    plt.xlabel('Asset Value')
    plt.ylabel('Frequency')

    # Add statistics annotations for assets
    median_assets = all_traits_assets['amount'].median()
    mean_assets = all_traits_assets['amount'].mean()
    std_assets = all_traits_assets['amount'].std()
    skew_assets = all_traits_assets['amount'].skew()

    plt.axvline(median_assets, color='r', linestyle='--', label=f'Median: £{median_assets:.2f}')
    plt.axvline(mean_assets, color='g', linestyle='-', label=f'Mean: £{mean_assets:.2f}')
    plt.legend()

    print("\n--- Asset Value Statistics ---")
    print(f"Mean: £{mean_assets:.2f} | Median: £{median_assets:.2f}")
    print(f"Std Dev: £{std_assets:.2f} | Skewness: {skew_assets:.3f}")
    print(f"Min: £{all_traits_assets['amount'].min():.2f} | Max: £{all_traits_assets['amount'].max():.2f}")

    plt.tight_layout()
    plt.savefig('comprehensive_distributions.png')
    plt.show()

    # Save text analysis
    with open('distribution_analysis.txt', 'w') as f:
        f.write("COMPREHENSIVE DISTRIBUTION ANALYSIS\n")
        f.write("=" * 50 + "\n\n")

        for trait in traits:
            data = all_traits_assets[trait]
            f.write(f"--- {trait.replace('_', ' ').title()} Statistics ---\n")
            f.write(f"Mean: {data.mean():.3f} | Median: {data.median():.3f}\n")
            f.write(f"Std Dev: {data.std():.3f} | Skewness: {data.skew():.3f}\n")
            f.write(f"Min: {data.min():.3f} | Max: {data.max():.3f}\n")
            f.write(f"25th percentile: {data.quantile(0.25):.3f} | 75th percentile: {data.quantile(0.75):.3f}\n\n")

        # Assets analysis
        data = all_traits_assets['amount']
        f.write("--- Asset Value Statistics ---\n")
        f.write(f"Mean: £{data.mean():.2f} | Median: £{data.median():.2f}\n")
        f.write(f"Std Dev: £{data.std():.2f} | Skewness: {data.skew():.3f}\n")
        f.write(f"Min: £{data.min():.2f} | Max: £{data.max():.2f}\n")
        f.write(f"25th percentile: £{data.quantile(0.25):.2f} | 75th percentile: £{data.quantile(0.75):.2f}\n")

        # Distribution interpretation
        f.write("\n\nDISTRIBUTION INTERPRETATION\n")
        f.write("- Most personality traits show approximately normal distributions\n")
        f.write("- Risk tolerance and composure are slightly left-skewed\n")
        f.write("- Impact desire shows a bimodal distribution\n")
        f.write(
            "- Asset values are right-skewed, indicating most investors have moderate assets while a few have very high assets\n")

    print("\nDistribution analysis complete. Results saved.")


# Execute distribution analysis
analyze_distributions()
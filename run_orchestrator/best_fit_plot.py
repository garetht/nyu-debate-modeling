# Re-run the analysis with variable names changed to training_duration / "Training Duration"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- Rough estimates of the 15 points (unchanged values, renamed variable) ---
training_duration = np.array([
    0.02, 0.05, 0.08, 0.10, 0.16,
    0.22, 0.28, 0.34, 0.46, 0.50,
    0.60, 0.70, 0.80, 0.92
])

accuracy = np.array([
    0.751, 0.767, 0.782, 0.773, 0.783,
    0.760, 0.790, 0.789, 0.755, 0.785,
    0.757, 0.755, 0.737, 0.779
])

df = pd.DataFrame({"training_duration": training_duration, "accuracy": accuracy})
# display_dataframe_to_user("Estimated data points (training_duration)", df)

# --- OLS with intercept ---
X = sm.add_constant(df["training_duration"])
model = sm.OLS(df["accuracy"], X).fit()

coef = model.params["training_duration"]
ci_low, ci_high = model.conf_int().loc["training_duration"].tolist()

print("OLS with intercept, predictor = training_duration")
print(f"Intercept: {model.params['const']:.4f}")
print(f"Slope (training_duration): {coef:.4f}")
print(f"95% CI for slope: [{ci_low:.4f}, {ci_high:.4f}]")

# --- Plot scatter + fitted line with 95% CI band ---
x_grid = np.linspace(0, 1, 200)
pred_exog = sm.add_constant(pd.DataFrame({"training_duration": x_grid}))
pred = model.get_prediction(pred_exog).summary_frame(alpha=0.05)  # 95%

plt.figure(figsize=(6,4))
plt.scatter(df["training_duration"], df["accuracy"], label="Estimated points")
plt.plot(x_grid, pred["mean"].astype(float).values, label="OLS fit")
plt.fill_between(
    x_grid,
    pred["mean_ci_lower"].astype(float).values,
    pred["mean_ci_upper"].astype(float).values,
    alpha=0.3,
    label="95% CI"
)
plt.xlabel("Training Duration")
plt.ylabel("Judge Accuracy")
plt.title("Debate: Estimated data & OLS with 95% CI")
plt.legend()
plt.tight_layout()
plt.show()

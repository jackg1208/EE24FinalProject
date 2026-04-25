import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("/Users/jackgeraghty/EE24Python/merged.csv")


formula = """
COLLISION_COUNT ~ TEMP + PRECIPITATION + WIND_SPEED +
C(PRECIPITATION_TYPE) + C(HOUR)
"""

model = smf.negativebinomial(formula=formula, data=df).fit(maxiters=200)


print(model.summary())

results = model.params.to_frame(name='Coefficient')
results['RiskMultiplier'] = np.exp(model.params)
print(results)


df['PREDICTED_COUNT'] = model.predict(df)

mu = np.array(df['PREDICTED_COUNT'])

alpha = model.params['alpha']

n = 1 / alpha
p = 1 / (1 + alpha * mu)


p = np.clip(p, 1e-10, 1 - 1e-10)

simulated_counts = np.random.negative_binomial(n, p)


df['SIMULATED_COUNT'] = simulated_counts


sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))


sns.histplot(
  df['COLLISION_COUNT'],
  color='royalblue',
  label='Actual Collisions',
  binwidth=1,
  stat='density',
  alpha=0.3
)




sns.histplot(
  df['PREDICTED_COUNT'],
  color='darkorange',
  label='Predicted Mean',
  binwidth=1,
  stat='density',
  alpha=0.3
)


sns.histplot(
  df['SIMULATED_COUNT'],
  color='green',
  label='Simulated (NB Model)',
  binwidth=1,
  stat='density',
  alpha=0.3
)




plt.title('Distribution Comparison: Actual vs NB Model', fontsize=15)
plt.xlabel('Number of Collisions per Hour', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 60)
plt.legend()


plt.savefig('collision_histogram.png', dpi=300)
plt.show()


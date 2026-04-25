import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

#Running this python code shows our full simulation (green bars on plot)

#Filepath for dataset
df = pd.read_csv("/Users/jackgeraghty/EE24Python/merged.csv")

#Model, can be adjusted for what conditions are being considered
formula = """
COLLISION_COUNT ~ TEMP + PRECIPITATION + WIND_SPEED +
C(PRECIPITATION_TYPE) + C(HOUR)
"""

#Fiitting NB distribution
model = smf.negativebinomial(formula=formula, data=df).fit()

print(model.summary())

#Beta coefficients
results = model.params.to_frame(name='Coefficient')
results['RiskMultiplier'] = np.exp(model.params)
print(results)

df['PREDICTED_COUNT'] = model.predict(df)

#Simulating from model
mu = np.array(df['PREDICTED_COUNT'])

#Alpha parameter (dispersion)
alpha = model.params['alpha']

#Convert to n and p for plotting
n = 1 / alpha
p = 1 / (1 + alpha * mu)

p = np.clip(p, 1e-10, 1 - 1e-10)

# Simulate counts
simulated_counts = np.random.negative_binomial(n, p)


df['SIMULATED_COUNT'] = simulated_counts


#Plotting
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


# Predicted means (not true counts, but useful comparison)
sns.histplot(
   df['PREDICTED_COUNT'],
   color='darkorange',
   label='Predicted Mean',
   binwidth=1,
   stat='density',
   alpha=0.3
)


# Simulated from NB model (THIS is the real model output)
sns.histplot(
   df['SIMULATED_COUNT'],
   color='green',
   label='Simulated (NB Model)',
   binwidth=1,
   stat='density',
   alpha=0.3
)


#Formatting
plt.title('Distribution Comparison: Actual vs NB Model', fontsize=15)
plt.xlabel('Number of Collisions per Hour', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlim(0, 60)
plt.legend()


plt.savefig('collision_histogram.png', dpi=300)
plt.show()

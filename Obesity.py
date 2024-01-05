import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pywaffle import Waffle


df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')

sns.set_theme(
    style='whitegrid',
    font_scale=1,
    context='paper',
    palette="muted"
)

print(df.dtypes)
print(df.columns.values)
print(df.describe())

sns.barplot(data=df, x='MTRANS', y='Weight')
plt.xlabel('MTRANS')
plt.ylabel('Weight')
plt.title('')
plt.savefig('barplot.png')
plt.close()

sns.pairplot(df.sample(200), hue='family_history_with_overweight')
plt.savefig('pairplot.png')

g = sns.PairGrid(df, hue='Gender')
g.map_upper(sns.histplot)
g.map_diag(sns.kdeplot, fill=True)
g.map_lower(sns.scatterplot)
g.savefig('pairgrid.png')
plt.show()
plt.close()

df['Gender'] = pd.get_dummies(df['Gender'], drop_first=True)
corr_matrix = df.corr(numeric_only=True)


mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
corr_matrix = corr_matrix.mask(mask)


sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.savefig('correlation_plot.png')
plt.show()
plt.close()

def corr_plot(df):
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    f, ax = plt.subplots(figsize=(10, 18))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1, vmin=-1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
        annot=True,
    )

    plt.savefig('custom_correlation_plot.png')
    plt.show()
    plt.close()

corr_plot(df)

sns.jointplot(
    data=df,
    x="Weight",
    y="Age"
)
plt.savefig('jp_plot.png')
plt.show()
plt.close()

sns.catplot(
    data=df.sample(200),
    x="SCC", y="Weight",
    hue="Gender",
    kind="swarm"
)
plt.savefig('catplot_plot.png')
plt.show()
plt.close()

sns.catplot(
    data=df.sample(100),
    x="family_history_with_overweight", y="Weight",
    hue="SMOKE",
    kind="box"
)
plt.savefig('catplot.png')
plt.show()
plt.close()



category_counts = df['NObeyesdad'].value_counts()
fig = plt.figure(
    FigureClass=Waffle,
    rows=20,
    columns=40,
    values=category_counts,
    labels=[f"{k} ({v})" for k, v in category_counts.items()],
    legend={'loc': 'center left', 'bbox_to_anchor': (0, -0.3), 'framealpha': 1},
     figsize=(10, 14)
)

plt.savefig('waffle.png')
plt.show()
plt.close()
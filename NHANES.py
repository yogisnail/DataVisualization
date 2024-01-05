import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('data/NHANES_age_prediction.csv')

sns.set_theme(
    style='whitegrid',
    font_scale=1,
    context='paper',
    palette="muted"
)

print(df.dtypes)
print(df.describe())

sns.relplot(
    data=df,
    x="LBXGLT", y="LBXIN", hue="age_group", style="age_group",
)
plt.savefig('relplot_plot.png')
plt.show()
plt.close()

#palette = sns.cubehelix_palette(light=0.8, n_colors=6)
sns.relplot(
    data=df, kind="line",
    x="DIQ010", y="LBXGLU",
    hue="age_group", style="RIAGENDR", )
plt.savefig('palette_plot.png')
plt.show()
plt.close()

sns.displot(
    df,
    x="LBXGLT",
    hue="age_group",
    stat="density",
    common_norm=False,
)

plt.savefig('displot_plot.png')
plt.show()
plt.close()

sns.displot(
    df,
    x="LBXIN",
    y="LBXGLU",
    hue="age_group",
    kind="kde"
)
plt.savefig('dp_plot.png')
plt.show()
plt.close()

sns.catplot(
    data=df.sample(100),
    x="RIAGENDR", y="LBXIN",
    hue="age_group",
    kind="violin",
    inner="stick",
    split=True,
    palette="pastel",
)
plt.savefig('ca_plot.png')
plt.show()
plt.close()

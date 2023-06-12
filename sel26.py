import staging as stage
import seaborn as sns
import matplotlib.pyplot as plt

# portfolio allocation with selection of 26 equally weighted stocks (sel26)
print("*" * 10, "Selection of 26 equally weighted stocks (sel26)", "*" * 10)
sns.heatmap(stage.trainX.corr(), cmap="Purples", vmin=0, square=True, linewidths=.4, cbar_kws={"shrink": .4})
plt.title("Multicollinearity Check")
plt.show()
print("\n")

# portfolio optimization evaluation
stage.valY['sel26'] = stage.Returns.mean(axis=1)
stage.evaluate(stage.valY, 'sel26')
stage.testY['sel26'] = stage.Returns.mean(axis=1)
stage.evaluate(stage.testY, 'sel26')
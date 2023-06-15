import staging as stage
import seaborn as sns
import matplotlib.pyplot as plt

# dow jones reconstruction with all 30 equally weighted stocks (recon)
print("*" * 10, "Dow Jones reconstruction with all 30 equally weighted stocks (recon)", "*" * 10)
sns.heatmap(stage.trainX.corr(), cmap="Purples", vmin=0, square=True, linewidths=.4, cbar_kws={"shrink": .4})
plt.title("Multicollinearity Check")
plt.show()
print("\n")

# portfolio optimization evaluation
stage.valY['recon'] = stage.valX.mean(axis=1)
stage.evaluate(stage.valY, 'recon')
stage.testY['recon'] = stage.testX.mean(axis=1)
stage.evaluate(stage.testY, 'recon')
import staging as stage

# recon optimization evaluation
stage.testY['recon'] = stage.testX.mean(axis=1)
stage.evaluate(stage.testY, 'recon')

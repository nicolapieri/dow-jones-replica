import staging as stage

# latest dow divisor (2021-11-04)
dow_div = 0.15172752595384

# recon optimization evaluation
stage.Opt['recon'] = stage.X.sum(axis=1) / dow_div
stage.evaluate(stage.Opt, 'recon')

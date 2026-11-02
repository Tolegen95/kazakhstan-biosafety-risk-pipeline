"""A CLI-only adapter exercising addition of a module without core changes."""
from pipeline.core import RiskModule, RiskResult, RunContext
from pipeline.modules.bayes_beta_binomial import BayesBetaBinomialModule
import pandas as pd

class PriorGridModule(RiskModule):
    name = 'prior_grid'
    input_kind = 'farm_survey'
    produces = 'posterior'

    def validate(self, ctx):
        return BayesBetaBinomialModule().validate(ctx)

    def run(self, ctx):
        tables = []
        for mean in ctx.params.get('means', [.01, .05, .10, .28]):
            for strength in ctx.params.get('strengths', [2, 20, 100, 209.26]):
                params = {'alpha0': mean * strength, 'beta0': (1-mean) * strength}
                result = BayesBetaBinomialModule().run(RunContext(ctx.disease, ctx.events, params))
                tables.append(result.tables['posterior'].assign(prior_mean=mean, prior_strength=strength))
        return RiskResult(None, {'posterior': pd.concat(tables, ignore_index=True)},
                          {'module': self.name, 'n_configurations': len(tables)})

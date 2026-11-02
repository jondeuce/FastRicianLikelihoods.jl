# Typically causes a lot of false positives with ambiguities and/or unbound args checks;
# unfortunately have to periodically check this manually
Aqua.test_all(FastRicianLikelihoods; ambiguities = false, unbound_args = false)

# rs-autoregression-prediction

Autoregressive model on resting state time series benchmark

Based on https://github.com/FrancoisPgm/fmri-autoreg

I have my own fork as lots of changes to optimise for large n dataset.

## Dataset structure

- All inputs (i.e. building blocks from other sources) are located in
  `inputs/`.
- All custom code is located in `code/`.


## Compute Canada and environment set up

Python version 3.10.x.
The reference model we are using is code in pytorch `1.13.1`.

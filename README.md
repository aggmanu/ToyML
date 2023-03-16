# ToyML

Broad aim: Play with Python and ML.

More specific aim: 'Analyze' noisy high-resolution time-series from multiple sources, say $s_i, 1 \leq i \leq N.$ Examples, stock market, brain resting-state MEG, heart ECG etc. Analysis here means, given an input of time-window length $T$ from all sources, can we predict anything in the future?

What do we mean 'predicting the future'? To begin with, it can be whether $s_i(T+k) > s_i(0),$ represented by $1$ during training and $0$ otherwise. We do not know what value of $k$ can be. Maybe there is an optimal value of $k$ for some patterns that ML recognizes?




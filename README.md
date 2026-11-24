# Calibration metrics in Python: expected calibration error, reliability table, Brier score and log loss

If a model says 70% you'd hope the thing happens about 70% of the time, and checking that is what calibration metrics are for. I wrote this small NumPy module because the most common one (expected calibration error) is short enough that people write it from memory, and in my experience they usually get one of two details wrong when they do.

`ece(p, y, m)` sorts the predictions into m equal-width bins and compares the mean prediction in each bin with how often the event actually happened there. Each gap is weighted by the share of predictions that fell into that bin. The first detail is the top edge, because if you work out the bin as `int(p * m)` then a prediction of exactly 1.0 lands in bin m (which doesn't exist), so here it gets clamped into the last bin. The second is the weighting. If you take a plain average of the gaps you've let a bin with three predictions in it count for as much as one with three thousand.

`reliability_table(p, y, m)` hands you the rows behind that number so you can draw a reliability diagram. I'd always look at one next to the ECE, since the number on its own shifts about depending on how many bins you choose.

I've also put in `brier(p, y)` and `log_loss(p, y)`, because calibration by itself can mislead you quite badly. A model that always predicts the base rate has an ECE of zero and tells you nothing at all, and both of those scores will pick that up.

```python
from calibration import ece, reliability_table, brier, log_loss

ece([0.2, 0.2, 0.8, 0.8], [0, 1, 1, 1])     # 0.25
reliability_table(p, y, 10)                  # [(0.0, 0.1, count, mean_p, freq), ...]
brier(p, y), log_loss(p, y)
```

## Tests

`python3 -m unittest test_calibration` runs 7 checks and needs only NumPy. The first is a four-prediction example you can work out by hand to get 0.25. Another puts 99 good predictions next to one bad one, which is where a wrong weighting shows up straight away. The last few simulate 5,000 events on a fixed seed: the true chances score an ECE of 0.013 and an overconfident model scores 0.090, while the base-rate model scores exactly 0 and still loses on the Brier score and on log loss.

## More practice

If you'd like to write the ECE function yourself, there's a version with hints and a worked solution here: https://academicquant.co.uk/ml-expected-calibration-error?utm_source=github&utm_medium=post&utm_campaign=github-calibration-metrics

MIT licence.

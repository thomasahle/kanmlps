Requirements:
```bash
$ pip install torch tqdm numpy matplotlib
```

To run:
```bash
$ python run_models.py
Training on Dataset 1
Training Simple on Dataset 1: 100%|████████████| 500/500 [00:00<00:00, 714.19it/s, loss=0.0214]
Training Expanding on Dataset 1: 100%|█████████| 500/500 [00:00<00:00, 648.52it/s, loss=0.0133]
Training Learned Act on Dataset 1: 100%|███████| 500/500 [00:00<00:00, 515.79it/s, loss=0.0117]
Training Gated Sine on Dataset 1: 100%|███████| 500/500 [00:00<00:00, 675.63it/s, loss=0.00923]
...
```

To plot:
```bash
$ python plot_data.py --plot_type error_bars --skip-items 1
Plot saved to loss_comparison_with_variance.png
$ python plot_data.py --plot_type loss
Plot saved to step_times.png
Plot saved to memory_usage.png
```

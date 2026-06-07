import matplotlib.pyplot as plt

def plot_loss_from_meters(meters, title='Loss Curve'):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for name, meter in meters.meters.items():
        if name in ['time', 'data']:
            continue
        plt.plot(meter.series, label=name)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

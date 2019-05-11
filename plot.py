import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0.9111, 0.6538, 0.6757, 0.6461, 0.7464, 0.7127, 0.7151, 0.6322, 0.7237, 0.6851],
              [0.875, 0.625, 0.609375, 0.59375, 0.734375, 0.6875, 0.6875, 0.59375, 0.671875, 0.59375],
              [0.9375, 0.6875, 0.796875, 0.78125, 0.78125, 0.765625, 0.734375, 0.65625, 0.78125, 0.78125]])

def plot_fn(first=True):
    fig, ax = plt.subplots()
    train = x[:, ::2]
    test = x[:, 1::2]
    labels = ['Vision', 'Audition', 'Multisensory', 'Multisensory Vision', 'Multisensory Audition']

    if first:
        train = train[:, :3]
        test = test[:, :3]
        labels = labels[:3]
    else:
        idxs = [0, 3, 1, 4]
        train = train[:, idxs]
        test = test[:, idxs]
        labels = [labels[i] for i in idxs]

    means_train = train[0]
    errs_train = np.abs(train[1:] - means_train)

    means_test = test[0]
    errs_test = np.abs(test[1:] - means_test)

    index = np.arange(train.shape[1])
    bar_width = 0.35

    opacity = 0.4
    error_config = {'ecolor': '0.3'}

    ax.bar(index + bar_width, means_train, bar_width, alpha=opacity / 2, color='r', yerr=errs_train, error_kw=error_config, label='Train')

    ax.bar(index, means_test, bar_width, alpha=opacity, color='b', yerr=errs_test, error_kw=error_config, label='Test')

    title = 'Testing in Unisensory Environments' if not first else 'Testing in Multisensory Environments'
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim((0, 1.0))

    fig.tight_layout()
    plt.savefig(f'{"first" if first else "second"}')
    plt.show()
    plt.clf()

plot_fn(True)
plot_fn(False)

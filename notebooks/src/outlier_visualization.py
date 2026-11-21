"""Plotting helpers for notebook-based outlier inspection."""

import numpy as np


def show_outlier_grid(
    images,
    labels,
    result,
    indices,
    title,
    *,
    consensus=False,
    max_images=100,
):
    """Display ranked grayscale images without unused cells in the final row."""
    import matplotlib.pyplot as plt

    images = np.asarray(images)
    labels = np.asarray(labels)
    indices = np.asarray(indices[:max_images], dtype=int)
    if not len(indices):
        print(f"No images to show for: {title}")
        return

    image_side = int(np.sqrt(images.shape[1]))
    if image_side * image_side != images.shape[1]:
        raise ValueError(f"Expected flattened square images, got {images.shape[1]} features.")

    outlier_set = set(np.asarray(result.outlier_indices, dtype=int))
    ncols = min(10, len(indices))
    nrows = int(np.ceil(len(indices) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(1.3 * ncols, 1.3 * nrows), squeeze=False
    )
    axes = list(axes.flat)
    for rank, (index, ax) in enumerate(zip(indices, axes), 1):
        ax.imshow(images[index].reshape(image_side, image_side), cmap="gray_r", vmin=0, vmax=255)
        detail = f"Avg rank={result.average_ranks[index]:.1f}\n"
        if consensus:
            detail += f"Consensus {result.outlier_frequency[index]}/{len(result.components)}"
        else:
            detail += f"Outlier dims={result.outlier_frequency[index]}/{len(result.components)}"
        ax.set_title(
            f"#{rank} y={labels[index]}\n{detail}",
            fontsize=6,
            color="crimson" if index in outlier_set else "black",
        )
        ax.axis("off")
    for ax in axes[len(indices):]:
        fig.delaxes(ax)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    plt.show()
    plt.close(fig)

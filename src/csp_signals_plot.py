import numpy as np
import mne
from moabb.datasets import Schirrmeister2017
from moabb.paradigms import LeftRightImagery
from mne.decoding import CSP
import matplotlib.pyplot as plt
import matplotlib.colors

# Set logging level to see output
mne.set_log_level("INFO")

# Load dataset
dataset = Schirrmeister2017()
subject_id = 3  # Let's use the first subject
paradigm = LeftRightImagery(fmin=8, fmax=30)  # for rh vs lh

# Get data
X, y, metadata = paradigm.get_data(dataset, [subject_id], return_epochs=True)
print("Data loaded successfully.")

# Fit CSP
csp = CSP(
    n_components=2,
    reg=None,
    component_order="alternate",
    transform_into="average_power",
    log=False,
)
X_csp = csp.fit_transform(X._data, y)
X_csp = csp.fit_transform(X._data, y)

palette = [
    [0.0, "#1f5af9"],
    [0.125, "#3e5ff2"],
    [0.25, "#7080f2"],
    [0.375, "#b7bff7"],
    [0.5, "#fdfdfd"],
    [0.625, "#f6bfba"],
    [0.75, "#ef8177"],
    [0.875, "#f04e46"],
    [1.0, "#f72726"],
]
fig, axes = plt.subplots(
    1, 3, figsize=(10, 3), dpi=600
)  # Create 4 subplots for 3 filters and 1 colorbar
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", palette)
fig = csp.plot_filters(X.info, ch_type="eeg", size=1.5, cmap=cmap, axes=axes)
fig.savefig(f"csp_filters.png", transparent=True)
plt.show()

# %% Plot the first component after applying CSP
# component = X_csp[:, 0, :]  # Extract the first component
for selected_filter in np.arange(np.shape(X_csp)[1]):
    pick_filters = csp.filters_[selected_filter]
    X_ = np.asarray([np.dot(pick_filters, epoch) for epoch in X])

    # Plot
    # X_reshaped = X_.reshape((288, 1001))
    # X_concatenated = X_reshaped.flatten().reshape(1, -1)
    X_concatenated = X_.flatten().reshape(1, -1)

    fig, ax = plt.subplots(figsize=(30, 10), dpi=300)

    # Add vertical lines separating each event
    selected_events = 520
    num_events = selected_events
    samples_per_event = X_.shape[1]

    plt.plot(
        X_concatenated[:, : samples_per_event * selected_events].T,
        linewidth=3,
        color="black",
    )

    for i in range(1, num_events):
        plt.axvline(
            x=i * samples_per_event, color="#FFDB45", linestyle="--", linewidth=6
        )
        event_name = y[i]  # Get event name from metadata
        # plt.text(
        #     i * samples_per_event + 100,
        #     2,  # max(X_concatenated[0]),
        #     event_name,
        #     color="white",
        #     # rotation=90,
        #     verticalalignment="bottom",
        #     fontsize=10,
        # )

    plt.grid(False)
    plt.ylim([-2.5, 2.5])
    plt.xlim([samples_per_event * 58 - 50, samples_per_event * 62 + 50])
    plt.axis("off")
    plt.show()
    fig.savefig(f"csp_signal_{selected_filter}.png", transparent=True)

import json
import numpy as np
from mne.viz.topomap import _get_pos_outlines
from mne.channels import make_dig_montage
import mne


def channel_pos(
    ch_names,
    dimension="2d",
    montage_type="file",
):
    """Get the 2D or 3D positions of EEG channels.

    Parameters
    ----------
    ch_names : list
        EEG channel names.
    dimension : {'2d', '3d'}, optional
        Dimension of the positions to retrieve, by default '2d'.

    montage_type : {'file', 'standard'}, optional
        Type of montage to use with different eeg positions, by default 'file'.

    Returns
    -------
    ch_pos : np.ndarray
        Array containing the positions of EEG channels.
    """
    # Create info structure using the selected montage
    if montage_type == "file":
        # Load montage from file
        with open("eeg_positions.json", "r") as f:
            eeg_positions = json.load(f)

        montage = make_dig_montage(
            ch_pos=eeg_positions["ch_pos"],
            nasion=eeg_positions["nasion"],
            lpa=eeg_positions["lpa"],
            rpa=eeg_positions["rpa"],
            hsp=eeg_positions["hsp"],
            hpi=eeg_positions["hpi"],
        )
        info = mne.create_info(montage.ch_names, sfreq=256, ch_types="eeg")
        with info._unlock():
            info["dig"] = montage.dig

    elif montage_type == "standard":
        # Load standard montage
        montage = mne.channels.make_standard_montage("standard_1005")
        info = mne.create_info(montage.ch_names, sfreq=256, ch_types="eeg")
        info.set_montage("standard_1005")  # Add digitization points

    # Get positions and outlines
    picks = list(range(len(montage.ch_names)))
    sphere = np.array([0, 0, 0, 0.95])
    pos, outlines = _get_pos_outlines(info, picks, sphere, to_sphere=True)

    # Initialize list to store channel positions
    ch_pos = []

    # Loop through each channel name
    for ch in ch_names:
        try:
            idx = montage.ch_names.index(ch)

            # Retrieve 2D or 3D position based on dimension
            if dimension == "2d":
                ch_pos.append(pos[idx])
            elif dimension == "3d":
                ch_pos.append(montage.dig[idx + 3]["r"])
        except ValueError:
            print(f"Channel '{ch}' not found in montage.")

    # Convert list to numpy array
    ch_pos = np.array(ch_pos)

    return ch_pos

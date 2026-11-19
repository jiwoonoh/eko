"""Lead order and single-lead conditions used by ECG-MTD."""

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
SINGLE_LEADS = {"single-lead-I": "I", "single-lead-II": "II"}


def keep_lead(signal, lead):
    """Keep one waveform in its original channel and zero the others."""
    index = LEADS.index(lead)
    masked = signal.copy()
    masked[:index] = 0
    masked[index + 1:] = 0
    return masked

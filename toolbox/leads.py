"""ECG lead order and waveform selection."""

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
DEFAULT_MEASUREMENT_LEAD = "II"


def add_lead_argument(parser):
    parser.add_argument("--leads", nargs="+", choices=(*LEADS, "all"), metavar="LEAD",
                        help=f"Waveform leads ({', '.join(LEADS)}). Default: all.")


def select_leads(leads=None):
    """Validate the selection and return leads in ECG channel order."""
    if leads is None:
        return LEADS
    if not isinstance(leads, (list, tuple)) or not leads:
        raise ValueError("--leads requires a nonempty list of ECG lead names or 'all'")
    if tuple(leads) == ("all",):
        return LEADS
    if any(lead not in LEADS for lead in leads):
        raise ValueError(f"--leads must be drawn from {', '.join(LEADS)}, or use 'all' alone")
    if len(set(leads)) != len(leads):
        raise ValueError("--leads must not contain duplicates")
    return tuple(lead for lead in LEADS if lead in leads)


def measurement_lead(leads):
    return DEFAULT_MEASUREMENT_LEAD if DEFAULT_MEASUREMENT_LEAD in leads else leads[0]


def describe_leads(leads):
    if leads == LEADS:
        return "all 12 leads"
    prefix = "lead" if len(leads) == 1 else "leads"
    return f"{prefix} {', '.join(leads)}"


def keep_leads(signal, leads):
    """Zero unselected waveforms while preserving the 12-channel layout."""
    if leads == LEADS:
        return signal
    masked = signal.copy()
    masked[[index for index, lead in enumerate(LEADS) if lead not in leads]] = 0
    return masked

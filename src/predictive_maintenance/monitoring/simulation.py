"""Create deterministic feature shifts for monitoring demonstrations."""

import pandas as pd

from predictive_maintenance.data.preprocessing import DataProcessor


def simulate_feature_drift(current: pd.DataFrame) -> pd.DataFrame:
    """Shift four input features without modifying the original sample."""
    shifted = DataProcessor().select_features(current)

    shifted["air_temperature_k"] = shifted["air_temperature_k"] + 10.0
    shifted["process_temperature_k"] = shifted["process_temperature_k"] + 10.0
    shifted["rotational_speed_rpm"] = shifted["rotational_speed_rpm"] * 1.4
    shifted["torque_nm"] = shifted["torque_nm"] + 20.0

    return shifted

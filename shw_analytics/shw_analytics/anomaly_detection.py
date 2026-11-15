"""Seasonally matched weekly CHR anomaly detection.

Compares a selected academic term with earlier years of the SAME term.
Requires an appointments DataFrame with start_time, service_line, outcome,
provider_key, and appointment_key (or a unique row per appointment).

Example:
    from shw_analytics import analysis
    from shw_analytics.anomaly_detection import detect_chr_anomalies
    con = analysis.connect()
    df = con.execute('SELECT * FROM appointments').df()
    con.close()
    result = detect_chr_anomalies(df, term='Summer', year=2026,
                                  service='Counselling')
    print(result['anomalies'].to_string(index=False))
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binomtest

METRICS = ('appointment_volume', 'cancellation_rate', 'no_show_rate', 'active_practitioners')


def _bh_adjust(p_values):
    """Benjamini-Hochberg adjustment across all eligible weekly rate tests."""
    p = np.asarray(p_values, dtype=float)
    if len(p) == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(p), dtype=float)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def _term_info(dates: pd.Series) -> pd.DataFrame:
    month = dates.dt.month
    term = np.select([month.between(1, 4), month.between(5, 8), month.between(9, 12)],
                     ['Winter', 'Summer', 'Fall'], default='Unknown')
    start_month = np.select([month.between(1, 4), month.between(5, 8), month.between(9, 12)],
                            [1, 5, 9], default=1)
    starts = pd.to_datetime({'year': dates.dt.year, 'month': start_month, 'day': 1})
    week = ((dates - starts).dt.days // 7 + 1).astype('Int64')
    return pd.DataFrame({'term': term, 'year': dates.dt.year, 'week': week}, index=dates.index)


def detect_chr_anomalies(
    appointments: pd.DataFrame,
    *,
    term: str,
    year: int,
    service: str | None = None,
    min_appointments: int = 25,
    rate_min_pp: float = 3.0,
    count_min_relative_change: float = 0.30,
    alpha: float = 0.05,
    historical_window: int = 1,
) -> dict:
    """Detect anomalies by comparing weeks with the same term in earlier years.

    Rate metrics use a binomial test against a historical pooled rate from
    the same week +/- historical_window weeks in prior matching terms. All
    eligible weekly rate p-values receive a Benjamini-Hochberg correction.
    Alerts require adjusted q < alpha and a change of at least rate_min_pp
    percentage points.

    Count metrics are *screening alerts*, not statistical significance claims:
    current weekly count must differ by >= count_min_relative_change from the
    median historical count for the matching week, and the current value must
    differ by >= 2 from that median. With only 1-2 previous years, this is a
    conservative operational heuristic rather than a calibrated prediction band.

    Returns weekly series for chart overlays, plus anomaly flags and reasons.
    """
    term = term.title().strip()
    if term not in ('Winter', 'Summer', 'Fall'):
        raise ValueError('term must be Winter, Summer, or Fall')
    if historical_window < 0:
        raise ValueError('historical_window must be >= 0')
    if not (0 < alpha < 1):
        raise ValueError('alpha must be between 0 and 1')
    needed = {'start_time', 'outcome', 'service_line', 'provider_key'}
    missing = needed.difference(appointments.columns)
    if missing:
        raise ValueError(f'Missing appointments columns: {sorted(missing)}')

    data = appointments.copy()
    data['start_time'] = pd.to_datetime(data['start_time'], errors='coerce')
    data = data.loc[data['start_time'].notna()].copy()
    data['outcome'] = data['outcome'].astype('string').str.lower()
    if service:
        data = data.loc[data['service_line'].astype('string').str.casefold() == service.casefold()].copy()
    if data.empty:
        raise ValueError('No valid dated appointments for this service')

    ti = _term_info(data['start_time'])
    data = pd.concat([data.drop(columns=['term', 'year', 'week'], errors='ignore'), ti], axis=1)
    data = data.loc[(data['term'] == term) & (data['year'] <= year)].copy()
    years = sorted(int(y) for y in data['year'].dropna().unique())
    historical_years = [y for y in years if y < year]
    if year not in years:
        raise ValueError(f'No appointments for {term} {year}')
    if not historical_years:
        raise ValueError(f'No previous {term} available for {year}')

    # For fair comparison of a partial term, restrict all prior years to
    # the number of days observed in the current term. Exclude the incomplete
    # last 7-day week unless it has all seven calendar days represented by
    # the requested observation window (the last calendar day may be quiet).
    current_dates = data.loc[data['year'] == year, 'start_time']
    start_month = {'Winter': 1, 'Summer': 5, 'Fall': 9}[term]
    term_start = pd.Timestamp(year, start_month, 1)
    term_end = term_start + pd.DateOffset(months=4)
    last_observed_day = current_dates.max().normalize()
    observed_days = (last_observed_day - term_start).days + 1
    complete_weeks = min(observed_days // 7, (term_end - term_start).days // 7)
    if complete_weeks < 1:
        raise ValueError('No complete week in the selected term')
    data = data.loc[data['week'].between(1, complete_weeks)].copy()

    weekly = (data.groupby(['year', 'week'], observed=True)
              .agg(appointment_volume=('start_time', 'size'),
                   cancellations=('outcome', lambda x: (x == 'cancelled').sum()),
                   no_shows=('outcome', lambda x: (x == 'no_show').sum()),
                   active_practitioners=('provider_key', 'nunique'))
              .reset_index())
    # Include zero-appointment weeks so a true volume collapse can be detected.
    full_index = pd.MultiIndex.from_product([years, range(1, complete_weeks + 1)], names=['year', 'week'])
    weekly = (weekly.set_index(['year', 'week']).reindex(full_index, fill_value=0).reset_index())
    weekly['cancellation_rate'] = np.where(weekly['appointment_volume'] > 0,
                                          100 * weekly['cancellations'] / weekly['appointment_volume'], np.nan)
    weekly['no_show_rate'] = np.where(weekly['appointment_volume'] > 0,
                                     100 * weekly['no_shows'] / weekly['appointment_volume'], np.nan)
    weekly['term'] = term
    weekly['service'] = service or 'All services'

    history = weekly.loc[weekly['year'] < year]
    current = weekly.loc[weekly['year'] == year]
    flags = []
    rate_tests = []
    for row in current.itertuples(index=False):
        same_week = history.loc[history['week'] == row.week]
        neighborhood = history.loc[history['week'].between(row.week - historical_window,
                                                          row.week + historical_window)]
        for metric in METRICS:
            actual = float(getattr(row, metric))
            if metric in ('no_show_rate', 'cancellation_rate'):
                successes = 'no_shows' if metric == 'no_show_rate' else 'cancellations'
                n_hist = int(neighborhood['appointment_volume'].sum())
                k_hist = int(neighborhood[successes].sum())
                n_cur = int(row.appointment_volume)
                if n_hist < min_appointments or n_cur < min_appointments:
                    continue
                baseline = 100 * k_hist / n_hist
                difference = actual - baseline
                p_value = float(binomtest(int(getattr(row, successes)), n_cur, baseline / 100).pvalue)
                rate_tests.append((int(row.week), metric, p_value))
                flagged = abs(difference) >= rate_min_pp
                measure = 'percentage points'
            else:
                baseline = float(same_week[metric].median())
                difference = actual - baseline
                p_value = np.nan
                # Protect against divide-by-zero; any count alert needs
                # a meaningful absolute difference as well.
                relative = abs(difference) / max(baseline, 1.0)
                flagged = abs(difference) >= 2 and relative >= count_min_relative_change
                measure = 'appointments' if metric == 'appointment_volume' else 'practitioners'
            if flagged:
                flags.append({
                    'term': term, 'year': year, 'week': int(row.week),
                    'service': service or 'All services', 'metric': metric,
                    'direction': 'spike' if difference > 0 else 'drop',
                    'observed': round(actual, 2), 'historical_baseline': round(baseline, 2),
                    'difference': round(difference, 2), 'difference_unit': measure,
                    'p_value': p_value, 'current_appointments': int(row.appointment_volume),
                    'historical_years': ', '.join(map(str, historical_years)),
                    'method': ('binomial rate test + minimum effect' if metric.endswith('_rate')
                               else 'historical median + minimum relative change (screening)'),
                })
    # Correct ALL eligible rate tests, including weeks with small differences.
    if rate_tests:
        adjusted = _bh_adjust([test[2] for test in rate_tests])
        q_by_test = {(week, metric): float(q) for (week, metric, _), q in zip(rate_tests, adjusted)}
    else:
        q_by_test = {}

    corrected_flags = []
    for flag in flags:
        if flag['metric'] in ('no_show_rate', 'cancellation_rate'):
            q = q_by_test[(flag['week'], flag['metric'])]
            flag['q_value'] = round(q, 6)
            if q >= alpha:
                continue
            flag['method'] = 'binomial rate test + BH correction + minimum effect'
        else:
            flag['q_value'] = np.nan
        corrected_flags.append(flag)
    flags = corrected_flags

    columns = ['term', 'year', 'week', 'service', 'metric', 'direction', 'observed',
               'historical_baseline', 'difference', 'difference_unit', 'p_value', 'q_value',
               'current_appointments', 'historical_years', 'method']
    anomalies = pd.DataFrame(flags, columns=columns)
    if not anomalies.empty:
        anomalies = anomalies.sort_values(['week', 'metric']).reset_index(drop=True)
    return {
        'weekly': weekly.reset_index(drop=True),
        'anomalies': anomalies,
        'historical_years': historical_years,
        'complete_weeks': complete_weeks,
        'note': (f'Rate alerts require a difference of at least {rate_min_pp} percentage points and '
                 f'Benjamini-Hochberg adjusted q < {alpha} across all eligible weekly rate tests. '
                 'Count alerts are operational screening flags, not statistical significance tests. '
                 'Historical baselines are estimates; confirm flagged periods before taking action.'),
    }

"""Preprocessing + feature engineering shared by validation and final training.

Every step here is label-free, so it is applied identically to train and test."""
import numpy as np
import pandas as pd

CLASSES = ["Availability", "Benign", "Reconnaissance", "Spoofing"]
WELL_KNOWN = [21, 22, 23, 53, 67, 68, 80, 123, 137, 443, 1883, 5353, 8080]

DUR_EPS_S = 1e-5      # flow_duration is in seconds (10 microseconds)
IAT_EPS_US = 10       # iat / active / idle columns are in microseconds
IAT_COLS_PREFIX = ("fwd_iat.", "bwd_iat.", "flow_iat.", "active.", "idle.")


def _safe_div(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / np.where(b > 0, b, 1), 0.0)


def normalize_timing(df, dur_eps=None, iat_eps=None):
    """Microsecond-level timing is a capture artifact: the same SYN flood has
    durations of 1-6 us in train but exactly 0 in test, and the *_per_sec columns
    are count/duration, so they explode to ~1e6 in train and are 0 in test.
    Snap anything below 10 us to 0 and recompute the rates so an 'instant' flow
    always has rate 0. (A 1 ms threshold was also tested: it erased real
    0.1-1 ms latency differences between MITM-spoofed and benign traffic.)"""
    dur_eps = DUR_EPS_S if dur_eps is None else dur_eps
    iat_eps = IAT_EPS_US if iat_eps is None else iat_eps
    df = df.copy()
    dur = df["flow_duration"].where(df["flow_duration"] >= dur_eps, 0.0)
    df["flow_duration"] = dur
    for c in df.columns:
        if c.startswith(IAT_COLS_PREFIX):
            df[c] = df[c].where(df[c] >= iat_eps, 0.0)
    df["fwd_pkts_per_sec"] = _safe_div(df["fwd_pkts_tot"], dur)
    df["bwd_pkts_per_sec"] = _safe_div(df["bwd_pkts_tot"], dur)
    df["flow_pkts_per_sec"] = _safe_div(df["fwd_pkts_tot"] + df["bwd_pkts_tot"], dur)
    df["payload_bytes_per_second"] = _safe_div(df["flow_pkts_payload.tot"], dur)
    return df


def build_features(df, orig_port_counts):
    """Raw dataframe -> numeric feature matrix (no labels used)."""
    df = normalize_timing(df)
    X = {}
    drop = ("id", "Target", "proto", "service", "id.orig_p", "id.resp_p")
    for c in df.columns:
        if c not in drop:
            X[c] = df[c].astype(float)

    # categorical: one-hot protocol and service (small fixed vocabulary)
    for p in ["tcp", "udp", "icmp"]:
        X[f"proto_{p}"] = (df["proto"] == p).astype(int)
    for s in ["-", "dns", "http", "ssl", "mqtt", "ntp", "dhcp", "irc"]:
        X[f"svc_{s}"] = (df["service"] == s).astype(int)

    # ports: raw numbers memorise specific hosts/tools, so encode behaviour instead
    op, rp = df["id.orig_p"], df["id.resp_p"]
    X["orig_port_reuse"] = np.log1p(op.map(orig_port_counts).fillna(1).values)  # scanners reuse one src port
    X["orig_port_wellknown"] = (op < 1024).astype(int)
    X["resp_port_wellknown"] = (rp < 1024).astype(int)
    X["resp_port_ephemeral"] = (rp >= 49152).astype(int)
    X["resp_port_common_service"] = rp.isin(WELL_KNOWN).astype(int)

    # direction / response behaviour
    fwd, bwd = df["fwd_pkts_tot"], df["bwd_pkts_tot"]
    tot = fwd + bwd
    X["has_response"] = (bwd > 0).astype(int)
    X["single_pkt_flow"] = (tot <= 2).astype(int)
    X["fwd_frac_pkts"] = _safe_div(fwd, tot)
    X["fwd_bytes_per_pkt"] = _safe_div(df["fwd_pkts_payload.tot"], fwd)
    X["bwd_bytes_per_pkt"] = _safe_div(df["bwd_pkts_payload.tot"], bwd)
    X["fwd_hdr_per_pkt"] = _safe_div(df["fwd_header_size_tot"], fwd)
    X["payload_ratio"] = _safe_div(df["bwd_pkts_payload.tot"], df["fwd_pkts_payload.tot"] + 1)
    X["data_pkt_frac"] = _safe_div(df["fwd_data_pkts_tot"] + df["bwd_data_pkts_tot"], tot)

    # TCP flag signatures (floods / scans)
    for f in ["flow_SYN_flag_count", "flow_ACK_flag_count", "flow_RST_flag_count",
              "flow_FIN_flag_count", "fwd_PSH_flag_count"]:
        X[f + "_rate"] = _safe_div(df[f], tot)
    X["syn_no_ack"] = ((df["flow_SYN_flag_count"] > 0) & (df["flow_ACK_flag_count"] == 0)).astype(int)
    X["rst_seen"] = (df["flow_RST_flag_count"] > 0).astype(int)

    # scale-free timing
    for c in ["flow_duration", "flow_pkts_per_sec", "payload_bytes_per_second", "flow_iat.avg"]:
        X["log_" + c] = np.log1p(df[c].clip(lower=0))

    X = pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in X.items()}, index=df.index)
    return X.replace([np.inf, -np.inf], 0).fillna(0)


def port_counts(df):
    return df["id.orig_p"].value_counts()

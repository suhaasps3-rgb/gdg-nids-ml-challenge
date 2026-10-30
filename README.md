# Network Intrusion Detection — GDG VNRVJIET AI/ML Challenge

Multiclass classification of network flows into **Availability**, **Benign**, **Reconnaissance** and **Spoofing**. The metric is **Macro F1**, and the hidden test set contains **behaviour profiles that are absent from training**.

The core idea of this solution: **random cross-validation is useless here (≈0.98 for everything), so I built a validation scheme that holds out whole behaviour profiles, studied exactly how the test traffic differs from train, and engineered features that describe *behaviour* instead of memorising *artifacts*.**

| | Macro F1 |
|---|---|
| Random stratified 5-fold CV | 0.978 |
| **Profile-holdout CV (3 repeats)** | **0.805 ± 0.021** |

---

## 1. How to run

```bash
pip install -r requirements.txt          # numpy, pandas, scikit-learn only
# put train.csv, test.csv in ./data/
python src/shift_analysis.py --data data          # train-vs-test shift findings (~1 min)
python src/validate.py       --data data          # random CV vs profile-holdout CV (~10 min)
python src/train_predict.py  --data data --out submission.csv   # final model (~1 min)
```

| File | Purpose |
|---|---|
| `src/features.py` | timing normalisation + feature engineering (label-free, same for train and test) |
| `src/model.py` | model definition, class weights, flood down-sampling |
| `src/validate.py` | random CV vs **profile-holdout CV** |
| `src/train_predict.py` | trains 5 seeds on all training data, averages probabilities, writes `submission.csv` |
| `src/shift_analysis.py` | reproduces the data findings below, including adversarial validation |
| `src/self_train_predict.py` | optional high-variance variant: self-training on confident test predictions -> `submission_selftrain.csv` |
| `src/compare_boosters.py` | optional: compares sklearn HGB vs LightGBM vs XGBoost and their probability-average ensemble (needs `pip install lightgbm xgboost`) |

---

## 2. Understanding the data

* 84,713 train rows, 33,161 test rows and 82 flow features (packet counts, payload sizes, inter-arrival times, TCP flags, window sizes, ports, protocol and service). No missing values and no duplicate rows.
* **Extreme imbalance:** Availability 80,072 (94.5%), Spoofing 2,323, Benign 1,583, Reconnaissance 735.
  * 79,698 of the Availability rows are one identical SYN-flood pattern on port 21 (1 packet out, 1 RST/ACK back, window 64, payload 120).
* **The test set looks completely different** (`shift_analysis.py`):

| proto/service | train share | test share |
|---|---|---|
| tcp/- | 94.5% | 46.0% |
| udp/dns | 1.3% | 24.6% |
| tcp/mqtt | 1.1% | 9.6% |
| tcp/http | 0.8% | 7.9% |
| udp/- | 0.8% | 6.5% |

So the test set is far less dominated by Availability, and macro F1 weights all four classes equally. A model that is good only at the majority class will score poorly.

### Key finding 1: the same attack is recorded differently (a timing artifact)

| port-21 SYN flood | train | test |
|---|---|---|
| rows | 79,699 | 10,396 |
| flows with a reply | 100% | 11.9% |
| `flow_duration == 0` | 0% | **99.3%** |
| median `flow_pkts_per_sec` | **524,288** | **0** |

In train, a single-packet flood flow lasts 1–6 µs, so `pkts_per_sec = packets / duration` explodes to around 10⁶. In test the duration is exactly 0 and the rate is 0. The attack is the same; only the capture's timing resolution differs.

My first model learned "huge packet rate ⇒ Availability", and on test it predicted **10,298 of these 10,396 flood rows as Benign**. That one mistake alone would have wrecked the Availability F1.

**Fix (`normalize_timing`):** durations below 10 µs and inter-arrival, active or idle times below 10 µs are snapped to 0. The rate features are then recomputed as `count / duration`, but only when the duration is greater than 0 (otherwise 0). After this fix, **all 10,396 flood rows are predicted Availability.**

How the threshold was chosen (profile-holdout CV, 3 repeats):

| threshold | macro F1 |
|---|---|
| none | 0.772 ± 0.032 |
| **10 µs** | **0.781 ± 0.028** |
| 100 µs | 0.669 ± 0.015 |
| 1 ms | 0.575 ± 0.006 |

A large threshold hurts because 0.1–1 ms latency is *real signal*: ARP spoofing places a man-in-the-middle on the path, which adds latency compared with benign traffic. The threshold therefore has to be just large enough to remove the microsecond artifact and no larger.

### Key finding 2: scanners reuse one source port

Nmap sends thousands of probes from one fixed source port. In test, 9 source ports cover 5,923 flows (each around 1,000). In train, the Reconnaissance rows reuse ports up to about 100 times, while normal clients use a fresh ephemeral port for each connection.

The raw source-port *number* is meaningless (it is random). The *number of flows sharing that source port* describes behaviour, so it became the feature `orig_port_reuse = log(1 + count)`.

* The count is computed within each file, train on train and test on test, and uses no labels.
* It lets the model recognise a new Nmap profile in test (TCP probes with window size 1024, which never appear in train): 4,025 of those 4,026 rows are predicted Reconnaissance.

### Key finding 3: raw port numbers memorise rather than generalise

`id.resp_p = 21` is a perfect shortcut in train (port 21 = DoS), but an attack on a different port would be missed. I replaced the raw ports with behavioural flags: well-known (<1024), ephemeral (≥49152), common-service-port, and source-port-is-well-known.

---

## 3. Validation strategy (the most important decision)

The organisers state that the test data contains behaviour profiles that are absent from training, so a random split leaks the profile: near-identical flows sit on both sides of the split. That is why random CV gives about 0.98 for every model and cannot tell good choices from bad ones.

**Profile-holdout CV (`validate.py`):**

1. Within each class, cluster the (log-scaled, standardised) features with KMeans into about 8 clusters. Each cluster approximates one behaviour profile (for example MQTT telemetry, DNS lookups, or a particular scan type).
2. Run `StratifiedGroupKFold(5)` with the clusters as groups. Each validation fold then contains **profiles the model has never seen**, while every class still appears in training.
3. Repeat with 3 different clusterings and split seeds, and report mean ± std. A single split was too noisy: differences of 0.03 to 0.05 came from the split alone.

Every decision below was made with this CV, never with random CV, and the public leaderboard was not used for tuning. I also checked each decision against the test *feature* distribution, with no labels involved:

* **Adversarial validation:** a classifier that separates train rows from test rows, with permutation importance showing which features carry the shift.
* **Per-group predictions:** checking what the model predicts for each identifiable test group (shown in section 6).

---

## 4. Experiments

Numbers are comparable **within** a table. Different tables used different clusterings.

**Early experiments (single profile-holdout split, before the timing fix):**

| experiment | random CV | profile-holdout CV |
|---|---|---|
| HistGradientBoosting | 0.977 | 0.724 |
| RandomForest | 0.978 | 0.524 |
| ExtraTrees | 0.977 | 0.565 |
| HGB without `orig_port_reuse` | – | 0.700 (vs 0.738 with it) |
| HGB without raw `id.resp_p` | – | **0.752** (vs 0.738 with it) |

**Model comparison (final features, 3 repeats):**

| model | profile-holdout macro F1 |
|---|---|
| **HGB, 31 leaves, balanced weights (final)** | **0.805 ± 0.021** |
| HGB, sqrt-balanced weights | 0.802 ± 0.005 |
| HGB, 63 leaves | 0.794 ± 0.033 |
| HGB, 15 leaves, min_samples_leaf 40 | 0.785 ± 0.037 |
| HGB, 50% feature subsampling | 0.775 ± 0.049 |
| ExtraTrees | 0.603 ± 0.009 |
| RandomForest | 0.526 ± 0.013 |

**Tried and dropped: "no-response" augmentation.** I generated copies of short TCP flows with the reply removed, to imitate a victim too overloaded to answer. It was neutral on CV (0.781 vs 0.781) and changed almost no test predictions once the timing fix was in, so I removed it for simplicity. The lesson: fixing the root cause (the timing artifact) beat patching the symptom.

---

## 5. Final model and why

* **Model:** `HistGradientBoostingClassifier` (max_iter 200 with early stopping, learning rate 0.1, 31 leaves, L2 = 1.0).
  * Gradient boosting fits tabular data with mixed scales and many interacting features without scaling.
  * It generalised far better than bagged trees on held-out profiles. RF and ET grow deep trees on bootstrap samples and memorise the exact training profiles.
  * It is in scikit-learn, so the code is fully reproducible. LightGBM and XGBoost were not installable in my sandbox at first, so I compared them afterwards on my own machine (section 7): neither beat it.
* **Class imbalance:** `class_weight="balanced"` sample weights, so each class contributes equally to the loss. This matches macro F1, which also weights each class equally. I chose weighting over SMOTE, because synthetic interpolated flows can be physically impossible (e.g. 2.5 packets).
* **Down-sampling:** the near-duplicate port-21 flood is cut from 79.7k to 15k rows. This loses no information, gives 4× faster training, and keeps every other Availability row.
* **Ensemble:** 5 seeds, each with a different flood subsample and a different early-stopping split. The probabilities are averaged, which reduces variance.
* **Feature set (113 features):** the original numeric columns after timing normalisation, plus:
  * one-hot `proto` and `service`
  * behavioural port flags and `orig_port_reuse`
  * response features: `has_response`, `single_pkt_flow`, forward fraction of packets, bytes per packet in each direction, header bytes per packet, payload ratio, data-packet fraction
  * TCP flag rates (SYN, ACK, RST, FIN, PSH per packet), `syn_no_ack`, `rst_seen`
  * log-scaled timing and rates

Final profile-holdout report (repeat #3):

| class | precision | recall | F1 |
|---|---|---|---|
| Availability | 0.999 | 0.986 | 0.993 |
| Benign | 0.576 | 0.673 | 0.621 |
| Reconnaissance | 0.746 | 0.722 | 0.734 |
| Spoofing | 0.763 | 0.746 | 0.754 |

The hardest pair is **Benign vs Spoofing**. ARP spoofing re-routes normal traffic through an attacker, so the flows look like normal DNS, HTTP and TLS traffic. The signal is in subtle latency and timing, which is exactly why the timing threshold had to stay small.

---

## 6. Sanity check on the test set (features only, no labels)

| test group | Avail. | Benign | Recon | Spoof | total |
|---|---|---|---|---|---|
| port-21 SYN, no reply | **9,159** | 0 | 0 | 0 | 9,159 |
| port-21 SYN, with reply | **1,232** | 0 | 5 | 0 | 1,237 |
| TCP probes, window 1024 (new Nmap profile) | 0 | 1 | **4,025** | 0 | 4,026 |
| tcp/mqtt | 0 | **3,181** | 0 | 0 | 3,181 |
| udp/- (mostly UDP scan, port reuse) | 0 | 63 | **2,026** | 47 | 2,136 |
| udp/dns | 0 | 3,574 | 0 | 2,586 | 6,160 |
| mDNS 5353 | 0 | 79 | 10 | 1,920 | 2,009 |
| tcp/http, ssl, dns, other tcp | 0 | 3,564 | 807 | 882 | 5,253 |
| **total** | 10,391 | 10,462 | 6,873 | 5,435 | 33,161 |

Every group whose behaviour is unambiguous is classified the way its traffic pattern implies.

* **mDNS rows (port 5353) → Spoofing:** in train, mDNS from an ephemeral source port with about 40-byte payloads is Spoofing. Benign mDNS uses source port 5353.
* **DNS and HTTPS:** this is where the remaining uncertainty is (Benign vs Spoofing).
* **DNS decision logic:** queries that repeat the same size (32 and 36 bytes, an IoT device looking up the same domain again and again) are predicted Benign. The wide spread of other query sizes (many different domains, typical of intercepted user traffic) is predicted Spoofing. That mirrors train exactly: in train, every 31-, 32- and 36-byte query is Benign.

**Public leaderboard (40% of test): 0.84788 on the first submission.** No leaderboard feedback was used to tune anything.

---

## 7. Round 2: experiments after the first submission (all rejected)

After the first submission I looked for the biggest remaining error source and tested fixes. Each was validated with the same 3-repeat profile-holdout CV, and **none beat the baseline (0.805)**, so the submitted model is unchanged.

**Diagnosis.**

* In held-out CV, udp/dns causes most errors: 88% of Benign and 43% of Spoofing DNS flows are misclassified when their profile is held out.
* Inside udp/dns, every absolute timing feature is **completely disjoint** between train and test (KS statistic = 1.00). The same 2-request, 2-response lookup takes 0.10–0.5 s in train and 0.0002–0.07 s in test, because the network is different.

| idea | why it might help | profile-holdout CV | decision |
|---|---|---|---|
| baseline (submitted) | – | **0.805 ± 0.021** | kept |
| drop all absolute timing features | timing doesn't transfer across networks | 0.649 ± 0.059 | rejected: timing carries real signal inside train |
| replace timing with scale-free ratios (IAT coefficient of variation, fwd/bwd gap ratio, active/idle fraction) | ratios survive a uniformly faster network | 0.627 without absolute timing, 0.788 with it | rejected |
| self-training (add confident predictions on the held-out fold, retrain) | adapts to unseen profiles | 0.762 ± 0.093 (one split collapsed to 0.63) | rejected: confidently wrong predictions reinforce themselves |
| self-training, threshold 0.99 / 0.9 at weight 0.5 / 2 rounds | gentler variants | 0.753 / 0.769 / 0.765 | rejected |
| no early stopping (400 rounds at lr 0.05, or 200 at lr 0.1) | early stopping stopped at ~55 rounds, maybe underfitting | 0.767 / 0.762 | rejected: more rounds overfit the training profiles |
| depth 6, L2 = 3, 150 rounds | stronger regularisation | 0.790 ± 0.035 | rejected |
| XGBoost (60 / 120 trees, depth 6) | the standard Kaggle booster | 0.737 ± 0.075 / 0.761 ± 0.069 | rejected (a one-split quick run had looked closer, 0.800 / 0.806, which turned out to be luck) |
| LightGBM (60 / 120 trees, 31 leaves) | faster, leaf-wise boosting | 0.779 ± 0.031 / 0.792 ± 0.021 | rejected |
| MLP neural network (scikit-learn, log + standard scaling, balanced weights), 128-64 and 64 hidden units | a different model family, more diversity | 0.714 ± 0.015 / 0.733 ± 0.019 | rejected |
| 50/50 probability blend of HGB and MLP | diversity from a different family | 0.771 ± 0.010 / 0.786 ± 0.017 | rejected: worse than HGB alone (0.805) |

The XGBoost / LightGBM comparison (`src/compare_boosters.py`, same profile-holdout splits, class weights and features, 3 repeats) put the scikit-learn model first (0.805 ± 0.021). The XGBoost and LightGBM submissions disagreed with the submitted file on only 102 and 151 of 33,161 rows (0.3% and 0.5%): the library choice barely matters here, because the gains came from the timing fix, the validation scheme and the features.

Two more observations:

* **The test predictions are stable.** Across every variant above, the per-group test predictions barely changed, for example udp/dns Benign/Spoofing stayed within about 3,570–3,620 / 2,540–2,590. The model's decisions do not hinge on the shifted timing features.
* **Row order carries no information.** In both files, the chance that a row has the same label or group as the next row equals the random baseline (train 0.895 vs 0.895; test 0.294 vs 0.295), so the files are shuffled.

**Alternative (high-variance) submission: `submission_selftrain.csv`.** `src/self_train_predict.py` trains the normal model, adds the test rows it is >= 95% confident about as pseudo-labels (using its own predictions, never any true labels), retrains for 2 rounds and averages 5 seeds. In profile-holdout CV it averaged 0.765 +/- 0.093 versus 0.805 for the normal model, with one split at 0.839 and another at 0.63: higher variance, lower mean. On the real test set it changes only 243 of 33,161 predictions (0.7%) relative to `submission.csv`. Its public leaderboard score was 0.84156 versus 0.84788 for the main model, which agrees with the CV result, so **I did not select it as a final submission.** The final submission is `submission.csv` (the main model).

I deliberately did **not** probe the leaderboard, for example by flipping a whole group's predictions and watching the public score. That amounts to extracting hidden labels through the scoreboard, which the rules prohibit, and it overfits to the 40% public split.

---

## 8. Rules and integrity

* Only the provided training data was used for training. No external labelled data was used.
* Test labels were never accessed and no manual labelling was done. The test set was used **only** in two ways, both label-free:
  * to diagnose the distribution shift: adversarial validation and per-group prediction counts
  * to compute the per-file source-port reuse count
* The dataset matches the public RT-IoT2022 dataset credited on the competition page. I used that knowledge only to understand what the classes are (for example, ARP poisoning as Spoofing and Nmap as Reconnaissance), not for any data.

---

## 9. Limitations and next steps

* KMeans profiles only approximate the organisers' hidden profiles. The absolute CV number is a guide, not a promise.
* Benign vs Spoofing on DNS and HTTPS is still the weak spot. Self-training, timing removal and timing ratios were tried and rejected (section 7).
* With source and destination IP addresses and timestamps (not provided), host-level context would help most. ARP spoofing is visible as one MAC answering for many IPs, which flow statistics alone cannot show.
* LightGBM, XGBoost and an MLP (alone and blended with HGB) were compared and did not help (section 7). CatBoost and per-class probability-threshold tuning for macro F1 remain untested.

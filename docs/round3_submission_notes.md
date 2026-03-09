# Round 3 Submission Notes

## 1. Current Internal Result Positioning

Source: `outputs/results/main_results_summary.csv`

Clean-condition mean results across 4 seeds:

| Model | Recall | Precision | F1 | FPR |
|---|---:|---:|---:|---:|
| Bi-ARL | 72.38% | 90.56% | 80.17% | 10.41% |
| MARL | 70.97% | 91.65% | 79.32% | 9.54% |
| LSTM-IDS | 65.05% | 96.73% | 77.78% | 2.91% |
| Vanilla PPO | 76.39% | 79.75% | 74.12% | 39.57% |

Interpretation:

- Bi-ARL is better than `Vanilla PPO` on `F1` and `FPR` by a large margin.
- Bi-ARL is slightly better than `MARL` on `F1`, but not on clean `FPR`.
- Bi-ARL is better than `LSTM-IDS` on `Recall` and `F1`, but clearly worse on `Precision` and `FPR`.
- Therefore, the current paper can claim:
  - `Bi-ARL improves the precision-recall trade-off over Vanilla PPO`
  - `Bi-ARL achieves the best mean F1 among the RL baselines`
- The current paper should NOT claim:
  - `lowest FPR overall`
  - `strong adversarial robustness`
  - `uniform superiority over all baselines`

## 2. What This Means for Submission

If you want a near-term submission, the safest framing is:

- The method is a `bi-level adversarial RL IDS`.
- The strongest evidence currently is on `clean-data trade-off improvement`, especially versus vanilla RL.
- The method remains `preliminary` on modern robustness claims because FGSM results degrade sharply.

If you want a stronger paper, the next iteration should add:

- at least one modern public dataset beyond `NSL-KDD`
- at least one recent `Transformer / SSL / AutoML` baseline
- one cross-dataset or train-test-shift experiment

## 3. Recent Related Papers Worth Comparing

### A. Generalization / stronger experimental framing

1. SSCL-IDS: Enhancing Generalization of Intrusion Detection with Self-Supervised Contrastive Learning
- Venue/source: IFIP Networking 2024
- Link: https://dl.ifip.org/db/conf/networking/networking2024/1570998643.pdf
- Why relevant:
  - focuses on generalization under distribution shift
  - argues that supervised IDS degrades on unseen distributions
  - reports gains over supervised and unsupervised baselines
- Use in your paper:
  - as motivation for moving beyond single-split accuracy
  - as a recent baseline family for future comparison

2. Machine Learning in Network Intrusion Detection: A Cross-Dataset Generalization Study
- Source: IEEE Access 2024
- Link: https://iris.unicas.it/retrieve/a40146bd-61d7-4a77-98c7-cf72f480c839/2024%20-%20Machine%20Learning%20in%20Network%20Intrusion%20Detection%20-%20A%20Cross-Dataset%20Generalization%20Study.pdf
- Why relevant:
  - directly studies inter-dataset generalization
  - useful to justify why only using NSL-KDD is weak for a modern submission

3. Network Intrusion Datasets: A Survey, Limitations, and Recommendations
- Source: arXiv 2025
- Link: https://arxiv.org/abs/2502.06688
- Why relevant:
  - reviews 89 public NIDS datasets
  - gives a good citation base for dataset choice and limitations

### B. Recent model families

4. VAEMax: Open-Set Intrusion Detection based on OpenMax and Variational Autoencoder
- Source: arXiv 2024
- Link: https://arxiv.org/abs/2403.04193
- Why relevant:
  - open-set / unknown-attack detection
  - evaluated on `CIC-IDS2017` and `CSE-CIC-IDS2018`
- Use in your paper:
  - as a recent non-RL baseline direction for unknown attack handling

5. Applying self-supervised learning to network intrusion detection for network flows with graph neural network
- Source: Computer Networks 2024
- Link: https://doi.org/10.1016/j.comnet.2024.110495
- Why relevant:
  - self-supervised GNN for flow-based IDS
  - reduces dependence on labels
- Use in your paper:
  - as a recent representation-learning baseline

6. A novel multi-scale network intrusion detection model with transformer
- Source: Scientific Reports 2024
- Link: https://www.nature.com/articles/s41598-024-74214-w
- Why relevant:
  - recent Transformer-based IDS
  - evaluated on `NSL-KDD`, `UNSW-NB15`, and `CIC-DDoS2019`
- Use in your paper:
  - as a modern deep baseline to cite when discussing Transformer-based IDS

7. Towards Autonomous Cybersecurity: An Intelligent AutoML Framework for Autonomous Intrusion Detection
- Source: ACM CCS AutonomousCyber 2024
- Paper/code: https://github.com/Western-OC2-Lab/AutonomousCyber-AutoML-based-Autonomous-Intrusion-Detection-System
- Why relevant:
  - open implementation
  - uses `CICIDS2017` and `5G-NIDD`
  - strong practical baseline because it automates feature engineering and ensemble selection
- Use in your paper:
  - as a practical industry-facing baseline family

## 4. Open Datasets Recommended for Your Next Stage

### Priority 1: easiest upgrade from current work

1. UNSW-NB15
- Official: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Why:
  - more modern than NSL-KDD
  - tabular, easy to adapt to your current pipeline
  - widely used in IDS literature
- Good for:
  - replacing or supplementing NSL-KDD with minimal engineering cost

2. CIC-IDS2017
- Official: https://www.unb.ca/cic/datasets/ids-2017.html
- Why:
  - flow CSVs are available
  - includes PCAP and labeled flow files
  - widely used and still easy to train on
- Good for:
  - direct train/test evaluation
  - stronger paper than NSL-KDD-only

3. CSE-CIC-IDS2018
- Official: https://www.unb.ca/cic/datasets/ids-2018.html
- Why:
  - CSVs available through AWS
  - larger and more modern than CIC-IDS2017
- Good for:
  - a second modern benchmark
  - robustness and scalability evaluation

### Priority 2: if you want a more topical submission

4. 5G-NIDD
- Official page: https://netslab.ucd.ie/5g-nidd-dataset/
- Dataset entry: IEEE DataPort link on that page
- Why:
  - more current networking scenario
  - already used by recent AutoML IDS work
- Good for:
  - next-generation network security framing

5. CICIoT2023
- Official: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Why:
  - 33 attacks, 105 devices
  - CSV data plus example notebook
  - realistic for IoT security papers
- Good for:
  - IoT / edge-security submission direction

### Priority 3: niche but useful

6. CIC-BCCC-NRC TabularIoTAttack-2024
- Official: https://www.unb.ca/cic/datasets/tabular-iot-attack-2024.html
- Why:
  - large tabular dataset
  - directly friendly to ML/RL pipelines

7. CIC-DDoS2019
- Official dataset index: https://www.unb.ca/cic/datasets/
- Why:
  - good if you narrow the paper topic from generic IDS to DDoS detection

## 5. Recommended Submission Strategy

### Option A: Shortest path to a defensible submission

- Keep the paper focused on `Bi-level RL vs RL baselines`.
- Update claims to:
  - best RL F1
  - large FPR reduction versus Vanilla PPO
  - preliminary robustness, not strong robustness
- Add one more dataset:
  - preferred order: `UNSW-NB15` then `CIC-IDS2017`

### Option B: Better paper, more work

- Add `CIC-IDS2017` and `UNSW-NB15`
- Compare against:
  - `LSTM`
  - `Transformer`
  - one `SSL / GNN / AutoML` recent method family
- Add:
  - cross-dataset test
  - stronger attack setting
  - variance and statistical significance

## 6. Immediate Engineering Recommendation

For the next coding round, the best cost/performance move is:

1. Add `UNSW-NB15` loader support.
2. Add one modern non-RL baseline:
   - simplest practical choice: `XGBoost/LightGBM`
   - strongest paper-story choice: `Transformer`
3. Re-run the full evaluation pipeline on `NSL-KDD + UNSW-NB15`.
4. Only after that decide whether to push into `CIC-IDS2017`.

That path gives you a submission story that is much stronger than the current `NSL-KDD`-only setup without forcing a full redesign first.

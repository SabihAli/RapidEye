# Benchmark summary

Run: `20260702T064650Z`

## ReID / face models (isolated)

| model_id         | backend   | family   |   acc_precision |   acc_recall |   acc_tp |   acc_fp |   acc_fn |   p95_ms |   throughput_ips |   acc_EER |   acc_TAR@FAR0.001 |   acc_TAR@FAR0.0001 |   acc_rank1 |   acc_rank5 |    acc_mAP |
|:-----------------|:----------|:---------|----------------:|-------------:|---------:|---------:|---------:|---------:|-----------------:|----------:|-------------------:|--------------------:|------------:|------------:|-----------:|
| face-det-scrfd   | pytorch   | face_det |               0 |            0 |        0 |       20 |       20 | 0.340435 |          3833.2  |     nan   |                nan |                 nan | nan         | nan         | nan        |
| face-emb-cosface | pytorch   | face_emb |             nan |          nan |      nan |      nan |      nan | 0.413125 |          3397.11 |       0.5 |                  0 |                   0 | nan         | nan         | nan        |
| reid-osnet-pt    | pytorch   | reid     |             nan |          nan |      nan |      nan |      nan | 0.41324  |          3421.24 |     nan   |                nan |                 nan |   0.0350877 |   0.0877193 |   0.034123 |

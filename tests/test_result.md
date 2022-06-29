# Test report for `sig_test.py`

## Test for `two-split` tests

### `two-split` with default settings
```bash
$ python test.py --s 'two-split'
```
    =========================== TEST `tuneHP` ============================
    (tuneHP: ratio) Est. Type 1 error: 0.020; inf sample ratio: 0.200
    (tuneHP: ratio) Done with inf sample ratio: 0.200
    ✅ results for tuned `test_params`:
    {'split': 'two-split', 'inf_ratio': 0.2, 'perturb': 0.0, 'cv_num': 2, 'cp': 'hommel', 'verbose': 2}
    ========================== TEST `test_base` ==========================
    cv: 0; p_value: 0.50000; loss_null: 0.00140(0.03744); loss_alter: 0.00140(0.03744)
    cv: 1; p_value: 0.15857; loss_null: 0.00211(0.04583); loss_alter: 0.00421(0.06475)
    🧪 0-th Hypothesis: accept H0 with p_value: 0.476
    ✅ results for `test_base`:
    p_value:

    0.475710966366659
    p_value_cv:

    [0.5        0.15857032]
    =========================== TEST `testing` ===========================
    ====================== two-split for 0-th Hypothesis =======================
    (tuneHP: ratio) Est. Type 1 error: 0.030; inf sample ratio: 0.200
    (tuneHP: ratio) Done with inf sample ratio: 0.200
    cv: 0; p_value: 0.50000; loss_null: 0.00140(0.03744); loss_alter: 0.00140(0.03744)
    cv: 1; p_value: 0.50000; loss_null: 0.00211(0.04583); loss_alter: 0.00211(0.04583)
    🧪 0-th Hypothesis: accept H0 with p_value: 0.750
    ====================== two-split for 1-th Hypothesis =======================
    cv: 0; p_value: 0.00000; loss_null: 0.00140(0.03744); loss_alter: 0.03509(0.18400)
    cv: 1; p_value: 0.00000; loss_null: 0.00140(0.03744); loss_alter: 0.02737(0.16315)
    🧪 1-th Hypothesis: reject H0 with p_value: 0.000
    ✅ results for `testing`:
    [0.75, 2.0363712057310684e-11]

### `two-split` with `inf_ratio=0.5`

```bash
python test.py --s "two-split" -r 0.5
```
    =========================== TEST `tuneHP` ============================
    ✅ results for tuned `test_params`:
    {'split': 'two-split', 'inf_ratio': 0.5, 'perturb': 0.0, 'cv_num': 2, 'cp': 'hommel', 'verbose': 2}
    ========================== TEST `test_base` ==========================
    cv: 0; p_value: 0.97952; loss_null: 0.00561(0.07472); loss_alter: 0.00253(0.05020)
    cv: 1; p_value: 0.71816; loss_null: 0.00421(0.06476); loss_alter: 0.00337(0.05794)
    🧪 0-th Hypothesis: accept H0 with p_value: 1.000
    ✅ results for `test_base`:
    p_value:

    1.0
    p_value_cv:

    [0.97951562 0.71815769]
    =========================== TEST `testing` ===========================
    2022-06-28 20:30:05.036734: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    ====================== two-split for 0-th Hypothesis =======================
    cv: 0; p_value: 0.96612; loss_null: 0.00561(0.07472); loss_alter: 0.00281(0.05291)
    cv: 1; p_value: 0.64727; loss_null: 0.00421(0.06476); loss_alter: 0.00365(0.06030)
    🧪 0-th Hypothesis: accept H0 with p_value: 1.000
    ====================== two-split for 1-th Hypothesis =======================
    cv: 0; p_value: 0.00000; loss_null: 0.00393(0.06257); loss_alter: 0.03397(0.18115)
    cv: 1; p_value: 0.00000; loss_null: 0.00421(0.06476); loss_alter: 0.03313(0.17897)
    🧪 1-th Hypothesis: reject H0 with p_value: 0.000
    ✅ results for `testing`:
    [1.0, 1.7332961916623114e-20]



### `two-split` with `tune_ratio_method="log-ratio"`

```bash
$ python test.py --s "two-split" -tr "log-ratio"
```

    =========================== TEST `tuneHP` ============================
    ✅ results for tuned `test_params`:
    {'split': 'two-split', 'inf_ratio': 0.09613243843898467, 'perturb': 0.0, 'cv_num': 2, 'cp': 'hommel', 'verbose': 2}
    ========================== TEST `test_base` ==========================
    2022-06-28 20:46:22.274285: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8101
    cv: 0; p_value: 0.15848; loss_null: 0.00000(0.00000); loss_alter: 0.00146(0.03821)
    cv: 1; p_value: 0.50000; loss_null: 0.00146(0.03821); loss_alter: 0.00146(0.03821)
    🧪 0-th Hypothesis: accept H0 with p_value: 0.475
    ✅ results for `test_base`:
    p_value:

    0.47543473606393216
    p_value_cv:

    [0.15847825 0.5       ]
    =========================== TEST `testing` ===========================
    ====================== two-split for 0-th Hypothesis =======================
    cv: 0; p_value: 0.50000; loss_null: 0.00146(0.03821); loss_alter: 0.00146(0.03821)
    cv: 1; p_value: 0.95871; loss_null: 0.00439(0.06608); loss_alter: 0.00000(0.00000)
    🧪 0-th Hypothesis: accept H0 with p_value: 1.000
    ====================== two-split for 1-th Hypothesis =======================
    cv: 0; p_value: 0.00001; loss_null: 0.00292(0.05399); loss_alter: 0.03509(0.18400)
    cv: 1; p_value: 0.00081; loss_null: 0.00439(0.06608); loss_alter: 0.02485(0.15568)
    🧪 1-th Hypothesis: reject H0 with p_value: 0.000
    ✅ results for `testing`:
    [1.0, 1.825729289126946e-05]

## Test for `one-split` test

### `one-split` with default settings

```bash
python test.py --s "one-split"
```
    =========================== TEST `tuneHP` ============================
    2022-06-29 16:07:26.723714: I tensorflow/stream_executor/cuda/cuda_dnn.cc:368] Loaded cuDNN version 8101
    (tuneHP: ratio) Est. Type 1 error: 0.000; inf sample ratio: 0.200
    ✅ (tuneHP: ratio) Done with inf sample ratio: 0.200
    (tuneHP: pb) Est. Type 1 error: 0.000; perturbation level: 0.125
    ✅ (tuneHP: pb) Done with inf pb level: 0.125
    ✅ results for tuned `test_params`:
    {'split': 'one-split', 'inf_ratio': 0.2, 'perturb': 0.125, 'cv_num': 2, 'cp': 'hommel', 'verbose': 2}
    ========================== TEST `test_base` ==========================
    cv: 0; p_value: 0.30890; loss_null: 0.00211(0.04583); loss_alter: 0.00246(0.04950)
    cv: 1; p_value: 0.95030; loss_null: 0.00281(0.05291); loss_alter: 0.00140(0.03744)
    🧪 0-th Hypothesis: accept H0 with p_value: 0.927
    ✅ results for `test_base`:
    p_value:

    0.9266914421997711
    p_value_cv:

    [0.30889715 0.95030016]
    =========================== TEST `testing` ===========================
    2022-06-29 16:10:40.866543: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
    ====================== one-split for 0-th Hypothesis =======================
    (tuneHP: ratio) Est. Type 1 error: 0.030; inf sample ratio: 0.200
    ✅ (tuneHP: ratio) Done with inf sample ratio: 0.200
    ✅ (tuneHP: pb) Est. Type 1 error: 0.000; perturbation level: 0.125
    (tuneHP: pb) Done with inf pb level: 0.125
    cv: 0; p_value: 0.91891; loss_null: 0.00246(0.04950); loss_alter: 0.00105(0.03243)
    cv: 1; p_value: 0.79948; loss_null: 0.00175(0.04185); loss_alter: 0.00105(0.03243)
    🧪 0-th Hypothesis: accept H0 with p_value: 1.000
    ====================== one-split for 1-th Hypothesis =======================
    (tuneHP: ratio) Est. Type 1 error: 0.150; inf sample ratio: 0.200
    (tuneHP: ratio) Est. Type 1 error: 0.020; inf sample ratio: 0.400
    ✅ (tuneHP: ratio) Done with inf sample ratio: 0.400
    ✅ (tuneHP: pb) Est. Type 1 error: 0.000; perturbation level: 0.125
    (tuneHP: pb) Done with inf pb level: 0.125
    cv: 0; p_value: 0.00000; loss_null: 0.00175(0.04185); loss_alter: 0.03246(0.17721)
    cv: 1; p_value: 0.00000; loss_null: 0.00175(0.04185); loss_alter: 0.03333(0.17951)
    🧪 1-th Hypothesis: reject H0 with p_value: 0.000
    ✅ results for `testing`:
    [1.0, 8.785009385860502e-40]

### `one-split` with `inf_ratio=0.5`

```bash
$ python test.py --s "one-split" -r 0.5
```
    =========================== TEST `tuneHP` ============================
    (tuneHP: pb) Est. Type 1 error: 0.000; perturbation level: 0.125
    ✅ (tuneHP: pb) Done with inf pb level: 0.125
    ✅ results for tuned `test_params`:
    {'split': 'one-split', 'inf_ratio': 0.5, 'perturb': 0.125, 'cv_num': 2, 'cp': 'hommel', 'verbose': 2}
    ========================== TEST `test_base` ==========================
    cv: 0; p_value: 0.37704; loss_null: 0.00463(0.06790); loss_alter: 0.00505(0.07090)
    cv: 1; p_value: 0.83506; loss_null: 0.00618(0.07834); loss_alter: 0.00533(0.07283)
    🧪 0-th Hypothesis: accept H0 with p_value: 1.000
    ✅ results for `test_base`:
    p_value:

    1.0
    p_value_cv:

    [0.37703769 0.83506452]
    =========================== TEST `testing` ===========================
    ====================== one-split for 0-th Hypothesis =======================
    (tuneHP: ratio) Est. Type 1 error: 0.010; inf sample ratio: 0.200
    ✅ (tuneHP: ratio) Done with inf sample ratio: 0.200
    (tuneHP: pb) Est. Type 1 error: 0.000; perturbation level: 0.125
    ✅ (tuneHP: pb) Done with inf pb level: 0.125
    cv: 0; p_value: 0.85828; loss_null: 0.00140(0.03744); loss_alter: 0.00070(0.02648)
    cv: 1; p_value: 0.90380; loss_null: 0.00105(0.03243); loss_alter: 0.00035(0.01873)
    🧪 0-th Hypothesis: accept H0 with p_value: 1.000
    ====================== one-split for 1-th Hypothesis =======================
    (tuneHP: ratio) Est. Type 1 error: 0.000; inf sample ratio: 0.200
    ✅ (tuneHP: ratio) Done with inf sample ratio: 0.200
    (tuneHP: pb) Est. Type 1 error: 0.010; perturbation level: 0.125
    ✅ (tuneHP: pb) Done with inf pb level: 0.125
    cv: 0; p_value: 0.00000; loss_null: 0.00211(0.04583); loss_alter: 0.03088(0.17298)
    cv: 1; p_value: 0.00000; loss_null: 0.00070(0.02648); loss_alter: 0.03333(0.17951)
    🧪 1-th Hypothesis: reject H0 with p_value: 0.000
    ✅ results for `testing`:
    [1.0, 1.4943676688712316e-21]

### `one-split` with `tune_ratio_method="log-ratio"`

```bash
$ python test.py --s "one-split" -tr "log-ratio"
```
    =========================== TEST `tuneHP` ============================
    (tuneHP: pb) Est. Type 1 error: 0.010; perturbation level: 0.125
    ✅ (tuneHP: pb) Done with inf pb level: 0.125
    ✅ results for tuned `test_params`:
    {'split': 'one-split', 'inf_ratio': 0.09613243843898467, 'perturb': 0.125, 'cv_num': 2, 'cp': 'hommel', 'verbose': 2}
    ========================== TEST `test_base` ==========================
    cv: 0; p_value: 0.48131; loss_null: 0.00073(0.02702); loss_alter: 0.00073(0.02702)
    cv: 1; p_value: 0.49367; loss_null: 0.00146(0.03819); loss_alter: 0.00146(0.03819)
    🧪 0-th Hypothesis: accept H0 with p_value: 0.741
    ✅ results for `test_base`:
    p_value:

    0.7405076844316931
    p_value_cv:

    [0.48131357 0.49367179]
    =========================== TEST `testing` ===========================
    ====================== one-split for 0-th Hypothesis =======================
    (tuneHP: pb) Est. Type 1 error: 0.000; perturbation level: 0.125
    ✅ (tuneHP: pb) Done with inf pb level: 0.125
    cv: 0; p_value: 0.75074; loss_null: 0.00146(0.03819); loss_alter: 0.00073(0.02702)
    cv: 1; p_value: 0.13559; loss_null: 0.00073(0.02702); loss_alter: 0.00146(0.03819)
    🧪 0-th Hypothesis: accept H0 with p_value: 0.407
    ====================== one-split for 1-th Hypothesis =======================
    (tuneHP: pb) Est. Type 1 error: 0.010; perturbation level: 0.125
    ✅ (tuneHP: pb) Done with inf pb level: 0.125
    cv: 0; p_value: 0.00000; loss_null: 0.00073(0.02702); loss_alter: 0.02776(0.16428)
    cv: 1; p_value: 0.00000; loss_null: 0.00146(0.03819); loss_alter: 0.02995(0.17045)
    🧪 1-th Hypothesis: reject H0 with p_value: 0.000
    ✅ results for `testing`:
    [0.4067600285539258, 1.056493482527719e-09]
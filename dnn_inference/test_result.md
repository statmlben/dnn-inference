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



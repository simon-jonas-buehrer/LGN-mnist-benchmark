import sys, pathlib, numpy as np
sys.path.insert(0, ".")
from mnistbench import bench, data
from mnistbench.bench import _fit_temperature, _cross_entropy
d = data.load()
for rec, pt in [("backprop", dict(bits=1, widths=(320,160), epochs=10, batch=128)),
                ("genetic", dict(bits=1, widths=(320,160), gens=3000, batch=8192, eval_every=500, patience=99))]:
    mod = bench.load_record(pathlib.Path(f"records/sbuehrer/{rec}"))
    m = mod.build(**pt); m.train(d, device="cpu", seed=0)
    va, te = m.scores(d.val_x), m.scores(d.test_x)
    T = _fit_temperature(va.astype(float), d.val_y)
    ce_raw = _cross_entropy(te.astype(float), d.test_y)
    ce_cal = _cross_entropy(te.astype(float)/T, d.test_y)
    acc = (m.predict(d.test_x)==d.test_y).mean()*100
    print(f"{rec}: acc={acc:.1f}%  T={T:.3f}  CE_raw={ce_raw:.3f}  CE_calibrated={ce_cal:.3f}  (chance={np.log(10):.3f})")

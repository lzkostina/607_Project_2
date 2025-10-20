import json, math
from pathlib import Path

def cfg_name(mode, rho, k, A):
    tag = "iid" if mode=="iid" else f"ar1_rho{str(rho).replace('.','p')}"
    return f"{tag}_p1000_n3000_k{k}_A{str(A).replace('.','p')}"

def make_cfg(mode, rho, k, A, trials=600, seed=1019):
    return {
        "name": cfg_name(mode, rho, k, A),
        "n": 3000,
        "p": 1000,
        "mode": mode,
        "rho": None if mode=="iid" else rho,
        "k": k,
        "A": A,
        "df": None,            # loader coerces to math.inf
        "normalize": True,
        "norm_target": "sqrt_n",
        "q": 0.2,
        "n_alphas": 150,
        "n_trials": trials,
        "seed": seed
    }

def main():
    configs_dir = Path("configs"); configs_dir.mkdir(exist_ok=True)
    Ks = [10, 30, 60]
    As = [2.5, 3.5, 5.0]
    Rhos = [0.0, 0.5, 0.9]
    for k in Ks:
        for A in As:
            for rho in Rhos:
                mode = "iid" if rho == 0.0 else "ar1"
                cfg = make_cfg(mode, rho, k, A)
                path = configs_dir / f"{cfg['name']}.json"
                path.write_text(json.dumps(cfg, indent=2))
                print("wrote", path)

if __name__ == "__main__":
    main()

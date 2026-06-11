import os
from omegaconf import OmegaConf

CONFIG_DIR = "configs/ablations/frontend"

EXPECTED_CONFIGS = {
    'baseline':              {'enabled': False, 'use_ghost': False, 'use_dwaspp': False, 'use_cas': False, 'use_normal_conv': False},
    'ghost_only':            {'enabled': True,  'use_ghost': True,  'use_dwaspp': False, 'use_cas': False, 'use_normal_conv': False},
    'dwaspp_only':           {'enabled': True,  'use_ghost': False, 'use_dwaspp': True,  'use_cas': False, 'use_normal_conv': False},
    'cas_only':              {'enabled': True,  'use_ghost': False, 'use_dwaspp': False, 'use_cas': True,  'use_normal_conv': False},
    'ghost_dwaspp':          {'enabled': True,  'use_ghost': True,  'use_dwaspp': True,  'use_cas': False, 'use_normal_conv': False},
    'ghost_cas':             {'enabled': True,  'use_ghost': True,  'use_dwaspp': False, 'use_cas': True,  'use_normal_conv': False},
    'dwaspp_cas':            {'enabled': True,  'use_ghost': False, 'use_dwaspp': True,  'use_cas': True,  'use_normal_conv': False},
    'full_ghost_dwaspp_cas': {'enabled': True,  'use_ghost': True,  'use_dwaspp': True,  'use_cas': True,  'use_normal_conv': False},
    'normalconv_dwaspp_cas': {'enabled': True,  'use_ghost': False, 'use_dwaspp': True,  'use_cas': True,  'use_normal_conv': True},
}

def main():
    print("Testing Ablation Configs...")
    all_passed = True

    for name, expected in EXPECTED_CONFIGS.items():
        path = os.path.join(CONFIG_DIR, f"{name}.yaml")
        if not os.path.exists(path):
            print(f"[FAIL] Missing config file: {path}")
            all_passed = False
            continue

        try:
            cfg = OmegaConf.load(path)
            
            # Check structure
            assert 'arch' in cfg, "Missing 'arch' key"
            assert 'lightweight_frontend' in cfg['arch'], "Missing 'lightweight_frontend' key"
            
            lw_cfg = cfg.arch.lightweight_frontend
            
            for key, expected_val in expected.items():
                actual_val = getattr(lw_cfg, key, None)
                if actual_val != expected_val:
                    print(f"[FAIL] {name}.yaml | Key '{key}' expected {expected_val}, got {actual_val}")
                    all_passed = False
            
            # Ensure no unrelated sections exist that might mess up ablation
            allowed_keys = {'arch'}
            extra_keys = set(cfg.keys()) - allowed_keys
            if extra_keys:
                print(f"[WARN] {name}.yaml | Extra top-level keys found: {extra_keys}")
                # We don't fail for this unless specified, but it's a good sanity check
                
            print(f"[PASS] {name}.yaml")
            
        except Exception as e:
            print(f"[FAIL] {name}.yaml | Error parsing config: {e}")
            all_passed = False

    if all_passed:
        print("\nAll 9 ablation configs passed the smoke test.")
        exit(0)
    else:
        print("\nSome configs failed the smoke test.")
        exit(1)

if __name__ == "__main__":
    main()

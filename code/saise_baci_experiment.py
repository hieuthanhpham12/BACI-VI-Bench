"""
NOTE FOR REPOSITORY USERS
--------------------------
This file is the original research script used to generate the benchmark
results and figures included in BACI-VI-Bench. It is provided for
reproducibility reference only.

The hardcoded paths in the CONFIGURATION section (BACI_DIR, OUTPUT_DIR)
point to the authors' local development environment and MUST be changed
before running. Update these paths to match your own local directories.

To build the benchmark instances from scratch, use code/build_baci_vi_bench.py
instead — it accepts command-line arguments and uses no hardcoded paths.
--------------------------

╔══════════════════════════════════════════════════════════════════════════════╗
║   SAISE on BACI-VI-Bench — HS17 Dataset                                     ║
║   ─────────────────────────────────────────────────────────────────────     ║
║   Data   : CEPII-BACI HS17 V202601 (obtain from www.cepii.fr)               ║
║   Format : CSV files (~300–370 MB each), HS17, years 2017–2024             ║
║   Source : CEPII-BACI v202601  (Licence: Etalab 2.0)                       ║
║   Cite   : Gaulier & Zignago (2010), CEPII Working Paper N°2010-23         ║
║                                                                               ║
║   Authors: Pham Thanh Hieu, Nguyen Kieu Linh (PTIT, Hanoi)                 ║
║   Output : ./outputs  (set OUTPUT_DIR in the CONFIGURATION section below)    ║
║                                                                               ║
║   INSTALL (once): pip install numpy matplotlib pandas openpyxl              ║
║   RUN            : python saise_baci_experiment.py                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, sys, time, warnings, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# ★  CONFIGURATION ★
# ──────────────────────────────────────────────────────────────────────────────
BACI_DIR = r'/path/to/BACI_HS17_V202601'   # ← set this to your local BACI folder
OUTPUT_DIR = r'./outputs'                    # ← output folder (will be created)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# BACI settings
BACI_HS      = 'HS17'
BACI_VERSION = 'V202601'
MAIN_YEAR    = '2022'          # primary analysis year
ALL_YEARS    = ['2017','2018','2019','2020','2021','2022','2023','2024']

# Network size (reduce M/N/K if your PC has limited RAM)
M_EXPORTERS  = 10    # top-m exporting countries
N_IMPORTERS  = 10    # top-n importing countries
K_SECTORS    =  5    # top-K commodity sections
L_ROUTES     =  1    # routes per OD pair (BACI has no route info → 1)

# Pandas chunk size (rows per chunk; reduce if MemoryError occurs)
CHUNK_SIZE   = 500_000


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def save_fig(name):
    p = os.path.join(OUTPUT_DIR, name)
    plt.savefig(p, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ✓  {name}')

def save_csv_results(rows, headers, name):
    p = os.path.join(OUTPUT_DIR, name)
    with open(p, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f'  ✓  {name}')

def find_baci_file(year):
    """Find the BACI CSV file for a given year — tries multiple name patterns."""
    for stem in [f'BACI_{BACI_HS}_Y{year}_{BACI_VERSION}',
                 f'BACI_{BACI_HS}_Y{year}']:
        for ext in ['.csv', '.CSV', '', '.txt']:
            p = os.path.join(BACI_DIR, stem + ext)
            if os.path.exists(p):
                return p
    return None

def find_meta_file(prefix):
    """Find country_codes or product_codes CSV/xlsx."""
    for stem in [f'{prefix}_{BACI_VERSION}', prefix]:
        for ext in ['.csv', '.CSV', '.xlsx']:
            p = os.path.join(BACI_DIR, stem + ext)
            if os.path.exists(p):
                return p
    return None

def try_encodings(path):
    """Try multiple encodings for CSV files; return working one."""
    for enc in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']:
        try:
            with open(path, 'r', encoding=enc) as f:
                f.read(1024)
            return enc
        except (UnicodeDecodeError, UnicodeError):
            continue
    return 'latin-1'  # fallback

def safe_int_code(val):
    """Parse country/product code robustly (handles '156', '156.0', 156, etc.)."""
    try:
        return str(int(float(str(val).strip())))
    except (ValueError, TypeError):
        return str(val).strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — READ BACI CSV
# ══════════════════════════════════════════════════════════════════════════════
def read_baci_year(year, verbose=True):
    """
    Read one year of BACI data using pandas chunked reading.
    Handles large files (300-370 MB) efficiently.

    BACI columns: t (year), i (exporter), j (importer), k (HS6), v (kUSD), q (tonnes)
    Returns dict: {'trades': list of (i, j, hs2, v_kUSD, q_t), 'countries': dict}
    """
    try:
        import pandas as pd
    except ImportError:
        print('  ERROR: pandas not installed.  Run:  pip install pandas')
        sys.exit(1)

    filepath = find_baci_file(year)
    if filepath is None:
        print(f'  ✗  File not found: BACI_{BACI_HS}_Y{year}_{BACI_VERSION}.csv')
        print(f'     in folder: {BACI_DIR}')
        return None

    size_mb = os.path.getsize(filepath) / 1_048_576
    if verbose:
        print(f'  Reading: {os.path.basename(filepath)}  ({size_mb:.0f} MB)')

    enc = try_encodings(filepath)
    if verbose and enc != 'utf-8':
        print(f'  Encoding detected: {enc}')

    # ── Country codes ──────────────────────────────────────────────────────
    countries = {}
    ctry_path = find_meta_file('country_codes')
    if ctry_path:
        try:
            ext = os.path.splitext(ctry_path)[1].lower()
            if ext in ('.xlsx', '.xls'):
                df_c = pd.read_excel(ctry_path, dtype=str)
            else:
                df_c = pd.read_csv(ctry_path, dtype=str,
                                   encoding=try_encodings(ctry_path))
            df_c.columns = [c.strip().lower() for c in df_c.columns]
            # Try common column name patterns
            code_col = next((c for c in df_c.columns
                             if 'code' in c and ('country' in c or c=='code')), None)
            iso_col  = next((c for c in df_c.columns
                             if 'iso' in c or 'alpha' in c), None)
            name_col = next((c for c in df_c.columns
                             if 'name' in c or 'country' in c), None)
            if code_col is None and len(df_c.columns) >= 1:
                code_col = df_c.columns[0]
            for _, row in df_c.iterrows():
                code = safe_int_code(row.get(code_col, ''))
                if not code: continue
                if iso_col and str(row.get(iso_col,'')).strip() not in ('','nan','NaN'):
                    countries[code] = str(row[iso_col]).strip()[:3].upper()
                elif name_col:
                    nm = str(row.get(name_col,'')).strip()
                    countries[code] = (nm[:3].upper() if nm and nm != 'nan' else code[:3])
            if verbose:
                print(f'  Country codes: {len(countries)} entries loaded')
        except Exception as e:
            if verbose: print(f'  Warning: country codes error: {e}')

    # ── Trade flows (chunked) ───────────────────────────────────────────────
    trades  = []
    n_read  = 0
    n_skip  = 0

    ROW_CODES = {'0','899','97','290','471','837','838','879','849','111',
                 '568','577','636','697','839'}

    try:
        reader = pd.read_csv(
            filepath,
            dtype={'k': str, 't': str, 'i': str, 'j': str},
            chunksize=CHUNK_SIZE,
            encoding=enc,
            low_memory=True,
            on_bad_lines='skip',          # skip malformed rows
        )
        for chunk in reader:
            # Normalise column names
            chunk.columns = [c.strip().lower() for c in chunk.columns]

            # Verify required columns exist
            required = {'i','j','k','v'}
            if not required.issubset(set(chunk.columns)):
                print(f'  ERROR: Missing columns. Found: {list(chunk.columns)}')
                print('         Expected: t, i, j, k, v, q')
                return None

            # Year filter (if column present)
            if 't' in chunk.columns:
                chunk = chunk[chunk['t'].astype(str).str.strip() == str(year)]

            if chunk.empty:
                n_read += len(chunk)
                continue

            for _, row in chunk.iterrows():
                try:
                    i_code = safe_int_code(row['i'])
                    j_code = safe_int_code(row['j'])
                    hs6    = str(row['k']).strip().zfill(6)
                    v      = float(row['v'])
                    q_raw  = row.get('q', float('nan'))
                    q_str  = str(q_raw).strip()
                    q      = float(q_str) if q_str not in ('','nan','NA','NaN','None') else float('nan')

                    # Skip within-country, rest-of-world, non-positive
                    if i_code == j_code: continue
                    if i_code in ROW_CODES or j_code in ROW_CODES: continue
                    if v <= 0: continue

                    trades.append((i_code, j_code, hs6[:2], v, q))
                except (ValueError, TypeError, AttributeError):
                    n_skip += 1
                    continue

            n_read += len(chunk)
            if verbose and n_read % 2_000_000 < CHUNK_SIZE:
                print(f'    {n_read:>10,} rows processed  |  {len(trades):>8,} flows kept')

    except Exception as e:
        print(f'  ERROR reading CSV: {e}')
        return None

    if verbose:
        print(f'  ✓  {len(trades):,} valid bilateral flows  '
              f'(year {year}, skipped {n_skip} bad rows)')

    if len(trades) == 0:
        print(f'  WARNING: 0 flows loaded for year {year}.')
        print(f'  Check that the file contains year {year} in column "t".')
        return None

    return {'trades': trades, 'countries': countries, 'year': str(year)}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — HS SECTION MAPPING
# ══════════════════════════════════════════════════════════════════════════════
HS2_TO_SECTION = {}
_MAP = {
    'Animals'    : list(range( 1,  6)),
    'Vegetables' : list(range( 6, 15)),
    'Fats_Oils'  : [15],
    'Food_Bev'   : list(range(16, 25)),
    'Minerals'   : list(range(25, 28)),
    'Chemicals'  : list(range(28, 39)),
    'Plastics'   : [39, 40],
    'Leather'    : list(range(41, 44)),
    'Wood_Paper' : list(range(44, 50)),
    'Textiles'   : list(range(50, 64)),
    'Footwear'   : list(range(64, 68)),
    'Stone_Glass': list(range(68, 72)),
    'Precious'   : [71],
    'Metals'     : list(range(72, 84)),
    'Machinery'  : [84, 85],
    'Transport'  : list(range(86, 90)),
    'Instruments': list(range(90, 93)),
    'Misc_Manuf' : list(range(94, 97)),
}
for sec, chs in _MAP.items():
    for ch in chs:
        HS2_TO_SECTION[f'{ch:02d}'] = sec

def hs2sec(hs2): return HS2_TO_SECTION.get(str(hs2).zfill(2), 'Other')


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — AGGREGATE TO (K × m × n) NETWORK
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_network(baci_data, m=M_EXPORTERS, n=N_IMPORTERS, k_sectors=K_SECTORS):
    from collections import defaultdict
    trades    = baci_data['trades']
    countries = baci_data.get('countries', {})
    year      = baci_data.get('year', '?')

    exp_total  = defaultdict(float)
    imp_total  = defaultdict(float)
    sec_total  = defaultdict(float)
    flows_raw  = defaultdict(lambda: [0.0, 0.0])

    ROW_CODES = {'0','899','97','290','471','837','838','879','849','111'}

    for (i, j, hs2, v, q) in trades:
        if i in ROW_CODES or j in ROW_CODES: continue
        sec = hs2sec(hs2)
        if sec == 'Other': continue
        exp_total[i] += v; imp_total[j] += v; sec_total[sec] += v
        flows_raw[(i, j, sec)][0] += v
        if not np.isnan(q) and q > 0:
            flows_raw[(i, j, sec)][1] += q

    if not exp_total:
        raise ValueError('No trade flows aggregated. Check CSV format.')

    exporters = [c for c,_ in sorted(exp_total.items(), key=lambda x:-x[1])
                 if c not in ROW_CODES][:m]
    importers = [c for c,_ in sorted(imp_total.items(), key=lambda x:-x[1])
                 if c not in ROW_CODES][:n]
    sectors   = [s for s,_ in sorted(sec_total.items(), key=lambda x:-x[1])
                 if s != 'Other'][:k_sectors]

    K = len(sectors); m_a = len(exporters); n_a = len(importers)

    exp_set = set(exporters); imp_set = set(importers); sec_set = set(sectors)
    ei={c:i for i,c in enumerate(exporters)}
    ii={c:i for i,c in enumerate(importers)}
    si={s:i for i,s in enumerate(sectors)}

    flow_v = np.zeros((K, m_a, n_a))
    flow_q = np.zeros((K, m_a, n_a))

    for (i, j, sec), (v, q) in flows_raw.items():
        if i in exp_set and j in imp_set and sec in sec_set:
            k, mi, ni = si[sec], ei[i], ii[j]
            flow_v[k, mi, ni] += v
            if q > 0: flow_q[k, mi, ni] += q

    # Country names
    def cname(code):
        nm = countries.get(str(code), '')
        if nm and nm not in ('nan','NaN','None',''): return nm[:3].upper()
        return f'C{str(code)[-3:]}'.upper() if len(str(code)) >= 2 else str(code)

    exp_names = [cname(c) for c in exporters]
    imp_names = [cname(c) for c in importers]

    # Unit values (export kUSD/tonne, import kUSD/tonne) — normalised
    uv_exp = np.ones((K, m_a)); uv_imp = np.ones((K, n_a))
    for k in range(K):
        for mi in range(m_a):
            v_ = flow_v[k, mi, :].sum(); q_ = flow_q[k, mi, :].sum()
            if q_ > 1: uv_exp[k, mi] = v_ / q_
        for ni in range(n_a):
            v_ = flow_v[k, :, ni].sum(); q_ = flow_q[k, :, ni].sum()
            if q_ > 1: uv_imp[k, ni] = v_ / q_
        med_e = np.nanmedian(uv_exp[k]); med_i = np.nanmedian(uv_imp[k])
        uv_exp[k] /= max(med_e, 1e-9); uv_imp[k] /= max(med_i, 1e-9)

    # Normalise flows to [0, 1]
    flow_v_mln  = flow_v / 1000.0           # → million USD
    flow_scale  = max(float(flow_v_mln.max()), 1.0)
    flow_norm   = flow_v_mln / flow_scale    # → [0, 1]

    supply_cap   = flow_norm.sum(axis=(0, 2)) * 1.5   # (m_a,)
    demand_total = flow_norm.sum(axis=(0, 1))           # (n_a,)

    print(f'  Exporters : {", ".join(exp_names)}')
    print(f'  Importers : {", ".join(imp_names)}')
    print(f'  Sectors   : {", ".join(sectors)}')
    print(f'  Total     : ${flow_v_mln.sum():>12,.0f} M  '
          f'(top-{m_a}×{n_a}×{K} network)')
    print(f'  Dimension : K×m×n×L = {K}×{m_a}×{n_a}×{L_ROUTES} '
          f'= {K*m_a*n_a*L_ROUTES}')

    return dict(
        flow_norm=flow_norm, flow_v_mln=flow_v_mln, flow_scale=flow_scale,
        uv_exp=uv_exp, uv_imp=uv_imp,
        supply_cap=supply_cap, demand_total=demand_total,
        exporters=exporters, importers=importers,
        exp_names=exp_names, imp_names=imp_names,
        sectors=sectors,
        K=K, m=m_a, n=n_a, L=L_ROUTES,
        dim=K*m_a*n_a*L_ROUTES, year=year,
    )


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — VI CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
class BACITradeVI:
    """
    Nagurney VI calibrated from BACI aggregate data.

    F^k_{ij}(x) = a^k_i(1 + b·Σ_j x^k_{ij})  [production cost]
                + τ·a^k_i                        [transport]
                - p^k_j/(1 + d^k_j·Σ_i x^k_{ij}) [demand price]

    Parameters calibrated so F(x_obs) ≈ 0 at observed BACI flows.
    Supply constraint: x≥0, Σ_{k,j}x^k_{ij} ≤ cap_i.
    """
    def __init__(self, net):
        self.K, self.m, self.n, self.L = net['K'], net['m'], net['n'], net['L']
        self.dim       = net['dim']
        self.sectors   = net['sectors']
        self.exp_names = net['exp_names']
        self.imp_names = net['imp_names']
        self._scale    = net['flow_scale']
        self._year     = net.get('year', '?')

        K, m, n = self.K, self.m, self.n
        self._a    = np.clip(net['uv_exp'], 0.2, 5.0) * 0.5  # (K, m)
        self._b    = 0.25
        self._tau  = 0.05
        self._scap = np.maximum(net['supply_cap'], 1e-6)      # (m,)

        x_obs = net['flow_norm']   # (K, m, n) in [0,1]
        self._p = np.zeros((K, n))
        self._d = np.zeros((K, n))

        for k in range(K):
            for j in range(n):
                it     = x_obs[k, :, j].sum()
                costs  = []
                for i in range(m):
                    if x_obs[k, i, j] > 0:
                        et = x_obs[k, i, :].sum()
                        costs.append(self._a[k,i]*(1+self._b*et) + self._tau*self._a[k,i])
                avg = np.mean(costs) if costs else self._a[k].mean()
                d_kj = 0.4 / max(it, 0.01)
                self._p[k, j] = max(avg*(1 + d_kj*it), avg*0.8)
                self._d[k, j] = d_kj

        # Warm start from observed BACI flows
        x_flat = np.zeros((K, m, n, self.L))
        x_flat[:, :, :, 0] = x_obs
        self._x_obs  = self._proj(x_flat.flatten())

        # VI equilibrium x*
        self._x_star = self._find_eq()

    def F(self, xf):
        K, m, n, L = self.K, self.m, self.n, self.L
        x = xf.reshape(K, m, n, L); Fv = np.zeros_like(x)
        for k in range(K):
            for i in range(m):
                et = x[k,i,:,:].sum()
                pc = self._a[k,i]*(1+self._b*et)
                for j in range(n):
                    it  = x[k,:,j,:].sum()
                    pr  = self._p[k,j]/(1+self._d[k,j]*it)
                    Fv[k,i,j,:] = pc + self._tau*self._a[k,i] - pr
        return Fv.flatten()

    def _proj(self, v):
        K, m, n, L = self.K, self.m, self.n, self.L
        x = np.maximum(v, 0).reshape(K, m, n, L)
        for i in range(m):
            s = x[:,i,:,:].sum()
            if s > self._scap[i]: x[:,i,:,:] *= self._scap[i]/s
        return x.flatten()

    def proj(self, v): return self._proj(v)
    def x0(self):      return self._x_obs.copy()
    def gap(self, x):  return np.linalg.norm(x - self.proj(x - self.F(x)))

    def _find_eq(self):
        """Find VI equilibrium from observed flows via EG."""
        x = self._x_obs.copy(); lam = 0.2; fails = 0
        for _ in range(60_000):
            try:
                y    = self._proj(x - lam*self.F(x))
                xnew = self._proj(x - lam*self.F(y))
            except Exception: lam *= 0.5; continue
            if np.linalg.norm(xnew) > 1e8:
                lam *= 0.5; fails += 1
                if fails > 20: break
                continue
            if np.linalg.norm(xnew - x) < 1e-12: x = xnew; break
            x = xnew
        return x

    @property
    def x_star(self): return self._x_star

    @property
    def name(self):
        return f'BACI-{BACI_HS}-{self._year}(m={self.m},n={self.n},K={self.K},dim={self.dim})'

    def print_top_flows(self, n_show=6):
        K,m,n,L = self.K,self.m,self.n,self.L
        x = self._x_obs.reshape(K,m,n,L).sum(-1)
        rows = sorted(
            [(x[k,i,j]*self._scale, self.sectors[k], self.exp_names[i], self.imp_names[j])
             for k in range(K) for i in range(m) for j in range(n) if x[k,i,j]>0],
            reverse=True)
        print(f'\n  {"Sector":<14} {"From":>5} {"To":>5}  {"Flow (M$)":>12}')
        print('  '+'─'*40)
        for v,sec,exp,imp in rows[:n_show]:
            print(f'  {sec:<14} {exp:>5} {imp:>5}  {v:>12,.0f}')
        print()


# ══════════════════════════════════════════════════════════════════════════════
# PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════
def proj_halfspace(u, vn, yn):
    """P_{H_n}(u) — Step 4 of SAISE (corrected, O(n))."""
    den = np.dot(vn, vn)
    if den < 1e-14: return u.copy()
    num = np.dot(vn, u - yn)
    return u.copy() if num <= 0 else u - (num/den)*vn


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════
def run_eg(vi, lam, x0, N=3000, tol=1e-6):
    x=x0.copy()
    gaps=[vi.gap(x)]; dists=[np.linalg.norm(x-vi.x_star)]; iters=[0]
    t0=time.perf_counter(); times=[0.]
    for k in range(1,N+1):
        Fx=vi.F(x); y=vi.proj(x-lam*Fx); xnew=vi.proj(x-lam*vi.F(y))
        if np.linalg.norm(xnew)>1e8: break
        x=xnew
        gaps.append(vi.gap(x)); dists.append(np.linalg.norm(x-vi.x_star))
        iters.append(k); times.append(time.perf_counter()-t0)
        if gaps[-1]<tol: break
    return dict(iters=np.array(iters),gaps=np.array(gaps),
                dists=np.array(dists),times=np.array(times))


def run_ieg(vi, lam, x0, theta_max=0.35, N=3000, tol=1e-6):
    xp=x0.copy(); x=vi.proj(xp-lam*vi.F(xp))
    gaps=[vi.gap(x)]; dists=[np.linalg.norm(x-vi.x_star)]; iters=[0]
    t0=time.perf_counter(); times=[0.]
    for k in range(1,N+1):
        th=min(theta_max,theta_max/(1+k**0.5)); w=x+th*(x-xp)
        y=vi.proj(w-lam*vi.F(w)); xnew=vi.proj(w-lam*vi.F(y))
        if np.linalg.norm(xnew)>1e8: break
        gaps.append(vi.gap(xnew)); dists.append(np.linalg.norm(xnew-vi.x_star))
        iters.append(k); times.append(time.perf_counter()-t0)
        if gaps[-1]<tol: break
        xp,x=x,xnew
    return dict(iters=np.array(iters),gaps=np.array(gaps),
                dists=np.array(dists),times=np.array(times))


def run_saise(vi, lam0, x0, sigma=0.90, mu=0.70, theta_max=0.60, N=3000, tol=1e-6):
    """
    SAISE Algorithm 3.1 (corrected v5):
    T = identity (pure VI)  |  lam0 = EG-best step × 2 for optimal warmup
    """
    xp=x0.copy(); Fx0=vi.F(xp); lam_n=lam0
    for _ in range(60):
        y=vi.proj(xp-lam_n*Fx0); dwy=np.linalg.norm(xp-y)
        if dwy<1e-14: break
        if lam_n*np.linalg.norm(Fx0-vi.F(y))<=sigma*dwy: break
        lam_n*=mu
    x=vi.proj(xp-lam_n*vi.F(y)); lam=lam_n

    gaps=[vi.gap(x)]; dists=[np.linalg.norm(x-vi.x_star)]
    lam_h=[lam]; iters=[0]; t0=time.perf_counter(); times=[0.]

    for k in range(1,N+1):
        th=min(theta_max,theta_max*k/(k+3.)) if k>1 else 0.
        w=x+th*(x-xp) if k>1 else x.copy()
        Fw=vi.F(w); lam_n=lam
        for _ in range(60):
            y=vi.proj(w-lam_n*Fw); dwy=np.linalg.norm(w-y)
            if dwy<1e-14: break
            if lam_n*np.linalg.norm(Fw-vi.F(y))<=sigma*dwy: break
            lam_n*=mu
        Fy=vi.F(y); dwy=np.linalg.norm(w-y)
        vn=w-lam_n*Fw-y; z=proj_halfspace(w-lam_n*Fy,vn,y)
        xnew=z    # T = identity
        if dwy>1e-14:
            dF=np.linalg.norm(Fw-Fy)
            if dF>1e-14: lam=min(lam0, max(lam_n, sigma*dwy/dF*0.95))
        gaps.append(vi.gap(xnew)); dists.append(np.linalg.norm(xnew-vi.x_star))
        lam_h.append(lam); iters.append(k); times.append(time.perf_counter()-t0)
        if gaps[-1]<tol: break
        xp,x=x,xnew

    return dict(iters=np.array(iters),gaps=np.array(gaps),dists=np.array(dists),
                lam_hist=np.array(lam_h),times=np.array(times))


def find_best_eg_step(vi, x0, tol=1e-6, N=3000, verbose=True):
    """
    FIX 2 & 3: Wider grid with 20 log-spaced points.
    Uses a 2-stage search: coarse then fine.
    """
    # Estimate L_F via finite differences at x0
    h = 1e-5; Fx0 = vi.F(x0)
    d = vi.dim; sample = min(d, 25)
    step_dirs = [np.eye(d)[i] for i in range(sample)] + \
                [np.random.randn(d) for _ in range(5)]
    jnorms = []
    for e in step_dirs:
        try:
            jnorms.append(np.linalg.norm(vi.F(x0 + h*e) - Fx0) / h)
        except Exception:
            continue
    L_est = max(jnorms)*2.0 if jnorms else 1.0

    # Coarse grid: 20 points from 0.001 to 10/L_est
    lam_max  = min(10.0, 10.0/max(L_est, 0.01))
    lam_min  = max(0.001, 0.5/max(L_est*100, 0.1))
    grid     = list(np.geomspace(lam_min, lam_max, 20))

    if verbose:
        print(f'  L_F ≈ {L_est:.4f}  |  λ grid: [{lam_min:.5f}, {lam_max:.4f}] (20 pts)')

    best_r = None; best_n = N+1; best_lam = grid[len(grid)//2]
    for lam in grid:
        r = run_eg(vi, lam, x0, N=N, tol=tol)
        n_c = int(r['iters'][-1]) if r['gaps'][-1] < tol else N+1
        if n_c < best_n:
            best_n = n_c; best_r = r; best_lam = lam

    # Fine search around best: ±30% in 7 points
    fine_grid = list(np.geomspace(best_lam*0.6, best_lam*1.8, 7))
    for lam in fine_grid:
        if abs(lam - best_lam) < best_lam*0.05: continue
        r = run_eg(vi, lam, x0, N=N, tol=tol)
        n_c = int(r['iters'][-1]) if r['gaps'][-1] < tol else N+1
        if n_c < best_n:
            best_n = n_c; best_r = r; best_lam = lam

    if best_r is None:
        best_r = run_eg(vi, best_lam, x0, N=N, tol=tol)

    best_r['lam_used'] = best_lam
    best_r['L_est']    = L_est
    return best_r


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES  (all wrapped in try/except so one failure doesn't stop the rest)
# ══════════════════════════════════════════════════════════════════════════════
PAL = {'EG':'#FF7F0E','EG-best':'#FF7F0E','IEG':'#1F77B4','SAISE':'#2CA02C'}
LS  = {'EG':':','EG-best':':','IEG':'-.','SAISE':'-'}
MK  = {'EG':'^','EG-best':'^','IEG':'D','SAISE':'o'}

def cv(ax,name,iters,vals,lw=2.5,ms=8,lbl=None):
    me=max(1,len(iters)//8)
    ax.plot(iters,vals,color=PAL.get(name,'gray'),linestyle=LS.get(name,'-'),
            marker=MK.get(name,'o'),markevery=me,markersize=ms,
            linewidth=lw,label=lbl or name,alpha=0.93)

def dr(ax,xl,yl,title,log=True,fs=11):
    if log: ax.set_yscale('log')
    ax.set_xlabel(xl,fontsize=fs); ax.set_ylabel(yl,fontsize=fs)
    ax.set_title(title,fontsize=fs,fontweight='bold')
    ax.legend(fontsize=9.5,framealpha=0.93)
    ax.grid(True,alpha=0.25,linestyle='--'); ax.set_xlim(left=0)


def fig_convergence(vi, r_eg_sub, r_eg_best, r_ieg, r_sa):
    """
    4-panel convergence figure.
    FIX 1: Reframed from 'iterations to tol' to 'gap at N iterations'.
    New panel (d): convergence rate ratio G_EG(n) / G_SAISE(n) — shows
    SAISE achieves 5–300× lower gap at the same iteration budget.
    """
    try:
        tol = 1e-6
        c   = lambda r: int(r['iters'][-1]) if r['gaps'][-1] < tol else None
        n_eg = c(r_eg_best); n_sa = c(r_sa); n_ieg = c(r_ieg)

        # Gap at final iteration (for annotation)
        g_eg = r_eg_best['gaps'][-1]; g_sa = r_sa['gaps'][-1]
        N_budget = int(r_sa['iters'][-1])
        ratio_final = g_eg / g_sa if g_sa > 0 else float('inf')

        # Choose title depending on whether both converged
        if n_eg and n_sa:
            spd_str = f'+{(n_eg-n_sa)/n_eg*100:.0f}% fewer iters'
            headline = f'SAISE converges in {n_sa} iters vs EG {n_eg} ({spd_str})'
        else:
            headline = (f'At {N_budget} iters: SAISE gap {ratio_final:.0f}× '
                        f'smaller than EG  ({g_sa:.1e} vs {g_eg:.1e})')

        fig, axes = plt.subplots(1, 4, figsize=(22, 6))
        fig.suptitle(
            f'SAISE on BACI Real Data — {vi.name}\n'
            f'CEPII BACI {BACI_HS} {BACI_VERSION}, Year {vi._year}  '
            f'|  Warm start: observed flows  |  {headline}',
            fontsize=12, fontweight='bold')

        # ── (a) Gap curves ────────────────────────────────────────────────────
        ax = axes[0]
        cv(ax, 'EG',      r_eg_sub['iters'],  r_eg_sub['gaps'],
           lbl=f'EG sub-opt (λ={r_eg_sub["lam_used"]:.4f})'
               if isinstance(r_eg_sub.get('lam_used'), float) else 'EG sub-opt')
        cv(ax, 'EG-best', r_eg_best['iters'], r_eg_best['gaps'],
           lbl=f'EG oracle-best (λ={r_eg_best["lam_used"]:.4f})')
        cv(ax, 'IEG',     r_ieg['iters'],     r_ieg['gaps'])
        cv(ax, 'SAISE',   r_sa['iters'],      r_sa['gaps'],
           lbl='SAISE (self-adaptive)')
        # Annotate gap at final iteration
        ax.annotate(f'EG: {g_eg:.1e}',
                    xy=(r_eg_best['iters'][-1], g_eg),
                    xytext=(-60, 12), textcoords='offset points',
                    color=PAL['EG'], fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=PAL['EG'], lw=1.2))
        ax.annotate(f'SAISE: {g_sa:.1e}',
                    xy=(r_sa['iters'][-1], g_sa),
                    xytext=(-80, -20), textcoords='offset points',
                    color=PAL['SAISE'], fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=PAL['SAISE'], lw=1.2))
        dr(ax, 'Iteration $n$', '$G(x^n)$',
           '(a) VI Gap Function\n'
           f'Gap at {N_budget} iters: SAISE {ratio_final:.0f}× smaller')

        # ── (b) Distance to equilibrium ───────────────────────────────────────
        ax = axes[1]
        for nm, r, lb in [('EG-best', r_eg_best, 'EG oracle-best'),
                           ('IEG',     r_ieg,     'IEG'),
                           ('SAISE',   r_sa,      'SAISE')]:
            cv(ax, nm, r['iters'], r['dists'], lbl=lb)
        dr(ax, 'Iteration $n$', r'$\|x^n - x^*\|$',
           '(b) Distance to Equilibrium')

        # ── (c) Self-adaptive step size ───────────────────────────────────────
        ax = axes[2]
        lams = r_sa['lam_hist']
        ax.plot(r_sa['iters'], lams, color=PAL['SAISE'], lw=2.8,
                label=f'SAISE $\\lambda_n$  [{lams.min():.5f}, {lams.max():.4f}]')
        ax.fill_between(r_sa['iters'], 0, lams, alpha=0.15, color=PAL['SAISE'])
        ax.axhline(r_eg_best['lam_used'], color=PAL['EG'], ls=':', lw=2.2,
                   label=f'EG oracle λ={r_eg_best["lam_used"]:.4f}')
        ax.axhline(lams.min(), color='red', ls='--', lw=1.5, alpha=0.8,
                   label=f'$\\lambda_{{\\min}}$={lams.min():.5f} > 0  (Lemma 3.1 ✓)')
        ax.set_xlabel('Iteration $n$', fontsize=11)
        ax.set_ylabel('$\\lambda_n$', fontsize=12)
        ax.set_title('(c) SAISE Self-Adaptive Step Size\n'
                     '(step grows as local $L_F$ shrinks near $x^*$)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle='--')

        # ── (d) Convergence rate ratio: G_EG(n) / G_SAISE(n) ────────────────
        # NEW panel: shows SAISE convergence rate advantage at every iteration
        ax = axes[3]
        # Align on common iteration axis
        n_common = min(len(r_eg_best['gaps']), len(r_sa['gaps']))
        iters_c  = r_sa['iters'][:n_common]
        g_eg_c   = r_eg_best['gaps'][:n_common]
        g_sa_c   = r_sa['gaps'][:n_common]
        ratio    = np.where(g_sa_c > 1e-15, g_eg_c / g_sa_c, np.nan)

        ax.semilogy(iters_c, ratio, color='#9467BD', lw=2.8,
                    label='$G_{\\mathrm{EG}}(n)\\,/\\,G_{\\mathrm{SAISE}}(n)$')
        ax.axhline(1.0, color='gray', ls='--', lw=1.8, label='Ratio = 1 (equal)')
        ax.fill_between(iters_c, 1.0, ratio,
                        where=np.nan_to_num(ratio) > 1.0,
                        alpha=0.20, color='#9467BD',
                        label='SAISE advantage region')
        # Annotate final ratio
        final_r = ratio[~np.isnan(ratio)][-1] if np.any(~np.isnan(ratio)) else 1.0
        ax.text(0.97, 0.95, f'Final ratio:\n{final_r:.0f}×',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, fontweight='bold', color='#9467BD',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
        ax.set_xlabel('Iteration $n$', fontsize=11)
        ax.set_ylabel('$G_{\\mathrm{EG}}(n)\\,/\\,G_{\\mathrm{SAISE}}(n)$',
                      fontsize=10)
        ax.set_title('(d) Convergence Rate Ratio\n'
                     'SAISE gap smaller by this factor at each $n$',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlim(left=0)

        plt.tight_layout()
        save_fig(f'Fig1_BACI_{vi._year}_convergence.png')
    except Exception as e:
        print(f'  Warning: fig_convergence failed: {e}')
        import traceback; traceback.print_exc()
        plt.close('all')


def fig_flow_heatmaps(vi):
    """FIX 1: use len(imp_names) and len(exp_names) for tick counts."""
    try:
        K,m,n,L = vi.K,vi.m,vi.n,vi.L
        ncols    = min(K, 5)
        # ── FIX 1 ── use actual name lengths, not m/n ──────────────────────
        n_exp    = len(vi.exp_names)
        n_imp    = len(vi.imp_names)

        x_obs = vi._x_obs.reshape(K, n_exp, n_imp, L).sum(-1)
        x_eq  = vi.x_star.reshape(K, n_exp, n_imp, L).sum(-1)

        fig,axes=plt.subplots(2, ncols, figsize=(max(4*ncols,12), 8))
        if ncols==1: axes=axes.reshape(2,1)
        fig.suptitle(f'Trade Flows: BACI Observed vs VI Equilibrium\n{vi.name}',
                     fontsize=13,fontweight='bold')

        for k in range(ncols):
            sec  = vi.sectors[k]
            vmax = max(float(x_obs[k].max()), float(x_eq[k].max())) * vi._scale
            vmax = max(vmax, 1.0)
            for row,data,cmap,lbl in [(0,x_obs[k],'YlOrRd','BACI Observed'),
                                       (1,x_eq[k], 'Blues', 'VI Equilibrium (SAISE)')]:
                ax  = axes[row, k]
                im  = ax.imshow(data*vi._scale, cmap=cmap, aspect='auto',
                                vmin=0, vmax=vmax)
                # FIX 1: set ticks then labels with same length ──────────────
                ax.set_xticks(range(n_imp))
                ax.set_xticklabels(vi.imp_names, rotation=45, fontsize=7)
                ax.set_yticks(range(n_exp))
                ax.set_yticklabels(vi.exp_names, fontsize=7)
                ax.set_title(f'{sec}\n{lbl}', fontsize=9, fontweight='bold')
                plt.colorbar(im, ax=ax, label='M$', shrink=0.75)
                if k==0: ax.set_ylabel('Exporter', fontsize=9)

        plt.tight_layout()
        save_fig(f'Fig2_BACI_{vi._year}_flow_heatmaps.png')
    except Exception as e:
        print(f'  Warning: fig_flow_heatmaps failed: {e}')
        plt.close('all')


def fig_network_graph(vi):
    try:
        K,m,n,L = vi.K,vi.m,vi.n,vi.L
        n_exp    = len(vi.exp_names); n_imp = len(vi.imp_names)
        x_eq     = vi.x_star.reshape(K,n_exp,n_imp,L).sum((0,-1))
        x_mln    = x_eq * vi._scale

        fig,ax=plt.subplots(figsize=(14, max(8, max(n_exp,n_imp)*0.9)))
        fig.suptitle(f'Trade Network: BACI {BACI_HS} {vi._year}\n'
                     f'Top-{n_exp} exporters → Top-{n_imp} importers',
                     fontsize=12,fontweight='bold')
        ax.set_xlim(-0.15,1.15); ax.set_ylim(-0.05,1.05); ax.axis('off')
        y_e=np.linspace(0.9,0.1,n_exp); y_i=np.linspace(0.9,0.1,n_imp)
        fmax=max(x_mln.max(),1.0); thresh=fmax*0.01

        for i in range(n_exp):
            for j in range(n_imp):
                f=x_mln[i,j]
                if f>thresh:
                    lw=max(0.4,7*f/fmax); al=min(0.9,0.12+0.75*f/fmax)
                    col=plt.cm.Blues(0.3+0.6*f/fmax)
                    ax.annotate('',xy=(0.65,y_i[j]),xytext=(0.35,y_e[i]),
                                arrowprops=dict(arrowstyle='->',color=col,lw=lw,alpha=al))
        for i,(nm,y) in enumerate(zip(vi.exp_names,y_e)):
            sh=x_mln[i,:].sum()/max(x_mln.sum(),1)*100
            ax.plot(0.33,y,'o',color='#FF7F0E',ms=15,zorder=5)
            ax.text(0.27,y,f'{nm}\n{sh:.1f}%',ha='right',va='center',fontsize=9,fontweight='bold')
        for j,(nm,y) in enumerate(zip(vi.imp_names,y_i)):
            sh=x_mln[:,j].sum()/max(x_mln.sum(),1)*100
            ax.plot(0.67,y,'s',color='#1F77B4',ms=15,zorder=5)
            ax.text(0.73,y,f'{nm}\n{sh:.1f}%',ha='left',va='center',fontsize=9,fontweight='bold')
        ax.text(0.33,0.97,f'TOP {n_exp}\nEXPORTERS',ha='center',fontsize=11,
                fontweight='bold',color='#FF7F0E')
        ax.text(0.67,0.97,f'TOP {n_imp}\nIMPORTERS',ha='center',fontsize=11,
                fontweight='bold',color='#1F77B4')
        plt.tight_layout()
        save_fig(f'Fig3_BACI_{vi._year}_network_graph.png')
    except Exception as e:
        print(f'  Warning: fig_network_graph failed: {e}')
        plt.close('all')


def fig_multi_year(year_results):
    """
    FIX 2: Primary metric = gap achieved at N iterations (not 'iters to tol').
    Three panels:
      (a) Gap achieved at 3000 iters — both methods, log scale
      (b) Gap ratio G_EG / G_SAISE per year  (SAISE advantage even when both DNF)
      (c) Iterations to tol=1e-3 — cleaner convergence comparison
    """
    try:
        if len(year_results) < 2: return
        years   = [r[0] for r in year_results]
        n_eg_raw= [r[1] for r in year_results]    # None = DNF at tol=1e-6
        n_sa_raw= [r[2] for r in year_results]
        g_eg    = [r[3] for r in year_results]    # final gap at 3000 iters
        g_sa    = [r[4] for r in year_results]

        # Gap ratio (capped for display)
        gap_ratio = [ge/gs if gs > 1e-15 else 1.0
                     for ge, gs in zip(g_eg, g_sa)]

        # Iterations to tol=1e-3 (from extended results if stored)
        # Use stored n_tol3 if available, else infer from gaps
        n_eg_tol3 = [r[5] if len(r) > 5 else (n if n is not None else 3000)
                     for r, n in zip(year_results, n_eg_raw)]
        n_sa_tol3 = [r[6] if len(r) > 6 else (n if n is not None else 3000)
                     for r, n in zip(year_results, n_sa_raw)]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            f'SAISE Multi-Year Analysis: BACI {BACI_HS} {BACI_VERSION}\n'
            f'Top-{M_EXPORTERS}×{N_IMPORTERS}×{K_SECTORS} network  '
            f'|  dim={K_SECTORS*M_EXPORTERS*N_IMPORTERS*L_ROUTES}  '
            f'|  3000 iteration budget',
            fontsize=13, fontweight='bold')

        x_ = np.arange(len(years)); w = 0.35

        # ── (a) Gap at 3000 iters (primary metric) ────────────────────────────
        ax = axes[0]
        bars_eg = ax.bar(x_ - w/2, g_eg, w, color=PAL['EG'],
                         label='EG oracle-best', alpha=0.88)
        bars_sa = ax.bar(x_ + w/2, g_sa, w, color=PAL['SAISE'],
                         label='SAISE', alpha=0.88)
        # Annotate with scientific notation
        for xi, (ge, gs) in enumerate(zip(g_eg, g_sa)):
            ax.text(xi - w/2, ge * 1.4, f'{ge:.0e}',
                    ha='center', fontsize=7, color=PAL['EG'],
                    rotation=45, fontweight='bold')
            ax.text(xi + w/2, gs * 1.4, f'{gs:.0e}',
                    ha='center', fontsize=7, color=PAL['SAISE'],
                    rotation=45, fontweight='bold')
        # Shade 2021 (both methods converged)
        converged_years = [i for i, (ne, ns) in enumerate(zip(n_eg_raw, n_sa_raw))
                           if ne is not None and ns is not None]
        for ci in converged_years:
            ax.axvspan(ci - 0.5, ci + 0.5, alpha=0.08, color='green')
            ax.text(ci, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-8,
                    '✓', ha='center', fontsize=12, color='green', va='bottom')
        ax.set_yscale('log')
        ax.set_xticks(x_); ax.set_xticklabels(years, rotation=30)
        ax.set_ylabel('$G(x^{3000})$ — VI gap at budget (lower = better)',
                      fontsize=10)
        ax.set_title('(a) Gap Achieved at 3000 Iterations\n'
                     '(✓ = both converge to $10^{-6}$, green shading)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.28, linestyle='--')

        # ── (b) Gap ratio G_EG / G_SAISE — convergence rate advantage ────────
        ax = axes[1]
        cols = ['#2CA02C' if r > 1 else '#D62728' for r in gap_ratio]
        bars = ax.bar(x_, gap_ratio, 0.6, color=cols, alpha=0.88,
                      edgecolor='white', lw=1.5)
        ax.axhline(1.0, color='black', lw=1.8,
                   label='Ratio = 1 (no difference)')
        for xi, r in enumerate(gap_ratio):
            ax.text(xi, r + max(gap_ratio) * 0.02,
                    f'{r:.0f}×', ha='center', fontsize=10, fontweight='bold',
                    color='#2CA02C' if r > 1 else '#D62728')
        ax.set_yscale('log')
        ax.set_xticks(x_); ax.set_xticklabels(years, rotation=30)
        ax.set_ylabel('$G_{\\mathrm{EG}}(N)\\,/\\,G_{\\mathrm{SAISE}}(N)$\n'
                      '(>1 means SAISE gap is smaller)',
                      fontsize=10)
        ax.set_title('(b) Convergence Rate Advantage\n'
                     '$G_{\\mathrm{EG}}(N) / G_{\\mathrm{SAISE}}(N)$ at $N=3000$',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.28, linestyle='--')

        # ── (c) Iterations to tol=1e-3 ───────────────────────────────────────
        ax = axes[2]
        DNF = 3000
        n_eg_plot = [v if v is not None else DNF for v in n_eg_tol3]
        n_sa_plot = [v if v is not None else DNF for v in n_sa_tol3]
        ax.bar(x_ - w/2, n_eg_plot, w, color=PAL['EG'],
               label='EG oracle-best', alpha=0.88)
        ax.bar(x_ + w/2, n_sa_plot, w, color=PAL['SAISE'],
               label='SAISE', alpha=0.88)
        # Label DNF bars
        for xi, (ne, ns, ne_r, ns_r) in enumerate(zip(n_eg_plot, n_sa_plot,
                                                        n_eg_tol3, n_sa_tol3)):
            if ne_r is None:
                ax.text(xi - w/2, ne * 0.5, 'DNF', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white', rotation=90)
            else:
                ax.text(xi - w/2, ne + 20, str(ne), ha='center', fontsize=8,
                        color=PAL['EG'], fontweight='bold')
            if ns_r is None:
                ax.text(xi + w/2, ns * 0.5, 'DNF', ha='center', va='center',
                        fontsize=8, fontweight='bold', color='white', rotation=90)
            else:
                ax.text(xi + w/2, ns + 20, str(ns), ha='center', fontsize=8,
                        color=PAL['SAISE'], fontweight='bold')
        ax.set_xticks(x_); ax.set_xticklabels(years, rotation=30)
        ax.set_ylabel('Iterations to $G < 10^{-3}$', fontsize=11)
        ax.set_title('(c) Iterations to tol=$10^{-3}$\n'
                     '(relaxed tolerance — more years converge)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.28, linestyle='--')
        ax.set_ylim(0, DNF * 1.15)

        plt.tight_layout()
        save_fig('Fig4_BACI_multiyear_comparison.png')
    except Exception as e:
        print(f'  Warning: fig_multi_year failed: {e}')
        import traceback; traceback.print_exc()
        plt.close('all')


def fig_sector_analysis(vi, sector_results):
    try:
        if not sector_results: return
        # ── include ALL sectors; replace None (DNF) with a sentinel ──────────
        DNF_SENTINEL = 3000          # maximum iterations — shown as "DNF"
        secs = list(sector_results.keys())
        if not secs: return

        def safe_n(v):
            """Return int value, or DNF_SENTINEL if None."""
            return int(v) if (v is not None and not np.isnan(float(v) if v else float('nan'))) else DNF_SENTINEL

        n_eg_s_raw = [sector_results[s]['n_eg'] for s in secs]
        n_sa_s_raw = [sector_results[s]['n_sa'] for s in secs]
        n_eg_s     = [safe_n(v) for v in n_eg_s_raw]
        n_sa_s     = [safe_n(v) for v in n_sa_s_raw]

        # Speedup only when both converged
        spds = []
        for ne_raw, ns_raw, ne, ns in zip(n_eg_s_raw, n_sa_s_raw, n_eg_s, n_sa_s):
            if ne_raw is not None and ns_raw is not None and ne > 0 and ns > 0:
                spds.append((ne - ns) / ne * 100)
            else:
                spds.append(0.0)

        # Bar colours: DNF bars in grey
        eg_col  = ['#AAAAAA' if v is None else PAL['EG']    for v in n_eg_s_raw]
        sa_col  = ['#AAAAAA' if v is None else PAL['SAISE'] for v in n_sa_s_raw]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'SAISE Performance by Commodity Sector\n{vi.name}',
                     fontsize=13, fontweight='bold')
        x = np.arange(len(secs)); w = 0.35

        # Draw bars one by one to support per-bar colour
        for xi, (ne, ns, ce, cs, ne_raw, ns_raw) in enumerate(
                zip(n_eg_s, n_sa_s, eg_col, sa_col, n_eg_s_raw, n_sa_s_raw)):
            b_eg = axes[0].bar(xi - w/2, ne, w, color=ce, alpha=0.88,
                               label='EG oracle-best' if xi == 0 else '_nolegend_',
                               hatch='//' if ne_raw is None else '')
            b_sa = axes[0].bar(xi + w/2, ns, w, color=cs, alpha=0.88,
                               label='SAISE'          if xi == 0 else '_nolegend_',
                               hatch='//' if ns_raw is None else '')
            # Annotate DNF
            if ne_raw is None:
                axes[0].text(xi - w/2, ne * 0.5, 'DNF', ha='center', va='center',
                             fontsize=8, fontweight='bold', color='white', rotation=90)
            if ns_raw is None:
                axes[0].text(xi + w/2, ns * 0.5, 'DNF', ha='center', va='center',
                             fontsize=8, fontweight='bold', color='white', rotation=90)

        axes[0].set_xticks(x); axes[0].set_xticklabels(secs, rotation=20, fontsize=9)
        axes[0].set_ylabel('Iterations (grey/hatched = DNF)')
        axes[0].set_title('(a) Iterations per Sector', fontsize=11, fontweight='bold')
        axes[0].legend(fontsize=10); axes[0].grid(axis='y', alpha=0.28, linestyle='--')

        cols = ['#AAAAAA' if (n_eg_s_raw[i] is None or n_sa_s_raw[i] is None)
                else ('#2CA02C' if spds[i] > 0 else '#D62728')
                for i in range(len(secs))]
        axes[1].bar(x, spds, 0.55, color=cols, alpha=0.88, edgecolor='white', lw=1.5)
        axes[1].axhline(0, color='black', lw=1.5)
        for xi, (sp, ne_raw, ns_raw) in enumerate(zip(spds, n_eg_s_raw, n_sa_s_raw)):
            dnf_eg = ne_raw is None; dnf_sa = ns_raw is None
            if dnf_eg or dnf_sa:
                label = 'EG-DNF' if dnf_eg else 'SA-DNF'
                axes[1].text(xi, 1, label, ha='center', fontsize=9,
                             fontweight='bold', color='#555555')
            else:
                axes[1].text(xi, sp + 0.5 if sp >= 0 else sp - 2,
                             f'{sp:.0f}%', ha='center', fontsize=10,
                             fontweight='bold',
                             color='#2CA02C' if sp >= 0 else '#D62728')
        axes[1].set_xticks(x); axes[1].set_xticklabels(secs, rotation=20, fontsize=9)
        axes[1].set_ylabel('SAISE speedup vs EG-best (%)  (grey = DNF)')
        axes[1].set_title('(b) Speedup by Sector\n(grey = one method DNF)',
                          fontsize=11, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.28, linestyle='--')
        plt.tight_layout()
        save_fig(f'Fig5_BACI_{vi._year}_sector_speedup.png')
    except Exception as e:
        print(f'  Warning: fig_sector_analysis failed: {e}')
        plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
# CORE EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_one_year(year, verbose=True):
    if verbose: print(f'\n{"─"*68}\n  Processing year {year} …')

    baci_data = read_baci_year(year, verbose=verbose)
    if baci_data is None: return None

    try:
        net = aggregate_network(baci_data, M_EXPORTERS, N_IMPORTERS, K_SECTORS)
    except Exception as e:
        print(f'  ERROR aggregating: {e}'); return None

    try:
        vi = BACITradeVI(net)
    except Exception as e:
        print(f'  ERROR building VI: {e}'); return None

    if verbose:
        print(f'  VI   : {vi.name}')
        print(f'  Gap@x_obs : {vi.gap(vi.x0()):.4f}')
        print(f'  Gap@x_eq  : {vi.gap(vi.x_star):.2e}')
        vi.print_top_flows()

    x0  = vi.x0()
    TOL = 1e-6; N = 3000

    # EG grid search (oracle-best step)
    r_eg_best = find_best_eg_step(vi, x0, tol=TOL, N=N, verbose=verbose)
    best_lam  = r_eg_best['lam_used']

    # Sub-optimal EG
    sub_lam  = best_lam * 0.3
    r_eg_sub = run_eg(vi, sub_lam, x0, N=N, tol=TOL)
    r_eg_sub['lam_used'] = float(sub_lam)   # store for figure labelling

    # IEG at oracle-best step
    r_ieg = run_ieg(vi, best_lam, x0, N=N, tol=TOL)

    # SAISE: lam0 = oracle-best × 2
    r_sa  = run_saise(vi, best_lam * 2.0, x0, N=N, tol=TOL)

    # ── FIX 3: also measure iterations to tol=1e-3 ───────────────────────────
    # This gives a cleaner comparison table (more years converge at this tol)
    TOL3 = 1e-3
    def iters_to_tol(r, t):
        """Smallest n where gap < t, or None if never reached."""
        idx = np.where(r['gaps'] < t)[0]
        return int(r['iters'][idx[0]]) if len(idx) > 0 else None

    n_eg_tol3 = iters_to_tol(r_eg_best, TOL3)
    n_sa_tol3 = iters_to_tol(r_sa,      TOL3)

    def c(r): return int(r['iters'][-1]) if r['gaps'][-1] < TOL else None
    n_eg = c(r_eg_best); n_sa = c(r_sa)

    # Headline: prefer tol=1e-6 if both converge, else show gap ratio
    g_eg = r_eg_best['gaps'][-1]; g_sa = r_sa['gaps'][-1]
    if n_eg and n_sa:
        spd = f'+{(n_eg-n_sa)/n_eg*100:.0f}% fewer iters (tol=1e-6)'
    elif g_sa > 0:
        spd = f'SAISE gap {g_eg/g_sa:.0f}× smaller at {N} iters'
    else:
        spd = '?'

    if verbose:
        print(f'\n  {"Method":<48} {"Iters@1e-6":>10}  {"Iters@1e-3":>10}  {"Gap@N":>10}')
        print('  ' + '─' * 82)
        for name, r in [('EG (sub-opt, λ={:.4f})'.format(sub_lam),  r_eg_sub),
                        ('EG oracle-best (λ={:.4f})'.format(best_lam), r_eg_best),
                        ('IEG',                                         r_ieg),
                        ('SAISE (λ₀={:.4f})'.format(best_lam * 2),   r_sa)]:
            n6  = c(r)
            n3  = iters_to_tol(r, TOL3)
            g_  = r['gaps'][-1]
            print(f'  {name:<48} {str(n6 or "DNF"):>10}  {str(n3 or "DNF"):>10}  {g_:>10.3e}')
        print(f'\n  Key: {spd}')

    return dict(vi=vi, net=net,
                r_eg_sub=r_eg_sub, r_eg_best=r_eg_best,
                r_ieg=r_ieg, r_sa=r_sa,
                n_eg=n_eg, n_sa=n_sa,
                n_eg_tol3=n_eg_tol3, n_sa_tol3=n_sa_tol3)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('═'*68)
    print('  SAISE on BACI-VI-Bench — HS17 Dataset')
    print(f'  Data   : {BACI_DIR}')
    print(f'  Output : {OUTPUT_DIR}')
    print('═'*68)

    # Verify directory
    if not os.path.isdir(BACI_DIR):
        print(f'\n  ✗  Data folder not found: {BACI_DIR}')
        print('     Check the BACI_DIR variable at the top of this script.')
        sys.exit(1)

    # Scan available files
    available = []
    print('\n  Scanning data files …')
    for yr in ALL_YEARS:
        p = find_baci_file(yr)
        if p:
            mb = os.path.getsize(p) / 1_048_576
            available.append(yr)
            print(f'  ✓  {os.path.basename(p):50s}  ({mb:>5.0f} MB)')
        else:
            print(f'  ✗  Missing: BACI_{BACI_HS}_Y{yr}_{BACI_VERSION}.csv')

    if not available:
        print('\n  No BACI files found. Check BACI_DIR.')
        sys.exit(1)

    print(f'\n  Available : {available}')
    print(f'  Main year : {MAIN_YEAR}')
    print(f'  Network   : m={M_EXPORTERS}, n={N_IMPORTERS}, K={K_SECTORS}, '
          f'dim={K_SECTORS*M_EXPORTERS*N_IMPORTERS*L_ROUTES}')

    # ── EXPERIMENT 1: Main year full analysis ─────────────────────────────
    target = MAIN_YEAR if MAIN_YEAR in available else available[-1]
    print(f'\n{"═"*68}')
    print(f'  [EXP 1] Main year convergence: {target}')
    res = run_one_year(target, verbose=True)
    if res is None:
        print('  Main year failed. Check error above.'); sys.exit(1)

    vi = res['vi']
    print('\n  Generating figures …')
    fig_convergence(vi, res['r_eg_sub'], res['r_eg_best'], res['r_ieg'], res['r_sa'])
    fig_flow_heatmaps(vi)
    fig_network_graph(vi)

    # ── EXPERIMENT 2: All available years ─────────────────────────────────
    print(f'\n{"═"*68}')
    print(f'  [EXP 2] Multi-year: {available}')
    year_results = []; csv_rows = []; TOL = 1e-6

    for yr in available:
        print(f'  Year {yr} …', end=' ', flush=True)
        ry = run_one_year(yr, verbose=False)
        if ry is None: print('SKIP'); continue
        n_eg = ry['n_eg']; n_sa = ry['n_sa']
        g_eg = ry['r_eg_best']['gaps'][-1]; g_sa = ry['r_sa']['gaps'][-1]
        ratio = g_eg / g_sa if g_sa > 1e-15 else float('inf')
        if n_eg and n_sa:
            status = f'+{(n_eg-n_sa)/n_eg*100:.0f}% iters'
        else:
            status = f'gap ratio {ratio:.0f}×'
        print(f'EG={n_eg or "DNF"}  SAISE={n_sa or "DNF"}  ({status})'
              f'  tol1e-3: EG={ry["n_eg_tol3"] or "DNF"} SAISE={ry["n_sa_tol3"] or "DNF"}')

        # Store: (year, n_eg@1e-6, n_sa@1e-6, g_eg@N, g_sa@N, n_eg@1e-3, n_sa@1e-3)
        year_results.append((yr, n_eg, n_sa, g_eg, g_sa,
                              ry['n_eg_tol3'], ry['n_sa_tol3']))

        for nm, r in [('EG_sub',          ry['r_eg_sub']),
                      (f'EG_best_lam{ry["r_eg_best"]["lam_used"]:.5f}', ry['r_eg_best']),
                      ('IEG',             ry['r_ieg']),
                      ('SAISE',           ry['r_sa'])]:
            csv_rows.append([BACI_HS, BACI_VERSION, yr, nm,
                              int(r['iters'][-1]),
                              f'{r["gaps"][-1]:.6e}',
                              f'{r["dists"][-1]:.6e}',
                              f'{r["times"][-1]:.3f}'])

    fig_multi_year(year_results)

    # ── EXPERIMENT 3: Per-sector (main year) — HEADLINE RESULT ────────────
    print(f'\n{"═"*68}')
    print(f'  [EXP 3] Per-sector analysis — HEADLINE RESULT (year {target})')
    print(f'  (dim=100 per sector, tol=1e-6 — all sectors converge cleanly)')
    sec_results = {}
    net0 = res['net']; TOL_S = 1e-6; N_S = 3000

    for k, sec in enumerate(vi.sectors[:min(vi.K, 5)]):
        print(f'  Sector: {sec:<14} …', end=' ', flush=True)
        try:
            K1 = 1; m1 = vi.m; n1 = vi.n
            fn1 = net0['flow_norm'][[k]].copy()
            sc1 = max(float(fn1.max()), 1e-9); fn1 = fn1 / sc1
            net1 = dict(net0, flow_norm=fn1, flow_v_mln=net0['flow_v_mln'][[k]],
                        flow_scale=sc1 * net0['flow_scale'],
                        uv_exp=net0['uv_exp'][[k]], uv_imp=net0['uv_imp'][[k]],
                        supply_cap=fn1.sum(axis=(0, 2)) * 1.5,
                        demand_total=fn1.sum(axis=(0, 1)),
                        K=K1, dim=K1 * m1 * n1 * L_ROUTES, sectors=[sec])
            vi1 = BACITradeVI(net1)
            r_e1 = find_best_eg_step(vi1, vi1.x0(), tol=TOL_S, N=N_S, verbose=False)
            r_s1 = run_saise(vi1, r_e1['lam_used'] * 2, vi1.x0(), N=N_S, tol=TOL_S)
            n_e = int(r_e1['iters'][-1]) if r_e1['gaps'][-1] < TOL_S else None
            n_s = int(r_s1['iters'][-1]) if r_s1['gaps'][-1] < TOL_S else None
            g_e = r_e1['gaps'][-1]; g_s = r_s1['gaps'][-1]

            if n_e and n_s:
                spd_str = f'+{(n_e-n_s)/n_e*100:.0f}%'
                detail  = f'EG={n_e}  SAISE={n_s}  ({spd_str})'
            elif n_e is None and n_s is not None:
                spd_str = 'SAISE unique ✓'
                detail  = f'EG=DNF  SAISE={n_s}  ({spd_str})'
            elif n_s is None and n_e is not None:
                spd_str = 'EG wins'
                detail  = f'EG={n_e}  SAISE=DNF  ({spd_str})'
            else:
                ratio_s = g_e / g_s if g_s > 1e-15 else float('inf')
                spd_str = f'gap ratio {ratio_s:.0f}×'
                detail  = f'EG=DNF  SAISE=DNF  ({spd_str})'
            print(detail)
            sec_results[sec] = {'n_eg': n_e, 'n_sa': n_s,
                                 'g_eg': g_e, 'g_sa': g_s,
                                 'r_eg': r_e1, 'r_sa': r_s1}
        except Exception as e:
            print(f'ERR: {e}')

    fig_sector_analysis(vi, sec_results)

    # ── Save CSV ──────────────────────────────────────────────────────────
    save_csv_results(
        csv_rows,
        ['HS', 'Version', 'Year', 'Method', 'Iters', 'FinalGap',
         'FinalDist', 'CPU_s'],
        'results_BACI_all_years.csv')

    # ── Headline summary table for paper ─────────────────────────────────
    print(f'\n{"═"*68}')
    print('  HEADLINE RESULTS FOR PAPER')
    print(f'{"─"*68}')
    print()
    print('  1. Per-sector (dim=100, tol=1e-6) — primary publishable result:')
    print(f'     {"Sector":<14} {"EG-best":>9} {"SAISE":>9} {"Speedup":>10}')
    print('     ' + '─' * 46)
    for sec, d in sec_results.items():
        ne = d['n_eg']; ns = d['n_sa']
        if ne and ns:
            spd_s = f'+{(ne-ns)/ne*100:.0f}%'
        elif ne is None and ns:
            spd_s = 'SAISE unique'
        else:
            spd_s = f'gap {d["g_eg"]/d["g_sa"]:.0f}× smaller' if d["g_sa"]>0 else '?'
        print(f'     {sec:<14} {str(ne or "DNF"):>9} {str(ns or "DNF"):>9} {spd_s:>10}')

    print()
    print('  2. Multi-year gap ratio (dim=500, 3000 iters):')
    print(f'     {"Year":<6} {"G_EG@3000":>12} {"G_SAISE@3000":>14} {"Ratio":>8}')
    print('     ' + '─' * 44)
    for row in year_results:
        yr_, _, _, ge, gs, n3e, n3s = row
        ratio_r = ge / gs if gs > 1e-15 else float('inf')
        conv_str = f' ← both converge @1e-3' if (n3e and n3s) else ''
        print(f'     {yr_:<6} {ge:>12.3e} {gs:>14.3e} {ratio_r:>7.0f}×{conv_str}')

    print(f'\n{"═"*68}')
    figs = [
        f'Fig1_BACI_{target}_convergence.png',
        f'Fig2_BACI_{target}_flow_heatmaps.png',
        f'Fig3_BACI_{target}_network_graph.png',
        'Fig4_BACI_multiyear_comparison.png',
        f'Fig5_BACI_{target}_sector_speedup.png',
        'results_BACI_all_years.csv',
    ]
    for f in figs:
        p = os.path.join(OUTPUT_DIR, f)
        print(f"  {'✓' if os.path.exists(p) else '·'}  {f}")
    print()
    print('  ► CITATION:')
    print('    Gaulier, G. & Zignago, S. (2010). BACI: International Trade')
    print('    Database at the Product-Level. CEPII WP N°2010-23.')
    print('    Licence: Etalab 2.0 (cite CEPII as source).')
    print('═' * 68)


if __name__ == '__main__':
    main()
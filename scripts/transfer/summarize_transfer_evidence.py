#!/usr/bin/env python3
"""Build minimal high-value transfer diagnostic tables from existing result CSVs."""
import os, csv, math
from statistics import mean, pstdev

OUT_DIR = 'transfer/summary'
STRICT = os.path.join(OUT_DIR, 'strict_common_gene_seed_results.csv')
COV = os.path.join(OUT_DIR, 'coverage_matched_native_results.csv')
GAP = os.path.join(OUT_DIR, 'transfer_gap_summary.csv')
SHIFT = os.path.join(OUT_DIR, 'score_shift_summary.csv')

EXP1 = os.path.join(OUT_DIR, 'exp1_protocol_matrix.csv')
EXP2 = os.path.join(OUT_DIR, 'exp2_size_vs_composition.csv')
EXP3 = os.path.join(OUT_DIR, 'exp3_margin_perf_link.csv')
NOTE = os.path.join(OUT_DIR, 'Current_problems_with_the_evidence.md')


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float('nan')


def write_csv(path, fields, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_stats(rows, keys, metric):
    d = {}
    for r in rows:
        k = tuple(r[kx] for kx in keys)
        d.setdefault(k, []).append(to_float(r[metric]))
    out = {}
    for k, v in d.items():
        vals = [x for x in v if not math.isnan(x)]
        if not vals:
            out[k] = (float('nan'), float('nan'), 0)
        else:
            out[k] = (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0, len(vals))
    return out


def build_exp1(strict_rows, cov_rows, gap_rows):
    # native means from GAP, strict from strict_rows, cov from cov_rows
    strict_stats_au = group_stats(strict_rows, ['train_dataset', 'test_dataset', 'embedding', 'clf'], 'auroc')
    strict_stats_ap = group_stats(strict_rows, ['train_dataset', 'test_dataset', 'embedding', 'clf'], 'auprc')
    cov_stats_au = group_stats(cov_rows, ['train_dataset', 'test_dataset', 'embedding', 'clf'], 'auroc')
    cov_stats_ap = group_stats(cov_rows, ['train_dataset', 'test_dataset', 'embedding', 'clf'], 'auprc')

    rows = []
    for r in gap_rows:
        k = (r['train_dataset'], r['test_dataset'], r['embedding'], r['clf'])
        s_au, s_au_std, s_n = strict_stats_au.get(k, (float('nan'), float('nan'), 0))
        s_ap, s_ap_std, _ = strict_stats_ap.get(k, (float('nan'), float('nan'), 0))
        c_au, c_au_std, c_n = cov_stats_au.get(k, (float('nan'), float('nan'), 0))
        c_ap, c_ap_std, _ = cov_stats_ap.get(k, (float('nan'), float('nan'), 0))
        rows.append({
            'train_dataset': r['train_dataset'],
            'test_dataset': r['test_dataset'],
            'embedding': r['embedding'],
            'clf': r['clf'],
            'native_mean_auroc': r.get('native_mean_auroc', ''),
            'native_mean_auprc': r.get('native_mean_auprc', ''),
            'strict_mean_auroc': '' if math.isnan(s_au) else f'{s_au:.6f}',
            'strict_std_auroc': '' if math.isnan(s_au_std) else f'{s_au_std:.6f}',
            'strict_mean_auprc': '' if math.isnan(s_ap) else f'{s_ap:.6f}',
            'strict_std_auprc': '' if math.isnan(s_ap_std) else f'{s_ap_std:.6f}',
            'strict_n': s_n,
            'covmatch_mean_auroc': '' if math.isnan(c_au) else f'{c_au:.6f}',
            'covmatch_std_auroc': '' if math.isnan(c_au_std) else f'{c_au_std:.6f}',
            'covmatch_mean_auprc': '' if math.isnan(c_ap) else f'{c_ap:.6f}',
            'covmatch_std_auprc': '' if math.isnan(c_ap_std) else f'{c_ap_std:.6f}',
            'covmatch_n': c_n,
        })
    return rows


def build_exp2(gap_rows):
    rows = []
    for r in gap_rows:
        na = to_float(r.get('native_mean_auroc', ''))
        sa = to_float(r.get('strict_mean_auroc', ''))
        ca = to_float(r.get('coverage_matched_mean_auroc', ''))
        np = to_float(r.get('native_mean_auprc', ''))
        sp = to_float(r.get('strict_mean_auprc', ''))
        cp = to_float(r.get('coverage_matched_mean_auprc', ''))

        size_proxy_au = na - ca if (not math.isnan(na) and not math.isnan(ca)) else float('nan')
        comp_proxy_au = ca - sa if (not math.isnan(ca) and not math.isnan(sa)) else float('nan')
        size_proxy_ap = np - cp if (not math.isnan(np) and not math.isnan(cp)) else float('nan')
        comp_proxy_ap = cp - sp if (not math.isnan(cp) and not math.isnan(sp)) else float('nan')

        rows.append({
            'train_dataset': r['train_dataset'],
            'test_dataset': r['test_dataset'],
            'embedding': r['embedding'],
            'clf': r['clf'],
            'native_minus_covmatch_auroc': '' if math.isnan(size_proxy_au) else f'{size_proxy_au:.6f}',
            'covmatch_minus_strict_auroc': '' if math.isnan(comp_proxy_au) else f'{comp_proxy_au:.6f}',
            'native_minus_covmatch_auprc': '' if math.isnan(size_proxy_ap) else f'{size_proxy_ap:.6f}',
            'covmatch_minus_strict_auprc': '' if math.isnan(comp_proxy_ap) else f'{comp_proxy_ap:.6f}',
        })
    return rows


def build_exp3(gap_rows, shift_rows):
    # join by (train,test,embedding,clf)
    m = {(r['train_dataset'], r['test_dataset'], r['embedding'], r['clf']): r for r in shift_rows}
    out = []
    for r in gap_rows:
        k = (r['train_dataset'], r['test_dataset'], r['embedding'], r['clf'])
        s = m.get(k)
        if not s:
            continue
        out.append({
            'train_dataset': r['train_dataset'],
            'test_dataset': r['test_dataset'],
            'embedding': r['embedding'],
            'clf': r['clf'],
            'gap_native_strict_auroc': r.get('gap_native_strict_auroc', ''),
            'gap_native_strict_auprc': r.get('gap_native_strict_auprc', ''),
            'abs_shift_margin': s.get('abs_shift_margin', ''),
            'abs_shift_pos': s.get('abs_shift_pos', ''),
            'abs_shift_neg': s.get('abs_shift_neg', ''),
        })
    return out


def build_note(gap_rows):
    # conservative diagnostic note
    with open(NOTE, 'w') as f:
        f.write('# Current problems with the evidence\n\n')
        f.write('- Ranking may be protocol-sensitive (native vs strict vs coverage-matched).\n')
        f.write('- Claims of overall superiority are not justified without robust cross-protocol consistency.\n')
        f.write('- Directional asymmetry (hESC->mESC vs mESC->hESC) must be reported explicitly.\n')
        f.write('- Margin/score shift suggests domain-shift calibration issues may mediate transfer failures.\n')
        f.write('- Coverage and composition remain entangled unless decomposition tables are explicitly used.\n')
        n = len(gap_rows)
        f.write(f'- Current rows seen in transfer_gap_summary: {n}.\n')


def main():
    strict_rows = read_csv(STRICT)
    cov_rows = read_csv(COV)
    gap_rows = read_csv(GAP)
    shift_rows = read_csv(SHIFT)

    exp1_rows = build_exp1(strict_rows, cov_rows, gap_rows)
    exp2_rows = build_exp2(gap_rows)
    exp3_rows = build_exp3(gap_rows, shift_rows)

    write_csv(EXP1,
              ['train_dataset','test_dataset','embedding','clf',
               'native_mean_auroc','native_mean_auprc',
               'strict_mean_auroc','strict_std_auroc','strict_mean_auprc','strict_std_auprc','strict_n',
               'covmatch_mean_auroc','covmatch_std_auroc','covmatch_mean_auprc','covmatch_std_auprc','covmatch_n'],
              exp1_rows)
    write_csv(EXP2,
              ['train_dataset','test_dataset','embedding','clf',
               'native_minus_covmatch_auroc','covmatch_minus_strict_auroc',
               'native_minus_covmatch_auprc','covmatch_minus_strict_auprc'],
              exp2_rows)
    write_csv(EXP3,
              ['train_dataset','test_dataset','embedding','clf',
               'gap_native_strict_auroc','gap_native_strict_auprc',
               'abs_shift_margin','abs_shift_pos','abs_shift_neg'],
              exp3_rows)
    build_note(gap_rows)


if __name__ == '__main__':
    main()

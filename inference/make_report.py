#!/usr/bin/env python3


import glob
import os
import re
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_palette(sns.color_palette(palette='Set1', n_colors=12))


MARKERS = '.ov^spP*+xDd<>X'
LINE_KINDS = '-'
# STYLES = [f'{m}{k}' for m in MARKERS for k in LINE_KINDS]
STYLES = '.- o- v- : -- : : D'.split(' ')
IMG_SIZE = (17, 5)
ROLLING_WINDOW = 1
LINEWIDTH = 2


def get_exp_id(path, namelen=2):
    return os.path.sep.join(path.split(os.path.sep)[-namelen:])


def read_exp_metrics(path):
    result = []
    for metrics_file in glob.glob(os.path.join(path, 'metrics*.tsv')):
        result.append(pd.read_csv(metrics_file, sep='\t', index_col=[0]))
    result = pd.concat(result, axis=0)
    return result


POINTWISE_METRICS = ('lpips_1_enc lpips_2_opt lpips_3_ft lpips_4_final '
                     'ssim_1_enc ssim_2_opt ssim_3_ft ssim_4_final '
                     'flow_l2 flow_l2_rel'.split(' '))
CHART_STAGE2NAME = {
    '1_enc': 'Encoder',
    '2_opt': 'Optimization',
    '3_ft' : 'Fine-Tune'
}
SEGM_PREFIXES = {
    # 'segm_1_enc'  : 'after Encoder',
    # 'segm_2_opt'  : 'after Optimization',
    'segm_3_ft'   : 'after Fine Tuning',
    'segm_4_final': 'finally'
}
SEGM_METRICS = {
    # 'kl': 'KL-Divergence 🡓',
    # 'd_ce': 'Cross Entropy against Discrete Segmentation 🡓',
    'acc': 'Per-pixel Accuracy of Discrete Segmentation 🡑',
    'lpips': 'LPIPS for all classes, except Sky and Water 🡓',
    'ssim': 'SSIM for all classes, except Sky and Water 🡑'
}


EXPERIMENT_NAMES = {
    '01_eoifs': 'EOIFS',
    '02_eoif': 'EOIF',
    '03_eoi': 'EOI',
    '04_eo': 'EO',
    '05_e': 'E',
    '06_mo': 'MO',
    '07_i2s': 'I2S',
}


FLOAT_RE = re.compile(r"tensor\((.*), device='.*")


def convert_float(x):
    if isinstance(x, str):
        if x.startswith('tensor'):
            if x.startswith('tensor(nan'):
                return np.nan
            else:
                return float(FLOAT_RE.match(x).group(1))
    return x


def format_mean_std(table):
    def _do_format(row):
        mean = row.xs('mean', level=1).iloc[0]
        return mean
        # std = row.xs('std', level=1).iloc[0]
        # return f'{mean:.3f}±{std:.3f}'

    def _transform_group(gr):
        return gr.apply(_do_format, axis=1)

    result = table.groupby(level=0, axis=1).apply(_transform_group)
    result.columns = pd.MultiIndex.from_tuples([c.split('_', 1) for c in result.columns])
    result.sort_index(axis=1, inplace=True)
    return result


def main(args):
    os.makedirs(os.path.join(args.outpath, 'img'), exist_ok=True)

    exp_paths = list(glob.glob(args.in_glob))
    exp_data = [read_exp_metrics(p) for p in exp_paths]
    exp_ids = [get_exp_id(p, namelen=args.namelen) for p in exp_paths]

    full_exp_metrics = pd.concat(exp_data, keys=exp_ids).applymap(convert_float)
    full_exp_metrics.dropna(axis=1, how='all', inplace=True)
    full_exp_metrics.dropna(axis=0, how='all', inplace=True)
    full_exp_metrics.index.set_names(('experiment', 'filename'), inplace=True)
    full_exp_metrics.reset_index(inplace=True)
    full_exp_metrics['img_type'] = [re.match(args.scenetype, f).group('scene') for f in full_exp_metrics['filename']]

    real_videos_index = full_exp_metrics['experiment'] == args.realname
    real_flow_l2 = full_exp_metrics.loc[real_videos_index, 'flow_l2'].mean(axis=0, skipna=True)
    full_exp_metrics['flow_l2_rel'] = full_exp_metrics['flow_l2'] / real_flow_l2

    full_exp_metrics.to_csv(os.path.join(args.outpath, 'joint_metrics.tsv'), sep='\t')

    metrics_by_exp = full_exp_metrics.set_index('experiment')
    metrics_by_st = full_exp_metrics.set_index('img_type')

    # ******************
    # Reconstruction
    # mean reconstruction metrics by experiment
    pointwise_exp_metrics = metrics_by_exp[POINTWISE_METRICS]
    pointwise_exp_mean = pointwise_exp_metrics.groupby(level=0, axis=0).describe()
    pointwise_exp_mean = pointwise_exp_mean[[c for c in pointwise_exp_mean.columns
                                             if c[1] in ('mean', 'std')]]
    pointwise_exp_mean = format_mean_std(pointwise_exp_mean)
    pointwise_exp_mean_html = pointwise_exp_mean[['lpips', 'ssim']].to_html(float_format=lambda x: f'{x:.4f}')
    flow_exp_mean_html = pointwise_exp_mean['flow'].rename(index=EXPERIMENT_NAMES).to_html()

    lpips_data = pointwise_exp_mean['lpips'][list(CHART_STAGE2NAME.keys())]
    lpips_data.rename(columns=CHART_STAGE2NAME, inplace=True)
    lpips_data.rename(index=EXPERIMENT_NAMES, inplace=True)
    lpips_data.dropna(axis=0, inplace=True)

    ssim_data = pointwise_exp_mean['ssim'][list(CHART_STAGE2NAME.keys())]
    ssim_data.rename(columns=CHART_STAGE2NAME, inplace=True)
    ssim_data.rename(index=EXPERIMENT_NAMES, inplace=True)
    ssim_data.dropna(axis=0, inplace=True)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches((7, 3.3))
    lpips_data.transpose().plot(ax=ax1, style=STYLES[:lpips_data.shape[0]], linewidth=LINEWIDTH)
    ssim_data.transpose().plot(ax=ax2, style=STYLES[:ssim_data.shape[0]], linewidth=LINEWIDTH)
    ax1.set_ylabel('LPIPS (less is better)')
    ax1.get_legend().remove()
    ax2.set_ylabel('SSIM (more is better)')
    ax2.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
    fig.tight_layout()
    fig.savefig(os.path.join(args.outpath, 'img', 'lpips_chart.png'))
    fig.savefig(os.path.join(args.outpath, 'img', 'lpips_chart.pdf'))
    plt.close(fig)


    # mean reconstruction metrics by scene type
    pointwise_st_mean_html = ''
    flow_st_mean_html = ''
    if not args.nost:
        pointwise_st_metrics = metrics_by_st[POINTWISE_METRICS]
        pointwise_st_mean = pointwise_st_metrics.groupby(level=0, axis=0).describe()
        pointwise_st_mean = pointwise_st_mean[[c for c in pointwise_st_mean.columns
                                               if c[1] in ('mean', 'std')]]
        pointwise_st_mean = format_mean_std(pointwise_st_mean)
        pointwise_st_mean_html = pointwise_st_mean[['lpips', 'ssim']].to_html()
        flow_st_mean_html = pointwise_st_mean['flow'].to_html(float_format=lambda x: f'{x:.4f}')

    os.makedirs(os.path.join(args.outpath, 'img'), exist_ok=True)

    # ******************
    # Dynamic Metrics - Segmentation
    # dynamic metrics by experiment
    chart_by_exp_html = []
    for mname, mtitle in SEGM_METRICS.items():
        chart_by_exp_html.append(f'<h2>{mtitle}</h2>')
        for prefix, ptitle in SEGM_PREFIXES.items():
            col_filter = f'{prefix}{mname}'
            ncols = sum(1 for c in metrics_by_exp.columns if c.startswith(col_filter))
            if ncols == 0:
                continue
            cur_values = metrics_by_exp[[f'{col_filter}_{i}' for i in range(ncols)]].groupby(level=0, axis=0).mean()
            cur_values.columns = list(range(ncols))
            cur_values = cur_values.rolling(window=ROLLING_WINDOW, axis=1, center=True).mean()

            fig, ax = plt.subplots()
            fig.set_size_inches(IMG_SIZE)
            cur_values.transpose().plot(ax=ax, style=STYLES[:cur_values.shape[0]], linewidth=LINEWIDTH)
            ax.set_title(f'{mtitle} {ptitle}')
            ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
            fig.tight_layout()

            curname = f'segm_by_exp_{col_filter}.png'
            fig.savefig(os.path.join(args.outpath, 'img', curname))
            plt.close(fig)
            chart_by_exp_html.append(f'<img src="img/{curname}" />')

    chart_by_exp_html = '\n'.join(chart_by_exp_html)

    # dynamic metrics by scene type
    chart_by_st_html = []
    if not args.nost:
        chart_by_st_html.append('<h1>Segmentation metrics by scene type</h1>')
        for mname, mtitle in SEGM_METRICS.items():
            chart_by_st_html.append(f'<h2>{mtitle}</h2>')
            for prefix, ptitle in SEGM_PREFIXES.items():
                col_filter = f'{prefix}{mname}'
                ncols = sum(1 for c in metrics_by_st.columns if c.startswith(col_filter))
                if ncols == 0:
                    continue
                cur_values = metrics_by_st[[f'{col_filter}_{i}' for i in range(ncols)]].groupby(level=0, axis=0).mean()
                cur_values.columns = list(range(ncols))
                cur_values = cur_values.rolling(window=ROLLING_WINDOW, axis=1, center=True).mean()

                fig, ax = plt.subplots()
                fig.set_size_inches(IMG_SIZE)
                cur_values.transpose().plot(ax=ax, style=STYLES[:cur_values.shape[0]], linewidth=LINEWIDTH)
                ax.set_title(f'{mtitle} {ptitle}')
                ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
                fig.tight_layout()

                curname = f'segm_by_st_{col_filter}.png'
                fig.savefig(os.path.join(args.outpath, 'img', curname))
                plt.close(fig)
                chart_by_st_html.append(f'<img src="img/{curname}" />')

    chart_by_st_html = '\n'.join(chart_by_st_html)

    # ******************
    # FID with time
    fid_columns = [c for c in metrics_by_exp.columns if c.startswith('fid_')]
    fid_int_names = {c: int(c[4:]) for c in fid_columns}
    fid_data = full_exp_metrics.loc[full_exp_metrics['filename'] == 'global_metrics']

    # FID by experiment
    fid_by_exp = fid_data.set_index('experiment')[fid_columns].rename(columns=fid_int_names) \
        .sort_index(axis=1).sort_index(axis=0).groupby(level=0, axis=0).mean() \
        .interpolate(axis=1) \
        .rolling(window=ROLLING_WINDOW, axis=1, center=True).mean()

    fig, ax = plt.subplots()
    fig.set_size_inches(IMG_SIZE)
    fid_by_exp.rename(columns=EXPERIMENT_NAMES, inplace=True)
    fid_by_exp.transpose().plot(ax=ax, style=STYLES[:fid_by_exp.shape[0]], linewidth=LINEWIDTH)
    ax.set_title('FID with time by experiment')
    ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
    fig.tight_layout()
    curname = 'fid_by_exp.png'
    fig.savefig(os.path.join(args.outpath, 'img', curname))
    plt.close(fig)
    fid_by_exp_html = f'<img src="img/{curname}" />'

    # FID by scene type
    fid_by_st_html = ''
    if not args.nost:
        fid_by_st = fid_data.set_index('img_type')[fid_columns].rename(columns=fid_int_names) \
            .sort_index(axis=1).sort_index(axis=0).groupby(level=0, axis=0).mean() \
            .interpolate(axis=1) \
            .rolling(window=ROLLING_WINDOW, axis=1, center=True).mean()
        fig, ax = plt.subplots()
        fig.set_size_inches(IMG_SIZE)
        fid_by_st.transpose().plot(ax=ax, style=STYLES[:fid_by_st.shape[0]], linewidth=LINEWIDTH)
        ax.set_title('FID with time by scene type')
        ax.legend(bbox_to_anchor=(1.01, 0.5), loc='center left')
        fig.tight_layout()
        curname = 'fid_by_st.png'
        fig.savefig(os.path.join(args.outpath, 'img', curname))
        plt.close(fig)
        fid_by_st_html = f'<h1>FID by scene type 🡓</h1><img src="img/{curname}" />'

    html = f'''
    <html>
    <style>
        * {{
          text-align: center;
        }}
        table {{
           margin-left: auto;
           margin-right:auto;
           border: 0.2px solid grey;
           border-spacing: 0px;
        }}
        td {{
           padding: 5px 10px;
        }}
    </style>
    <body>
        <h1>Reconstruction quality by experiment</h1>
        {pointwise_exp_mean_html}
        <br />
        <img src="img/lpips_chart.png" />
        <br />
        {pointwise_st_mean_html}
        <hr />

        <h1>Animation amount by experiment</h1>
        {flow_exp_mean_html}
        <br />
        {flow_st_mean_html}
        <hr />

        <h1>Segmentation metrics by experiment</h1>
        {chart_by_exp_html}
        <hr />

        {chart_by_st_html}
        
        <h1>FID by experiment 🡓</h1>
        {fid_by_exp_html}
        <hr />
        
        {fid_by_st_html}
    </body>
    </html>
    '''
    with open(os.path.join(args.outpath, 'index.html'), 'w') as f:
        f.write(html)


if __name__ == "__main__":
    import argparse

    aparser = argparse.ArgumentParser()
    aparser.add_argument('in_glob', type=str, help='Glob to get all paths to predictions for comparison')
    aparser.add_argument('outpath', type=str, help='Where to store resulting report')
    aparser.add_argument('--namelen', type=int, default=2, help='How many last subpath items to use as experiment id')
    aparser.add_argument('--scenetype', type=str, default=r'^(\d-)?(?P<scene>[^_]+)_', help='How to extract scene type from filename')
    aparser.add_argument('--nost', action='store_true', help='Whether to disable comparison over scene type')
    aparser.add_argument('--realname', type=str, default='test_videos',
                         help='Name of experiment to calculate comparison against')

    main(aparser.parse_args())

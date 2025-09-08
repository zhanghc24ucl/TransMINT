from TransMINT.data_utils.datamodule import DataLoaderConfig
from TransMINT.model.transformer import MINTransformer
from TransMINT.tasks.cn_futs.data import CNFutDataProvider, build_input_spec, load_data, random_spec

version = 'v2'
raw_data = load_data('../data', version=version)

data_provider = CNFutDataProvider(raw_data)

def get_n_params(is_lite, d_model, n_features):
    args = dict(
        d_model=d_model,
        num_heads=4,
        dropout=0.2,
        trainable_skip_add=False,
        is_lite=is_lite,
    )
    model = MINTransformer(random_spec(2, n_features), **args)
    return model.n_parameters()


def model_size_tabular():
    tab = []
    for n_feature in [10, 18, 30, 100]:
        for d_model in [16, 32, 64]:
            row = [n_feature, d_model]
            a = get_n_params(False, d_model, n_feature)[0]
            row.append(f'{a/1e3:.02f} K')
            b = get_n_params(True, d_model, n_feature)[0]
            row.append(f'{b/1e3:.02f} K')
            row.append(f'{a/b:.02f}')
            tab.append(row)
    headers = ['n_features', 'd_model', 'with VSN', 'with VFN', 'ratio']
    from tabulate import tabulate
    print(tabulate(tab, headers=headers, tablefmt='latex'))


# model_size_tabular()

def data_size():
    input_spec = build_input_spec(version)

    data_cfg = DataLoaderConfig(
        input_spec=input_spec,
        batch_size=128,
        time_step=180,  # 15 hours
    )

    d = data_provider.get_dataloader(data_cfg, '2017-01-01', '2019-07-01')
    print(len(d), len(d) * 128)

data_size()

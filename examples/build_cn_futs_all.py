from TransMINT.tasks.cn_futs.build_features import build_all_1m_features
from TransMINT.tasks.cn_futs.build_targets import build_1m_price, build_all_1m_tabular


build_1m_price('../data')
build_all_1m_features('../data')
build_all_1m_tabular('../data/', horizon=5, phase=0, offset=0, clip=5.)

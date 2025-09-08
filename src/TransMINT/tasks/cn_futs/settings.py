
CN_FUTS_TICKERS_FULL = [
    # stock index futures
    'IC',  # CSI 500 Index Futures
    'IF',  # CSI 300 Index Futures
    'IH',  # SSE 50 Index Futures

    # precise metals
    'AG',  # Silver
    'AU',  # Gold

    # non-ferrous metals
    'AL',  # Aluminum
    'CU',  # Copper
    'ZN',  # Zinc
    'SN',  # Tin
    'NI',  # Nickel
    'PB',  # Lead

    # black commodities & building materials
    'HC',  # Hot-Rolled Coil
    'RB',  # Rebar
    'JC',  # Coke
    'JM',  # Coking Coal
    'IO',  # Iron Ore
    'FG',  # Float Glass

    # energy & chemicals
    'ME',  # Methanol
    'BU',  # Bitumen
    'RU',  # Rubber
    'LP',  # Polyethylene (LLDPE)
    'PP',  # Polypropylene
    'PV',  # PVC
    'TA',  # PTA

    # Agricultural
    'CN',  # Corn
    'CS',  # Corn Starch
    'HA',  # Soybean

    'HY',  # Soybean Oil
    'RO',  # Rapeseed Oil
    'PO',  # Palm Oil

    'HM',  # Soybean Meal
    'RM',  # Rapeseed Meal

    'JD',  # Egg
    'SR',  # White Sugar
    'CF',  # Cotton
]


CN_FUTS_TICKERS_SMALL = [
    # stock index futures
    'IC',  # CSI 500 Index Futures
    'IF',  # CSI 300 Index Futures
    'IH',  # SSE 50 Index Futures

    # precise metals
    'AG',  # Silver
    'AU',  # Gold

    # non-ferrous metals
    'AL',  # Aluminum
    'CU',  # Copper
    'NI',  # Nickel
    'ZN',  # Zinc
]


CN_FUTS_TICKER_NAME = {
    # stock index futures
    'IC': 'CSI500',
    'IF': 'CSI300',
    'IH': 'SSE50',
    # precise metals
    'AG': 'Silver',
    'AU': 'Gold',
    # non-ferrous metals
    'AL': 'Aluminum',
    'CU': 'Copper',
    'ZN': 'Zinc',
    'SN': 'Tin',
    'NI': 'Nickel',
    'PB': 'Lead',

    # black commodities & building materials
    'HC': 'Hot-Rolled Coil',
    'RB': 'Rebar',
    'JC': 'Coke',
    'JM': 'Coking Coal',
    'IO': 'Iron Ore',
    'FG': 'Float Glass',

    # energy & chemicals
    'ME': 'Methanol',
    'BU': 'Bitumen',
    'RU': 'Rubber',
    'LP': 'Polyethylene',
    'PP': 'Polypropylene',
    'PV': 'PVC',
    'TA': 'PTA',

    # Agricultural
    'CN': 'Corn',
    'CS': 'Corn Starch',
    'HA': 'Soybean',

    'HY': 'Soybean Oil',
    'RO': 'Rapeseed Oil',
    'PO': 'Palm Oil',

    'HM': 'Soybean Meal',
    'RM': 'Rapeseed Meal',

    'JD': 'Egg',
    'SR': 'White Sugar',
    'CF': 'Cotton',
}


CN_FUTS_SECTORS = {
    # stock index futures
    'IC': 'Finance',  # CSI 500 Index Futures
    'IF': 'Finance',  # CSI 300 Index Futures
    'IH': 'Finance',  # SSE 50 Index Futures

    # precious metals
    'AG': 'PreciousMetal',  # Silver
    'AU': 'PreciousMetal',  # Gold

    # non-ferrous metals
    'AL': 'BaseMetal',  # Aluminum
    'CU': 'BaseMetal',  # Copper
    'NI': 'BaseMetal',  # Nickel
    'PB': 'BaseMetal',  # Lead
    'ZN': 'BaseMetal',  # Zinc
    'SN': 'BaseMetal',  # Tin

    # black commodities & building materials
    'HC': 'Ferrous',  # Hot-Rolled Coil
    'RB': 'Ferrous',  # Rebar
    'JC': 'Ferrous',  # Coke
    'JM': 'Ferrous',  # Coking Coal
    'IO': 'Ferrous',  # Iron Ore
    'FG': 'Construction',  # Float Glass

    # energy & chemicals
    'ME': 'EnergyChemical',  # Methanol
    'BU': 'EnergyChemical',  # Bitumen
    'RU': 'EnergyChemical',  # Rubber
    'LP': 'EnergyChemical',  # Polyethylene (LLDPE)
    'PP': 'EnergyChemical',  # Polypropylene
    'PV': 'EnergyChemical',  # PVC
    'TA': 'EnergyChemical',  # PTA

    # agricultural
    'CN': 'Grains',  # Corn
    'CS': 'Grains',  # Corn Starch

    'HA': 'OilSeeds',  # Soybean
    'HY': 'OilSeeds',  # Soybean Oil
    'RO': 'OilSeeds',  # Rapeseed Oil
    'PO': 'OilSeeds',  # Palm Oil

    'HM': 'FeedStuff',  # Soybean Meal
    'RM': 'FeedStuff',  # Rapeseed Meal

    'JD': 'Softs',  # Egg
    'SR': 'Softs',  # White Sugar
    'CF': 'Softs',  # Cotton
}


CN_FUTS_MINUTES_PER_DAY = {
    'AG': [(20160101, 555), (20200203, 225), (20200506, 555)],
    'AL': [(20160101, 465), (20200203, 225), (20200506, 465)],
    'AU': [(20160101, 555), (20200203, 225), (20200506, 555)],
    'BU': [(20160101, 465), (20160503, 345), (20200203, 225), (20200506, 345)],
    'CF': [(20160101, 375), (20200203, 225), (20200506, 345)],
    'CN': [(20160101, 225), (20190329, 345), (20200203, 225), (20200506, 345)],
    'CS': [(20160101, 225), (20190329, 345), (20200203, 225), (20200506, 345)],
    'CU': [(20160101, 465), (20200203, 225), (20200506, 465)],
    'FG': [(20160101, 375), (20200203, 225), (20200506, 345)],
    'HA': [(20160101, 375), (20190329, 345), (20200203, 225), (20200506, 345)],
    'HC': [(20160101, 465), (20160503, 345), (20200203, 225), (20200506, 345)],
    'HM': [(20160101, 375), (20190329, 345), (20200203, 225), (20200506, 345)],
    'HY': [(20160101, 375), (20190329, 345), (20200203, 225), (20200506, 345)],
    'IC': [(20160101, 240)],
    'IF': [(20160101, 240)],
    'IH': [(20160101, 240)],
    'IO': [(20160101, 375), (20190329, 345), (20200203, 225), (20200506, 345)],
    'JC': [(20160101, 375), (20190329, 345), (20200203, 225), (20200506, 345)],
    'JD': [(20160101, 225)],
    'JM': [(20160101, 375), (20190329, 345), (20200203, 225), (20200506, 345)],
    'LP': [(20160101, 225), (20190329, 345), (20200203, 225), (20200506, 345)],
    'ME': [(20160101, 375), (20200203, 225), (20200506, 345)],
    'NI': [(20160101, 465), (20200203, 225), (20200506, 465)],
    'PB': [(20160101, 465), (20200203, 225), (20200506, 465)],
    'PO': [(20160101, 375), (20190329, 345), (20200203, 225), (20200506, 345)],
    'PP': [(20160101, 225), (20190329, 345), (20200203, 225), (20200506, 345)],
    'PV': [(20160101, 225), (20190329, 345), (20200203, 225), (20200506, 345)],
    'RB': [(20160101, 465), (20160503, 345), (20200203, 225), (20200506, 345)],
    'RM': [(20160101, 375), (20200203, 225), (20200506, 345)],
    'RO': [(20160101, 375), (20200203, 225), (20200506, 345)],
    'RU': [(20160101, 345), (20200203, 225), (20200506, 345)],
    'SN': [(20160101, 465), (20200203, 225), (20200506, 465)],
    'SR': [(20160101, 375), (20200203, 225), (20200506, 345)],
    'TA': [(20160101, 375), (20200203, 225), (20200506, 345)],
    'ZN': [(20160101, 465), (20200203, 225), (20200506, 465)],
}


InSampleWindows = [
        ('2016-03-01', '2018-07-01', '2019-01-01', '2019-07-01'),
        ('2016-07-01', '2019-01-01', '2019-07-01', '2020-01-01'),
        ('2017-01-01', '2019-07-01', '2020-01-01', '2020-07-01'),
        ('2017-07-01', '2020-01-01', '2020-07-01', '2021-01-01'),
]
OutOfSampleWindows = [
        ('2018-01-01', '2020-07-01', '2021-01-01', '2021-07-01'),
        ('2018-07-01', '2021-01-01', '2021-07-01', '2022-01-01'),
        ('2019-01-01', '2021-07-01', '2022-01-01', '2022-07-01'),
        ('2019-07-01', '2022-01-01', '2022-07-01', '2023-01-01'),
]


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
    'NI',  # Nickel
    'PB',  # Lead
    'ZN',  # Zinc
    'SN',  # Tin

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
    'HM',  # Soybean Meal
    'HY',  # Soybean Oil

    'RM',  # Rapeseed Meal
    'RO',  # Rapeseed Oil

    'JD',  # Egg
    'SR',  # White Sugar
    'PO',  # Palm Oil
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

    'HA': 'Oilseeds',  # Soybean
    'HY': 'Oilseeds',  # Soybean Oil
    'RO': 'Oilseeds',  # Rapeseed Oil
    'PO': 'Oilseeds',  # Palm Oil

    'HM': 'Feedstuff',  # Soybean Meal
    'RM': 'Feedstuff',  # Rapeseed Meal

    'JD': 'Softs',  # Egg
    'SR': 'Softs',  # White Sugar
    'CF': 'Softs',  # Cotton
}

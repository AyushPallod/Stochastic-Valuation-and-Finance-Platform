from src.data_loader import StockDataLoader
from src.simulator import MonteCarloSimulator
from src.risk import RiskAnalyzer
from src.visualizations import MonteCarloVisualizer

loader = StockDataLoader(
    ticker="SETL.NS",
    start_date="2020-01-01",
    end_date="2026-06-30"
)

loader.download_data()
loader.calculate_log_returns()
loader.summary()

simulator = MonteCarloSimulator(
    last_price=loader.get_last_price(),
    log_returns=loader.log_returns,
    days=252,
    simulations=10000,
    seed=42
)

simulator.simulate()
simulator.summary()

risk = RiskAnalyzer(simulator)

risk.summary()

visualizer = MonteCarloVisualizer(simulator)

visualizer.plot_price_paths().show()
visualizer.plot_distribution().show()
visualizer.plot_fan_chart().show()
visualizer.plot_return_distribution().show()
visualizer.plot_var().show()
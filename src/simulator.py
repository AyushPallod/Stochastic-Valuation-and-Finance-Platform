import numpy as np
import pandas as pd
from numba import njit


@njit
def simulate_gbm_single_numba(last_price, mu, sigma, days, simulations):
    """JIT-accelerated single-asset GBM simulation."""
    prices = np.zeros((days + 1, simulations))
    prices[0, :] = last_price
    for s in range(simulations):
        current = last_price
        for d in range(days):
            z = np.random.normal()
            ret = np.exp((mu - 0.5 * sigma**2) + sigma * z)
            current = current * ret
            prices[d + 1, s] = current
    return prices


@njit
def simulate_jump_diffusion_single_numba(last_price, mu, sigma, days, simulations, jump_lambda, jump_mu, jump_sigma):
    """JIT-accelerated single-asset Merton Jump Diffusion simulation."""
    prices = np.zeros((days + 1, simulations))
    prices[0, :] = last_price
    daily_lambda = jump_lambda / 252.0
    for s in range(simulations):
        current = last_price
        for d in range(days):
            z = np.random.normal()
            
            # Poisson jump simulation inside compiled Numba loop
            L_val = np.exp(-daily_lambda)
            k = 0
            p = 1.0
            while p > L_val:
                k += 1
                p *= np.random.uniform(0.0, 1.0)
            jumps = k - 1
            
            jump_effect = 0.0
            for _ in range(jumps):
                jump_effect += np.random.normal(jump_mu, jump_sigma)
                
            ret = np.exp((mu - 0.5 * sigma**2) + sigma * z + jump_effect)
            current = current * ret
            prices[d + 1, s] = current
    return prices


@njit
def simulate_gbm_portfolio_numba(last_price, mu, L, days, simulations):
    """JIT-accelerated multi-asset correlated GBM portfolio simulation."""
    num_assets = len(last_price)
    prices = np.zeros((days + 1, num_assets, simulations))
    for a in range(num_assets):
        prices[0, a, :] = last_price[a]
        
    for s in range(simulations):
        current = np.copy(last_price)
        for d in range(days):
            z = np.zeros(num_assets)
            for a in range(num_assets):
                z[a] = np.random.normal()
                
            correlated_ret = np.zeros(num_assets)
            for i in range(num_assets):
                dot_sum = 0.0
                for j in range(num_assets):
                    dot_sum += L[i, j] * z[j]
                correlated_ret[i] = dot_sum
                
            for a in range(num_assets):
                ret = np.exp(correlated_ret[a] + mu[a])
                current[a] = current[a] * ret
                prices[d + 1, a, s] = current[a]
    return prices


@njit
def simulate_jump_diffusion_portfolio_numba(last_price, mu, L, days, simulations, jump_lambda, jump_mu, jump_sigma):
    """JIT-accelerated multi-asset correlated Merton Jump Diffusion portfolio simulation."""
    num_assets = len(last_price)
    prices = np.zeros((days + 1, num_assets, simulations))
    daily_lambda = jump_lambda / 252.0
    for a in range(num_assets):
        prices[0, a, :] = last_price[a]
        
    for s in range(simulations):
        current = np.copy(last_price)
        for d in range(days):
            z = np.zeros(num_assets)
            for a in range(num_assets):
                z[a] = np.random.normal()
                
            correlated_ret = np.zeros(num_assets)
            for i in range(num_assets):
                dot_sum = 0.0
                for j in range(num_assets):
                    dot_sum += L[i, j] * z[j]
                correlated_ret[i] = dot_sum
                
            for a in range(num_assets):
                L_val = np.exp(-daily_lambda)
                k = 0
                p = 1.0
                while p > L_val:
                    k += 1
                    p *= np.random.uniform(0.0, 1.0)
                jumps = k - 1
                
                jump_effect = 0.0
                for _ in range(jumps):
                    jump_effect += np.random.normal(jump_mu, jump_sigma)
                    
                ret = np.exp(correlated_ret[a] + mu[a] + jump_effect)
                current[a] = current[a] * ret
                prices[d + 1, a, s] = current[a]
    return prices


class MonteCarloSimulator:
    """
    Simulation engine supporting single asset and multi-asset portfolio simulations
    using Geometric Brownian Motion (GBM), Historical Bootstrap, and Merton Jump Diffusion.
    Optimized via Numba JIT compilers.
    """

    def __init__(self, last_price, log_returns, days=65, simulations=1000, seed=None,
                 model_type='gbm', jump_lambda=0.1, jump_mu=-0.05, jump_sigma=0.1, weights=None):
        
        # Check if portfolio or single asset
        if isinstance(last_price, (pd.Series, np.ndarray, list)):
            self.last_price = pd.Series(last_price)
            self.is_portfolio = True
            self.num_assets = len(self.last_price)
        else:
            self.last_price = float(last_price)
            self.is_portfolio = False
            self.num_assets = 1

        self.log_returns = log_returns
        self.days = days
        self.simulations = simulations
        self.model_type = model_type.lower()
        self.jump_lambda = jump_lambda
        self.jump_mu = jump_mu
        self.jump_sigma = jump_sigma

        # Calculate mean (drift) and covariance
        if self.is_portfolio:
            self.mu = log_returns.mean()
            self.cov = log_returns.cov()
            
            # Setup weights
            if weights is None:
                self.weights = np.ones(self.num_assets) / self.num_assets
            else:
                self.weights = np.array(weights)
                # Normalize weights to sum to 1.0
                if np.sum(self.weights) > 0:
                    self.weights = self.weights / np.sum(self.weights)
                else:
                    self.weights = np.ones(self.num_assets) / self.num_assets
        else:
            self.mu = log_returns.mean()
            self.sigma = log_returns.std()
            self.weights = np.array([1.0])

        if seed is not None:
            np.random.seed(seed)
        
        self.simulated_prices = None
    
    def simulate(self, use_numba=True):
        if use_numba:
            try:
                if not self.is_portfolio:
                    if self.model_type == 'gbm':
                        prices = simulate_gbm_single_numba(
                            self.last_price, self.mu, self.sigma, self.days, self.simulations
                        )
                        self.simulated_prices = pd.DataFrame(prices)
                        return self.simulated_prices
                    elif self.model_type == 'jump_diffusion':
                        prices = simulate_jump_diffusion_single_numba(
                            self.last_price, self.mu, self.sigma, self.days, self.simulations,
                            self.jump_lambda, self.jump_mu, self.jump_sigma
                        )
                        self.simulated_prices = pd.DataFrame(prices)
                        return self.simulated_prices
                else:
                    try:
                        L = np.linalg.cholesky(self.cov.values)
                    except np.linalg.LinAlgError:
                        regularized_cov = self.cov.values + 1e-6 * np.eye(self.num_assets)
                        L = np.linalg.cholesky(regularized_cov)
                    
                    if self.model_type == 'gbm':
                        prices = simulate_gbm_portfolio_numba(
                            self.last_price.values, self.mu.values, L, self.days, self.simulations
                        )
                        portfolio_prices = np.sum(prices * self.weights[np.newaxis, :, np.newaxis], axis=1)
                        self.simulated_prices = pd.DataFrame(portfolio_prices)
                        return self.simulated_prices
                    elif self.model_type == 'jump_diffusion':
                        prices = simulate_jump_diffusion_portfolio_numba(
                            self.last_price.values, self.mu.values, L, self.days, self.simulations,
                            self.jump_lambda, self.jump_mu, self.jump_sigma
                        )
                        portfolio_prices = np.sum(prices * self.weights[np.newaxis, :, np.newaxis], axis=1)
                        self.simulated_prices = pd.DataFrame(portfolio_prices)
                        return self.simulated_prices
            except Exception:
                # Fall back transparently to NumPy vectorized model on any compilation or runtime exception
                pass

        # NumPy standard vector fallbacks
        if not self.is_portfolio:
            if self.model_type == 'gbm':
                random_returns = np.random.normal(
                    loc=self.mu,
                    scale=self.sigma,
                    size=(self.days, self.simulations)
                )
            elif self.model_type == 'bootstrap':
                random_returns = np.random.choice(
                    self.log_returns.values,
                    size=(self.days, self.simulations),
                    replace=True
                )
            elif self.model_type == 'jump_diffusion':
                gbm_returns = np.random.normal(
                    loc=self.mu,
                    scale=self.sigma,
                    size=(self.days, self.simulations)
                )
                daily_lambda = self.jump_lambda / 252.0
                jumps = np.random.poisson(
                    lam=daily_lambda,
                    size=(self.days, self.simulations)
                )
                jump_sizes = np.random.normal(
                    loc=self.jump_mu,
                    scale=self.jump_sigma,
                    size=(self.days, self.simulations)
                ) * jumps
                random_returns = gbm_returns + jump_sizes
            else:
                raise ValueError(f"Unknown simulation model: {self.model_type}")

            cummulative_returns = np.cumsum(random_returns, axis=0)
            prices = self.last_price * np.exp(cummulative_returns)
            prices = np.vstack([np.full(self.simulations, self.last_price), prices])
            self.simulated_prices = pd.DataFrame(prices)
        else:
            try:
                L = np.linalg.cholesky(self.cov.values)
            except np.linalg.LinAlgError:
                regularized_cov = self.cov.values + 1e-6 * np.eye(self.num_assets)
                L = np.linalg.cholesky(regularized_cov)

            if self.model_type in ('gbm', 'jump_diffusion'):
                Z = np.random.normal(
                    size=(self.days, self.num_assets, self.simulations)
                )
                correlated_returns = np.einsum('ij,kjl->kil', L, Z)
                gbm_returns = correlated_returns + self.mu.values[np.newaxis, :, np.newaxis]
                
                if self.model_type == 'gbm':
                    random_returns = gbm_returns
                else:
                    daily_lambda = self.jump_lambda / 252.0
                    jumps = np.random.poisson(
                        lam=daily_lambda,
                        size=(self.days, self.num_assets, self.simulations)
                    )
                    jump_sizes = np.random.normal(
                        loc=self.jump_mu,
                        scale=self.jump_sigma,
                        size=(self.days, self.num_assets, self.simulations)
                    ) * jumps
                    random_returns = gbm_returns + jump_sizes

            elif self.model_type == 'bootstrap':
                indices = np.random.choice(
                    len(self.log_returns),
                    size=(self.days, self.simulations),
                    replace=True
                )
                sampled_returns = self.log_returns.values[indices]
                random_returns = np.transpose(sampled_returns, (0, 2, 1))
            else:
                raise ValueError(f"Unknown simulation model: {self.model_type}")

            cummulative_returns = np.cumsum(random_returns, axis=0)
            start_prices = self.last_price.values[np.newaxis, :, np.newaxis]
            prices = start_prices * np.exp(cummulative_returns)
            t0_prices = np.repeat(start_prices, self.simulations, axis=2)
            prices = np.vstack([t0_prices, prices])
            
            portfolio_prices = np.sum(prices * self.weights[np.newaxis, :, np.newaxis], axis=1)
            self.simulated_prices = pd.DataFrame(portfolio_prices)

        return self.simulated_prices
    
    def get_final_prices(self):
        if self.simulated_prices is None:
            raise ValueError("Simulations not run. Please call simulate() first.")
        return self.simulated_prices.iloc[-1]
    
    def summary(self):
        final = self.get_final_prices()
        print("="*50)
        print(f"Simulations : {self.simulations:,}")
        print(f"Days        : {self.days}")
        if self.is_portfolio:
            print(f"Portfolio Assets: {', '.join(self.log_returns.columns)}")
        else:
            print(f"Current     : {self.last_price:.2f}")
        print(f"Expected    : {final.mean():.2f}")
        print(f"Median      : {final.median():.2f}")
        print(f"Minimum     : {final.min():.2f}")
        print(f"Maximum     : {final.max():.2f}")
        print(f"Std Dev     : {final.std():.2f}")
# qf627-project



## Machine-Learning Review

We predicted daily and weekly (5-day log returns) using supervised learning models, incorporating features from S&P 500 price and volume data.

While our machine learning strategies seemed promising when tested on the training data, they ran into real problems when faced with new, unseen data. The best results, at least by RMSE, came from linear models like Linear Regression, LASSO, and Elastic Net, as well as K-Nearest Neighbors. But in practice, these models didn’t give us useful or realistic trading signals. 

The linear models tended to make almost the same prediction over and over—usually a small, positive value—so the strategy basically just bought and held the index. In fact, our actual trading rule was simple: buy when the model predicted returns above zero, and otherwise just hold. But since the predictions rarely dropped below zero, we ended up almost always in a buy position, which didn’t really add any practical value over a basic buy-and-hold approach. 

That’s a sign the models couldn’t find meaningful patterns in the data, probably because predicting daily S&P 500 returns is extremely tough, with very little signal compared to the noise. K-Nearest Neighbors, meanwhile, made predictions that hovered around zero, with the occasional odd spike—likely just picking up on random noise rather than anything real. 

These problems point to bigger challenges: risks of using future information by accident (lookahead bias), models that break down when the market changes, and the fact that stock index returns are notoriously hard to predict and often behave almost randomly. 

There are also issues like possible survivorship bias in our data, overfitting to features, and assuming that past relationships will hold in the future—even when economic conditions shift, which is highly likely for our case, as we had a long prediction period of almost 5 years, and also we were unable to use macroeconomic features that could have possibly helped with the better prediction


## Momentum Strategy Review

This momentum-based trading strategy is implemented on the S&P 500 index (^GSPC) using moving average crossovers from November 2006 to November 2025. The core strategy employs a 20-day short moving average and an 80-day long moving average to generate buy and sell signals.

Strategy Logic: The system generates a buy signal when the 20-day MA crosses above the 80-day MA, indicating upward momentum, and a sell signal when it crosses below, suggesting downward momentum. This crossover approach aims to capture trending market movements while avoiding sideways market noise.

Position Sizing Analysis: The project compares two distinct approaches:

Fixed Position Sizing: Uses a constant 8 shares per trade, resulting in a final portfolio value of $126,732 (26.73% return)(8 is an arbitrary number)
Risk-Based Position Sizing: Allocates 10% of current portfolio value per trade, achieving $111,852 (11.85% return)
Performance Metrics: Both approaches show identical Sharpe ratios (0.49), but the risk-based method demonstrates superior risk management with significantly lower volatility (1.23% vs 2.64%) and maximum drawdown (-3.31% vs -9.29%). The fixed approach generates higher absolute returns but with substantially increased risk.

Issues we faced/approaches that could be sharpen:
- Which Moving average pair should we incorporate: Since we are testing on the last 25% of the timeframe, we wanted trades to be executed more often. Whereas, if the MA's were far greater (say 100,200 MA), would equate to fewer trade opportunities.
- Transaction costs not considered: Our analysis assumes zero transaction costs, but in reality, each buy/sell would incur brokerage fees and bid-ask spreads, which could significantly impact the strategy's profitability, especially with frequent trading.
- Lagging nature of moving averages: The crossover signals inherently lag behind actual trend changes, meaning we often enter positions after significant price movements have already occurred, potentially missing optimal entry/exit points.
- Risk management limitations: The strategy lacks stop-loss mechanisms or position sizing based on volatility, making it vulnerable to prolonged drawdowns during volatile market periods or false breakouts.
- Market regime sensitivity: The 20/80 MA combination may perform differently across various market conditions (bull, bear, sideways), suggesting the need for adaptive parameters or regime-detection mechanisms.
- Optimization and overfitting risk: While we tested 20/80 MA, more systematic parameter optimization across different MA combinations could improve performance, though this risks overfitting to historical data. 
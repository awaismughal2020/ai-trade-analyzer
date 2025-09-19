"""
Comprehensive Exploratory Data Analysis (EDA) for Meme Coin Trading Data
Dynamically analyzes any processed_crypto_data_*.csv file in the data directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class MemeCoinEDA:
    def __init__(self, data_dir="data"):
        """
        Initialize EDA analyzer
        
        Args:
            data_dir (str): Directory containing processed CSV files
        """
        self.data_dir = data_dir
        self.df = None
        self.csv_file = None
        
    def find_latest_processed_file(self):
        """
        Find the latest processed_crypto_data_*.csv file
        
        Returns:
            str: Path to the latest processed CSV file
        """
        pattern = os.path.join(self.data_dir, "processed_crypto_data_*.csv")
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            raise FileNotFoundError(f"No processed_crypto_data_*.csv files found in {self.data_dir}")
        
        # Sort by modification time (newest first)
        csv_files.sort(key=os.path.getmtime, reverse=True)
        
        latest_file = csv_files[0]
        print(f"Found {len(csv_files)} processed files, using latest: {os.path.basename(latest_file)}")
        
        return latest_file
    
    def load_data(self, csv_file=None):
        """
        Load data from CSV file
        
        Args:
            csv_file (str): Path to CSV file (if None, auto-detect latest)
        """
        if csv_file is None:
            csv_file = self.find_latest_processed_file()
        
        self.csv_file = csv_file
        print(f"Loading data from: {os.path.basename(csv_file)}")
        
        self.df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"Data loaded successfully:")
        print(f"- Records: {len(self.df):,}")
        print(f"- Columns: {len(self.df.columns)}")
        print(f"- Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"- Unique coins: {self.df['coin_id'].nunique()}")
        
    def basic_data_overview(self):
        """
        Basic Data Overview Analysis
        """
        print("\n" + "="*60)
        print("BASIC DATA OVERVIEW")
        print("="*60)
        
        # Data shape and info
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        # Missing values
        print(f"\nMissing Values:")
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing_data,
            'Missing %': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        # Summary statistics for numerical columns
        print(f"\nSummary Statistics for Numerical Columns:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())
        
        # Date range analysis
        print(f"\nDate Range Analysis:")
        print(f"- Start date: {self.df['timestamp'].min()}")
        print(f"- End date: {self.df['timestamp'].max()}")
        print(f"- Total days: {(self.df['timestamp'].max() - self.df['timestamp'].min()).days}")
        
        # Check for gaps in time series
        time_gaps = self.df.groupby('coin_id')['timestamp'].apply(
            lambda x: x.diff().dt.total_seconds() / 3600  # Convert to hours
        )
        expected_interval = 4  # Assuming 4-hour intervals
        gaps = time_gaps[time_gaps > expected_interval * 1.5]  # More than 1.5x expected
        
        if len(gaps) > 0:
            print(f"\nTime Series Gaps Detected:")
            print(f"- {len(gaps)} gaps found where interval > {expected_interval * 1.5} hours")
            print(f"- Largest gap: {gaps.max():.1f} hours")
        else:
            print(f"\nNo significant time series gaps detected")
    
    def target_variable_analysis(self):
        """
        Target Variable Analysis (Label Distribution)
        """
        print("\n" + "="*60)
        print("TARGET VARIABLE ANALYSIS")
        print("="*60)
        
        # Label distribution
        label_counts = self.df['label'].value_counts().sort_index()
        total = len(self.df)
        
        print(f"\nLabel Distribution:")
        label_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        for label, count in label_counts.items():
            pct = count / total * 100
            print(f"- {label_names[label]} ({label}): {count:,} ({pct:.1f}%)")
        
        # Class imbalance analysis
        print(f"\nClass Imbalance Analysis:")
        max_count = label_counts.max()
        min_count = label_counts.min()
        imbalance_ratio = max_count / min_count
        print(f"- Imbalance ratio: {imbalance_ratio:.2f}")
        print(f"- Most common: {label_names[label_counts.idxmax()]} ({label_counts.max():,})")
        print(f"- Least common: {label_names[label_counts.idxmin()]} ({label_counts.min():,})")
        
        # Visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        label_counts.plot(kind='bar', ax=axes[0], color=['red', 'orange', 'green'])
        axes[0].set_title('Label Distribution (Count)')
        axes[0].set_xlabel('Label')
        axes[0].set_ylabel('Count')
        axes[0].set_xticklabels([label_names[i] for i in label_counts.index], rotation=0)
        
        # Pie chart
        axes[1].pie(label_counts.values, labels=[label_names[i] for i in label_counts.index], 
                   autopct='%1.1f%%', colors=['red', 'orange', 'green'])
        axes[1].set_title('Label Distribution (Percentage)')
        
        plt.tight_layout()
        plt.show()
    
    def price_data_exploration(self):
        """
        Price Data Exploration and Outlier Analysis
        """
        print("\n" + "="*60)
        print("PRICE DATA EXPLORATION")
        print("="*60)
        
        # OHLC price statistics
        price_cols = ['open', 'high', 'low', 'close']
        print(f"\nOHLC Price Statistics:")
        price_stats = self.df[price_cols].describe()
        print(price_stats)
        
        # Price distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(price_cols):
            axes[i].hist(self.df[col], bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{col.title()} Price Distribution')
            axes[i].set_xlabel('Price')
            axes[i].set_ylabel('Frequency')
            axes[i].axvline(self.df[col].mean(), color='red', linestyle='--', label=f'Mean: {self.df[col].mean():.6f}')
            axes[i].axvline(self.df[col].median(), color='green', linestyle='--', label=f'Median: {self.df[col].median():.6f}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Outlier detection using IQR method
        print(f"\nOutlier Detection (IQR Method):")
        for col in price_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            print(f"- {col}: {len(outliers)} outliers ({len(outliers)/len(self.df)*100:.2f}%)")
        
        # Price volatility analysis
        print(f"\nPrice Volatility Analysis:")
        self.df['price_range'] = (self.df['high'] - self.df['low']) / self.df['close']
        self.df['price_change_abs'] = abs(self.df['price_change_pct'])
        
        print(f"- Average price range: {self.df['price_range'].mean():.4f} ({self.df['price_range'].mean()*100:.2f}%)")
        print(f"- Average absolute price change: {self.df['price_change_abs'].mean():.4f} ({self.df['price_change_abs'].mean()*100:.2f}%)")
        print(f"- Max price range: {self.df['price_range'].max():.4f} ({self.df['price_range'].max()*100:.2f}%)")
        print(f"- Max absolute price change: {self.df['price_change_abs'].max():.4f} ({self.df['price_change_abs'].max()*100:.2f}%)")
        
        # Correlation between OHLC prices
        print(f"\nOHLC Price Correlations:")
        ohlc_corr = self.df[price_cols].corr()
        print(ohlc_corr)
        
        # Visualization
        plt.figure(figsize=(10, 8))
        sns.heatmap(ohlc_corr, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.3f')
        plt.title('OHLC Price Correlation Matrix')
        plt.show()
    
    def technical_indicators_analysis(self):
        """
        Technical Indicators Analysis
        """
        print("\n" + "="*60)
        print("TECHNICAL INDICATORS ANALYSIS")
        print("="*60)
        
        # RSI analysis
        print(f"\nRSI Analysis:")
        rsi_stats = self.df['rsi_14'].describe()
        print(rsi_stats)
        
        # RSI distribution analysis
        oversold_count = len(self.df[self.df['rsi_14'] < 30])
        overbought_count = len(self.df[self.df['rsi_14'] > 70])
        total = len(self.df)
        
        print(f"- Oversold (< 30): {oversold_count:,} ({oversold_count/total*100:.1f}%)")
        print(f"- Overbought (> 70): {overbought_count:,} ({overbought_count/total*100:.1f}%)")
        print(f"- Neutral (30-70): {total - oversold_count - overbought_count:,} ({(total - oversold_count - overbought_count)/total*100:.1f}%)")
        
        # EMA analysis
        print(f"\nEMA Analysis:")
        print(f"- EMA-20 mean: {self.df['ema_20'].mean():.6f}")
        print(f"- Close/EMA ratio mean: {self.df['close_ema_ratio'].mean():.4f}")
        print(f"- Price above EMA: {self.df['price_above_ema'].sum():,} ({self.df['price_above_ema'].mean()*100:.1f}%)")
        
        # Volatility analysis
        print(f"\nVolatility Analysis:")
        vol_stats = self.df['volatility_20'].describe()
        print(vol_stats)
        
        # Volume analysis
        print(f"\nVolume Analysis:")
        print(f"- Volume ratio mean: {self.df['volume_ratio'].mean():.2f}")
        print(f"- Volume above average: {self.df['volume_above_avg'].sum():,} ({self.df['volume_above_avg'].mean()*100:.1f}%)")
        
        # Technical indicators visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RSI distribution
        axes[0,0].hist(self.df['rsi_14'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].axvline(30, color='red', linestyle='--', label='Oversold (30)')
        axes[0,0].axvline(70, color='red', linestyle='--', label='Overbought (70)')
        axes[0,0].set_title('RSI Distribution')
        axes[0,0].set_xlabel('RSI')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # Close/EMA ratio
        axes[0,1].hist(self.df['close_ema_ratio'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,1].axvline(1, color='red', linestyle='--', label='EMA Line')
        axes[0,1].set_title('Close/EMA Ratio Distribution')
        axes[0,1].set_xlabel('Close/EMA Ratio')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        
        # Volatility
        axes[1,0].hist(self.df['volatility_20'], bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Volatility Distribution')
        axes[1,0].set_xlabel('Volatility (20-period)')
        axes[1,0].set_ylabel('Frequency')
        
        # Volume ratio
        axes[1,1].hist(self.df['volume_ratio'], bins=50, alpha=0.7, edgecolor='black')
        axes[1,1].axvline(1, color='red', linestyle='--', label='Average Volume')
        axes[1,1].set_title('Volume Ratio Distribution')
        axes[1,1].set_xlabel('Volume Ratio')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def coin_specific_analysis(self):
        """
        Coin-Specific Analysis
        """
        print("\n" + "="*60)
        print("COIN-SPECIFIC ANALYSIS")
        print("="*60)
        
        # Unique coins count
        unique_coins = self.df['coin_id'].nunique()
        print(f"\nDataset contains {unique_coins} unique coins")
        
        # Data distribution across coins
        coin_counts = self.df['coin_id'].value_counts()
        print(f"\nTop 10 coins by record count:")
        print(coin_counts.head(10))
        
        print(f"\nBottom 10 coins by record count:")
        print(coin_counts.tail(10))
        
        # Coin dominance analysis
        total_records = len(self.df)
        top_coin_pct = coin_counts.iloc[0] / total_records * 100
        top_5_pct = coin_counts.head(5).sum() / total_records * 100
        
        print(f"\nCoin Dominance Analysis:")
        print(f"- Top coin represents: {top_coin_pct:.1f}% of all records")
        print(f"- Top 5 coins represent: {top_5_pct:.1f}% of all records")
        
        # Price statistics by coin
        print(f"\nPrice Statistics by Coin (Top 10):")
        top_coins = coin_counts.head(10).index
        coin_price_stats = self.df[self.df['coin_id'].isin(top_coins)].groupby('coin_id')['close'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(8)
        print(coin_price_stats)
        
        # Volatility by coin
        print(f"\nVolatility by Coin (Top 10):")
        coin_vol_stats = self.df[self.df['coin_id'].isin(top_coins)].groupby('coin_id')['volatility_20'].agg([
            'mean', 'std', 'min', 'max'
        ]).round(8)
        print(coin_vol_stats)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Records per coin (top 20)
        coin_counts.head(20).plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Records per Coin (Top 20)')
        axes[0,0].set_xlabel('Coin ID')
        axes[0,0].set_ylabel('Record Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Price distribution by top coins
        top_5_coins = coin_counts.head(5).index
        for coin in top_5_coins:
            coin_data = self.df[self.df['coin_id'] == coin]['close']
            axes[0,1].hist(coin_data, alpha=0.6, label=coin, bins=30)
        axes[0,1].set_title('Price Distribution by Top 5 Coins')
        axes[0,1].set_xlabel('Close Price')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].legend()
        axes[0,1].set_yscale('log')
        
        # Volatility by top coins
        coin_vol_data = []
        coin_labels = []
        for coin in top_5_coins:
            coin_data = self.df[self.df['coin_id'] == coin]['volatility_20']
            coin_vol_data.append(coin_data)
            coin_labels.append(coin)
        
        axes[1,0].boxplot(coin_vol_data, labels=coin_labels)
        axes[1,0].set_title('Volatility Distribution by Top 5 Coins')
        axes[1,0].set_xlabel('Coin ID')
        axes[1,0].set_ylabel('Volatility')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Records distribution pie chart
        top_10_counts = coin_counts.head(10)
        other_count = coin_counts.iloc[10:].sum()
        pie_data = list(top_10_counts.values) + [other_count]
        pie_labels = list(top_10_counts.index) + ['Others']
        
        axes[1,1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%')
        axes[1,1].set_title('Data Distribution by Coin')
        
        plt.tight_layout()
        plt.show()
    
    def feature_relationships(self):
        """
        Feature Relationships and Correlation Analysis
        """
        print("\n" + "="*60)
        print("FEATURE RELATIONSHIPS")
        print("="*60)
        
        # Select numerical features for correlation analysis
        numerical_features = [
            'rsi_14', 'ema_20', 'volume_ratio', 'price_change_pct',
            'high_low_ratio', 'close_ema_ratio', 'volatility_20',
            'price_above_ema', 'volume_above_avg', 'future_return'
        ]
        
        # Filter features that exist in the dataset
        available_features = [col for col in numerical_features if col in self.df.columns]
        
        print(f"\nAnalyzing correlations for {len(available_features)} features")
        
        # Correlation matrix
        corr_matrix = self.df[available_features].corr()
        
        print(f"\nCorrelation Matrix:")
        print(corr_matrix.round(3))
        
        # Strong correlations (> 0.5 or < -0.5)
        print(f"\nStrong Correlations (|r| > 0.5):")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
        
        for feat1, feat2, corr in strong_corr:
            print(f"- {feat1} vs {feat2}: {corr:.3f}")
        
        # Correlation with target variable (future_return)
        if 'future_return' in self.df.columns:
            print(f"\nCorrelation with Future Return:")
            future_return_corr = corr_matrix['future_return'].drop('future_return').sort_values(key=abs, ascending=False)
            for feature, corr in future_return_corr.items():
                print(f"- {feature}: {corr:.3f}")
        
        # Correlation with labels
        print(f"\nCorrelation with Labels:")
        label_corr = []
        for feature in available_features:
            if feature != 'future_return':
                corr = self.df[feature].corr(self.df['label'])
                label_corr.append((feature, corr))
        
        label_corr.sort(key=lambda x: abs(x[1]), reverse=True)
        for feature, corr in label_corr:
            print(f"- {feature}: {corr:.3f}")
        
        # Visualization
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
        # Scatter plots for top correlations with future return
        if 'future_return' in self.df.columns:
            top_corr_features = future_return_corr.head(4).index
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, feature in enumerate(top_corr_features):
                axes[i].scatter(self.df[feature], self.df['future_return'], alpha=0.5, s=1)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Future Return')
                axes[i].set_title(f'{feature} vs Future Return (r={future_return_corr[feature]:.3f})')
                
                # Add trend line
                z = np.polyfit(self.df[feature].dropna(), self.df['future_return'].dropna(), 1)
                p = np.poly1d(z)
                axes[i].plot(self.df[feature], p(self.df[feature]), "r--", alpha=0.8)
            
            plt.tight_layout()
            plt.show()
    
    def time_series_patterns(self):
        """
        Time Series Patterns Analysis
        """
        print("\n" + "="*60)
        print("TIME SERIES PATTERNS")
        print("="*60)
        
        # Daily aggregation
        daily_stats = self.df.groupby(self.df['timestamp'].dt.date).agg({
            'close': ['mean', 'std', 'min', 'max'],
            'volume': 'sum',
            'label': lambda x: x.value_counts().to_dict()
        }).round(6)
        
        print(f"\nDaily Statistics (first 10 days):")
        print(daily_stats.head(10))
        
        # Label distribution over time
        daily_labels = self.df.groupby(self.df['timestamp'].dt.date)['label'].value_counts().unstack(fill_value=0)
        daily_labels_pct = daily_labels.div(daily_labels.sum(axis=1), axis=0) * 100
        
        print(f"\nLabel Distribution Over Time (first 10 days):")
        print(daily_labels_pct.head(10))
        
        # Time series visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Price trend over time
        daily_price = self.df.groupby(self.df['timestamp'].dt.date)['close'].mean()
        axes[0,0].plot(daily_price.index, daily_price.values)
        axes[0,0].set_title('Average Close Price Over Time')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Close Price')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # Volume trend over time
        daily_volume = self.df.groupby(self.df['timestamp'].dt.date)['volume'].sum()
        axes[0,1].plot(daily_volume.index, daily_volume.values)
        axes[0,1].set_title('Total Volume Over Time')
        axes[0,1].set_xlabel('Date')
        axes[0,1].set_ylabel('Volume')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Label distribution over time
        daily_labels_pct.plot(kind='area', ax=axes[1,0], stacked=True, 
                            color=['red', 'orange', 'green'], alpha=0.7)
        axes[1,0].set_title('Label Distribution Over Time')
        axes[1,0].set_xlabel('Date')
        axes[1,0].set_ylabel('Percentage')
        axes[1,0].legend(['SELL', 'HOLD', 'BUY'])
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # RSI trend over time
        daily_rsi = self.df.groupby(self.df['timestamp'].dt.date)['rsi_14'].mean()
        axes[1,1].plot(daily_rsi.index, daily_rsi.values)
        axes[1,1].axhline(y=30, color='red', linestyle='--', alpha=0.7, label='Oversold')
        axes[1,1].axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        axes[1,1].set_title('Average RSI Over Time')
        axes[1,1].set_xlabel('Date')
        axes[1,1].set_ylabel('RSI')
        axes[1,1].legend()
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Seasonality analysis (hourly patterns)
        print(f"\nHourly Patterns Analysis:")
        hourly_stats = self.df.groupby(self.df['timestamp'].dt.hour).agg({
            'close': 'mean',
            'volume': 'mean',
            'rsi_14': 'mean',
            'label': lambda x: x.value_counts().to_dict()
        })
        
        print(hourly_stats.head(10))
        
        # Day of week patterns
        print(f"\nDay of Week Patterns:")
        dow_stats = self.df.groupby(self.df['timestamp'].dt.day_name()).agg({
            'close': 'mean',
            'volume': 'mean',
            'rsi_14': 'mean',
            'label': lambda x: x.value_counts().to_dict()
        })
        print(dow_stats)
    
    def future_return_analysis(self):
        """
        Future Return Analysis
        """
        print("\n" + "="*60)
        print("FUTURE RETURN ANALYSIS")
        print("="*60)
        
        if 'future_return' not in self.df.columns:
            print("Future return data not available in this dataset")
            return
        
        # Future return statistics
        print(f"\nFuture Return Statistics:")
        future_return_stats = self.df['future_return'].describe()
        print(future_return_stats)
        
        # Future return distribution
        print(f"\nFuture Return Distribution:")
        positive_returns = len(self.df[self.df['future_return'] > 0])
        negative_returns = len(self.df[self.df['future_return'] < 0])
        zero_returns = len(self.df[self.df['future_return'] == 0])
        total = len(self.df)
        
        print(f"- Positive returns: {positive_returns:,} ({positive_returns/total*100:.1f}%)")
        print(f"- Negative returns: {negative_returns:,} ({negative_returns/total*100:.1f}%)")
        print(f"- Zero returns: {zero_returns:,} ({zero_returns/total*100:.1f}%)")
        
        # Extreme returns analysis
        extreme_positive = len(self.df[self.df['future_return'] > 0.1])  # >10%
        extreme_negative = len(self.df[self.df['future_return'] < -0.1])  # <-10%
        
        print(f"\nExtreme Returns Analysis:")
        print(f"- Returns > 10%: {extreme_positive:,} ({extreme_positive/total*100:.1f}%)")
        print(f"- Returns < -10%: {extreme_negative:,} ({extreme_negative/total*100:.1f}%)")
        
        # Label validation
        print(f"\nLabel Validation:")
        print(f"- BUY labels (2) with positive returns: {len(self.df[(self.df['label'] == 2) & (self.df['future_return'] > 0)]):,}")
        print(f"- SELL labels (0) with negative returns: {len(self.df[(self.df['label'] == 0) & (self.df['future_return'] < 0)]):,}")
        print(f"- HOLD labels (1) with small returns: {len(self.df[(self.df['label'] == 1) & (abs(self.df['future_return']) <= 0.02)]):,}")
        
        # Future return by label
        print(f"\nFuture Return by Label:")
        label_return_stats = self.df.groupby('label')['future_return'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(4)
        print(label_return_stats)
        
        # Visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Future return distribution
        axes[0,0].hist(self.df['future_return'], bins=50, alpha=0.7, edgecolor='black')
        axes[0,0].axvline(0, color='red', linestyle='--', label='Zero Return')
        axes[0,0].set_title('Future Return Distribution')
        axes[0,0].set_xlabel('Future Return')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        
        # Future return by label (box plot)
        label_data = [self.df[self.df['label'] == i]['future_return'].values for i in [0, 1, 2]]
        axes[0,1].boxplot(label_data, labels=['SELL', 'HOLD', 'BUY'])
        axes[0,1].set_title('Future Return by Label')
        axes[0,1].set_xlabel('Label')
        axes[0,1].set_ylabel('Future Return')
        axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Future return vs RSI
        axes[1,0].scatter(self.df['rsi_14'], self.df['future_return'], alpha=0.5, s=1)
        axes[1,0].set_xlabel('RSI')
        axes[1,0].set_ylabel('Future Return')
        axes[1,0].set_title('Future Return vs RSI')
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        # Future return vs Volume Ratio
        axes[1,1].scatter(self.df['volume_ratio'], self.df['future_return'], alpha=0.5, s=1)
        axes[1,1].set_xlabel('Volume Ratio')
        axes[1,1].set_ylabel('Future Return')
        axes[1,1].set_title('Future Return vs Volume Ratio')
        axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_eda(self, csv_file=None):
        """
        Run complete EDA analysis
        
        Args:
            csv_file (str): Path to CSV file (if None, auto-detect latest)
        """
        print("="*80)
        print("COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("MEME COIN TRADING DATA")
        print("="*80)
        
        # Load data
        self.load_data(csv_file)
        
        if self.df is None or self.df.empty:
            print("No data available for analysis")
            return
        
        # Run all analysis sections
        self.basic_data_overview()
        self.target_variable_analysis()
        self.price_data_exploration()
        self.technical_indicators_analysis()
        self.coin_specific_analysis()
        self.feature_relationships()
        self.time_series_patterns()
        self.future_return_analysis()
        
        print("\n" + "="*80)
        print("EDA ANALYSIS COMPLETE")
        print("="*80)
        
        # Summary statistics
        print(f"\nDataset Summary:")
        print(f"- File analyzed: {os.path.basename(self.csv_file)}")
        print(f"- Total records: {len(self.df):,}")
        print(f"- Unique coins: {self.df['coin_id'].nunique()}")
        print(f"- Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"- Features analyzed: {len(self.df.columns)}")
        
        # Label distribution summary
        label_counts = self.df['label'].value_counts().sort_index()
        label_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        print(f"\nLabel Distribution:")
        for label, count in label_counts.items():
            pct = count / len(self.df) * 100
            print(f"- {label_names[label]}: {count:,} ({pct:.1f}%)")


def main():
    """
    Main function to run EDA analysis
    """
    import sys
    
    # Initialize EDA analyzer
    eda = MemeCoinEDA()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            # List available files
            pattern = os.path.join(eda.data_dir, "processed_crypto_data_*.csv")
            csv_files = glob.glob(pattern)
            
            if not csv_files:
                print(f"No processed_crypto_data_*.csv files found in {eda.data_dir}")
                return
            
            csv_files.sort(key=os.path.getmtime, reverse=True)
            
            print(f"Available processed files:")
            for i, file in enumerate(csv_files, 1):
                basename = os.path.basename(file)
                mod_time = os.path.getmtime(file)
                mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
                print(f"{i}. {basename} (modified: {mod_time_str})")
            return
        elif sys.argv[1] == "--file" and len(sys.argv) > 2:
            # Analyze specific file
            csv_file = sys.argv[2]
            if not os.path.exists(csv_file):
                print(f"File not found: {csv_file}")
                return
            eda.run_complete_eda(csv_file)
            return
        else:
            print("Usage:")
            print("  python eda_analysis.py                    # Analyze latest file")
            print("  python eda_analysis.py --list             # List available files")
            print("  python eda_analysis.py --file <path>      # Analyze specific file")
            return
    else:
        # Run complete EDA on latest file
        try:
            eda.run_complete_eda()
        except Exception as e:
            print(f"Error during EDA analysis: {e}")
            raise


if __name__ == "__main__":
    main()

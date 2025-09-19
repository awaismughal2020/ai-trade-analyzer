"""
AI-Powered EDA Report Generator
Uses OpenAI to generate comprehensive analytical reports from EDA data
Generates PDF reports with detailed insights and recommendations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from datetime import datetime
import json
import warnings
from openai import OpenAI
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import io
import base64
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')

class AIEDAReporter:
    def __init__(self, data_dir="data", reports_dir="data/reports"):
        """
        Initialize AI EDA Reporter
        
        Args:
            data_dir (str): Directory containing processed CSV files
            reports_dir (str): Directory to save generated reports
        """
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.df = None
        self.csv_file = None
        self.eda_results = {}
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Create reports directory if it doesn't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        
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
    
    def extract_eda_insights(self):
        """
        Extract comprehensive EDA insights from the dataset
        """
        print("Extracting EDA insights...")
        
        # Basic data overview
        self.eda_results['basic_info'] = {
            'dataset_shape': self.df.shape,
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'date_range': {
                'start': str(self.df['timestamp'].min()),
                'end': str(self.df['timestamp'].max()),
                'total_days': (self.df['timestamp'].max() - self.df['timestamp'].min()).days
            },
            'unique_coins': self.df['coin_id'].nunique(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.astype(str).to_dict()
        }
        
        # Target variable analysis
        label_counts = self.df['label'].value_counts().sort_index()
        total = len(self.df)
        self.eda_results['target_variable'] = {
            'label_distribution': {
                'SELL (0)': {'count': int(label_counts.get(0, 0)), 'percentage': float(label_counts.get(0, 0) / total * 100)},
                'HOLD (1)': {'count': int(label_counts.get(1, 0)), 'percentage': float(label_counts.get(1, 0) / total * 100)},
                'BUY (2)': {'count': int(label_counts.get(2, 0)), 'percentage': float(label_counts.get(2, 0) / total * 100)}
            },
            'class_imbalance_ratio': float(label_counts.max() / label_counts.min()),
            'most_common_label': int(label_counts.idxmax()),
            'least_common_label': int(label_counts.idxmin())
        }
        
        # Price data analysis
        price_cols = ['open', 'high', 'low', 'close']
        price_stats = self.df[price_cols].describe()
        
        # Outlier analysis using IQR
        outliers = {}
        for col in price_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_count = len(self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)])
            outliers[col] = {
                'count': outlier_count,
                'percentage': float(outlier_count / len(self.df) * 100)
            }
        
        self.eda_results['price_analysis'] = {
            'ohlc_statistics': price_stats.to_dict(),
            'outliers': outliers,
            'price_volatility': {
                'avg_price_range_pct': float(self.df['high_low_ratio'].mean() * 100),
                'avg_abs_price_change_pct': float(abs(self.df['price_change_pct']).mean() * 100),
                'max_price_range_pct': float(self.df['high_low_ratio'].max() * 100),
                'max_abs_price_change_pct': float(abs(self.df['price_change_pct']).max() * 100)
            },
            'ohlc_correlations': self.df[price_cols].corr().to_dict()
        }
        
        # Technical indicators analysis
        self.eda_results['technical_indicators'] = {
            'rsi_analysis': {
                'mean': float(self.df['rsi_14'].mean()),
                'std': float(self.df['rsi_14'].std()),
                'min': float(self.df['rsi_14'].min()),
                'max': float(self.df['rsi_14'].max()),
                'oversold_count': int(len(self.df[self.df['rsi_14'] < 30])),
                'oversold_percentage': float(len(self.df[self.df['rsi_14'] < 30]) / len(self.df) * 100),
                'overbought_count': int(len(self.df[self.df['rsi_14'] > 70])),
                'overbought_percentage': float(len(self.df[self.df['rsi_14'] > 70]) / len(self.df) * 100)
            },
            'ema_analysis': {
                'ema_20_mean': float(self.df['ema_20'].mean()),
                'close_ema_ratio_mean': float(self.df['close_ema_ratio'].mean()),
                'price_above_ema_count': int(self.df['price_above_ema'].sum()),
                'price_above_ema_percentage': float(self.df['price_above_ema'].mean() * 100)
            },
            'volatility_analysis': {
                'volatility_20_mean': float(self.df['volatility_20'].mean()),
                'volatility_20_std': float(self.df['volatility_20'].std()),
                'volatility_20_min': float(self.df['volatility_20'].min()),
                'volatility_20_max': float(self.df['volatility_20'].max())
            },
            'volume_analysis': {
                'volume_ratio_mean': float(self.df['volume_ratio'].mean()),
                'volume_above_avg_count': int(self.df['volume_above_avg'].sum()),
                'volume_above_avg_percentage': float(self.df['volume_above_avg'].mean() * 100)
            }
        }
        
        # Coin-specific analysis
        coin_counts = self.df['coin_id'].value_counts()
        top_coins = coin_counts.head(10).to_dict()
        bottom_coins = coin_counts.tail(10).to_dict()
        
        self.eda_results['coin_analysis'] = {
            'total_unique_coins': int(self.df['coin_id'].nunique()),
            'top_10_coins': {str(k): int(v) for k, v in top_coins.items()},
            'bottom_10_coins': {str(k): int(v) for k, v in bottom_coins.items()},
            'coin_dominance': {
                'top_coin_percentage': float(coin_counts.iloc[0] / len(self.df) * 100),
                'top_5_coins_percentage': float(coin_counts.head(5).sum() / len(self.df) * 100)
            }
        }
        
        # Feature relationships
        numerical_features = [
            'rsi_14', 'ema_20', 'volume_ratio', 'price_change_pct',
            'high_low_ratio', 'close_ema_ratio', 'volatility_20',
            'price_above_ema', 'volume_above_avg'
        ]
        
        available_features = [col for col in numerical_features if col in self.df.columns]
        corr_matrix = self.df[available_features + ['future_return']].corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    strong_correlations.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_val)
                    })
        
        # Future return correlations
        future_return_corr = {}
        if 'future_return' in self.df.columns:
            future_return_corr = corr_matrix['future_return'].drop('future_return').to_dict()
            future_return_corr = {k: float(v) for k, v in future_return_corr.items()}
        
        self.eda_results['feature_relationships'] = {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'future_return_correlations': future_return_corr,
            'label_correlations': {col: float(self.df[col].corr(self.df['label'])) for col in available_features}
        }
        
        # Time series patterns
        daily_stats = self.df.groupby(self.df['timestamp'].dt.date).agg({
            'close': ['mean', 'std'],
            'volume': 'sum',
            'label': lambda x: x.value_counts().to_dict()
        })
        
        # Hourly patterns
        hourly_stats = self.df.groupby(self.df['timestamp'].dt.hour).agg({
            'close': 'mean',
            'volume': 'mean',
            'rsi_14': 'mean'
        })
        
        self.eda_results['time_series_patterns'] = {
            'daily_statistics_sample': daily_stats.head(10).to_dict(),
            'hourly_patterns': hourly_stats.to_dict(),
            'day_of_week_patterns': self.df.groupby(self.df['timestamp'].dt.day_name()).agg({
                'close': 'mean',
                'volume': 'mean',
                'rsi_14': 'mean'
            }).to_dict()
        }
        
        # Future return analysis
        if 'future_return' in self.df.columns:
            future_return_stats = self.df['future_return'].describe()
            
            positive_returns = len(self.df[self.df['future_return'] > 0])
            negative_returns = len(self.df[self.df['future_return'] < 0])
            zero_returns = len(self.df[self.df['future_return'] == 0])
            
            extreme_positive = len(self.df[self.df['future_return'] > 0.1])
            extreme_negative = len(self.df[self.df['future_return'] < -0.1])
            
            # Label validation
            buy_positive = len(self.df[(self.df['label'] == 2) & (self.df['future_return'] > 0)])
            sell_negative = len(self.df[(self.df['label'] == 0) & (self.df['future_return'] < 0)])
            hold_small = len(self.df[(self.df['label'] == 1) & (abs(self.df['future_return']) <= 0.02)])
            
            self.eda_results['future_return_analysis'] = {
                'statistics': future_return_stats.to_dict(),
                'distribution': {
                    'positive_returns': {'count': positive_returns, 'percentage': float(positive_returns / len(self.df) * 100)},
                    'negative_returns': {'count': negative_returns, 'percentage': float(negative_returns / len(self.df) * 100)},
                    'zero_returns': {'count': zero_returns, 'percentage': float(zero_returns / len(self.df) * 100)}
                },
                'extreme_returns': {
                    'returns_above_10pct': {'count': extreme_positive, 'percentage': float(extreme_positive / len(self.df) * 100)},
                    'returns_below_10pct': {'count': extreme_negative, 'percentage': float(extreme_negative / len(self.df) * 100)}
                },
                'label_validation': {
                    'buy_labels_with_positive_returns': buy_positive,
                    'sell_labels_with_negative_returns': sell_negative,
                    'hold_labels_with_small_returns': hold_small
                },
                'future_return_by_label': self.df.groupby('label')['future_return'].agg(['count', 'mean', 'std', 'min', 'max']).to_dict()
            }
        
        print("EDA insights extracted successfully!")
    
    def create_enhanced_prompt(self):
        """
        Create an enhanced prompt for OpenAI analysis
        """
        prompt = f"""You are a senior data scientist and financial analyst specializing in cryptocurrency markets and machine learning. Generate a comprehensive analytical report for cryptocurrency trading signal prediction based on the EDA findings below.

**Dataset Context:**
- Crypto meme coins OHLC data with technical indicators
- Target: Trading signals (SELL=0, HOLD=1, BUY=2) 
- Records: {self.eda_results['basic_info']['dataset_shape'][0]:,} across {self.eda_results['basic_info']['unique_coins']} coins
- Period: {self.eda_results['basic_info']['date_range']['start']} to {self.eda_results['basic_info']['date_range']['end']} ({self.eda_results['basic_info']['date_range']['total_days']} days)

**Key EDA Findings:**

1. **Target Distribution:**
   - SELL: {self.eda_results['target_variable']['label_distribution']['SELL (0)']['percentage']:.1f}%, HOLD: {self.eda_results['target_variable']['label_distribution']['HOLD (1)']['percentage']:.1f}%, BUY: {self.eda_results['target_variable']['label_distribution']['BUY (2)']['percentage']:.1f}%
   - Class imbalance ratio: {self.eda_results['target_variable']['class_imbalance_ratio']:.2f}

2. **Price Volatility:**
   - Average range: {self.eda_results['price_analysis']['price_volatility']['avg_price_range_pct']:.2f}%
   - Max range: {self.eda_results['price_analysis']['price_volatility']['max_price_range_pct']:.2f}%
   - Outliers: {sum(v['count'] for v in self.eda_results['price_analysis']['outliers'].values())} total

3. **Technical Indicators:**
   - RSI: Mean {self.eda_results['technical_indicators']['rsi_analysis']['mean']:.1f}, Oversold {self.eda_results['technical_indicators']['rsi_analysis']['oversold_percentage']:.1f}%, Overbought {self.eda_results['technical_indicators']['rsi_analysis']['overbought_percentage']:.1f}%
   - Price above EMA: {self.eda_results['technical_indicators']['ema_analysis']['price_above_ema_percentage']:.1f}%
   - Volume above avg: {self.eda_results['technical_indicators']['volume_analysis']['volume_above_avg_percentage']:.1f}%

4. **Market Structure:**
   - Top coin dominance: {self.eda_results['coin_analysis']['coin_dominance']['top_coin_percentage']:.1f}%
   - Top 5 coins: {self.eda_results['coin_analysis']['coin_dominance']['top_5_coins_percentage']:.1f}% of data

5. **Future Returns:**
   - Positive: {self.eda_results['future_return_analysis']['distribution']['positive_returns']['percentage']:.1f}%, Negative: {self.eda_results['future_return_analysis']['distribution']['negative_returns']['percentage']:.1f}%
   - Extreme (>10%): {self.eda_results['future_return_analysis']['extreme_returns']['returns_above_10pct']['percentage']:.1f}%

6. **Strong Correlations:**
   {', '.join([f"{c['feature1']}-{c['feature2']} ({c['correlation']:.2f})" for c in self.eda_results['feature_relationships']['strong_correlations'][:5]])}

**Required Report Sections:**

1. **Executive Summary** - Key insights and data quality assessment
2. **Market Behavior Analysis** - Meme coin dynamics and volatility patterns  
3. **Technical Indicator Effectiveness** - RSI, EMA, volume analysis with statistical backing
4. **Target Variable Analysis** - Label distribution and class imbalance handling
5. **Feature Engineering** - New features, transformations, selection strategies
6. **Model Development Strategy** - ML approaches, validation, evaluation metrics
7. **Risk Assessment** - Data limitations, market risks, overfitting concerns
8. **Action Plan** - Preprocessing steps, development roadmap, benchmarks

**Requirements:**
- Provide specific numerical evidence for all claims
- Include actionable recommendations with implementation details
- Address meme coin volatility challenges
- Suggest Python libraries and code examples
- Consider risk management and position sizing
- Include realistic performance expectations

Generate a detailed, data-driven report with clear guidance for building robust cryptocurrency trading models."""
        
        return prompt
    
    def generate_ai_report(self):
        """
        Generate AI-powered analytical report using OpenAI
        """
        print("Generating AI-powered analytical report...")
        
        try:
            prompt = self.create_enhanced_prompt()
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior data scientist and financial analyst specializing in cryptocurrency markets and machine learning. Generate comprehensive, data-driven analytical reports with specific actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=3000,
                temperature=0.3
            )
            
            report_content = response.choices[0].message.content
            print("AI report generated successfully!")
            
            return report_content
            
        except Exception as e:
            print(f"Error generating AI report: {e}")
            return None
    
    def create_pdf_report(self, report_content, filename=None):
        """
        Create PDF report from the AI-generated content
        
        Args:
            report_content (str): AI-generated report content
            filename (str): Output filename (if None, auto-generate)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eda_report_{timestamp}.pdf"
        
        filepath = os.path.join(self.reports_dir, filename)
        
        print(f"Creating PDF report: {filename}")
        
        # Create PDF document
        doc = SimpleDocTemplate(filepath, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # Build content
        story = []
        
        # Title
        story.append(Paragraph("Comprehensive EDA Analysis Report", title_style))
        story.append(Paragraph("AI-Powered Cryptocurrency Trading Signal Analysis", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Dataset info
        story.append(Paragraph("Dataset Information", heading_style))
        story.append(Paragraph(f"<b>File:</b> {os.path.basename(self.csv_file)}", body_style))
        story.append(Paragraph(f"<b>Records:</b> {self.eda_results['basic_info']['dataset_shape'][0]:,}", body_style))
        story.append(Paragraph(f"<b>Unique Coins:</b> {self.eda_results['basic_info']['unique_coins']}", body_style))
        story.append(Paragraph(f"<b>Date Range:</b> {self.eda_results['basic_info']['date_range']['start']} to {self.eda_results['basic_info']['date_range']['end']}", body_style))
        story.append(Paragraph(f"<b>Analysis Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
        story.append(Spacer(1, 20))
        
        # Split report content into sections
        sections = report_content.split('\n\n')
        current_section = ""
        
        for section in sections:
            if section.strip():
                # Check if this is a main heading
                if section.strip().startswith('**') and section.strip().endswith('**'):
                    # Add previous section if exists
                    if current_section.strip():
                        story.append(Paragraph(current_section.strip(), body_style))
                        story.append(Spacer(1, 12))
                    
                    # Add new heading
                    heading_text = section.strip().replace('**', '').strip()
                    story.append(Paragraph(heading_text, heading_style))
                    current_section = ""
                else:
                    # Add to current section
                    current_section += section + "\n\n"
        
        # Add final section
        if current_section.strip():
            story.append(Paragraph(current_section.strip(), body_style))
        
        # Add page break and summary
        story.append(PageBreak())
        story.append(Paragraph("Key Statistics Summary", heading_style))
        
        # Add key statistics
        stats_text = f"""
        <b>Label Distribution:</b><br/>
        ¢ SELL (0): {self.eda_results['target_variable']['label_distribution']['SELL (0)']['count']:,} ({self.eda_results['target_variable']['label_distribution']['SELL (0)']['percentage']:.1f}%)<br/>
        ¢ HOLD (1): {self.eda_results['target_variable']['label_distribution']['HOLD (1)']['count']:,} ({self.eda_results['target_variable']['label_distribution']['HOLD (1)']['percentage']:.1f}%)<br/>
        ¢ BUY (2): {self.eda_results['target_variable']['label_distribution']['BUY (2)']['count']:,} ({self.eda_results['target_variable']['label_distribution']['BUY (2)']['percentage']:.1f}%)<br/><br/>
        
        <b>Price Volatility:</b><br/>
        ¢ Average Price Range: {self.eda_results['price_analysis']['price_volatility']['avg_price_range_pct']:.2f}%<br/>
        ¢ Maximum Price Range: {self.eda_results['price_analysis']['price_volatility']['max_price_range_pct']:.2f}%<br/><br/>
        
        <b>Technical Indicators:</b><br/>
        ¢ RSI Oversold Signals: {self.eda_results['technical_indicators']['rsi_analysis']['oversold_percentage']:.1f}%<br/>
        ¢ RSI Overbought Signals: {self.eda_results['technical_indicators']['rsi_analysis']['overbought_percentage']:.1f}%<br/>
        ¢ Price Above EMA: {self.eda_results['technical_indicators']['ema_analysis']['price_above_ema_percentage']:.1f}%<br/><br/>
        
        <b>Market Structure:</b><br/>
        ¢ Top Coin Dominance: {self.eda_results['coin_analysis']['coin_dominance']['top_coin_percentage']:.1f}%<br/>
        ¢ Top 5 Coins Represent: {self.eda_results['coin_analysis']['coin_dominance']['top_5_coins_percentage']:.1f}% of data<br/>
        """
        
        story.append(Paragraph(stats_text, body_style))
        
        # Build PDF
        doc.build(story)
        
        print(f"PDF report created successfully: {filepath}")
        return filepath
    
    def generate_complete_report(self, csv_file=None):
        """
        Generate complete AI-powered EDA report
        
        Args:
            csv_file (str): Path to CSV file (if None, auto-detect latest)
        """
        print("="*80)
        print("AI-POWERED EDA REPORT GENERATOR")
        print("="*80)
        
        # Load data
        self.load_data(csv_file)
        
        if self.df is None or self.df.empty:
            print("No data available for analysis")
            return None
        
        # Extract EDA insights
        self.extract_eda_insights()
        
        # Generate AI report
        report_content = self.generate_ai_report()
        
        if report_content is None:
            print("Failed to generate AI report")
            return None
        
        # Create PDF report
        pdf_path = self.create_pdf_report(report_content)
        
        print("\n" + "="*80)
        print("REPORT GENERATION COMPLETE")
        print("="*80)
        print(f"PDF Report saved to: {pdf_path}")
        print(f"Report contains comprehensive analysis of {len(self.df):,} records")
        print(f"Analysis covers {self.df['coin_id'].nunique()} unique meme coins")
        
        return pdf_path


def main():
    """
    Main function to generate AI-powered EDA report
    """
    import sys
    
    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in the .env file")
        return
    
    # Initialize AI EDA Reporter
    reporter = AIEDAReporter()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            # List available files
            pattern = os.path.join(reporter.data_dir, "processed_crypto_data_*.csv")
            csv_files = glob.glob(pattern)
            
            if not csv_files:
                print(f"No processed_crypto_data_*.csv files found in {reporter.data_dir}")
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
            reporter.generate_complete_report(csv_file)
            return
        else:
            print("Usage:")
            print("  python ai_eda_reporter.py                    # Generate report for latest file")
            print("  python ai_eda_reporter.py --list             # List available files")
            print("  python ai_eda_reporter.py --file <path>      # Generate report for specific file")
            return
    else:
        # Generate complete report for latest file
        try:
            reporter.generate_complete_report()
        except Exception as e:
            print(f"Error during report generation: {e}")
            raise


if __name__ == "__main__":
    main()

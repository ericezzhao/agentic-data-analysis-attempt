"""
Agentic AI Data Analysis - Demo with Sample Data

This demo launcher creates sample datasets and launches the Streamlit UI
so users can immediately try the system without needing their own data files.
"""

import pandas as pd
import numpy as np
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import json

def create_sample_datasets():
    """Create comprehensive sample datasets for demonstration."""
    
    # Ensure data/sample directory exists
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Creating sample datasets for demonstration...")
    
    # 1. Sales Performance Dataset
    print("   Creating: Sales Performance Dataset...")
    sales_data = {
        'date': pd.date_range('2023-01-01', '2024-12-31', freq='D'),
    }
    
    # Generate realistic sales data
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    sales_records = []
    
    regions = ['North', 'South', 'East', 'West', 'Central']
    products = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    sales_reps = [f'Rep_{i:02d}' for i in range(1, 21)]
    
    for i, date in enumerate(dates):
        # Generate 5-15 sales per day
        daily_sales = np.random.randint(5, 16)
        
        for _ in range(daily_sales):
            # Seasonal effects
            month = date.month
            seasonal_multiplier = 1.0
            if month in [11, 12]:  # Holiday season
                seasonal_multiplier = 1.4
            elif month in [6, 7, 8]:  # Summer boost
                seasonal_multiplier = 1.2
            elif month in [1, 2]:  # Post-holiday dip
                seasonal_multiplier = 0.8
            
            base_price = np.random.uniform(50, 500)
            quantity = np.random.randint(1, 10)
            
            sales_records.append({
                'date': date,
                'region': np.random.choice(regions),
                'product': np.random.choice(products),
                'sales_rep': np.random.choice(sales_reps),
                'quantity': quantity,
                'unit_price': round(base_price * seasonal_multiplier, 2),
                'total_revenue': round(quantity * base_price * seasonal_multiplier, 2),
                'customer_id': f'CUST_{np.random.randint(1000, 9999)}',
                'discount_pct': round(np.random.uniform(0, 0.15), 3),
                'profit_margin': round(np.random.uniform(0.15, 0.45), 3)
            })
    
    sales_df = pd.DataFrame(sales_records)
    sales_df.to_csv(sample_dir / 'sales_performance_2023_2024.csv', index=False)
    
    # 2. Customer Demographics Dataset
    print("   Creating: Customer Demographics Dataset...")
    n_customers = 2000
    
    customer_data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1, n_customers + 1)],
        'age': np.random.normal(40, 15, n_customers).astype(int).clip(18, 80),
        'income': np.random.lognormal(10.5, 0.5, n_customers).astype(int).clip(25000, 200000),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_customers, p=[0.48, 0.48, 0.04]),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_customers, p=[0.3, 0.45, 0.2, 0.05]),
        'city': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'], n_customers),
        'subscription_type': np.random.choice(['Basic', 'Premium', 'Enterprise'], n_customers, p=[0.5, 0.35, 0.15]),
        'signup_date': pd.to_datetime('2020-01-01') + pd.to_timedelta(np.random.randint(0, 1461, n_customers), unit='D'),  # 4 years of signups
        'lifetime_value': np.random.lognormal(6, 1, n_customers).astype(int).clip(100, 50000),
        'satisfaction_score': np.random.normal(4.2, 0.8, n_customers).clip(1, 5).round(1),
        'is_churned': np.random.choice([0, 1], n_customers, p=[0.85, 0.15])
    }
    
    customer_df = pd.DataFrame(customer_data)
    customer_df.to_csv(sample_dir / 'customer_demographics.csv', index=False)
    
    # 3. Financial Performance Dataset
    print("   Creating: Financial Performance Dataset...")
    financial_records = []
    
    quarters = pd.date_range('2020-Q1', '2024-Q4', freq='Q')
    departments = ['Sales', 'Marketing', 'R&D', 'Operations', 'HR', 'Finance']
    
    for quarter in quarters:
        for dept in departments:
            # Base budget with growth
            base_budget = np.random.uniform(500000, 2000000)
            growth_rate = 1.05 if quarter.year > 2020 else 1.0
            
            financial_records.append({
                'quarter': quarter,
                'year': quarter.year,
                'department': dept,
                'budgeted_amount': round(base_budget * growth_rate, 2),
                'actual_spending': round(base_budget * growth_rate * np.random.uniform(0.85, 1.15), 2),
                'revenue_generated': round(base_budget * growth_rate * np.random.uniform(1.2, 2.5) if dept in ['Sales', 'Marketing'] else 0, 2),
                'headcount': np.random.randint(10, 50),
                'projects_completed': np.random.randint(2, 12),
                'efficiency_score': round(np.random.uniform(0.7, 0.95), 3)
            })
    
    financial_df = pd.DataFrame(financial_records)
    financial_df.to_csv(sample_dir / 'financial_performance.csv', index=False)
    
    # 4. Website Analytics Dataset
    print("   Creating: Website Analytics Dataset...")
    analytics_records = []
    
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    pages = ['Home', 'Products', 'About', 'Contact', 'Blog', 'Pricing', 'Features', 'Support']
    traffic_sources = ['Organic', 'Paid Search', 'Social Media', 'Direct', 'Referral', 'Email']
    
    for date in dates:
        for page in pages:
            for source in traffic_sources:
                # Weekend effect
                weekend_multiplier = 0.7 if date.weekday() >= 5 else 1.0
                
                analytics_records.append({
                    'date': date,
                    'page': page,
                    'traffic_source': source,
                    'page_views': int(np.random.poisson(100) * weekend_multiplier),
                    'unique_visitors': int(np.random.poisson(80) * weekend_multiplier),
                    'bounce_rate': round(np.random.uniform(0.2, 0.8), 3),
                    'avg_session_duration': round(np.random.uniform(30, 300), 1),
                    'conversion_rate': round(np.random.uniform(0.01, 0.15), 4) if page in ['Products', 'Pricing'] else round(np.random.uniform(0.001, 0.05), 4)
                })
    
    analytics_df = pd.DataFrame(analytics_records)
    analytics_df.to_csv(sample_dir / 'website_analytics.csv', index=False)
    
    # Create a dataset info file
    dataset_info = {
        "datasets": [
            {
                "name": "Sales Performance 2023-2024",
                "file": "sales_performance_2023_2024.csv",
                "description": "Comprehensive sales data with regional performance, product sales, and revenue metrics",
                "records": len(sales_df),
                "columns": list(sales_df.columns),
                "sample_queries": [
                    "What are our top performing regions by revenue?",
                    "How do sales vary by season?",
                    "Which sales reps are most effective?",
                    "What's the average profit margin by product?"
                ]
            },
            {
                "name": "Customer Demographics", 
                "file": "customer_demographics.csv",
                "description": "Customer profile data including demographics, subscription info, and satisfaction metrics",
                "records": len(customer_df),
                "columns": list(customer_df.columns),
                "sample_queries": [
                    "What's the age distribution of our customers?",
                    "How does income correlate with lifetime value?",
                    "Which customer segments have highest satisfaction?",
                    "What factors predict customer churn?"
                ]
            },
            {
                "name": "Financial Performance",
                "file": "financial_performance.csv", 
                "description": "Quarterly financial data by department with budget vs actual analysis",
                "records": len(financial_df),
                "columns": list(financial_df.columns),
                "sample_queries": [
                    "Which departments exceed their budgets most often?",
                    "How has spending efficiency changed over time?",
                    "What's the ROI for marketing spend?",
                    "Show budget variance trends by year"
                ]
            },
            {
                "name": "Website Analytics",
                "file": "website_analytics.csv",
                "description": "Daily website traffic and performance metrics across pages and traffic sources", 
                "records": len(analytics_df),
                "columns": list(analytics_df.columns),
                "sample_queries": [
                    "Which pages have the highest conversion rates?",
                    "How does traffic vary by source and day of week?",
                    "What's the average bounce rate by page?",
                    "Show seasonal traffic patterns"
                ]
            }
        ],
        "created": datetime.now().isoformat(),
        "total_records": len(sales_df) + len(customer_df) + len(financial_df) + len(analytics_df)
    }
    
    with open(sample_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"✅ Created {len(dataset_info['datasets'])} sample datasets with {dataset_info['total_records']:,} total records")
    return dataset_info

def print_demo_instructions(dataset_info):
    """Print instructions for using the demo."""
    print("\n🎯 DEMO INSTRUCTIONS")
    print("=" * 60)
    print("Your AI Data Analysis system is ready with sample data!")
    print(f"\n📊 Available Datasets ({len(dataset_info['datasets'])} files):")
    
    for i, dataset in enumerate(dataset_info['datasets'], 1):
        print(f"\n{i}. 📈 {dataset['name']}")
        print(f"   📄 File: {dataset['file']}")
        print(f"   📝 {dataset['description']}")
        print(f"   📊 Records: {dataset['records']:,}")
        print("   💡 Try asking:")
        for query in dataset['sample_queries'][:2]:  # Show first 2 sample queries
            print(f"      • {query}")
    
    print(f"\n🚀 Getting Started:")
    print("1. 🌐 The web interface will open at: http://localhost:8503")
    print("2. 📁 Upload any of the sample datasets from data/sample/")
    print("3. 💬 Ask natural language questions about your data")
    print("4. 📊 Create custom visualizations in the 'Custom Charts' tab")
    print("5. 💾 Export results in multiple formats (CSV, Excel, JSON, ZIP)")
    
    print(f"\n🔧 System Features:")
    print("✨ Multi-modal AI query processing (Statistical + Semantic + SQL)")
    print("🧠 ChromaDB vector memory for conversation context")
    print("📊 Interactive chart builder with 7+ chart types") 
    print("💾 Professional export system with multiple formats")
    print("🎨 Custom themes and styling options")
    
    print(f"\n🛑 To stop: Press Ctrl+C in this terminal")

def launch_streamlit():
    """Launch the Streamlit application."""
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = "8503"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    
    # Launch Streamlit app
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "ui/streamlit_app.py",
        "--server.port", "8503",
        "--server.address", "localhost", 
        "--server.headless", "false",
        "--server.runOnSave", "true",
        "--theme.base", "light"
    ]
    
    try:
        print("\n🔄 Launching Streamlit application...")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n✅ Demo session ended successfully!")
    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        return False
    
    return True

def main():
    """Main demo launcher function."""
    print("🚀 Agentic AI Data Analysis - Interactive Demo")
    print("=" * 60)
    print("🎯 This demo creates sample datasets and launches the full system")
    print("🎮 No need to upload your own data - everything is ready to try!")
    print("=" * 60)
    
    # Create sample datasets
    dataset_info = create_sample_datasets()
    
    # Print instructions
    print_demo_instructions(dataset_info)
    
    # Launch the application
    success = launch_streamlit()
    
    if not success:
        print("❌ Demo failed to start. Please check your environment setup.")
        return False
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1) 
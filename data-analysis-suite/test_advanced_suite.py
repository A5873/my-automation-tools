#!/usr/bin/env python3
"""
Test script for the Advanced Data Analysis Suite
"""

from advanced_data_suite import AdvancedDataAnalysisSuite

def main():
    # Initialize the advanced suite
    suite = AdvancedDataAnalysisSuite({
        'output_dir': 'analysis_output',
        'auto_insights': True
    })
    
    # Display banner
    suite.display_banner()
    
    # Load data
    if not suite.load_data('example_data.csv'):
        print("Failed to load data. Exiting.")
        return
    
    # Comprehensive EDA
    print("\n🔍 Starting comprehensive exploratory data analysis...")
    eda_results = suite.comprehensive_eda()
    
    # Create visualizations
    print("\n📊 Creating advanced visualizations...")
    viz_files = suite.create_advanced_visualizations()
    
    # AutoML
    print("\n🤖 Running AutoML for target: weather_type")
    ml_results = suite.automl_model_selection('weather_type')
    
    if 'error' not in ml_results:
        best_model = ml_results['best_model']
        print(f"🏆 Best model: {best_model['name']}")
    
    # Generate comprehensive report
    print("\n📄 Generating insights report...")
    report_path = suite.generate_insights_report()
    
    # Final summary
    print("\n🎉 Analysis completed successfully!")
    print(f"📁 Results saved in: {suite.output_dir}")
    print(f"📄 View report: {report_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluation File Manager
=======================

Utility script to manage evaluation result files in the evaluation folder.
"""

import os
import json
import glob
from datetime import datetime
import argparse

def list_evaluation_files():
    """List all evaluation files in the evaluation folder"""
    evaluation_folder = "evaluation"
    
    if not os.path.exists(evaluation_folder):
        print("❌ Evaluation folder not found!")
        return
    
    files = glob.glob(os.path.join(evaluation_folder, "evaluation_results_*.json"))
    
    if not files:
        print("📁 No evaluation files found in evaluation folder")
        return
    
    print(f"📁 Found {len(files)} evaluation file(s):")
    print("-" * 80)
    
    # Sort by modification time (newest first)
    files.sort(key=os.path.getmtime, reverse=True)
    
    for i, file_path in enumerate(files, 1):
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        
        print(f"{i:2d}. {filename}")
        print(f"    📊 Size: {file_size // 1024} KB")
        print(f"    🕒 Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Try to load and show basic metrics
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "aggregate_metrics" in data:
                metrics = data["aggregate_metrics"]
                total_questions = data.get("total_questions", 0)
                successful = data.get("successful_evaluations", 0)
                
                print(f"    📈 Questions: {successful}/{total_questions}")
                
                if "f1_score" in metrics and "error" not in metrics["f1_score"]:
                    f1_mean = metrics["f1_score"]["mean"]
                    print(f"    🎯 F1 Score: {f1_mean:.3f}")
                
                if "keyword_coverage" in metrics and "error" not in metrics["keyword_coverage"]:
                    keyword_mean = metrics["keyword_coverage"]["mean"]
                    print(f"    🔑 Keyword Coverage: {keyword_mean:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error reading file: {e}")
        
        print()

def show_evaluation_details(filename):
    """Show detailed information about a specific evaluation file"""
    evaluation_folder = "evaluation"
    file_path = os.path.join(evaluation_folder, filename)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"📄 Evaluation Details: {filename}")
        print("=" * 60)
        
        # Basic info
        print(f"📅 Timestamp: {data.get('evaluation_timestamp', 'Unknown')}")
        print(f"📊 Total Questions: {data.get('total_questions', 0)}")
        print(f"✅ Successful: {data.get('successful_evaluations', 0)}")
        print(f"❌ Failed: {data.get('failed_evaluations', 0)}")
        
        # Metrics
        if "aggregate_metrics" in data:
            metrics = data["aggregate_metrics"]
            print("\n📈 Aggregate Metrics:")
            
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict) and "error" not in metric_data:
                    mean_val = metric_data.get("mean", 0)
                    std_val = metric_data.get("std", 0)
                    print(f"  • {metric_name}: {mean_val:.3f} ± {std_val:.3f}")
                elif isinstance(metric_data, dict) and "error" in metric_data:
                    print(f"  • {metric_name}: {metric_data['error']}")
        
        # Sample questions
        if "individual_results" in data:
            results = data["individual_results"]
            print(f"\n📋 Sample Questions ({min(3, len(results))} shown):")
            
            for i, result in enumerate(results[:3]):
                if "error" not in result:
                    question = result.get("question", "Unknown")
                    print(f"  {i+1}. {question[:60]}...")
                    
                    if "f1_metrics" in result:
                        f1_score = result["f1_metrics"].get("f1", 0)
                        print(f"     F1 Score: {f1_score:.3f}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

def cleanup_old_evaluations(days=30):
    """Remove evaluation files older than specified days"""
    evaluation_folder = "evaluation"
    
    if not os.path.exists(evaluation_folder):
        print("❌ Evaluation folder not found!")
        return
    
    files = glob.glob(os.path.join(evaluation_folder, "evaluation_results_*.json"))
    
    if not files:
        print("📁 No evaluation files found")
        return
    
    cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)
    old_files = [f for f in files if os.path.getmtime(f) < cutoff_time]
    
    if not old_files:
        print(f"✅ No files older than {days} days found")
        return
    
    print(f"🗑️  Found {len(old_files)} file(s) older than {days} days:")
    
    for file_path in old_files:
        filename = os.path.basename(file_path)
        mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"  • {filename} (modified: {mod_time.strftime('%Y-%m-%d')})")
    
    response = input(f"\n❓ Delete these {len(old_files)} file(s)? (y/N): ")
    
    if response.lower() == 'y':
        for file_path in old_files:
            try:
                os.remove(file_path)
                print(f"✅ Deleted: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"❌ Error deleting {os.path.basename(file_path)}: {e}")
    else:
        print("❌ Cleanup cancelled")

def main():
    parser = argparse.ArgumentParser(description="Manage evaluation result files")
    parser.add_argument("action", choices=["list", "show", "cleanup"], 
                       help="Action to perform")
    parser.add_argument("--file", "-f", help="Filename for show action")
    parser.add_argument("--days", "-d", type=int, default=30, 
                       help="Days threshold for cleanup (default: 30)")
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_evaluation_files()
    elif args.action == "show":
        if not args.file:
            print("❌ Please specify a filename with --file")
            return
        show_evaluation_details(args.file)
    elif args.action == "cleanup":
        cleanup_old_evaluations(args.days)

if __name__ == "__main__":
    main() 
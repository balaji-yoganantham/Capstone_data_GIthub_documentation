# Evaluation Results Folder

This folder contains all evaluation results from the RAG system performance testing.

## 📁 Contents

- **evaluation_results_*.json**: Evaluation result files with timestamps
- **evaluation_manager.py**: Utility script to manage evaluation files
- **README.md**: This documentation file

## 📊 File Format

Each evaluation result file contains:

```json
{
  "total_questions": 8,
  "successful_evaluations": 8,
  "failed_evaluations": 0,
  "aggregate_metrics": {
    "f1_score": {
      "mean": 0.243,
      "std": 0.033,
      "min": 0.0,
      "max": 0.5
    },
    "rouge1_f": {
      "mean": 0.243,
      "std": 0.033
    },
    "keyword_coverage": {
      "mean": 0.676,
      "std": 0.322
    }
  },
  "individual_results": [...],
  "evaluation_timestamp": "2025-08-01T12:26:26"
}
```

## 🛠️ Management Tools

### Using the Evaluation Manager Script

The `evaluation_manager.py` script provides several useful commands:

#### List all evaluation files:
```bash
python evaluation/evaluation_manager.py list
```

#### Show details of a specific file:
```bash
python evaluation/evaluation_manager.py show --file evaluation_results_20250801_122626.json
```

#### Clean up old files (older than 30 days):
```bash
python evaluation/evaluation_manager.py cleanup
```

#### Clean up files older than specific days:
```bash
python evaluation/evaluation_manager.py cleanup --days 7
```

### Using the Streamlit Interface

1. **View Previous Results**: In the Streamlit app, go to the "🧪 Evaluation" tab
2. **See File List**: Previous evaluation files are automatically listed
3. **Load Results**: Use the dropdown to select and view previous evaluation results
4. **Run New Evaluation**: Click "🚀 Run Evaluation" to perform a new evaluation

## 📈 Metrics Explained

- **F1 Score**: Measures the harmonic mean of precision and recall for text similarity
- **ROUGE-1 F1**: Measures unigram overlap between predicted and ground truth responses
- **Keyword Coverage**: Percentage of expected keywords found in the response
- **Success Rate**: Percentage of questions that were successfully evaluated

## 🔄 Automatic Saving

When you run an evaluation in the Streamlit app:
1. Results are automatically saved to this folder
2. Files are named with timestamps: `evaluation_results_YYYYMMDD_HHMMSS.json`
3. The folder is created automatically if it doesn't exist

## 📋 Best Practices

1. **Regular Cleanup**: Use the cleanup command to remove old evaluation files
2. **Backup Important Results**: Copy important evaluation files to a backup location
3. **Compare Results**: Use the Streamlit interface to compare different evaluation runs
4. **Track Performance**: Monitor metrics over time to track system improvements

## 🚀 Quick Start

1. Run the Streamlit app: `streamlit run app.py`
2. Go to the "🧪 Evaluation" tab
3. Click "🚀 Run Evaluation" to perform a new evaluation
4. View results and compare with previous evaluations
5. Use the evaluation manager script for file management

## 📞 Support

For issues or questions about evaluation results:
- Check the evaluation manager script help: `python evaluation/evaluation_manager.py --help`
- Review the main application logs for evaluation errors
- Ensure the evaluation folder has proper write permissions 
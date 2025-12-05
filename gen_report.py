import os
import csv
import pandas as pd

OUTPUT_DIR = "evaluation/evaluation_results_1500_pairs"
REPORT_FILE = os.path.join(OUTPUT_DIR, "evaluation_report.html")

MODELS_SHORT = {
    "cvlface_adaface_ir101_webface4m": "AdaFace IR101 WebFace4M",
    "cvlface_arcface_ir101_webface4m": "ArcFace IR101 WebFace4M",
    "cvlface_adaface_ir101_webface12m": "AdaFace IR101 WebFace12M",
    # "cvlface_adaface_vit_base_kprpe_webface4m": "AdaFace ViT-Base KPRPE WebFace4M",
    "cvlface_adaface_vit_base_webface4m": "AdaFace ViT-Base WebFace4M",
}

def generate_html_report():
    """Generate a comprehensive HTML report from all results"""
    html_content = """
    <html>
    <head>
        <title>Face Recognition Model Evaluation Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            h2 {
                color: #555;
                border-bottom: 2px solid #ddd;
                padding-bottom: 10px;
                margin-top: 30px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: white;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            th {
                background-color: #4CAF50;
                color: white;
                padding: 12px;
                text-align: left;
                font-weight: bold;
            }
            td {
                padding: 10px 12px;
                border-bottom: 1px solid #ddd;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            .metric {
                font-weight: bold;
                color: #2196F3;
            }
            .accuracy {
                color: #4CAF50;
                font-weight: bold;
            }
            .summary {
                background-color: #e8f5e9;
                padding: 15px;
                border-left: 4px solid #4CAF50;
                margin: 15px 0;
            }
        </style>
    </head>
    <body>
        <h1>Face Recognition Model Evaluation Report</h1>
        <div class="summary">
            <strong>Dataset:</strong> 1500 pairs (500 same-ID, 1,000 different-ID)<br>
            <strong>Models:</strong> 4 pretrained models from CVLface<br>
            <strong>Thresholds:</strong> [0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.4]<br>
            <strong>Metric:</strong> Cosine similarity on L2-normalized embeddings
        </div>
    """
    
    # Read and display results for each model
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if not filename.startswith("results_") or not filename.endswith(".csv"):
            continue
        
        model_key = filename.replace("results_", "").replace(".csv", "")
        model_name = MODELS_SHORT.get(model_key, model_key)
        
        df = pd.read_csv(os.path.join(OUTPUT_DIR, filename))
        
        html_content += f"\n        <h2>{model_name}</h2>\n"
        html_content += "        <table>\n"
        html_content += "            <tr>\n"
        html_content += "                <th>Threshold</th>\n"
        html_content += "                <th>Avg Time (ms)</th>\n"
        html_content += "                <th>Positive Right (out of 500)</th>\n"
        html_content += "                <th>Positive Wrong</th>\n"
        html_content += "                <th>Positive Accuracy (%)</th>\n"
        html_content += "                <th>Negative Right (out of 1,000)</th>\n"
        html_content += "                <th>Negative Wrong</th>\n"
        html_content += "                <th>Negative Accuracy (%)</th>\n"
        html_content += "                <th>Overall Accuracy (%)</th>\n"
        html_content += "            </tr>\n"
        
        for _, row in df.iterrows():
            threshold = row["threshold"]
            avg_time = row["avg_time_ms"]
            pos_right = row["positive_right"]
            pos_wrong = row["positive_wrong"]
            neg_right = row["negative_right"]
            neg_wrong = row["negative_wrong"]
            
            pos_acc = (pos_right / 500) * 100
            neg_acc = (neg_right / 1000) * 100
            overall_acc = ((pos_right + neg_right) / 1500) * 100
            
            html_content += f"            <tr>\n"
            html_content += f"                <td class='metric'>{threshold:.2f}</td>\n"
            html_content += f"                <td>{avg_time:.2f}</td>\n"
            html_content += f"                <td>{pos_right:.0f}</td>\n"
            html_content += f"                <td>{pos_wrong:.0f}</td>\n"
            html_content += f"                <td class='accuracy'>{pos_acc:.2f}%</td>\n"
            html_content += f"                <td>{neg_right:.0f}</td>\n"
            html_content += f"                <td>{neg_wrong:.0f}</td>\n"
            html_content += f"                <td class='accuracy'>{neg_acc:.2f}%</td>\n"
            html_content += f"                <td class='accuracy'><strong>{overall_acc:.2f}%</strong></td>\n"
            html_content += f"            </tr>\n"
        
        html_content += "        </table>\n"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(REPORT_FILE, "w") as f:
        f.write(html_content)
    
    print(f"HTML report saved to {REPORT_FILE}")

def generate_console_report():
    """Print a summary table to console"""
    print("\n" + "="*100)
    print("EVALUATION REPORT SUMMARY")
    print("="*100 + "\n")
    
    for filename in sorted(os.listdir(OUTPUT_DIR)):
        if not filename.startswith("results_") or not filename.endswith(".csv"):
            continue
        
        model_key = filename.replace("results_", "").replace(".csv", "")
        model_name = MODELS_SHORT.get(model_key, model_key)
        
        print(f"\n{model_name}")
        print("-" * 100)
        
        df = pd.read_csv(os.path.join(OUTPUT_DIR, filename))
        
        print(f"{'Threshold':>10} {'Avg Time':>12} {'Pos Right':>12} {'Pos Acc':>12} {'Neg Right':>12} {'Neg Acc':>12} {'Overall':>12}")
        print("-" * 100)
        
        for _, row in df.iterrows():
            threshold = row["threshold"]
            avg_time = row["avg_time_ms"]
            pos_right = row["positive_right"]
            neg_right = row["negative_right"]
            
            pos_acc = (pos_right / 500) * 100
            neg_acc = (neg_right / 1000) * 100
            overall_acc = ((pos_right + neg_right) / 1500) * 100
            
            print(f"{threshold:>10.2f} {avg_time:>12.2f}ms {pos_right:>12}/500 {pos_acc:>11.2f}% {neg_right:>12}/1000 {neg_acc:>11.2f}% {overall_acc:>11.2f}%")

def main():
    if not os.path.exists(OUTPUT_DIR):
        print(f"No results found in {OUTPUT_DIR}")
        return
    
    # Generate console report
    generate_console_report()
    
    # Generate HTML report
    generate_html_report()
    
    print("\n" + "="*100)
    print(f"Report saved to {REPORT_FILE}")
    print("="*100 + "\n")

if __name__ == "__main__":
    main()
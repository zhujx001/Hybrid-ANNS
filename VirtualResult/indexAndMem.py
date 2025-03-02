import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Path configuration
SEARCH_MEM_PATH = '/data/searchMem'
INDEX_DATA_PATH = '/data/indexdata'
OUTPUT_PATH = './visualizations'  # Output directory

# Ensure output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Dataset list
datasets = ['audio', 'enron', 'gist', 'glove-100', 'msong', 'sift']

# Color configuration
colors = ['#1e40af', '#047857', '#b91c1c', '#7c3aed', '#f59e0b', '#10b981', '#6366f1', '#ef4444']

def get_algorithms():
    """Get all algorithm names"""
    algorithms = []
    # Search all subdirectories
    if os.path.exists(SEARCH_MEM_PATH):
        for algo in os.listdir(SEARCH_MEM_PATH):
            algo_path = os.path.join(SEARCH_MEM_PATH, algo)
            if os.path.isdir(algo_path) and os.path.exists(os.path.join(algo_path, 'mem.csv')):
                algorithms.append(algo)
    return algorithms

def load_search_memory_data(algorithms):
    """Load search memory data"""
    memory_data = {}
    
    for algo in algorithms:
        mem_file = os.path.join(SEARCH_MEM_PATH, algo, 'mem.csv')
        if os.path.exists(mem_file):
            try:
                # Read CSV file
                df = pd.read_csv(mem_file)
                # Convert data to dictionary format
                memory_data[algo] = {}
                for _, row in df.iterrows():
                    dataset = row['Dataset']
                    memory = row['Memory(MB)']
                    memory_data[algo][dataset] = memory
            except Exception as e:
                print(f"Error reading memory data for {algo}: {e}")
    
    return memory_data

def load_index_data(algorithms, datasets):
    """Load index building data"""
    index_data = {}
    
    for algo in algorithms:
        index_data[algo] = {}
        for dataset in datasets:
            index_file = os.path.join(INDEX_DATA_PATH, algo, f'{dataset}.csv')
            if os.path.exists(index_file):
                try:
                    # Read CSV file
                    df = pd.read_csv(index_file)
                    if not df.empty:
                        # Convert first row to dictionary
                        index_data[algo][dataset] = df.iloc[0].to_dict()
                except Exception as e:
                    print(f"Error reading index data for {algo}/{dataset}: {e}")
    
    return index_data

def visualize_search_memory_by_dataset(memory_data, algorithms):
    """Create search memory usage comparison charts for each dataset"""
    for dataset in datasets:
        # Filter algorithms with data for this dataset
        valid_algos = [algo for algo in algorithms if algo in memory_data and dataset in memory_data[algo]]
        if not valid_algos:
            continue
            
        # Prepare data
        algo_memories = [(algo, memory_data[algo][dataset]) for algo in valid_algos]
        algo_memories.sort(key=lambda x: x[1])  # Sort by memory usage
        
        algos, memories = zip(*algo_memories)
        
        # Create chart
        plt.figure(figsize=(12, 6))
        plt.bar(algos, memories, color=colors[:len(algos)])
        plt.title(f'Search Memory Usage Comparison - {dataset} Dataset', fontsize=14)
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel('Memory Usage (MB)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels
        for i, v in enumerate(memories):
            plt.text(i, v + max(memories) * 0.01, f'{v:.2f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f'search_memory_{dataset}.png'), dpi=300)
        plt.close()

def visualize_index_metric_by_dataset(index_data, algorithms, metric):
    """Create index building metric comparison charts for each dataset"""
    metric_labels = {
        'build_time': 'Build Time (seconds)',
        'memory_mb': 'Peak Memory (MB)',
        'index_size_mb': 'Index Size (MB)'
    }
    
    for dataset in datasets:
        # Filter algorithms with metric data for this dataset
        valid_algos = [algo for algo in algorithms 
                      if algo in index_data 
                      and dataset in index_data[algo] 
                      and metric in index_data[algo][dataset]]
        
        if not valid_algos:
            continue
            
        # Prepare data
        algo_metrics = [(algo, index_data[algo][dataset][metric]) for algo in valid_algos]
        algo_metrics.sort(key=lambda x: x[1])  # Sort by metric value
        
        algos, metrics = zip(*algo_metrics)
        
        # Create chart
        plt.figure(figsize=(12, 6))
        plt.bar(algos, metrics, color=colors[:len(algos)])
        plt.title(f'{dataset} Dataset Index {metric_labels[metric]} Comparison', fontsize=14)
        plt.xlabel('Algorithm', fontsize=12)
        plt.ylabel(metric_labels[metric], fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels
        for i, v in enumerate(metrics):
            plt.text(i, v + max(metrics) * 0.01, f'{v:.2f}', ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, f'index_{metric}_{dataset}.png'), dpi=300)
        plt.close()

def visualize_metric_across_datasets(index_data, algorithms, metric):
    """Create metric comparison charts across datasets"""
    metric_labels = {
        'build_time': 'Build Time (seconds)',
        'memory_mb': 'Peak Memory (MB)',
        'index_size_mb': 'Index Size (MB)'
    }
    
    # Extract metric values for each algorithm across datasets
    data_for_plot = []
    
    for dataset in datasets:
        dataset_data = {'Dataset': dataset}
        for algo in algorithms:
            if (algo in index_data and 
                dataset in index_data[algo] and 
                metric in index_data[algo][dataset]):
                dataset_data[algo] = index_data[algo][dataset][metric]
            else:
                dataset_data[algo] = 0
        data_for_plot.append(dataset_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_for_plot)
    
    # Use dataset as index
    df.set_index('Dataset', inplace=True)
    
    # Create chart
    plt.figure(figsize=(14, 8))
    
    # Create a group of bars for each algorithm
    bar_width = 0.8 / len(algorithms)
    x = np.arange(len(datasets))
    
    for i, algo in enumerate(algorithms):
        if algo in df.columns:
            plt.bar(x + i * bar_width - 0.4 + bar_width/2, 
                   df[algo], 
                   width=bar_width, 
                   label=algo, 
                   color=colors[i % len(colors)])
    
    plt.title(f'{metric_labels[metric]} Comparison Across Datasets', fontsize=14)
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel(metric_labels[metric], fontsize=12)
    plt.xticks(x, datasets, rotation=45)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'comparison_{metric}_across_datasets.png'), dpi=300)
    plt.close()

def create_heatmap(memory_data, algorithms):
    """Create search memory usage heatmap"""
    # Prepare heatmap data
    heatmap_data = []
    
    for dataset in datasets:
        row_data = {'Dataset': dataset}
        for algo in algorithms:
            if algo in memory_data and dataset in memory_data[algo]:
                row_data[algo] = memory_data[algo][dataset]
            else:
                row_data[algo] = np.nan
        heatmap_data.append(row_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(heatmap_data)
    df.set_index('Dataset', inplace=True)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, cmap="YlGnBu")
    plt.title('Search Memory Usage by Algorithm and Dataset (MB)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'search_memory_heatmap.png'), dpi=300)
    plt.close()

def main():
    # Get algorithm list
    algorithms = get_algorithms()
    if not algorithms:
        print("No algorithm data found, please check the path")
        return
    
    print(f"Discovered algorithms: {algorithms}")
    
    # Load search memory data
    memory_data = load_search_memory_data(algorithms)
    
    # Load index building data
    index_data = load_index_data(algorithms, datasets)
    
    # Create search memory usage comparison charts for each dataset
    visualize_search_memory_by_dataset(memory_data, algorithms)
    
    # Create index building metric comparison charts
    metrics = ['build_time', 'memory_mb', 'index_size_mb']
    for metric in metrics:
        visualize_index_metric_by_dataset(index_data, algorithms, metric)
        
        # Create metric comparison charts across datasets
        visualize_metric_across_datasets(index_data, algorithms, metric)
    
    # Create search memory usage heatmap
    create_heatmap(memory_data, algorithms)
    
    print(f"Visualization charts saved to {os.path.abspath(OUTPUT_PATH)}")

if __name__ == "__main__":
    main()

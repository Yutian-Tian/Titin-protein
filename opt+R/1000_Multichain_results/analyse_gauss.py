import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_txt_data(filepath):
    """Read data from .txt file"""
    try:
        with open(filepath, 'r') as f:
            all_data = []
            for line in f:
                if line.strip():
                    numbers = [float(x) for x in line.strip().split()]
                    all_data.extend(numbers)
        
        data = np.array(all_data)
        print(f"✓ Read {len(data)} data points")
        return data
        
    except Exception as e:
        print(f"✗ Failed to read file: {e}")
        return None

def analyze_gaussian(data):
    """Analyze if data follows Gaussian distribution"""
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    
    # KS test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(mean, std))
    is_gaussian = ks_p > 0.05
    
    return {
        'n': n, 'mean': mean, 'std': std,
        'ks_p': ks_p, 'is_gaussian': is_gaussian
    }

def plot_distribution(data, result, filename="gaussian_analysis.png"):
    """Plot distribution and save to current directory"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Histogram + Normal curve
    ax1.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    x = np.linspace(data.min(), data.max(), 100)
    p = stats.norm.pdf(x, result['mean'], result['std'])
    ax1.plot(x, p, 'r-', linewidth=2)
    ax1.set_title('Histogram with Normal Fit')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot
    stats.probplot(data, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot
    ax3.boxplot(data, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightgreen'),
                medianprops=dict(color='red', linewidth=2))
    ax3.set_title('Box Plot')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    
    # 4. Density comparison
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 200)
    ax4.plot(x_range, kde(x_range), 'b-', linewidth=2, label='Empirical')
    ax4.plot(x_range, stats.norm.pdf(x_range, result['mean'], result['std']), 
             'r--', linewidth=2, label='Theoretical Normal')
    ax4.set_title('Probability Density Comparison')
    ax4.set_xlabel('Value')
    ax4.set_ylabel('Probability Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add statistics information
    stats_text = f"""Statistics:
Samples: {result['n']:,}
Mean: {result['mean']:.4f}
Std Dev: {result['std']:.4f}
KS test p-value: {result['ks_p']:.4f}
Gaussian: {'Yes' if result['is_gaussian'] else 'No'}"""
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    
    plt.suptitle(f"Gaussian Distribution Analysis", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to current directory
    plt.savefig(f'/home/tyt/project/Single-chain/opt+R/1000_Multichain_results/gaussian_analysis_result.png', dpi=150, bbox_inches='tight')
    print(f"✓ Image saved: {filename}")
    plt.show()

def main():
    """Main function"""
    # File path (modify this to your file path)
    filepath = "/home/tyt/project/Single-chain/opt+R/1000_Multichain_results/energy.txt"
    
    print("="*50)
    print("Gaussian Distribution Analysis Tool")
    print("="*50)
    
    # 1. Read data
    data = read_txt_data(filepath)
    if data is None:
        return
    
    # 2. Analyze
    result = analyze_gaussian(data)
    
    # 3. Output results
    print(f"\nAnalysis Results:")
    print(f"  Samples: {result['n']:,}")
    print(f"  Mean: {result['mean']:.4f}")
    print(f"  Std Dev: {result['std']:.4f}")
    print(f"  KS test p-value: {result['ks_p']:.4f}")
    print(f"  Gaussian: {'Yes' if result['is_gaussian'] else 'No'}")
    
    # 4. Plot and save image
    output_filename = "gaussian_analysis_result.png"
    plot_distribution(data, result, output_filename)
    
    print("\n✓ Analysis completed!")

if __name__ == "__main__":    
    main()
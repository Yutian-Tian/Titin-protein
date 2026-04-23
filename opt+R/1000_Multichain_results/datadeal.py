import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    f_file_path = '/home/tyt/project/Single-chain/opt+R/1000_Multichain_results/f_multichain.csv'
    n_file_path = '/home/tyt/project/Single-chain/opt+R/1000_Multichain_results/n_multichain.csv'
    
    # 2. Read CSV file
    df = pd.read_csv(f_file_path)
    dn = pd.read_csv(n_file_path)
    
    # 3. Calculate row average f_a
    f_a = df.iloc[:, :].mean(axis=1)
    n_a = dn.iloc[:, :].mean(axis=1)
    
    # 4. Create R values [0, 400]
    L = 350.0
    R = np.linspace(0, L, len(f_a))
    
    # 5. Plot with English labels
    plt.figure(figsize=(10, 6))
    plt.plot(R[:800], f_a[:800], 'b-', linewidth=2)
    plt.xlabel('R', fontsize=12)
    plt.ylabel('$<f>$ (Row Average)', fontsize=12)
    plt.title('$<f>$ vs R Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 6. Save and show plot
    f_output_file = f_file_path.replace('.csv', '_plot.png')
    plt.savefig(f_output_file, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as: {f_output_file}")
    print(f"Processed {len(f_a)} data points")

    # 5. Plot with English labels
    plt.figure(figsize=(10, 6))
    plt.plot(R[:800], n_a[:800], 'b-', linewidth=2)
    plt.xlabel('R', fontsize=12)
    plt.ylabel('$<n_{u}>$ (Row Average)', fontsize=12)
    plt.title('$<n_{u}>$ vs R Curve', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 6. Save and show plot
    n_output_file = n_file_path.replace('.csv', '_plot.png')
    plt.savefig(n_output_file, dpi=300, bbox_inches='tight')
    
    print(f"Plot saved as: {n_output_file}")
    print(f"Processed {len(n_a)} data points")

if __name__ == "__main__":
    main()
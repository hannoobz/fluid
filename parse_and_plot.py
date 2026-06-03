#!/usr/bin/env python3
import os
import re
import sys

def parse_log_file(filepath):
    if not os.path.exists(filepath):
        return None
        
    pattern = re.compile(
        r'\[Timing\s+(?:CPU|GPU)\]\s+Frame\s+\d+\s*\|\s*'
        r'T1:\s*([\d\.]+)\s*\|\s*'
        r'T2:\s*([\d\.]+)\s*\|\s*'
        r'T3:\s*([\d\.]+)\s*\|\s*'
        r'T4:\s*([\d\.]+)\s*\|\s*'
        r'T5:\s*([\d\.]+)\s*\|\s*'
        r'T6:\s*([\d\.]+)\s*\(iters:\s*(\d+)\)\s*\|\s*'
        r'T7:\s*([\d\.]+)\s*\|\s*'
        r'T8:\s*([\d\.]+)\s*\|\s*'
        r'T9:\s*([\d\.]+)\s*\|\s*'
        r'T10:\s*([\d\.]+)\s*\|\s*'
        r'T_total:\s*([\d\.]+)\s*ms'
    )
    
    with open(filepath, 'r') as f:
        content = f.read()
        
    matches = pattern.findall(content)
    if not matches:
        return None
        
    # Discard first timing line as warmup
    if len(matches) > 1:
        matches = matches[1:]
        
    num_entries = len(matches)
    sums = [0.0] * 11
    iters_sum = 0
    
    for match in matches:
        for idx in range(6):
            sums[idx] += float(match[idx])
        iters_sum += int(match[6])
        for idx in range(7, 12):
            sums[idx-1] += float(match[idx])
            
    averages = [s / num_entries for s in sums]
    avg_iters = iters_sum / num_entries
    
    keys = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10', 'T_total']
    result = {keys[i]: averages[i] for i in range(len(keys))}
    result['iters'] = int(round(avg_iters))
    result['raw_count'] = num_entries
    return result

def main():
    resolutions = [50, 100, 150, 200]
    cpu_data = {}
    gpu_data = {}
    
    # Parse CPU Logs
    for res in resolutions:
        path = f'benchmark_cpu_{res}.log'
        stats = parse_log_file(path)
        if stats:
            cpu_data[res] = stats
            
    # Parse GPU Logs
    for res in resolutions:
        path = f'benchmark_gpu_{res}.log'
        stats = parse_log_file(path)
        if stats:
            gpu_data[res] = stats
            
    # Print markdown table
    print("# === HASIL EKSTRAKSI BENCHMARK ===")
    print("\n## Tabel Waktu CPU (ms per frame)")
    print("| Resolusi | T1 (Integrate) | T2 (PushApart) | T3 (Collide) | T4 (P2G) | T5 (Density) | T6 (Pressure) | T7 (G2P) | T8 (Colors) | T9 (Render) | T10 (D2H) | T_total | Iters |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for res in resolutions:
        if res in cpu_data:
            d = cpu_data[res]
            print(f"| **{res}** | {d['T1']:.2f} | {d['T2']:.2f} | {d['T3']:.2f} | {d['T4']:.2f} | {d['T5']:.2f} | {d['T6']:.2f} | {d['T7']:.2f} | {d['T8']:.2f} | {d['T9']:.2f} | {d['T10']:.2f} | {d['T_total']:.2f} | {d['iters']} |")
        else:
            print(f"| **{res}** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            
    print("\n## Tabel Waktu GPU/CUDA (ms per frame)")
    print("| Resolusi | T1 (Integrate) | T2 (PushApart) | T3 (Collide) | T4 (P2G) | T5 (Density) | T6 (Pressure) | T7 (G2P) | T8 (Colors) | T9 (Render) | T10 (D2H) | T_total | Iters |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for res in resolutions:
        if res in gpu_data:
            d = gpu_data[res]
            print(f"| **{res}** | {d['T1']:.2f} | {d['T2']:.2f} | {d['T3']:.2f} | {d['T4']:.2f} | {d['T5']:.2f} | {d['T6']:.2f} | {d['T7']:.2f} | {d['T8']:.2f} | {d['T9']:.2f} | {d['T10']:.2f} | {d['T_total']:.2f} | {d['iters']} |")
        else:
            print(f"| **{res}** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")

    print("\n## Perbandingan Speedup (CPU Time / GPU Time)")
    print("| Resolusi | T_total CPU (ms) | T_total GPU (ms) | Speedup Total | Speedup T2 (Push) | Speedup T6 (Pressure) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for res in resolutions:
        if res in cpu_data and res in gpu_data:
            c = cpu_data[res]
            g = gpu_data[res]
            sp_total = c['T_total'] / g['T_total']
            sp_t2 = c['T2'] / g['T2'] if g['T2'] > 0 else 0
            sp_t6 = c['T6'] / g['T6'] if g['T6'] > 0 else 0
            print(f"| **{res}** | {c['T_total']:.2f} | {g['T_total']:.2f} | **{sp_total:.2f}x** | {sp_t2:.2f}x | {sp_t6:.2f}x |")
        else:
            print(f"| **{res}** | N/A | N/A | N/A | N/A | N/A |")

    # Attempt to plot with matplotlib
    try:
        import matplotlib.pyplot as plt
        
        valid_res = [r for r in resolutions if r in cpu_data and r in gpu_data]
        if not valid_res:
            print("\n[Peringatan] Data CPU dan GPU kurang lengkap untuk membuat plot.")
            return
            
        cpu_totals = [cpu_data[r]['T_total'] for r in valid_res]
        gpu_totals = [gpu_data[r]['T_total'] for r in valid_res]
        speedups = [cpu_data[r]['T_total'] / gpu_data[r]['T_total'] for r in valid_res]
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Total Frame Time (Log Scale)
        plt.subplot(1, 2, 1)
        plt.plot(valid_res, cpu_totals, 'o-', color='crimson', label='CPU Baseline (Gauss-Seidel)')
        plt.plot(valid_res, gpu_totals, 's-', color='dodgerblue', label='GPU CUDA (Jacobi)')
        plt.xlabel('Resolusi Grid (N)')
        plt.ylabel('Rata-rata Waktu Frame (ms) - Log Scale')
        plt.yscale('log')
        plt.xticks(resolutions)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.title('Skalabilitas Resolusi vs Waktu Frame')
        plt.legend()
        
        # Plot 2: Speedup
        plt.subplot(1, 2, 2)
        plt.bar([str(r) for r in valid_res], speedups, color='seagreen', alpha=0.8, width=0.4)
        plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Batas Break-even (1.0x)')
        plt.xlabel('Resolusi Grid (N)')
        plt.ylabel('Faktor Speedup (CPU / GPU)')
        plt.title('Faktor Percepatan (Speedup) GPU vs CPU')
        plt.grid(axis='y', ls="--", alpha=0.5)
        plt.legend()
        
        plt.tight_layout()
        chart_path = 'benchmark_comparison.png'
        plt.savefig(chart_path, dpi=200)
        print(f"\n[Sukses] Grafik komparasi performa berhasil disimpan ke: {os.path.abspath(chart_path)}")
        
    except ImportError:
        print("\n[Info] Library 'matplotlib' belum terinstall. Grafik tidak dapat digambar secara otomatis.")
        print("Silakan install matplotlib terlebih dahulu dengan menjalankan:")
        print("    pip3 install matplotlib")
        print("Kemudian jalankan kembali script ini: python3 parse_and_plot.py")

if __name__ == '__main__':
    main()

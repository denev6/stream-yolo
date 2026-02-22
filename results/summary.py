import csv
import statistics

def analyze_and_print_metrics(file_path):
    delay_ms = []
    fps = []
    
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            delay_ms.append(float(row['delay_ms']))
            fps.append(float(row['fps']))
            
    avg_latency = statistics.mean(delay_ms)
    avg_fps = statistics.mean(fps)
    
    # 100분위수 계산 (Python 3.8 이상 필요)
    quantiles = statistics.quantiles(delay_ms, n=100)
    p95_latency = quantiles[94]
    p99_latency = quantiles[98]
    
    print(f"=== {file_path} 분석 결과 ===")
    print(f"평균 Latency: {avg_latency:.2f} ms")
    print(f"평균 FPS: {avg_fps:.2f}")
    print(f"P95 Latency: {p95_latency:.2f} ms")
    print(f"P99 Latency: {p99_latency:.2f} ms\n")

if __name__ == "__main__":
    analyze_and_print_metrics('go_result.csv')
    analyze_and_print_metrics('py_result.csv')
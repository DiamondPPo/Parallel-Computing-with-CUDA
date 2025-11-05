# CUDA Parallel Computing - Cryptocurrency Risk Analysis

## ไฟล์ที่ใช้งาน

- `sequential.c` - Sequential implementation (CPU)
- `parallel.cu` - Parallel implementation (GPU + CPU hybrid)
- `crypto_prices.csv` - Dataset (390 cryptocurrencies, 465,146 data points)

## การคอมไพล์

### Sequential Version
```bash
gcc -o sequential sequential.c
```

### Parallel Version
```bash
nvcc -o parallel parallel.cu
```

## การรันโปรแกรม

### รัน Sequential Version
```bash
./sequential crypto_prices.csv
```

### รัน Parallel Version
```bash
./parallel crypto_prices.csv
```


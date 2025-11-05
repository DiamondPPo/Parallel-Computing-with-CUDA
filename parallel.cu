#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <time.h>

double get_time() {
    return (double)clock() / CLOCKS_PER_SEC;
}

#define BLOCK_SIZE 256
#define TRADING_DAYS_PER_YEAR 365.0f
#define MAX_LINE 1024

typedef struct {
    char symbol[16];
    float *prices;
    int num_days;
} Stock;

typedef struct {
    char symbol[16];
    float mean;
    float std_dev;
    float volatility;
    float var_95;
    float var_99;
} RiskMetrics;

int load_csv_data(const char *filename, Stock **stocks, int *num_stocks);
void parallel_merge_sort_batch(float *d_data, float *d_temp, int *d_num_returns, int max_days, int num_stocks);
float calculate_percentile(float *sorted_returns, int n, float percentile);

__global__ void batch_merge_sort_kernel(float *data, float *temp, int *num_returns_per_stock, int max_days);

__global__ void calculate_kernel(float *prices, int *days_per_stock, float *means,
                                float *variances, float *all_returns, int max_days) {
    int stock_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_days = days_per_stock[stock_idx];
    int num_returns = num_days - 1;

    if (num_returns <= 0) {
        if (tid == 0) {
            means[stock_idx] = 0.0f;
            variances[stock_idx] = 0.0f;
        }
        return;
    }

    __shared__ float s_sum[BLOCK_SIZE];
    __shared__ float s_sum_sq[BLOCK_SIZE];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    for (int i = tid; i < num_returns; i += blockDim.x) {
        float price_curr = prices[stock_idx * max_days + i + 1];
        float price_prev = prices[stock_idx * max_days + i];
        float return_val = (price_curr - price_prev) / price_prev;

        all_returns[stock_idx * max_days + i] = return_val;
        local_sum += return_val;
        local_sum_sq += return_val * return_val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float sum = s_sum[0];
        float sum_sq = s_sum_sq[0];
        float n = (float)num_returns;

        float mean = sum / n;
        float variance = (sum_sq - sum * sum / n) / (n - 1.0f);

        means[stock_idx] = mean;
        variances[stock_idx] = variance;
    }
}


__global__ void batch_merge_sort_kernel(float *data, float *temp, int *num_returns_per_stock, int max_days) {
    // Kernel initialization
    int stock_idx = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    int num_returns = num_returns_per_stock[stock_idx];
    if (num_returns <= 1) return;

    // Memory pointers for this crypto
    float *input_data = data + stock_idx * max_days;
    float *output_data = temp + stock_idx * max_days;

    // Iterative merge sort
    for (int width = 1; width < num_returns; width *= 2) {
        // Calculate number of merges needed
        int num_merges = (num_returns + width * 2 - 1) / (width * 2);

        // Distribute work to threads
        for (int merge_idx = tid; merge_idx < num_merges; merge_idx += block_size) {
            // Calculate merge boundaries
            int left = merge_idx * width * 2;
            if (left >= num_returns) break;

            int mid = min(left + width, num_returns);
            int right = min(left + width * 2, num_returns);

            // Perform parallel merge
            if (mid < right) {
                int i = left, j = mid, k = left;

                while (i < mid && j < right) {
                    if (input_data[i] <= input_data[j]) {
                        output_data[k++] = input_data[i++];
                    } else {
                        output_data[k++] = input_data[j++];
                    }
                }

                while (i < mid) {
                    output_data[k++] = input_data[i++];
                }

                while (j < right) {
                    output_data[k++] = input_data[j++];
                }
            }
        }

        // Synchronize all threads
        __syncthreads();

        // Buffer swapping for next iteration
        if (width < num_returns / 2) {
            for (int idx = tid; idx < num_returns; idx += block_size) {
                input_data[idx] = output_data[idx];
            }
            __syncthreads();
        }
    }

    // Final copy to original data array
    if ((num_returns & (num_returns - 1)) != 0) {
        for (int idx = tid; idx < num_returns; idx += block_size) {
            data[stock_idx * max_days + idx] = output_data[idx];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_csv>\n", argv[0]);
        return 1;
    }

    printf("Parallel:\n");

    Stock *stocks = NULL;
    int num_stocks = 0;


    if (load_csv_data(argv[1], &stocks, &num_stocks) != 0) {
        printf("Failed to load data\n");
        return 1;
    }

    int max_days = 0;
    int total_days = 0;
    for (int i = 0; i < num_stocks; i++) {
        if (stocks[i].num_days > max_days) {
            max_days = stocks[i].num_days;
        }
        total_days += stocks[i].num_days;
    }

    printf("Dataset: %d stocks, %d total days\n",
           num_stocks, total_days);

    float *h_prices = (float*)malloc(num_stocks * max_days * sizeof(float));
    int *h_days_per_stock = (int*)malloc(num_stocks * sizeof(int));
    float *h_means = (float*)malloc(num_stocks * sizeof(float));
    float *h_variances = (float*)malloc(num_stocks * sizeof(float));

    for (int i = 0; i < num_stocks; i++) {
        h_days_per_stock[i] = stocks[i].num_days;
        for (int j = 0; j < stocks[i].num_days; j++) {
            h_prices[i * max_days + j] = stocks[i].prices[j];
        }
        for (int j = stocks[i].num_days; j < max_days; j++) {
            h_prices[i * max_days + j] = 0.0f;
        }
    }

    float *d_prices, *d_all_returns;
    int *d_days_per_stock;
    float *d_means, *d_variances;
    float *d_temp_sort;

    cudaMalloc(&d_prices, num_stocks * max_days * sizeof(float));
    cudaMalloc(&d_all_returns, num_stocks * max_days * sizeof(float));
    cudaMalloc(&d_days_per_stock, num_stocks * sizeof(int));
    cudaMalloc(&d_means, num_stocks * sizeof(float));
    cudaMalloc(&d_variances, num_stocks * sizeof(float));
    cudaMalloc(&d_temp_sort, num_stocks * max_days * sizeof(float));

    double start = get_time();

    cudaMemcpy(d_prices, h_prices, num_stocks * max_days * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_days_per_stock, h_days_per_stock, num_stocks * sizeof(int), cudaMemcpyHostToDevice);

    dim3 grid(num_stocks);
    dim3 block(BLOCK_SIZE);

    calculate_kernel<<<grid, block>>>(d_prices, d_days_per_stock, d_means, d_variances, d_all_returns, max_days);
    cudaDeviceSynchronize();

    parallel_merge_sort_batch(d_all_returns, d_temp_sort, d_days_per_stock, max_days, num_stocks);
    cudaDeviceSynchronize();

    float *h_all_returns = (float*)malloc(num_stocks * max_days * sizeof(float));
    float *h_var_95 = (float*)malloc(num_stocks * sizeof(float));
    float *h_var_99 = (float*)malloc(num_stocks * sizeof(float));

    cudaMemcpy(h_all_returns, d_all_returns, num_stocks * max_days * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_means, d_means, num_stocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variances, d_variances, num_stocks * sizeof(float), cudaMemcpyDeviceToHost);

    for (int stock_idx = 0; stock_idx < num_stocks; stock_idx++) {
        int num_returns = h_days_per_stock[stock_idx] - 1;

        if (num_returns <= 0) {
            h_var_95[stock_idx] = 0.0f;
            h_var_99[stock_idx] = 0.0f;
            continue;
        }

        float *stock_returns = h_all_returns + stock_idx * max_days;

        h_var_95[stock_idx] = calculate_percentile(stock_returns, num_returns, 0.05f);
        h_var_99[stock_idx] = calculate_percentile(stock_returns, num_returns, 0.01f);
    }

    double end = get_time();
    double time_taken = end - start;

    RiskMetrics *results = (RiskMetrics*)malloc(num_stocks * sizeof(RiskMetrics));
    for (int i = 0; i < num_stocks; i++) {
        strcpy(results[i].symbol, stocks[i].symbol);
        results[i].mean = h_means[i];
        results[i].std_dev = sqrtf(h_variances[i]);
        results[i].volatility = results[i].std_dev * sqrtf(TRADING_DAYS_PER_YEAR);
        results[i].var_95 = h_var_95[i];
        results[i].var_99 = h_var_99[i];
    }

    printf("Time: %.4f seconds\n", time_taken);

    printf("\nTop 5 stocks:\n");

    for (int i = 0; i < 5 && i < num_stocks; i++) {
        printf("%s %.6f %.6f %.6f %.6f\n",
               results[i].symbol,
               results[i].std_dev,
               results[i].volatility,
               results[i].var_95,
               results[i].var_99);
    }

    cudaFree(d_prices);
    cudaFree(d_all_returns);
    cudaFree(d_days_per_stock);
    cudaFree(d_means);
    cudaFree(d_variances);
    cudaFree(d_temp_sort);
    
    free(h_prices);
    free(h_days_per_stock);
    free(h_means);
    free(h_variances);
    free(h_all_returns);
    free(h_var_95);
    free(h_var_99);
    free(results);

    for (int i = 0; i < num_stocks; i++) {
        free(stocks[i].prices);
    }
    free(stocks);

    return 0;
}


int load_csv_data(const char *filename, Stock **stocks, int *num_stocks) {
    FILE *file = fopen(filename, "r");
    if (!file) return -1;

    char header[MAX_LINE];
    fgets(header, MAX_LINE, file);

    int max_capacity = 100;
    *stocks = (Stock*)malloc(max_capacity * sizeof(Stock));
    *num_stocks = 0;

    char line[MAX_LINE];
    while (fgets(line, MAX_LINE, file)) {
        char *token;
        char symbol[16];
        float price;

        token = strtok(line, ",");
        token = strtok(NULL, ",");
        strncpy(symbol, token, 15);
        symbol[15] = '\0';

        token = strtok(NULL, ",");
        price = atof(token);

        int found = 0;
        for (int i = 0; i < *num_stocks; i++) {
            if (strcmp((*stocks)[i].symbol, symbol) == 0) {
                (*stocks)[i].prices = (float*)realloc((*stocks)[i].prices, ((*stocks)[i].num_days + 1) * sizeof(float));
                (*stocks)[i].prices[(*stocks)[i].num_days] = price;
                (*stocks)[i].num_days++;
                found = 1;
                break;
            }
        }

        if (!found) {
            if (*num_stocks >= max_capacity) {
                max_capacity *= 2;
                *stocks = (Stock*)realloc(*stocks, max_capacity * sizeof(Stock));
            }

            strcpy((*stocks)[*num_stocks].symbol, symbol);
            (*stocks)[*num_stocks].prices = (float*)malloc(sizeof(float));
            (*stocks)[*num_stocks].prices[0] = price;
            (*stocks)[*num_stocks].num_days = 1;
            (*num_stocks)++;
        }
    }

    fclose(file);

    int valid_stocks = 0;
    for (int i = 0; i < *num_stocks; i++) {
        if ((*stocks)[i].num_days >= 100) {
            (*stocks)[valid_stocks] = (*stocks)[i];
            valid_stocks++;
        } else {
            free((*stocks)[i].prices);
        }
    }
    *num_stocks = valid_stocks;

    return 0;
}


void parallel_merge_sort_batch(float *d_data, float *d_temp, int *d_num_returns, int max_days, int num_stocks) {
    batch_merge_sort_kernel<<<num_stocks, 256>>>(d_data, d_temp, d_num_returns, max_days);
    cudaDeviceSynchronize();
}

float calculate_percentile(float *sorted_returns, int n, float percentile) {
    if (n == 0) return 0.0f;

    float index = percentile * (n - 1);
    int lower = (int)index;
    int upper = lower + 1;
    float weight = index - lower;

    if (upper >= n) {
        return sorted_returns[n - 1];
    }

    return sorted_returns[lower] * (1.0f - weight) + sorted_returns[upper] * weight;
}
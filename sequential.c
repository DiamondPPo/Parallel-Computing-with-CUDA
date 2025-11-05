#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

double get_time() {
    return (double)clock() / CLOCKS_PER_SEC;
}

#define MAX_LINE 1024
#define TRADING_DAYS_PER_YEAR 365.0f

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
void merge_sort(float *arr, int left, int right);
void merge(float *arr, int left, int mid, int right);
float calculate_percentile(float *sorted_returns, int n, float percentile);

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_csv>\n", argv[0]);
        return 1;
    }

    printf("Sequential:\n");

    Stock *stocks = NULL;
    int num_stocks = 0;

    if (load_csv_data(argv[1], &stocks, &num_stocks) != 0) {
        printf("Failed to load data\n");
        return 1;
    }

    int total_days = 0;
    for (int i = 0; i < num_stocks; i++) {
        total_days += stocks[i].num_days;
    }

    printf("Dataset: %d stocks, %d total days\n", num_stocks, total_days);

    double start = get_time();

    RiskMetrics *results = (RiskMetrics*)malloc(num_stocks * sizeof(RiskMetrics));

    for (int i = 0; i < num_stocks; i++) {
        strcpy(results[i].symbol, stocks[i].symbol);
        int num_returns = stocks[i].num_days - 1;

        if (num_returns <= 0) {
            results[i].mean = 0.0f;
            results[i].std_dev = 0.0f;
            results[i].volatility = 0.0f;
            results[i].var_95 = 0.0f;
            results[i].var_99 = 0.0f;
            continue;
        }

        float *returns = (float*)malloc(num_returns * sizeof(float));
        float sum = 0.0f;
        float sum_sq = 0.0f;

        for (int j = 0; j < num_returns; j++) {
            returns[j] = (stocks[i].prices[j + 1] - stocks[i].prices[j]) / stocks[i].prices[j];
            sum += returns[j];
            sum_sq += returns[j] * returns[j];
        }

        float mean = sum / num_returns;
        float variance = (sum_sq - sum * sum / num_returns) / (num_returns - 1.0f);
        float std_dev = sqrtf(variance);

        results[i].mean = mean;
        results[i].std_dev = std_dev;
        results[i].volatility = std_dev * sqrtf(TRADING_DAYS_PER_YEAR);

        merge_sort(returns, 0, num_returns - 1);
        results[i].var_95 = calculate_percentile(returns, num_returns, 0.05f);
        results[i].var_99 = calculate_percentile(returns, num_returns, 0.01f);

        free(returns);
    }

    double end = get_time();
    double time_taken = end - start;

    printf("Time: %.4f seconds\n", time_taken);

    printf("Top 5 stocks:\n");

    for (int i = 0; i < 5 && i < num_stocks; i++) {
        printf("%s %.6f %.6f %.6f %.6f\n",
               results[i].symbol,
               results[i].std_dev,
               results[i].volatility,
               results[i].var_95,
               results[i].var_99);
    }

    for (int i = 0; i < num_stocks; i++) {
        free(stocks[i].prices);
    }
    free(stocks);
    free(results);

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

void merge_sort(float *arr, int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void merge(float *arr, int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    float *L = (float*)malloc(n1 * sizeof(float));
    float *R = (float*)malloc(n2 * sizeof(float));

    for (i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    i = 0;
    j = 0;
    k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
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
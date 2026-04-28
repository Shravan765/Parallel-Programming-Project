// parallel_sorts.cpp
//
// Implements three parallel sorting algorithms faithful to their published references:
//
// 1. Parallel Merge Sort with binary-search-based parallel merge
//    Reference: R. Cole, "Parallel Merge Sort," SIAM J. Comput., 17(4), pp.770-785, 1988.
//    Key idea: The merge step itself is parallelized by using binary search to split
//    the merge into independent sub-merges (Cole's core contribution).
//
// 2. Parallel Quick Sort with cooperative multi-thread partitioning
//    Reference: D.M.W. Powers, "Parallelized QuickSort and RadixSort with Optimal Speedup,"
//    Proc. Int. Conf. on Parallel Computing Technologies, 1991.
//    Key idea: The partition step is parallelized — multiple threads cooperate to classify
//    and rearrange elements around the pivot simultaneously (performing partitioning implicitly).
//
// 3. Parallel Odd-Even Transposition Sort with merge-splitting
//    References:
//      - A.N. Habermann, "Parallel Neighbor Sort (or the Glory of the Induction Principle),"
//        CMU Computer Science Report, Aug. 1972.
//      - G. Baudet & D. Stevenson, "Optimal Sorting Algorithms for Parallel Computers,"
//        IEEE Trans. Comput., vol. C-27, pp.84-87, Jan. 1978.
//    Key idea: Each processor holds a sublist, sorts locally, then performs merge-split
//    operations with neighbors in alternating odd/even phases (Baudet-Stevenson extension).
//
// Compile: g++ -std=c++17 -pthread -O2 -o parallel_sorts parallel_sorts.cpp
// Run:     ./parallel_sorts <num_elements> <num_threads> <log_flag(0/1)>
// Example: ./parallel_sorts 10000000 8 1

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <algorithm>
#include <random>
#include <chrono>
#include <functional>
#include <cmath>
#include <fstream>
#include <iomanip>

// ===================== GLOBALS =====================
bool LOG_ENABLED = false;
std::mutex log_mutex;
std::ofstream log_file;

// ===================== LOGGING =====================
void log_msg(int thread_id, const std::string& algo, const std::string& msg) {
    if (!LOG_ENABLED) return;
    std::lock_guard<std::mutex> lock(log_mutex);
    auto now = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
    log_file << "[" << us << " us] "
             << "[" << algo << "] "
             << "[Thread " << thread_id << "] "
             << msg << "\n";
}

// ===================== UTILITY =====================
std::vector<int> generate_random_array(int n, unsigned seed) {
    std::vector<int> arr(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(1, 1000000);
    for (int i = 0; i < n; i++) {
        arr[i] = dist(rng);
    }
    return arr;
}

bool is_sorted_check(const std::vector<int>& arr) {
    for (size_t i = 1; i < arr.size(); i++) {
        if (arr[i] < arr[i - 1]) return false;
    }
    return true;
}

// ====================================================
// 1. PARALLEL MERGE SORT
//    Cole-inspired: binary-search-based parallel merge
//    
//    The key insight from Cole (1988) is that the MERGE
//    step itself must be parallelized. We use binary search
//    to find the rank of the median of one sorted half in
//    the other, splitting the merge into two independent
//    sub-merges that can proceed in parallel.
//
//    This is the practical shared-memory adaptation of
//    Cole's CREW PRAM approach using rank-based splitting.
// ====================================================
namespace ParallelMergeSort {

    static const int SEQ_THRESHOLD = 2000;

    // ---- Parallel Merge (Cole's core idea) ----
    // Merges two sorted regions src[la..ra] and src[lb..rb]
    // into dst starting at index lc.
    //
    // Strategy: pick the median of the larger half, binary-search
    // for its rank in the smaller half. This yields two independent
    // sub-merge problems that can be solved in parallel.
    void parallel_merge(const std::vector<int>& src, std::vector<int>& dst,
                        int la, int ra, int lb, int rb, int lc,
                        int depth, int max_merge_depth, int tid) {
        int na = ra - la + 1;
        int nb = rb - lb + 1;

        // Base cases: one or both ranges empty
        if (na <= 0 && nb <= 0) return;
        if (na <= 0) {
            std::copy(src.begin() + lb, src.begin() + rb + 1, dst.begin() + lc);
            return;
        }
        if (nb <= 0) {
            std::copy(src.begin() + la, src.begin() + ra + 1, dst.begin() + lc);
            return;
        }

        // Base case: fall back to sequential merge
        if (depth >= max_merge_depth || (na + nb) < SEQ_THRESHOLD) {
            log_msg(tid, "MergeSort",
                    "Sequential merge: [" + std::to_string(la) + ".." +
                    std::to_string(ra) + "] + [" + std::to_string(lb) +
                    ".." + std::to_string(rb) + "] → dst[" + std::to_string(lc) + "..]");
            int i = la, j = lb, k = lc;
            while (i <= ra && j <= rb) {
                if (src[i] <= src[j]) dst[k++] = src[i++];
                else dst[k++] = src[j++];
            }
            while (i <= ra) dst[k++] = src[i++];
            while (j <= rb) dst[k++] = src[j++];
            return;
        }

        // Ensure range A is the LARGER range (swap if needed)
        int ala = la, ara = ra, alb = lb, arb = rb;
        if (na < nb) {
            std::swap(ala, alb);
            std::swap(ara, arb);
            std::swap(na, nb);
        }

        // Pick median of larger range A
        int mid_a = ala + na / 2;

        // Binary search: find position in range B where src[mid_a] would be inserted
        // This is the rank-based splitting from Cole's approach
        int mid_b = static_cast<int>(
            std::upper_bound(src.begin() + alb, src.begin() + arb + 1, src[mid_a])
            - src.begin());

        // Compute destination position for the median element
        int mid_c = lc + (mid_a - ala) + (mid_b - alb);
        dst[mid_c] = src[mid_a];

        log_msg(tid, "MergeSort",
                "Parallel merge split: median src[" + std::to_string(mid_a) +
                "]=" + std::to_string(src[mid_a]) +
                ", rank in B at " + std::to_string(mid_b) +
                " → dst[" + std::to_string(mid_c) + "]");

        // LEFT sub-merge:  A[ala..mid_a-1]  +  B[alb..mid_b-1]  → dst[lc..mid_c-1]
        // RIGHT sub-merge: A[mid_a+1..ara]  +  B[mid_b..arb]    → dst[mid_c+1..]
        // These are INDEPENDENT — can run in parallel (Cole's key contribution)
        std::thread left_thread(parallel_merge, std::cref(src), std::ref(dst),
                                ala, mid_a - 1, alb, mid_b - 1, lc,
                                depth + 1, max_merge_depth, tid * 2 + 1);

        parallel_merge(src, dst,
                       mid_a + 1, ara, mid_b, arb, mid_c + 1,
                       depth + 1, max_merge_depth, tid * 2 + 2);

        left_thread.join();
    }

    // ---- Parallel Merge Sort (recursive) ----
    // Uses fork-join for the recursive sort AND parallel merge for combining.
    void p_merge_sort(std::vector<int>& arr, std::vector<int>& temp,
                      int left, int right,
                      int depth, int max_sort_depth, int max_merge_depth, int tid) {
        if (left >= right) return;

        // Below threshold or max depth: use optimized sequential sort
        if (depth >= max_sort_depth || (right - left) < SEQ_THRESHOLD) {
            log_msg(tid, "MergeSort",
                    "Sequential sort [" + std::to_string(left) + ".." +
                    std::to_string(right) + "] (" +
                    std::to_string(right - left + 1) + " elements)");
            std::sort(arr.begin() + left, arr.begin() + right + 1);
            return;
        }

        int mid = left + (right - left) / 2;

        log_msg(tid, "MergeSort",
                "Fork-sort [" + std::to_string(left) + ".." +
                std::to_string(right) + "], split at " + std::to_string(mid));

        // Fork: sort left and right halves in parallel
        std::thread left_thread(p_merge_sort, std::ref(arr), std::ref(temp),
                                left, mid,
                                depth + 1, max_sort_depth, max_merge_depth,
                                tid * 2 + 1);

        p_merge_sort(arr, temp, mid + 1, right,
                     depth + 1, max_sort_depth, max_merge_depth,
                     tid * 2 + 2);

        left_thread.join();

        // ---- PARALLEL MERGE (Cole's contribution) ----
        // Merge depth decreases at deeper sort levels to control thread count.
        // At sort depth d, merge gets max_merge_depth - d levels of parallelism.
        // This bounds total concurrent threads to ~2^max_merge_depth.
        int merge_depth = std::max(0, max_merge_depth - depth);

        log_msg(tid, "MergeSort",
                "Parallel merge [" + std::to_string(left) + ".." +
                std::to_string(right) + "], merge_depth=" +
                std::to_string(merge_depth));

        parallel_merge(arr, temp, left, mid, mid + 1, right, left,
                       0, merge_depth, tid);

        // Copy merged result back from temp to arr
        std::copy(temp.begin() + left, temp.begin() + right + 1,
                  arr.begin() + left);
    }

    // ---- Entry point ----
    void sort(std::vector<int>& arr, int num_threads) {
        std::vector<int> temp(arr.size());
        int max_sort_depth = std::max(1, static_cast<int>(std::log2(num_threads)));
        int max_merge_depth = std::max(1, static_cast<int>(std::log2(num_threads)));

        log_msg(0, "MergeSort",
                "Starting: threads=" + std::to_string(num_threads) +
                " sort_depth=" + std::to_string(max_sort_depth) +
                " merge_depth=" + std::to_string(max_merge_depth));

        p_merge_sort(arr, temp, 0, arr.size() - 1,
                     0, max_sort_depth, max_merge_depth, 0);

        log_msg(0, "MergeSort", "Completed.");
    }
}


// ====================================================
// 2. PARALLEL QUICKSORT
//    Powers-inspired: cooperative multi-thread partitioning
//
//    The key insight from Powers (1991) is that the PARTITION
//    step itself must be parallelized — multiple processors
//    cooperate to classify elements relative to the pivot
//    simultaneously ("performing partitioning implicitly").
//
//    We implement this as a 3-phase parallel partition:
//      Phase 1: Each thread counts elements ≤ pivot in its chunk
//      Phase 2: Prefix sum to compute global write offsets
//      Phase 3: Each thread scatters its elements to correct positions
//
//    This is the practical shared-memory adaptation of Powers'
//    CRCW PRAM implicit partitioning using parallel prefix sums.
// ====================================================
namespace ParallelQuickSort {

    static const int SEQ_THRESHOLD = 2000;

    // ---- Parallel Partition (Powers' core idea) ----
    // Multiple threads cooperate to partition arr[low..high] around a pivot.
    // Returns the final index of the pivot element.
    //
    // Phase 1: Each thread counts elements ≤ pivot in its chunk (parallel classification)
    // Phase 2: Prefix sum over counts to determine global write positions
    // Phase 3: Each thread scatters its classified elements to computed positions
    int parallel_partition(std::vector<int>& arr, std::vector<int>& temp,
                           int low, int high, int num_part_threads, int tid) {
        int n = high - low + 1;

        // Median-of-three pivot selection
        int mid_idx = low + (high - low) / 2;
        if (arr[mid_idx] < arr[low]) std::swap(arr[low], arr[mid_idx]);
        if (arr[high] < arr[low]) std::swap(arr[low], arr[high]);
        if (arr[mid_idx] < arr[high]) std::swap(arr[mid_idx], arr[high]);
        int pivot = arr[high];

        int data_n = n - 1; // elements to partition (excluding pivot at arr[high])

        // Fall back to sequential for small inputs or single thread
        if (num_part_threads <= 1 || data_n < SEQ_THRESHOLD) {
            log_msg(tid, "QuickSort",
                    "Sequential partition [" + std::to_string(low) + ".." +
                    std::to_string(high) + "], pivot=" + std::to_string(pivot));
            int i = low - 1;
            for (int j = low; j < high; j++) {
                if (arr[j] <= pivot) {
                    i++;
                    std::swap(arr[i], arr[j]);
                }
            }
            std::swap(arr[i + 1], arr[high]);
            return i + 1;
        }

        int nt = std::min(num_part_threads, std::max(2, data_n / 500));
        int chunk = data_n / nt;

        log_msg(tid, "QuickSort",
                "Parallel partition [" + std::to_string(low) + ".." +
                std::to_string(high) + "], pivot=" + std::to_string(pivot) +
                ", using " + std::to_string(nt) + " partition threads");

        // ---- Phase 1: Parallel classification (count) ----
        // Each thread counts how many elements in its chunk are ≤ pivot
        // This is the "implicit partitioning" from Powers — each processor
        // independently determines the classification of its elements.
        std::vector<int> less_count(nt, 0);
        std::vector<std::thread> threads;

        for (int t = 0; t < nt; t++) {
            int start = low + t * chunk;
            int end = (t == nt - 1) ? high - 1 : low + (t + 1) * chunk - 1;
            threads.emplace_back([&arr, &less_count, t, start, end, pivot]() {
                int cnt = 0;
                for (int i = start; i <= end; i++) {
                    if (arr[i] <= pivot) cnt++;
                }
                less_count[t] = cnt;
            });
        }
        for (auto& th : threads) th.join();
        threads.clear();

        log_msg(tid, "QuickSort", "Phase 1 complete: counts computed");

        // ---- Phase 2: Prefix sum (sequential — O(num_threads), negligible) ----
        // Compute write offsets for each thread's ≤pivot and >pivot elements
        // This mirrors the parallel prefix sum from Powers' PRAM model,
        // done sequentially here since nt is small (≤ num_threads).
        std::vector<int> less_offset(nt, 0);
        std::vector<int> greater_offset(nt, 0);
        int total_less = 0;
        int total_greater = 0;

        for (int t = 0; t < nt; t++) {
            less_offset[t] = total_less;
            int chunk_start = low + t * chunk;
            int chunk_end = (t == nt - 1) ? high - 1 : low + (t + 1) * chunk - 1;
            int chunk_size = chunk_end - chunk_start + 1;
            int greater_count_t = chunk_size - less_count[t];

            total_less += less_count[t];
            greater_offset[t] = total_greater;
            total_greater += greater_count_t;
        }

        int pivot_pos = low + total_less;

        log_msg(tid, "QuickSort",
                "Phase 2 complete: pivot_pos=" + std::to_string(pivot_pos) +
                " (less=" + std::to_string(total_less) +
                ", greater=" + std::to_string(total_greater) + ")");

        // ---- Phase 3: Parallel scatter ----
        // Each thread writes its elements to the globally correct positions
        // in the temp buffer. Elements ≤ pivot go to [low..pivot_pos-1],
        // elements > pivot go to [pivot_pos+1..high].
        for (int t = 0; t < nt; t++) {
            int start = low + t * chunk;
            int end = (t == nt - 1) ? high - 1 : low + (t + 1) * chunk - 1;
            threads.emplace_back(
                [&arr, &temp, t, start, end, low, pivot, pivot_pos,
                 &less_offset, &greater_offset]() {
                    int li = low + less_offset[t];
                    int gi = pivot_pos + 1 + greater_offset[t];
                    for (int i = start; i <= end; i++) {
                        if (arr[i] <= pivot)
                            temp[li++] = arr[i];
                        else
                            temp[gi++] = arr[i];
                    }
                });
        }
        for (auto& th : threads) th.join();

        // Place pivot at its final position
        temp[pivot_pos] = pivot;

        // Copy partitioned result back to arr
        std::copy(temp.begin() + low, temp.begin() + high + 1, arr.begin() + low);

        log_msg(tid, "QuickSort",
                "Phase 3 complete: partition done, pivot at " +
                std::to_string(pivot_pos));

        return pivot_pos;
    }

    // ---- Parallel Quick Sort (recursive) ----
    void p_quicksort(std::vector<int>& arr, std::vector<int>& temp,
                     int low, int high,
                     int depth, int max_depth, int num_threads, int tid) {
        if (low >= high) return;

        // Below threshold or max depth: use optimized sequential sort
        if (depth >= max_depth || (high - low) < SEQ_THRESHOLD) {
            log_msg(tid, "QuickSort",
                    "Sequential sort [" + std::to_string(low) + ".." +
                    std::to_string(high) + "] (" +
                    std::to_string(high - low + 1) + " elements)");
            std::sort(arr.begin() + low, arr.begin() + high + 1);
            return;
        }

        // Threads available for partition at this recursion depth.
        // At depth d, we have 2^d concurrent quicksort calls,
        // so each gets num_threads / 2^d threads for its partition.
        int part_threads = std::max(1, num_threads / (1 << depth));

        int pi = parallel_partition(arr, temp, low, high, part_threads, tid);

        log_msg(tid, "QuickSort",
                "Fork at depth " + std::to_string(depth) +
                ": left=[" + std::to_string(low) + ".." + std::to_string(pi - 1) +
                "], right=[" + std::to_string(pi + 1) + ".." + std::to_string(high) + "]");

        // Fork: sort left and right partitions in parallel
        std::thread left_thread(p_quicksort, std::ref(arr), std::ref(temp),
                                low, pi - 1,
                                depth + 1, max_depth, num_threads,
                                tid * 2 + 1);

        p_quicksort(arr, temp, pi + 1, high,
                    depth + 1, max_depth, num_threads,
                    tid * 2 + 2);

        left_thread.join();
    }

    // ---- Entry point ----
    void sort(std::vector<int>& arr, int num_threads) {
        std::vector<int> temp(arr.size());
        int max_depth = std::max(1, static_cast<int>(std::log2(num_threads))) + 1;

        log_msg(0, "QuickSort",
                "Starting: threads=" + std::to_string(num_threads) +
                " max_depth=" + std::to_string(max_depth));

        p_quicksort(arr, temp, 0, arr.size() - 1,
                    0, max_depth, num_threads, 0);

        log_msg(0, "QuickSort", "Completed.");
    }
}


// ====================================================
// 3. PARALLEL ODD-EVEN TRANSPOSITION SORT
//    Baudet-Stevenson extension of Habermann (1972)
//
//    Original (Habermann 1972): N processors, each holding ONE element,
//    alternate between odd-phase and even-phase compare-and-swap with neighbors.
//
//    Baudet-Stevenson (1978) extension: k processors, each holding n/k elements.
//    Each processor sorts its own sublist, then performs merge-splitting with
//    its neighbor: merge the two sublists, give the smaller half to the left
//    processor and the larger half to the right processor.
//
//    After k phases (alternating odd/even), the array is guaranteed sorted.
// ====================================================
namespace ParallelOddEvenSort {

    // ---- Merge-Split Operation (Baudet-Stevenson) ----
    // Given two adjacent sorted blocks, merge them and split so that
    // the left block retains the smaller half and the right block retains the larger half.
    // This replaces the single-element compare-and-swap from Habermann's original.
    void merge_split(std::vector<int>& arr, int block_size, int n,
                     int tid_left, int tid_right) {
        int start_left = tid_left * block_size;
        int end_left = std::min(start_left + block_size, n);
        int start_right = tid_right * block_size;
        int end_right = std::min(start_right + block_size, n);

        if (start_left >= n || start_right >= n) return;

        int size_left = end_left - start_left;
        int size_right = end_right - start_right;
        int total = size_left + size_right;

        // Merge both sorted blocks into a temporary array
        std::vector<int> merged(total);
        int i = start_left, j = start_right, k = 0;
        while (i < end_left && j < end_right) {
            if (arr[i] <= arr[j]) merged[k++] = arr[i++];
            else merged[k++] = arr[j++];
        }
        while (i < end_left) merged[k++] = arr[i++];
        while (j < end_right) merged[k++] = arr[j++];

        // Split: left block gets smaller elements, right block gets larger
        for (int p = 0; p < size_left; p++) {
            arr[start_left + p] = merged[p];
        }
        for (int p = 0; p < size_right; p++) {
            arr[start_right + p] = merged[size_left + p];
        }
    }

    // ---- Entry point ----
    void sort(std::vector<int>& arr, int num_threads) {
        int n = arr.size();
        int block_size = (n + num_threads - 1) / num_threads;

        log_msg(0, "OddEvenSort",
                "Starting Baudet-Stevenson variant: threads=" +
                std::to_string(num_threads) +
                " block_size=" + std::to_string(block_size) +
                " elements=" + std::to_string(n));

        // ---- Step 1: Each processor sorts its local block ----
        // (Baudet-Stevenson: "each processor sorts its own sublist")
        {
            std::vector<std::thread> threads;
            for (int t = 0; t < num_threads; t++) {
                threads.emplace_back([&arr, t, block_size, n]() {
                    int start = t * block_size;
                    int end = std::min(start + block_size, n);
                    if (start < n) {
                        std::sort(arr.begin() + start, arr.begin() + end);
                    }
                });
            }
            for (auto& th : threads) th.join();
        }

        log_msg(0, "OddEvenSort", "Local sorts complete. Beginning transposition phases.");

        // ---- Step 2: Alternate odd/even merge-split phases ----
        // Habermann (1972) proved that after p phases (where p = num_processors),
        // the array is guaranteed sorted. Each phase alternates between:
        //   Even phase: merge-split pairs (0,1), (2,3), (4,5), ...
        //   Odd phase:  merge-split pairs (1,2), (3,4), (5,6), ...
        for (int phase = 0; phase < num_threads; phase++) {
            std::vector<std::thread> threads;

            if (phase % 2 == 0) {
                // EVEN phase: pairs (0,1), (2,3), (4,5), ...
                log_msg(0, "OddEvenSort",
                        "Phase " + std::to_string(phase) + " (EVEN)");
                for (int t = 0; t + 1 < num_threads; t += 2) {
                    threads.emplace_back([&arr, t, block_size, n]() {
                        merge_split(arr, block_size, n, t, t + 1);
                    });
                }
            } else {
                // ODD phase: pairs (1,2), (3,4), (5,6), ...
                log_msg(0, "OddEvenSort",
                        "Phase " + std::to_string(phase) + " (ODD)");
                for (int t = 1; t + 1 < num_threads; t += 2) {
                    threads.emplace_back([&arr, t, block_size, n]() {
                        merge_split(arr, block_size, n, t, t + 1);
                    });
                }
            }

            for (auto& th : threads) th.join();

            log_msg(0, "OddEvenSort",
                    "Phase " + std::to_string(phase) + " complete.");
        }

        log_msg(0, "OddEvenSort", "Completed.");
    }
}


// ====================================================
// TIMING HARNESS
// ====================================================
using SortFunction = std::function<void(std::vector<int>&, int)>;

double benchmark(SortFunction sort_fn, const std::vector<int>& original,
                 int num_threads, int runs, const std::string& name) {
    double total_time = 0.0;

    for (int r = 0; r < runs; r++) {
        std::vector<int> arr = original; // fresh copy each run

        auto start = std::chrono::high_resolution_clock::now();
        sort_fn(arr, num_threads);
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += elapsed;

        if (!is_sorted_check(arr)) {
            std::cerr << "  *** ERROR: " << name << " run " << r + 1
                      << " produced UNSORTED output! ***\n";
        }

        log_msg(0, name, "Run " + std::to_string(r + 1) + ": " +
                std::to_string(elapsed) + " ms");
    }

    return total_time / runs;
}


// ====================================================
// MAIN
// ====================================================
int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <num_elements> <num_threads> <log_flag(0/1)>\n";
        std::cerr << "Example: " << argv[0] << " 10000000 8 1\n";
        return 1;
    }

    int num_elements = std::atoi(argv[1]);
    int num_threads  = std::atoi(argv[2]);
    int log_flag     = std::atoi(argv[3]);

    if (num_elements <= 0 || num_threads <= 0) {
        std::cerr << "Error: num_elements and num_threads must be > 0\n";
        return 1;
    }

    LOG_ENABLED = (log_flag == 1);

    if (LOG_ENABLED) {
        std::string log_filename = "parallel_sort_log_n" +
            std::to_string(num_elements) + "_t" +
            std::to_string(num_threads) + ".log";
        log_file.open(log_filename);
        if (!log_file.is_open()) {
            std::cerr << "Error: Could not open log file: " << log_filename << "\n";
            return 1;
        }
        std::cout << "Logging to: " << log_filename << "\n";
    }

    const int NUM_RUNS = 5;
    unsigned seed = 42;

    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "     PARALLEL SORTING ALGORITHM BENCHMARK (Paper-Faithful)\n";
    std::cout << "========================================================\n";
    std::cout << "  Elements  : " << num_elements << "\n";
    std::cout << "  Threads   : " << num_threads << "\n";
    std::cout << "  Runs      : " << NUM_RUNS << "\n";
    std::cout << "  Logging   : " << (LOG_ENABLED ? "ON" : "OFF") << "\n";
    std::cout << "========================================================\n";
    std::cout << "  Algorithms:\n";
    std::cout << "    1. Parallel Merge Sort    [Cole 1988 — parallel merge]\n";
    std::cout << "    2. Parallel Quick Sort    [Powers 1991 — parallel partition]\n";
    std::cout << "    3. Odd-Even Trans. Sort   [Habermann 1972 / Baudet-Stevenson 1978]\n";
    std::cout << "========================================================\n\n";

    std::vector<int> original = generate_random_array(num_elements, seed);

    // --- Benchmark 1: Parallel Merge Sort ---
    std::cout << "[1/3] Benchmarking Parallel Merge Sort (Cole)..." << std::flush;
    double avg_merge = benchmark(ParallelMergeSort::sort, original,
                                  num_threads, NUM_RUNS, "MergeSort");
    std::cout << " done.\n";

    // --- Benchmark 2: Parallel Quick Sort ---
    std::cout << "[2/3] Benchmarking Parallel Quick Sort (Powers)..." << std::flush;
    double avg_quick = benchmark(ParallelQuickSort::sort, original,
                                  num_threads, NUM_RUNS, "QuickSort");
    std::cout << " done.\n";

    // --- Benchmark 3: Parallel Odd-Even Transposition Sort ---
    std::cout << "[3/3] Benchmarking Parallel Odd-Even Trans. Sort (Baudet-Stevenson)..." << std::flush;
    double avg_oddeven = benchmark(ParallelOddEvenSort::sort, original,
                                    num_threads, NUM_RUNS, "OddEvenSort");
    std::cout << " done.\n";

    // --- Results ---
    std::cout << "\n";
    std::cout << "========================================================\n";
    std::cout << "              RESULTS (avg of " << NUM_RUNS << " runs)\n";
    std::cout << "========================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Parallel Merge Sort (Cole)           : " << avg_merge   << " ms\n";
    std::cout << "  Parallel Quick Sort (Powers)         : " << avg_quick   << " ms\n";
    std::cout << "  Odd-Even Trans. Sort (Baudet-Stev.)  : " << avg_oddeven << " ms\n";
    std::cout << "========================================================\n";

    double slowest = std::max({avg_merge, avg_quick, avg_oddeven});
    double fastest = std::min({avg_merge, avg_quick, avg_oddeven});
    std::cout << "\n  Relative Speedup (vs slowest):\n";
    std::cout << "    Merge Sort         : " << std::setprecision(2) << (slowest / avg_merge)   << "x\n";
    std::cout << "    Quick Sort         : " << (slowest / avg_quick)   << "x\n";
    std::cout << "    Odd-Even Trans.    : " << (slowest / avg_oddeven) << "x\n";
    std::cout << "========================================================\n\n";

    if (LOG_ENABLED) {
        log_file.close();
        std::cout << "Log file written successfully.\n";
    }

    return 0;
}
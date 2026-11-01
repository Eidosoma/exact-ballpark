// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Eidosoma
// libstdc++'s unordered containers as Destiny's compiled code uses them: the oracle of the
// implementation in exact_ballpark.space (StdHashMap, StdHashSet). Not Destiny's code: a probe of the C++ library the
// compiled Destiny harness links (conformance/build_space.py's binary imports _M_need_rehash from
// libstdc++.so.6, which calls _M_next_bkt and reads __prime_list there). Compiled with the harness's g++ and flags.
//
//   hashtable_probe primes N        the first N entries of std::__detail::__prime_list (N from the symbol's size)
//   hashtable_probe next_bkt        stdin: n per line; out: n, _M_next_bkt(n), _M_next_resize after the call
//   hashtable_probe need_rehash     stdin: next_resize n_bkt n_elt n_ins; out: the inputs, the pair, _M_next_resize
//   hashtable_probe tape            stdin: one operation a line, "KIND INDEX OP [KEY]"; out: one line a container
//                                   state after each: bucket_count, size, the keys in iteration order
//
// KIND: m, std::unordered_map<int64_t, int64_t> (MapOfLongLongs, a sensor's balls: MapOfLongLongs::insert is
// operator[]); d, std::unordered_map<int64_t, void*> (DictOfBalls, mProximityBalls: insert(pair)); s,
// std::unordered_set<int64_t> (SetOfIDs, a ball's followers). OP: i (insert(pair) or insert(key)), o (operator[] = 0,
// maps m only), e (erase(key)), c (clear()), y (the state of a copy-constructed container instead of the container's,
// the container unchanged), w D (a walk erasing, as CheckForProximity_Sensor erases while it walks, each key whose
// (uint64_t)key / 7 is a multiple of D).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace std {
namespace __detail {
extern const unsigned long __prime_list[];
}
}  // namespace std

typedef std::unordered_map<int64_t, int64_t> MapLL;
typedef std::unordered_map<int64_t, void*> DictB;
typedef std::unordered_set<int64_t> SetI;

static int64_t key_of(const std::pair<const int64_t, int64_t>& kv) { return kv.first; }
static int64_t key_of(const std::pair<const int64_t, void*>& kv) { return kv.first; }
static int64_t key_of(const int64_t& k) { return k; }

template <typename C>
static void print_state(const C& c) {
    std::printf("%zu %zu", c.bucket_count(), c.size());
    for (const auto& kv : c) std::printf(" %lld", (long long)key_of(kv));
    std::printf("\n");
}


template <typename C>
static void walk_erase(C& c, unsigned long long d) {
    for (auto it = c.begin(); it != c.end();) {
        auto store = it++;
        if (d && ((unsigned long long)key_of(*store) / 7) % d == 0) c.erase(store);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    const std::string mode = argv[1];
    if (mode == "primes") {
        const long n = std::atol(argv[2]);
        for (long i = 0; i < n; ++i) std::printf("%lu\n", std::__detail::__prime_list[i]);
        return 0;
    }
    if (mode == "next_bkt") {
        unsigned long n;
        while (std::scanf("%lu", &n) == 1) {
            std::__detail::_Prime_rehash_policy pol;
            const std::size_t r = pol._M_next_bkt(n);
            std::printf("%lu %zu %zu\n", n, r, pol._M_next_resize);
        }
        return 0;
    }
    if (mode == "need_rehash") {
        unsigned long nr, b, e, k;
        while (std::scanf("%lu %lu %lu %lu", &nr, &b, &e, &k) == 4) {
            std::__detail::_Prime_rehash_policy pol;
            pol._M_next_resize = nr;
            const std::pair<bool, std::size_t> res = pol._M_need_rehash(b, e, k);
            std::printf("%lu %lu %lu %lu %d %zu %zu\n", nr, b, e, k, (int)res.first, res.second, pol._M_next_resize);
        }
        return 0;
    }
    if (mode == "tape") {
        std::map<long, MapLL> ms;
        std::map<long, DictB> ds;
        std::map<long, SetI> ss;
        char kind[8], op[8], arg[4096];
        long index;
        char line[4200];
        while (std::fgets(line, sizeof line, stdin)) {
            arg[0] = 0;
            const int got = std::sscanf(line, "%7s %ld %7s %4095s", kind, &index, op, arg);
            if (got < 3) continue;
            const long long key = got > 3 ? std::atoll(arg) : 0;
            if (kind[0] == 'm') {
                MapLL& c = ms[index];
                if (op[0] == 'i') c.insert(std::make_pair((int64_t)key, (int64_t)0));
                else if (op[0] == 'o') c[(int64_t)key] = 0;
                else if (op[0] == 'e') c.erase((int64_t)key);
                else if (op[0] == 'c') c.clear();
                else if (op[0] == 'w') walk_erase(c, (unsigned long long)key);
                if (op[0] == 'y') { MapLL copy = c; print_state(copy); } else print_state(c);
            } else if (kind[0] == 'd') {
                DictB& c = ds[index];
                if (op[0] == 'i') c.insert(std::make_pair((int64_t)key, (void*)nullptr));
                else if (op[0] == 'e') c.erase((int64_t)key);
                else if (op[0] == 'c') c.clear();
                else if (op[0] == 'w') walk_erase(c, (unsigned long long)key);
                if (op[0] == 'y') { DictB copy = c; print_state(copy); } else print_state(c);
            } else {
                SetI& c = ss[index];
                if (op[0] == 'i') c.insert((int64_t)key);
                else if (op[0] == 'e') c.erase((int64_t)key);
                else if (op[0] == 'c') c.clear();
                else if (op[0] == 'w') walk_erase(c, (unsigned long long)key);
                if (op[0] == 'y') { SetI copy = c; print_state(copy); } else print_state(c);
            }
        }
        return 0;
    }
    return 2;
}

/********************************************************************************************************
        Author: Coder LSF (Liu Shaofeng)
          Date: 2023/11/09
         Brief: Implements a class template of hashmap with high performance
 ********************************************************************************************************/
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <map>
#include <unordered_map>
#include <vector>
#include <functional>
#include <algorithm>
#include <type_traits>

#include "hash.h"
#include "string_view.hpp"

#ifndef unlikely
#define unlikely(x) (__builtin_expect(!!(x), 0))
#endif

#ifndef likely
#define likely(exp) (__builtin_expect(!!(exp), 1))
#endif

namespace ftl {

bool is_prime(size_t num) {
	if (num < 2) return false;
	for (size_t i = 2; i * i <= num; ++i) {
		if (num % i == 0) return false;
	}
	return true;
}
size_t next_prime(size_t n) {
	while (!is_prime(n)) {
		++n;
	}
	return n;
}

template<typename T, typename = void> struct is_string_class : std::false_type {};
template<> struct is_string_class<std::string> : std::true_type {};
template<> struct is_string_class<std::string_view> : std::true_type {};

template<typename T, typename = void> struct is_string_type : std::false_type {};
template<> struct is_string_type<const char*> : std::true_type {};
template<> struct is_string_type<std::string> : std::true_type {};
template<> struct is_string_type<std::string_view> : std::true_type {};

// 对于部分需要将数据放在哈希桶之外的派生Value类型，提供内置对象池，以减少对系统malloc和free的调用频率，
// 从而显著加快构建和析构的性能，同时也能一定程度提升数据访问locaitily从而加速查找性能
template <typename T>
class HashmapWithNodeArena {
public:
    HashmapWithNodeArena():
        _reserved_nodes(nullptr), _reserved_pos(1), _reserved_size(0),
        _bkt_idx(0), _bkt_pos(0) {}
    HashmapWithNodeArena(size_t reserved_size):
        _reserved_nodes(nullptr), _reserved_pos(1), _reserved_size(reserved_size+1),
        _bkt_idx(0), _bkt_pos(0) {}
    ~HashmapWithNodeArena() {
        if (_reserved_nodes != nullptr) {
            ::free(_reserved_nodes);
        }
        for (size_t i = 0; i < sizeof(_nodes)/sizeof(_nodes[0]); ++i) {
            if (_nodes[i] != nullptr) {
                ::free(_nodes[i]);
            }
        }
    }

protected:
    inline T& get_node(uint32_t idx) noexcept {
        if (idx < _reserved_size) {
            return _reserved_nodes[idx];
        }
        idx = idx - _reserved_pos + 32;
        const auto bi = 32 - __builtin_clz(idx);
        return _nodes[bi-5][idx & ((1u << bi) - 1u)];
    }

    static inline uint32_t node_bucket_size(uint32_t bkt_idx) noexcept {
        return 1u << (5 + bkt_idx);
    }
    inline uint32_t node_bucket_size() noexcept {
        return node_bucket_size(_bkt_idx);
    }
    std::pair<T*, uint32_t> alloc_node() {
        if (_reserved_pos <= 0) {
            _reserved_pos = 1;
        }
        if (_reserved_pos < _reserved_size) {
            if unlikely(_reserved_nodes == nullptr) {
                _reserved_nodes = malloc(sizeof(T)*_reserved_size);
                if unlikely(_reserved_nodes == nullptr) {
                    return std::pair<T*, uint32_t>(nullptr, 0);
                }
            }
            auto res = std::pair<T*, uint32_t>(&_reserved_nodes[_reserved_pos], _reserved_pos);
            ++_reserved_pos;
            return res;
        }
        auto bs = node_bucket_size();
        if (_bkt_pos >= bs) {
            ++_bkt_idx;
            _bkt_pos = 0;
            bs <<= 1;
        }
        if unlikely(_nodes[_bkt_idx] == nullptr) {
            _nodes[_bkt_idx] = malloc(sizeof(T)*bs);
            if (_nodes[_bkt_idx] == nullptr) {
                return std::pair<T*, uint32_t>(nullptr, 0);
            }
        }
        auto node = _nodes[_bkt_idx] + _bkt_pos;
        auto indx = bs - 32 + _bkt_pos + _reserved_pos;
        ++_bkt_idx;
        return std::pair<T*, uint32_t>(node, indx);
    }

    inline uint32_t reserved_size() const noexcept {
        return _reserved_size;
    }

private:
    T*          _reserved_nodes;
    uint32_t    _reserved_size;
    uint32_t    _reserved_pos;
    uint32_t    _bkt_idx;
    uint32_t    _bkt_pos;
    T*          _nodes[16];
};

// 对于字符串类型，由此类提供内存的分配，可以极大降低字符串类型的构造和析构成本（趋近于0）
class HashmapWithSeqMempool {
public:
    HashmapWithSeqMempool(size_t reserve_size):
        _mem(nullptr), _pos(0), _cap(0), _rsv_size(reserve_size) {}
    HashmapWithSeqMempool():
        _mem(nullptr), _pos(0), _cap(0), _rsv_size(0) {}
    ~HashmapWithSeqMempool() {
        if (_mem != nullptr) {
            ::free(_mem);
        }
    }

protected:
    inline char* get_ptr(uint32_t offset) noexcept {
        return _mem + offset;
    }

    struct Mem {
        uint32_t    offset;
        uint32_t    size;
    };
    inline Mem dup(const char* s, size_t len, size_t cur_kv_size) {
        if (s == nullptr || len == 0) {
            return Mem(8u, 0u);
        }
        auto off = alloc_mem(len+1, cur_kv_size);
        copy(off, len, s);
        return Mem(off, len);
    }
    inline Mem dup(const StringView& s, size_t cur_kv_size) {
        return dup(s.data(), s.size(), cur_kv_size);
    }
    inline Mem dup(const std::string& s, size_t cur_kv_size) {
        return dup(s.c_str(), s.size(), cur_kv_size);
    }
    inline Mem dup(const char* s, size_t cur_kv_size) {
        return dup(s, s==nullptr?0:strlen(s), cur_kv_size);
    }
    inline void copy(uint32_t off, size_t len, const char* s) {
        auto ptr = reinterpret_cast<uint64_t*>(_mem+off);
        auto r = len & 7ull;
        auto n = (len+7) >> 3;
        if (len <= 32 && (reinterpret_cast<uint64_t>(s)&7ull) == 0) {
            for (size_t i = 0; i < n; ++i) {
                ptr[i] = reinterpret_cast<const uint64_t*>(s)[i];
            }
        } else {
            memcpy(ptr, s, (len+7)&~7ull);
        }
        ptr[n] &= (1ull << (r<<3)) - 1ull;
    }

    inline static const char* str_ptr(const std::string& s) noexcept {
        return s.c_str();
    }
    inline static bool str_len(const std::string& s) noexcept {
        return s.size();
    }
    inline static const char* str_ptr(const StringView& s) noexcept {
        return s.c_str();
    }
    inline static bool str_len(const StringView& s) noexcept {
        return s.size();
    }
    inline static const char* str_ptr(const char* s) noexcept {
        return s;
    }
    inline static bool str_len(const char* s) noexcept {
        return s == nullptr ? 0 : strlen(s);
    }
    bool compare_string(uint32_t offset, uint32_t size, const char* s, size_t len) const noexcept {
        if (len != size) {
            return false;
        } else if (len == 0) {
            return true;
        }
        if (len >= 32 || (reinterpret_cast<uint64_t>(s)&7ull) != 0) {
            return strncmp(_mem+offset, s, len) == 0;
        }

        auto d1 = reinterpret_cast<const uint64_t*>(_mem+offset);
        auto d2 = reinterpret_cast<const uint64_t*>(s);
        size &= 7u;
        len >>= 3;
        for (size_t i = 0; i < len; ++i) {
            if (d1[i] != d2[i]) {
                return false;
            }
        }
        return size == 0 || d1[len] == (d2[len]&((1ull<<(size<<3))-1ull));
    }
    inline bool compare_string(uint32_t offset, uint32_t size, const StringView& s) const noexcept {
        return compare_string(offset, size, s.data(), s.size());
    }

    bool compare_string(uint32_t off1, uint32_t off2, uint32_t size) const noexcept {
        auto d1 = reinterpret_cast<const uint64_t*>(_mem+off1);
        auto d2 = reinterpret_cast<const uint64_t*>(_mem+off2);
        size = (size+7)>>3;
        for (size_t i = 0; i < size; ++i) {
            if (d1[i] != d2[i]) {
                return false;
            }
        }
        return true;
    }

    uint32_t alloc_mem(size_t size, size_t cur_kv_size) {
        size = (size+7)&~7u;
        if unlikely(_pos + size + 24 > _cap || _mem == nullptr) {
            if (cur_kv_size >= 100 && cur_kv_size < _rsv_size) {
                _cap = static_cast<uint32_t>((_pos * 1.1 / cur_kv_size) * _rsv_size + size + 128);
            } else {
                _cap = static_cast<uint32_t>(_cap * 1.5 + ((size + 8) << 5) + (_rsv_size << 3));
            }
            auto mem = (char*)aligned_alloc(32, _cap);
            if (_mem != nullptr) {
                memcpy(mem, _mem, _pos);
            } else if (_pos <= 0) {
                _pos = 8u;
            }
            _mem = mem;
        }
        auto res = size < 32 ? _pos : ((_pos + 31u) & ~31u);
        _pos = res + size;
        return res;
    }

private:
    char*       _mem;
    uint32_t    _cap; // memory pool capacity
    uint32_t    _pos;
    uint32_t    _rsv_size;
};

template <typename T, bool IS_SIMPLE_ASSIGN, bool MOVE_CONSTRUCTIBLE>
struct MoveValue {
    inline void operator()(T& dst, T&& src) {
        dst = src;
    }
};
template <typename T>
struct MoveValue<T, false, true> {
    inline void operator()(T& dst, T&& src) {
        new(&dst) T(std::forward<T>(src));
    }
};
template <typename T>
struct MoveValue<T, false, false> {
    inline void operator()(T& dst, T&& src) {
        new(&dst) T(src);
    }
};
template <typename T>
inline void move_value(T& dst, T&& src) {
    MoveValue<T, std::is_fundamental<T>::value || std::is_pointer<T>::value,
        !(std::is_fundamental<T>::value || std::is_pointer<T>::value)
            && std::is_move_constructible<T>::value>()(dst, std::forward<T>(src));
}

template <typename T, bool IS_SIMPLE_ASSIGN>
struct Constructor {
    inline void operator()(T& dst, T&& src) {
        dst = src;
    }
    inline void operator()(T& dst, const T& src) {
        dst = src;
    }
};
template <typename T>
struct Constructor<T, false> {
    inline void operator()(T& dst, T&& src) {
        new(&dst) T(src);
    }
    inline void operator()(T& dst, const T& src) {
        new(&dst) T(src);
    }
};
template <typename T>
inline void construct(T& dst, T&& src) {
    Constructor<T, std::is_fundamental<T>::value || std::is_pointer<T>::value>()(
            dst, std::forward<T>(src));
}
template <typename T>
inline void construct(T& dst, const T& src) {
    Constructor<T, std::is_fundamental<T>::value || std::is_pointer<T>::value>()(dst, src);
}

template <typename T, bool NEED_DESTRUCT>
struct Destructor {
    inline void operator()(T& v) {
        v.~T();
    }
};
template <typename T>
struct Destructor<T, false> {
    inline void operator()(T& v) {}
};
template <typename T>
inline void destruct(T& v) {
    Destructor<T, !std::is_trivially_destructible<T>::value>()(v);
}

template <typename T>
class HashmapIterator {
public:
    friend T;

    HashmapIterator(const T* hm, uint32_t idx): _hm(hm), _idx(idx) {}
    HashmapIterator(HashmapIterator<T>& other): _hm(other._hm), _idx(other._idx) {}
    HashmapIterator<T>& operator=(const HashmapIterator<T>& other) noexcept {
        _hm = other._hm;
        _idx = other._idx;
        update_data();
        return *this;
    }
    inline bool operator==(const HashmapIterator<T>& other) const noexcept {
        return _idx == other._idx;
    }
    inline bool operator!=(const HashmapIterator<T>& other) const noexcept {
        return _idx != other._idx;
    }
    ~HashmapIterator() {
        if (_data != nullptr) {
            delete _data;
        }
    }

    using KeyType   = typename T::InnerKeyType;
    using ValueType = typename T::InnerValType;

    struct Data {
        const KeyType     first;
        const ValueType   second;
    };

    inline operator bool() const noexcept {
        return _idx < _hm->hash_size();
    }
    inline bool is_end() const noexcept {
        return _idx >= _hm->hash_size();
    }
    inline Data& operator*() const noexcept {
        return *_data;
    }
    inline Data* operator->() const noexcept {
        return _data;
    }
    inline HashmapIterator<T>& operator++() noexcept {
        if (_idx < _hm->hash_size()) {
            for (++_idx; _idx < _hm->hash_size() && _hm->get_bucket(_idx).empty(); ++_idx);
        }
        update_data();
        return *this;
    }

protected:
    void update_data() noexcept {
        if (_idx >= _hm->hash_size()) {
            return;
        }
        if (_data == nullptr) {
            _data = malloc(sizeof(Data));
        }
        new(_data) Data(_hm->get_key(_idx), _hm->get_value(_idx));
    }

private:
    uint32_t    _idx;
    const T*    _hm;
    Data*       _data = nullptr;
};

template <typename Hasher>
class HashmapBase {
public:
    HashmapBase(): _data_size(0), _hash_size(0) {}
    HashmapBase(size_t max_data_size):
        _data_size(0), _hash_size(static_cast<size_t>(max_data_size*3)) {}

    inline uint32_t size() const noexcept {
        return _data_size;
    }
    inline uint32_t hash_size() const noexcept {
        return _hash_size;
    }
    inline bool empty() const noexcept {
        return _data_size > 0;
    }
    template <typename T>
    inline uint32_t hash(const T& v) const noexcept {
        return Hasher()(v);
    }

protected:
    uint32_t _data_size;
    uint32_t _hash_size;
};

// 对于不同类型的KeyType和ValueType，以不同的方式设计哈希桶的结构，从而最大化性能
template <typename KeyType, typename ValueType, typename Hasher,
    bool STRING_KEY, bool STRING_VAL, bool RESIDENT_VAL>
class FastHashmapBase {};

template <typename KeyType, typename ValueType, typename Hasher>
class FastHashmapBase<KeyType, ValueType, Hasher, false, false, true>:
        public HashmapBase<Hasher> {
protected:
    using HashmapBase<Hasher>::_hash_size;
    using HashmapBase<Hasher>::_data_size;
    using HashmapBase<Hasher>::hash;

    struct Bucket {
        KeyType     key;
        ValueType   val;
        void clear() { key = KeyType(); }
        inline bool empty() const noexcept {
            return key == KeyType(); // TODO: 此处为临时方案，需要替换为更好的判空方法
        }
    };

    FastHashmapBase(): _buckets(nullptr) {}
    FastHashmapBase(size_t reserve_size):
        HashmapBase<Hasher>(reserve_size), _buckets(nullptr) {}
    ~FastHashmapBase() {
        if (_buckets == nullptr) {
            return;
        }
        if (!std::is_trivially_destructible<KeyType>::value
                || !std::is_trivially_destructible<ValueType>::value) {
            for (size_t i = 0; i < this->_hash_size; ++i) {
                auto& bkt = _buckets[i];
                if (!std::is_trivially_destructible<KeyType>::value) {
                    destruct(bkt.key);
                }
                if (!std::is_trivially_destructible<ValueType>::value) {
                    destruct(bkt.val);
                }
            }
        }
        ::free(_buckets);
    }

    using InnerKeyType = KeyType&;
    using InnerValType = ValueType&;
    using ConstInnerKeyType = const KeyType&;
    using ConstInnerValType = const ValueType&;
    inline KeyType& get_key(uint32_t idx) noexcept {
        return _buckets[idx].key;
    }
    inline ValueType& get_value(uint32_t idx) noexcept {
        return _buckets[idx].val;
    }

    inline bool put_slot(Bucket& bkt, const KeyType& key, const ValueType& val, uint32_t) {
        if (bkt.empty()) {
            bkt.key = key;
            construct(bkt.val, val);
            ++_data_size;
        } else {
            bkt.val = val;
        }
        return true;
    }

    bool move_slot(Bucket& dst, Bucket& src) {
        dst.key = src.key;
        move_value(dst.val, std::move(src.val));
        destruct(src.val);
        return true;
    }

    static inline uint32_t get_hash_code(Bucket& bkt) noexcept {
        return Hasher()(bkt.key);
    }

    inline bool compare_key(const Bucket& bkt, const KeyType& key, uint32_t) const noexcept {
        return bkt.key == key;
    }
    inline bool compare_key(const Bucket& a, const Bucket&& b) const noexcept {
        return a.key == b.key;
    }

protected:
    Bucket*     _buckets;
};

template <typename KeyType, typename ValueType, typename Hasher>
class FastHashmapBase<KeyType, ValueType, Hasher, false, false, false>:
            public HashmapBase<Hasher>,
            public HashmapWithNodeArena<ValueType> {
protected:
    using HashmapBase<Hasher>::_hash_size;
    using HashmapBase<Hasher>::_data_size;
    using HashmapBase<Hasher>::hash;
    using HashmapWithNodeArena<ValueType>::alloc_node;
    using HashmapWithNodeArena<ValueType>::get_node;

    struct Bucket {
        KeyType    key;
        uint32_t   val_index;
        void clear() { val_index = 0; }
        inline bool empty() const noexcept {
            return val_index == 0;
        }
    };

    FastHashmapBase(): _buckets(nullptr) {}
    FastHashmapBase(size_t reserve_size):
        HashmapBase<Hasher>(reserve_size),
        HashmapWithNodeArena<ValueType>(reserve_size),
        _buckets(nullptr) {}
    ~FastHashmapBase() {
        if (_buckets == nullptr) {
            return;
        }
        if (!std::is_trivially_destructible<KeyType>::value
                || !std::is_trivially_destructible<ValueType>::value) {
            for (size_t i = 0; i < this->_hash_size; ++i) {
                auto& bkt = _buckets[i];
                if (!std::is_trivially_destructible<KeyType>::value) {
                    destruct(bkt.key);
                }
                if (!std::is_trivially_destructible<ValueType>::value) {
                    destruct(get_node(bkt.val_index));
                }
            }
        }
        ::free(_buckets);
    }

    using InnerKeyType = KeyType&;
    using InnerValType = ValueType&;
    using ConstInnerKeyType = const KeyType&;
    using ConstInnerValType = const ValueType&;
    inline KeyType& get_key(uint32_t idx) noexcept {
        return _buckets[idx].key;
    }
    inline ValueType& get_value(uint32_t idx) noexcept {
        return get_node(_buckets[idx].val_index);
    }

    inline bool put_slot(Bucket& bkt, const KeyType& key, const ValueType& val, uint32_t) {
        if (bkt.empty()) {
            auto ni = this->alloc_node();
            if (ni.first == nullptr) {
                return false;
            }
            bkt.key = key;
            bkt.val_index = ni.second;
            construct(*ni.first, val);
            ++_data_size;
        } else {
            get_node(bkt.val_index) = val;
        }
        return true;
    }

    bool move_slot(Bucket& dst, Bucket& src) {
        dst = src;
        return true;
    }

    static inline uint32_t get_hash_code(Bucket& bkt) noexcept {
        return Hasher()(bkt.key);
    }

    inline bool compare_key(const Bucket& bkt, const KeyType& key, uint32_t) const noexcept {
        return bkt.key == key;
    }
    inline bool compare_key(const Bucket& a, const Bucket& b) const noexcept {
        return a.key == b.key;
    }

protected:
    Bucket*     _buckets;
};

template <typename KeyType, typename ValueType, typename Hasher>
class FastHashmapBase<KeyType, ValueType, Hasher, false, true, true>:
        public HashmapBase<Hasher>, public HashmapWithSeqMempool {
protected:
    using HashmapBase<Hasher>::_hash_size;
    using HashmapBase<Hasher>::_data_size;
    using HashmapBase<Hasher>::hash;
    using HashmapWithSeqMempool::dup;
    using HashmapWithSeqMempool::copy;
    using HashmapWithSeqMempool::str_len;
    using HashmapWithSeqMempool::str_ptr;

    struct Bucket {
        KeyType    key;
        uint32_t   val_offset;
        uint32_t   val_size;
        void clear() { val_offset = 0; }
        inline bool empty() const noexcept {
            return val_offset == 0;
        }
        inline bool operator==(const Bucket& other) const noexcept {
            return key == other.key;
        }
    };
    FastHashmapBase(): _buckets(nullptr) {}
    FastHashmapBase(size_t reserve_size): HashmapBase<Hasher>(reserve_size),
        HashmapWithSeqMempool(reserve_size), _buckets(nullptr) {}
    ~FastHashmapBase() {
        if (_buckets == nullptr) {
            return;
        }
        if (!std::is_trivially_destructible<KeyType>::value) {
            for (size_t i = 0; i < this->_hash_size; ++i) {
                destruct(_buckets[i].key);
            }
        }
        ::free(_buckets);
    }

    using InnerKeyType = KeyType&;
    using InnerValType = StringView;
    using ConstInnerKeyType = const KeyType&;
    using ConstInnerValType = const StringView;
    inline KeyType& get_key(uint32_t idx) noexcept {
        return _buckets[idx].key;
    }
    inline StringView get_value(uint32_t idx) noexcept {
        auto& bkt = _buckets[idx];
        return StringView(get_ptr(bkt.val_offset), bkt.val_size);
    }

    inline bool put_slot(Bucket& bkt, const KeyType& key, const StringView& val, uint32_t) {
        if (bkt.empty()) {
            auto vmem = dup(val, this->_data_size);
            if (vmem.offset == 0) {
                return false;
            }
            bkt.key = key;
            bkt.val_offset = vmem.offset;
            bkt.val_size = vmem.size;
            ++_data_size;
        } else {
            auto len = str_len(val);
            if (len <= bkt.val_size) {
                this->copy(bkt.val_offset, len, str_ptr(val));
            } else {
                auto vmem = dup(val, _data_size);
                if (vmem.offset == 0) {
                    return false;
                }
                bkt.val_offset = vmem.offset;
                bkt.val_size   = vmem.size;
            }
        }
        return true;
    }

    inline bool move_slot(Bucket& dst, Bucket& src) {
        dst = src;
        return true;
    }

    static inline uint32_t get_hash_code(Bucket& bkt) noexcept {
        return Hasher()(bkt.key);
    }

    inline bool compare_key(const Bucket& bkt, const KeyType& key, uint32_t) const noexcept {
        return bkt.key == key;
    }
    inline bool compare_key(const Bucket& a, const Bucket& b) const noexcept {
        return a.key == b.key;
    }

protected:
    Bucket*     _buckets;
};

template <typename KeyType, typename ValueType, typename Hasher>
class FastHashmapBase<KeyType, ValueType, Hasher, true, true, true>:
        public HashmapBase<Hasher>, public HashmapWithSeqMempool {
protected:
    using HashmapBase<Hasher>::_hash_size;
    using HashmapBase<Hasher>::_data_size;
    using HashmapBase<Hasher>::hash;
    using HashmapWithSeqMempool::dup;
    using HashmapWithSeqMempool::copy;
    using HashmapWithSeqMempool::str_len;
    using HashmapWithSeqMempool::str_ptr;

    struct Bucket {
        uint32_t    hash_code;
        uint32_t    key_size:10;
        uint32_t    val_size:22;
        uint32_t    key_offset;
        uint32_t    val_offset;
        void clear() { key_size = 0; }
        inline bool empty() const noexcept {
            return key_size == 0;
        }
    };

    FastHashmapBase(): _buckets(nullptr) {}
    FastHashmapBase(size_t reserve_size): HashmapBase<Hasher>(reserve_size),
            HashmapWithSeqMempool(reserve_size), _buckets(nullptr) {}
    ~FastHashmapBase() {
        if (_buckets != nullptr) {
            ::free(_buckets);
        }
    }

    using InnerKeyType = StringView;
    using InnerValType = StringView;
    using ConstInnerKeyType = const StringView;
    using ConstInnerValType = const StringView;
    inline StringView get_key(uint32_t idx) noexcept {
        auto& bkt = _buckets[idx];
        return StringView(get_ptr(bkt.key_offset), bkt.key_size);
    }
    inline StringView get_value(uint32_t idx) noexcept {
        auto& bkt = _buckets[idx];
        return StringView(get_ptr(bkt.val_offset), bkt.val_size);
    }

    bool put_slot(Bucket& bkt, const StringView& key, const StringView& val, uint32_t hash_code) {
        if (bkt.empty()) {
            auto kmem = dup(key, _data_size);
            auto vmem = dup(val, _data_size);
            if (kmem.offset == 0 || vmem.offset == 0) {
                return false;
            }
            bkt.hash_code = hash_code;
            bkt.key_size = kmem.size;
            bkt.val_size = vmem.size;
            bkt.key_offset = kmem.offset;
            bkt.val_offset = vmem.offset;
            ++_data_size;
        } else {
            auto len = str_len(val);
            if (len <= bkt.val_size) {
                copy(bkt.val_offset, len, str_ptr(val));
            } else {
                auto vmem = dup(val, _data_size);
                if (vmem.offset == 0) {
                    return false;
                }
                bkt.val_size = vmem.size;
                bkt.val_offset = vmem.offset;
            }
        }
        return true;
    }

    inline bool move_slot(Bucket& dst, Bucket& src) {
        dst = src;
        return true;
    }

    static inline uint32_t get_hash_code(Bucket& bkt) noexcept {
        return bkt.hash_code;
    }

    inline bool compare_key(
            const Bucket& bkt, const StringView& key, uint32_t hash_code) const noexcept {
        return hash_code == bkt.hash_code
                && compare_string(bkt.key_offset, bkt.key_size, key.data(), key.size());
    }
    inline bool compare_key(const Bucket& a, const Bucket& b) const noexcept {
        return a.hash_code == b.hash_code && a.key_size == b.key_size
            && compare_string(a.key_offset, b.key_offset, a.key_size);
    }

protected:
    Bucket*     _buckets;
};

template <typename KeyType, typename ValueType, typename Hasher>
class FastHashmapBase<KeyType, ValueType, Hasher, true, false, true>:
        public HashmapBase<Hasher>, public HashmapWithSeqMempool {
protected:
    using HashmapBase<Hasher>::_hash_size;
    using HashmapBase<Hasher>::_data_size;
    using HashmapBase<Hasher>::hash;
    using HashmapWithSeqMempool::dup;
    using HashmapWithSeqMempool::copy;
    using HashmapWithSeqMempool::str_len;
    using HashmapWithSeqMempool::str_ptr;

    struct Bucket {
        uint32_t    hash_code;
        uint32_t    key_size;
        uint32_t    key_offset;
        ValueType   val;
        void clear() { key_size = 0; }
        inline bool empty() const noexcept {
            return key_size == 0;
        }
    };

    FastHashmapBase(): _buckets(nullptr) {}
    FastHashmapBase(size_t reserve_size):
        HashmapBase<Hasher>(reserve_size), HashmapWithSeqMempool(reserve_size),
        _buckets(nullptr) {}
    ~FastHashmapBase() {
        if (_buckets == nullptr) {
            return;
        }
        if (!std::is_trivially_destructible<ValueType>::value) {
            for (size_t i = 0; i < this->_hash_size; ++i) {
                destruct(_buckets[i].val);
            }
        }
        ::free(_buckets);
    }

    using InnerKeyType = StringView;
    using InnerValType = const ValueType&;
    using ConstInnerKeyType = const StringView;
    using ConstInnerValType = const ValueType&;
    inline StringView get_key(uint32_t idx) noexcept {
        auto& bkt = _buckets[idx];
        return StringView(get_ptr(bkt.key_offset), bkt.key_size);
    }
    inline ValueType& get_value(uint32_t idx) noexcept {
        return _buckets[idx].val;
    }

    bool put_slot(Bucket& bkt, const StringView& key, const ValueType& val, uint32_t hash_code) {
        if (bkt.empty()) {
            auto kmem = dup(key, _data_size);
            if (kmem.offset == 0) {
                return false;
            }
            bkt.hash_code = hash_code;
            bkt.key_size = kmem.size;
            bkt.key_offset = kmem.offset;
            construct(bkt.val, val);
            ++_data_size;
        } else {
            bkt.val = val;
        }
        return true;
    }

    inline bool move_slot(Bucket& dst, Bucket& src) {
        dst.hash_code = src.hash_code;
        dst.key_size = src.key_size;
        dst.key_offset = src.key_offset;
        move_value(dst.val, std::move(src.val));
        destruct(src.val);
        return true;
    }

    static inline uint32_t get_hash_code(Bucket& bkt) noexcept {
        return bkt.hash_code;
    }

    inline bool compare_key(Bucket& bkt, const StringView& key, uint32_t hash_code) const noexcept {
        return hash_code == bkt.hash_code
                && compare_string(bkt.key_offset, bkt.key_size, key.data(), key.size());
    }
    inline bool compare_key(const Bucket& a, const Bucket& b) const noexcept {
        return a.hash_code == b.hash_code && a.key_size == b.key_size
            && compare_string(a.key_offset, b.key_offset, a.key_size);
    }

protected:
    Bucket*     _buckets;
};

template <typename KeyType, typename ValueType, typename Hasher>
class FastHashmapBase<KeyType, ValueType, Hasher, true, false, false>:
        public HashmapBase<Hasher>,
        public HashmapWithSeqMempool,
        public HashmapWithNodeArena<ValueType> {
protected:
    using HashmapBase<Hasher>::_hash_size;
    using HashmapBase<Hasher>::_data_size;
    using HashmapBase<Hasher>::hash;
    using HashmapWithNodeArena<ValueType>::alloc_node;
    using HashmapWithNodeArena<ValueType>::get_node;
    using HashmapWithSeqMempool::dup;
    using HashmapWithSeqMempool::copy;
    using HashmapWithSeqMempool::str_len;
    using HashmapWithSeqMempool::str_ptr;

    struct Bucket {
        union {
            struct {
                uint32_t    hash_code;
                uint32_t    key_size;
                uint32_t    key_offset;
            };
            char resident[12];
        };
        uint32_t    val_index;
        void clear() { key_size = 0; }
        inline bool empty() const noexcept {
            return key_size == 0;
        }
    };

    FastHashmapBase(): _buckets(nullptr) {
        std::cout << "sizeof(Bucket):" << sizeof(Bucket) << std::endl;
    }
    FastHashmapBase(size_t reserve_size):
        HashmapBase<Hasher>(reserve_size), HashmapWithSeqMempool(reserve_size),
        HashmapWithNodeArena<ValueType>(reserve_size), _buckets(nullptr) {}
    ~FastHashmapBase() {
        if (_buckets == nullptr) {
            return;
        }
        if (!std::is_trivially_destructible<ValueType>::value) {
            for (size_t i = 0; i < this->_hash_size; ++i) {
                destruct(get_node(_buckets[i].val_index));
            }
        }
        ::free(_buckets);
    }

    using InnerKeyType = StringView;
    using InnerValType = const ValueType&;
    using ConstInnerKeyType = const StringView;
    using ConstInnerValType = const ValueType&;
    inline StringView get_key(uint32_t idx) noexcept {
        auto& bkt = _buckets[idx];
        return StringView(get_ptr(bkt.key_offset), bkt.key_size);
    }
    inline ValueType& get_value(uint32_t idx) noexcept {
        return get_node(_buckets[idx].val);
    }

    inline bool put_slot(Bucket& bkt, const StringView& key, const ValueType& val, uint32_t hash_code) {
        if (bkt.empty()) {
            auto kmem = dup(key, _data_size);
            auto ni = this->alloc_node();
            if (kmem.offset == 0 || ni.first == nullptr) {
                return false;
            }

            bkt.hash_code = hash_code;
            bkt.key_size = kmem.size;
            bkt.key_offset = kmem.offset;
            bkt.val_index = ni.second;
            construct(*ni.first, val);
            ++_data_size;
        } else {
            get_node(bkt.val_index) = val;
        }
        return true;
    }

    inline bool move_slot(Bucket& dst, Bucket& src) {
        dst = src;
        return true;
    }

    static inline uint32_t get_hash_code(Bucket& bkt) noexcept {
        return bkt.hash_code;
    }

    inline bool compare_key(Bucket& bkt, const StringView& key, uint32_t hash_code) const noexcept {
        return hash_code == bkt.hash_code
                && compare_string(bkt.offset, bkt.size, key.data(), key.size());
    }
    inline bool compare_key(const Bucket& a, const Bucket& b) const noexcept {
        return a.hash_code == b.hash_code && a.key_size == b.key_size
            && compare_string(a.key_offset, b.key_offset, a.key_size);
    }

protected:
    Bucket*     _buckets;
};

template <typename T>
struct Hasher_ {
    inline uint64_t operator()(const T& v, uint64_t seed=0) noexcept {
        return std::hash<T>()(v);
    }
};

template <>
struct Hasher_<std::string> {
    inline uint64_t operator()(const std::string& v, uint64_t seed=0) noexcept {
        return murmurhash64(v.data(), v.size(), seed);
    }
};
template <>
struct Hasher_<StringView> {
    inline uint64_t operator()(const StringView& v, uint64_t seed=0) noexcept {
        return murmurhash64(v.data(), v.size(), seed);
    }
};
template <>
struct Hasher_<uint64_t> {
    inline uint64_t operator()(uint64_t v, uint64_t seed=0) noexcept {
        return ((v+1) * 1048583u) + (v>>35);
    }
};
template <>
struct Hasher_<int64_t> {
    inline uint64_t operator()(int64_t v, uint64_t seed=0) noexcept {
        return Hasher_<uint64_t>()(static_cast<uint64_t>(v), seed);
    }
};
template <>
struct Hasher_<uint32_t> {
    inline uint64_t operator()(uint32_t v, uint64_t seed=0) noexcept {
        return Hasher_<uint64_t>()(static_cast<uint64_t>(v), seed);
    }
};
template <>
struct Hasher_<int> {
    inline uint64_t operator()(int v, uint64_t seed=0) noexcept {
        return Hasher_<uint64_t>()(static_cast<uint64_t>(v), seed);
    }
};

struct Hasher {
    template <typename T>
    inline uint64_t operator()(const T& v, uint64_t seed=0) noexcept {
        return Hasher_<T>()(v, seed);
    }
};

template <typename KeyType, typename ValueType, typename Hasher=Hasher>
class FastHashmap final:
        public FastHashmapBase<KeyType, ValueType, Hasher,
            is_string_type<KeyType>::value, is_string_class<ValueType>::value,
            is_string_class<ValueType>::value || sizeof(ValueType) <= 16> {
    using Base = FastHashmapBase<KeyType, ValueType, Hasher,
            is_string_type<KeyType>::value, is_string_class<ValueType>::value,
            is_string_class<ValueType>::value || sizeof(ValueType) <= 16>;
    using InnerKeyType = typename Base::InnerKeyType;
    using InnerValType = typename Base::InnerValType;
    using ConstInnerKeyType = typename Base::ConstInnerKeyType;
    using ConstInnerValType = typename Base::ConstInnerValType;
    using Bucket = typename Base::Bucket;
    using Base::_buckets;
    using Base::_hash_size;
    using Base::_data_size;
    using Base::put_slot;
    using Base::move_slot;
    using Base::compare_key;
    using Base::get_hash_code;
    using Base::hash;
public:
    using key_type = KeyType;
    using value_type = ValueType;
    using iterator = HashmapIterator<FastHashmap<KeyType, ValueType, Hasher>>;
    friend class HashmapIterator<FastHashmap<KeyType, ValueType, Hasher>>;

    FastHashmap() {}
    FastHashmap(size_t max_data_size): Base(max_data_size) {}
    ~FastHashmap() {}

    bool insert(ConstInnerKeyType const key, ConstInnerValType val) {
        if (need_rehash() && !rehash(get_rehash_size())) {
            return false;
        }
        auto hash_code = hash(key);
        auto idx = find_insertion_slot(key, hash_code);
        return idx < _hash_size && put_slot(_buckets[idx], key, val, hash_code);
    }

    iterator find(ConstInnerKeyType key) {
        if (_buckets == nullptr || _data_size == 0) {
            return end();
        }
        auto hash_code = hash(key);
        auto idx = get_hash_index(hash_code);
        for (uint32_t i = 0; i < this->_hash_size; ++i) {
            if (_buckets[idx].empty()) {
                return end();
            }
            if (compare_key(_buckets[idx], key, hash_code)) {
                return iterator(this, idx);
            }
            if ((++idx) >= this->_hash_size) {
                idx = 0;
            }
        }
        return end();
    }

    inline iterator begin() const noexcept {
        return iterator(this, next(0));
    }
    inline iterator end() const noexcept {
        return iterator(this, this->_hash_size);
    }

protected:
    uint32_t next(uint32_t idx) const noexcept {
        for (; idx < this->_hash_size && _buckets[idx].empty(); ++idx) {}
        return idx;
    }

    static inline uint32_t get_hash_index(uint32_t hash_code, uint32_t hash_size) noexcept {
        return hash_code % hash_size;
    }
    inline uint32_t get_hash_index(uint32_t hash_code) const noexcept {
        return get_hash_index(hash_code, _hash_size);
    }

    inline bool need_rehash() const noexcept {
        return _buckets == nullptr || _hash_size <= (_data_size<<1);
    }
    inline uint32_t get_rehash_size() const noexcept {
        if (_buckets == nullptr && _hash_size > 0) {
            return _hash_size;
        }
        return static_cast<uint32_t>((_data_size+11)*3.3);
    }

private:
    bool rehash(uint32_t hash_size) {
        if (hash_size == _hash_size && _buckets != nullptr) {
            return true;
        }

        Bucket* buckets = (Bucket*)aligned_alloc(32, sizeof(Bucket)*hash_size);
        if (buckets == nullptr) {
            return false;
        }
        for (uint32_t i = 0; i < hash_size; ++i) {
            buckets[i].clear();
        }
        if (_data_size > 0) {
            for (uint32_t i = 0; i < _hash_size; ++i) {
                if (!buckets[i].empty()) {
                    auto j = find_empty_slot(buckets, hash_size, get_hash_code(_buckets[i]));
                    move_slot(buckets[j], _buckets[i]);
                }
            }
        }
        if (_buckets != nullptr) {
            ::free(_buckets);
        }
        _buckets = buckets;
        _hash_size = hash_size;
        return true;
    }
 
    inline uint32_t find_insertion_slot(ConstInnerKeyType& key, uint32_t hash_code) const noexcept {
        auto idx = hash_code % _hash_size;
        for (uint32_t i = 0; i < _hash_size; ++i) {
            if (_buckets[idx].empty() || compare_key(_buckets[idx], key, hash_code)) {
                return idx;
            }
            if ((++idx) >= _hash_size) {
                idx = 0;
            }
        }
        return _hash_size;
    }
    static inline uint32_t find_empty_slot(
            const Bucket* buckets, size_t hash_size, uint32_t hash_code) noexcept {
        auto idx = get_hash_index(hash_code, hash_size);
        for (uint32_t i = 0; i < hash_size && !buckets[idx].empty(); ++i) {
            if ((++idx) >= hash_size) {
                idx = 0;
            }
        }
        return idx;
    }
};

} // namespace ftl


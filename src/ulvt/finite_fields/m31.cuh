#pragma once
#include <cstdint>
#include <iostream>
#include <string>

class M31 {
public:
    uint32_t val;

    static constexpr uint32_t BITS = 31;

    static constexpr uint32_t P = ((uint32_t) 1<<BITS) - 1;

    __host__ __device__ constexpr M31() noexcept: val(0) {}
    // Assumes val is in the range [0, P)
    __host__ __device__ constexpr M31(uint32_t val) noexcept: val(val) {}

    // Assumes val is in the range [0, P^2)
    __host__ __device__ constexpr M31(uint64_t val) noexcept: val(
        ((((val >> BITS) + val + 1) >> BITS) + val) & P
    ) {}

    __host__ __device__ constexpr M31 operator+(M31 rhs) const { 
        uint32_t sum = val + rhs.val;
        uint32_t carry = sum >> BITS;
        return M31((sum + carry) & P);
     }

     __host__ __device__ constexpr M31& operator+=(M31 rhs) {
        uint32_t sum = val + rhs.val;
        uint32_t carry = sum >> BITS;
        val = (sum + carry) & P;
        return *this;
     }

    __host__ __device__ constexpr M31 operator-(M31 rhs) const { 
        uint32_t diff = val - rhs.val;
        uint32_t carry = diff >> BITS;
        return M31((diff - carry) & P);
     }

     __host__ __device__ constexpr M31& operator-=(M31 rhs) {
        uint32_t diff = val - rhs.val;
        uint32_t carry = diff >> BITS;
        val = (diff - carry) & P;
        return *this;
     }
      
    __host__ __device__ constexpr M31 operator*(M31 rhs) const { 
        return M31( (uint64_t) val * (uint64_t)rhs.val);
     }

     __host__ __device__ constexpr M31& operator*=(M31 rhs) {
        *this = *this * rhs;
        return *this;
     }

     __host__ __device__ constexpr bool operator==(M31 rhs) const { 
        return val == rhs.val;
     }

    __host__ __device__ constexpr bool operator!=(M31 rhs) const { 
        return val != rhs.val;
     }

     __host__ __device__ void write_to_u64(uint64_t* dst) const {
        *dst = (uint64_t) val;
     }

     __host__ __device__ void sum_into_u64(uint64_t* dst) const {
        *dst += (uint64_t) val;
     }

     __host__ std::string to_string() const {
        return std::to_string(val);
     }
};

class CM31 {
public:
    static constexpr uint32_t BITS = 31;

    static constexpr uint32_t P = ((uint32_t) 1<<BITS) - 1;

    M31 subfield_elements[2];

    __host__ __device__ constexpr CM31() noexcept: subfield_elements{M31(), M31()} {}

    __host__ __device__ constexpr CM31(uint32_t val) noexcept: subfield_elements{M31(val), M31()} {}

    __host__ __device__ constexpr CM31(uint64_t val[2]) noexcept: subfield_elements{M31(val[0]), M31(val[1])} {}

    __host__ __device__ constexpr CM31(M31 lo, M31 hi) noexcept: subfield_elements{lo, hi} {}

    __host__ __device__ constexpr CM31 operator+(CM31 rhs) const {
        return CM31(
            subfield_elements[0] + rhs.subfield_elements[0],
            subfield_elements[1] + rhs.subfield_elements[1]
        );
     }

    __host__ __device__ constexpr CM31 operator-(CM31 rhs) const {
        return CM31(
            subfield_elements[0] - rhs.subfield_elements[0],
            subfield_elements[1] - rhs.subfield_elements[1]
        );
     }

    __host__ __device__ constexpr CM31& operator+=(CM31 rhs) {
        subfield_elements[0] += rhs.subfield_elements[0];
        subfield_elements[1] += rhs.subfield_elements[1];
        return *this;
     }

     __host__ __device__ constexpr CM31& operator-=(CM31 rhs) {
        subfield_elements[0] -= rhs.subfield_elements[0];
        subfield_elements[1] -= rhs.subfield_elements[1];
        return *this;
     }

    __host__ __device__ constexpr CM31 operator*(CM31 rhs) const {
        return CM31(
            subfield_elements[0] * rhs.subfield_elements[0] - subfield_elements[1] * rhs.subfield_elements[1],
            subfield_elements[0] * rhs.subfield_elements[1] + subfield_elements[1] * rhs.subfield_elements[0]
        );
     }

    __host__ __device__ constexpr CM31& operator*=(CM31 rhs) {
        *this = *this * rhs;
        return *this;
     }

    __host__ __device__ constexpr bool operator==(CM31 rhs) const {
        return subfield_elements[0] == rhs.subfield_elements[0] && subfield_elements[1] == rhs.subfield_elements[1];
     }

    __host__ __device__ constexpr bool operator!=(CM31 rhs) const {
        return subfield_elements[0] != rhs.subfield_elements[0] || subfield_elements[1] != rhs.subfield_elements[1];
     }

     __host__ __device__ void write_to_u64(uint64_t dst[2]) const {
        subfield_elements[0].write_to_u64(&dst[0]);
        subfield_elements[1].write_to_u64(&dst[1]);
     }

     __host__ __device__ void sum_into_u64(uint64_t dst[2]) const {
        subfield_elements[0].sum_into_u64(&dst[0]);
        subfield_elements[1].sum_into_u64(&dst[1]);
     }

     __host__ std::string to_string() const {
        return "(" + subfield_elements[0].to_string() + ", " + subfield_elements[1].to_string() + ")";
     }
};

__device__ constexpr CM31 R = CM31(M31((uint32_t)2), M31((uint32_t)1));

class QM31 {
public:
    static constexpr uint32_t BITS = 31;

    static constexpr uint32_t P = ((uint32_t) 1<<BITS) - 1;

    CM31 subfield_elements[2];

    __host__ __device__ constexpr QM31(CM31 lo, CM31 hi) noexcept: subfield_elements{lo, hi} {}

    __host__ __device__ constexpr QM31() noexcept: subfield_elements{CM31(), CM31()} {}

    __host__ __device__ constexpr QM31(uint32_t val) noexcept: subfield_elements{CM31(val), CM31()} {}

    __host__ __device__ constexpr QM31(uint64_t val[4]) noexcept: subfield_elements{CM31(val), CM31(val+2)} {}

    __host__ __device__ constexpr QM31 operator+(QM31 rhs) const {
        return QM31(
            subfield_elements[0] + rhs.subfield_elements[0],
            subfield_elements[1] + rhs.subfield_elements[1]
        );
     }

    __host__ __device__ constexpr QM31 operator-(QM31 rhs) const {
        return QM31(
            subfield_elements[0] - rhs.subfield_elements[0],
            subfield_elements[1] - rhs.subfield_elements[1]
        );
     }

    __host__ __device__ constexpr QM31 operator*(QM31 rhs) const {
        return QM31(
            subfield_elements[0] * rhs.subfield_elements[0] + R * subfield_elements[1] * rhs.subfield_elements[1],
            subfield_elements[0] * rhs.subfield_elements[1] + subfield_elements[1] * rhs.subfield_elements[0]
        );
     }

    __host__ __device__ constexpr bool operator==(QM31 rhs) const {
        return subfield_elements[0] == rhs.subfield_elements[0] && subfield_elements[1] == rhs.subfield_elements[1];
     }

     __host__ __device__ constexpr bool operator!=(QM31 rhs) const {
        return subfield_elements[0] != rhs.subfield_elements[0] || subfield_elements[1] != rhs.subfield_elements[1];
     }

     __host__ __device__ constexpr QM31& operator+=(QM31 rhs) {
        subfield_elements[0] += rhs.subfield_elements[0];
        subfield_elements[1] += rhs.subfield_elements[1];
        return *this;
     }

     __host__ __device__ constexpr QM31& operator-=(QM31 rhs) {
        subfield_elements[0] -= rhs.subfield_elements[0];
        subfield_elements[1] -= rhs.subfield_elements[1];
        return *this;
     }

     __host__ __device__ constexpr QM31& operator*=(QM31 rhs) {
        *this = *this * rhs;
        return *this;
     }

     __host__ __device__ void write_to_u64(uint64_t dst[4]) const {
        subfield_elements[0].write_to_u64(&dst[0]);
        subfield_elements[1].write_to_u64(&dst[2]);
     }

     __host__ __device__ void sum_into_u64(uint64_t dst[4]) const {
        subfield_elements[0].sum_into_u64(&dst[0]);
        subfield_elements[1].sum_into_u64(&dst[2]);
     }

      __host__ std::string to_string() const {
        return "(" + subfield_elements[0].to_string() + ", " + subfield_elements[1].to_string() + ")";
     }
};
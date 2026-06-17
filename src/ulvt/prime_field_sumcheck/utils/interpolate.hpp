#include "../../finite_fields/m31.cuh"

constexpr QM31 one_half = (uint32_t) 0x40000000;

inline QM31 interpolate_at(QM31 challenge, QM31 evals[3]) {
  return (challenge * (challenge - 1) * evals[2] * one_half) -
         (challenge * (challenge - 2) * evals[1]) +
         ((challenge - 1) * (challenge - 2) * evals[0] * one_half);
}
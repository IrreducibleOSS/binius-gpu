#include "../finite_fields/qm31.cuh"
#include "./utils/interpolate.hpp"
#include "sumcheck.cuh"
#include <array>
#include <iostream>
#include <vector>


int main() {

  QM31 points[3] = {(uint32_t) 4, (uint32_t) 4, (uint32_t) 4};
  QM31 result = interpolate_at((uint32_t) 7, points);
  std::cout << "result" << result.to_string() << std::endl;

  constexpr uint32_t NUM_VARS = 24;

  QM31 expected_claim = QM31(1 << (NUM_VARS -1)) * QM31((1 << NUM_VARS) - 1);

  std::cout << "expected claim" << expected_claim.to_string() << std::endl;
  std::vector<QM31> evals;
  for (std::size_t i = 0; i < 1 << NUM_VARS; ++i) {
    evals.push_back(QM31(i));
  }

  for (std::size_t i = 0; i < 1 << NUM_VARS; ++i) {
    evals.push_back(QM31((uint32_t)1));
  }

  Sumcheck<NUM_VARS> sumcheck(evals, true);

  for (std::size_t i = 0; i < 4; ++i) {
    std::array<QM31, 3> this_round_points;
    uint64_t a[4] = {32482843, 85864538, 8348234, 9544334};
    QM31 challenge = QM31(a);

    if (i == 0){
      sumcheck.first_round_messages<2048, 32>(this_round_points);
    } else {
      sumcheck.subsequent_round_messages<2048, 32>(this_round_points, challenge);
    }

    QM31 this_round_claim = this_round_points[0] + this_round_points[1];

    std::cout << "this round claim" << this_round_claim.to_string()
              << std::endl;

    // std::cout << this_round_points[0].to_string() << std::endl;
    // std::cout << this_round_points[1].to_string() << std::endl;
    // std::cout << this_round_points[2].to_string() << std::endl;

    

    QM31 next_round_claim = interpolate_at(challenge, this_round_points.data());

    std::cout << "next round claim" << next_round_claim.to_string()
              << std::endl;
  }

  std::cout << (std::chrono::high_resolution_clock::now() - sumcheck.start_raw).count() << std::endl;

  return 0;
}
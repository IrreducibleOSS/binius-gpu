#include "../finite_fields/m31.cuh"
#include "./utils/interpolate.hpp"
#include "sumcheck.cuh"
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <vector>

TEST_CASE("Prime Field Sumcheck Test", "[prime_field]") {
  QM31 points[3] = {(uint32_t)4, (uint32_t)4, (uint32_t)4};
  QM31 result = interpolate_at((uint32_t)7, points);

  constexpr uint32_t NUM_VARS = 24;

  QM31 expected_claim = (uint32_t)0;

  std::vector<QM31> evals;
  for (std::size_t i = 0; i < 1 << NUM_VARS; ++i) {
    evals.push_back(QM31(i));
  }

  for (std::size_t i = 0; i < 1 << NUM_VARS; ++i) {
    evals.push_back(QM31(i));
  }

  for (std::size_t i = 0; i < 1 << NUM_VARS; ++i) {
    expected_claim += evals[i] * evals[i + (1 << NUM_VARS)];
  }

  Sumcheck<NUM_VARS> sumcheck(evals, false);

  for (std::size_t i = 0; i < NUM_VARS; ++i) {
    std::array<QM31, 3> this_round_points;

    if (NUM_VARS - i > 16) {
      sumcheck.this_round_messages<2048, 32>(this_round_points);
    } else if (NUM_VARS - i > 15) {
      sumcheck.this_round_messages<1024, 32>(this_round_points);
    } else if (NUM_VARS - i > 14) {
      sumcheck.this_round_messages<512, 32>(this_round_points);
    } else if (NUM_VARS - i > 13) {
      sumcheck.this_round_messages<256, 32>(this_round_points);
    } else if (NUM_VARS - i > 12) {
      sumcheck.this_round_messages<128, 32>(this_round_points);
    } else if (NUM_VARS - i > 11) {
      sumcheck.this_round_messages<64, 32>(this_round_points);
    } else if (NUM_VARS - i > 10) {
      sumcheck.this_round_messages<32, 32>(this_round_points);
    } else if (NUM_VARS - i > 9) {
      sumcheck.this_round_messages<16, 32>(this_round_points);
    } else if (NUM_VARS - i > 8) {
      sumcheck.this_round_messages<8, 32>(this_round_points);
    } else if (NUM_VARS - i > 7) {
      sumcheck.this_round_messages<4, 32>(this_round_points);
    } else if (NUM_VARS - i > 6) {
      sumcheck.this_round_messages<2, 32>(this_round_points);
    } else if (NUM_VARS - i > 5) {
      sumcheck.this_round_messages<1, 32>(this_round_points);
    } else {
      sumcheck.this_round_messages<1, 1>(this_round_points);
    }

    QM31 this_round_claim = this_round_points[0] + this_round_points[1];

    REQUIRE(this_round_claim == expected_claim);

    uint64_t a[4] = {32482843, 85864538, 8348234, 9544334};
    QM31 challenge = QM31(a);

    expected_claim = interpolate_at(challenge, this_round_points.data());

    if (NUM_VARS - i > 16) {
      sumcheck.fold<2048, 32>(challenge);
    } else if (NUM_VARS - i > 15) {
      sumcheck.fold<1024, 32>(challenge);
    } else if (NUM_VARS - i > 14) {
      sumcheck.fold<512, 32>(challenge);
    } else if (NUM_VARS - i > 13) {
      sumcheck.fold<256, 32>(challenge);
    } else if (NUM_VARS - i > 12) {
      sumcheck.fold<128, 32>(challenge);
    } else if (NUM_VARS - i > 11) {
      sumcheck.fold<64, 32>(challenge);
    } else if (NUM_VARS - i > 10) {
      sumcheck.fold<32, 32>(challenge);
    } else if (NUM_VARS - i > 9) {
      sumcheck.fold<16, 32>(challenge);
    } else if (NUM_VARS - i > 8) {
      sumcheck.fold<8, 32>(challenge);
    } else if (NUM_VARS - i > 7) {
      sumcheck.fold<4, 32>(challenge);
    } else if (NUM_VARS - i > 6) {
      sumcheck.fold<2, 32>(challenge);
    } else if (NUM_VARS - i > 5) {
      sumcheck.fold<1, 32>(challenge);
    } else {
      sumcheck.fold<1, 1>(challenge);
    }
  }
}
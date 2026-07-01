#include "g1_cpp/sim2real_controller.hpp"

int main(int argc, char** argv) {
  return g1::run_sim2real_main(argc, argv, g1::RealPolicyKind::Walk, "g1_walk.yaml", "g1_flat_1.onnx");
}

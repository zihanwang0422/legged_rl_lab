#pragma once

#include <filesystem>
#include <string>

namespace g1 {

enum class RealPolicyKind {
  Walk,
  Amp,
  Mimic,
};

struct RealRunOptions {
  RealPolicyKind kind = RealPolicyKind::Walk;
  std::string net = "enp108s0";
  int domain_id = 0;
  std::string config_name;
  std::string model_name;
  std::string flat_config_name = "g1_walk.yaml";
  std::string flat_model_name = "g1_flat_1.onnx";
  bool debug_policy = false;
};

int run_sim2real_main(int argc, char** argv,
                      RealPolicyKind kind,
                      const char* default_config,
                      const char* default_model);

}  // namespace g1

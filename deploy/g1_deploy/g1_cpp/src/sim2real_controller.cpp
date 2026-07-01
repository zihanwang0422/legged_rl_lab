#include "g1_cpp/sim2real_controller.hpp"

#include "g1_cpp/config.hpp"
#include "g1_cpp/math_utils.hpp"
#include "g1_cpp/onnx_policy.hpp"

#include <unitree/idl/go2/SportModeState_.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace g1 {
namespace {

using LowCmdHg = unitree_hg::msg::dds_::LowCmd_;
using LowStateHg = unitree_hg::msg::dds_::LowState_;
using SportModeState = unitree_go::msg::dds_::SportModeState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;
using unitree::robot::ChannelSubscriber;

struct Button {
  bool pressed = false;
  bool on_press = false;

  void update(bool state) {
    on_press = state && !pressed;
    pressed = state;
  }
};

struct Gamepad {
  Button start;
  Button select;
  Button a;
  Button b;
  float lx = 0.0f;
  float ly = 0.0f;
  float rx = 0.0f;
};

#pragma pack(push, 1)
struct RemoteButtons {
  uint8_t r1 : 1;
  uint8_t l1 : 1;
  uint8_t start : 1;
  uint8_t select : 1;
  uint8_t r2 : 1;
  uint8_t l2 : 1;
  uint8_t f1 : 1;
  uint8_t f2 : 1;
  uint8_t a : 1;
  uint8_t b : 1;
  uint8_t x : 1;
  uint8_t y : 1;
  uint8_t up : 1;
  uint8_t right : 1;
  uint8_t down : 1;
  uint8_t left : 1;
};

struct RemotePacket {
  uint8_t head[2];
  RemoteButtons btn;
  float lx;
  float rx;
  float ry;
  float l2;
  float ly;
  uint8_t idle[16];
};
#pragma pack(pop)

uint32_t crc32_core(uint32_t* ptr, uint32_t len) {
  uint32_t xbit = 0;
  uint32_t data = 0;
  uint32_t crc = 0xFFFFFFFF;
  constexpr uint32_t poly = 0x04c11db7;
  for (uint32_t i = 0; i < len; ++i) {
    xbit = 1u << 31;
    data = ptr[i];
    for (uint32_t bits = 0; bits < 32; ++bits) {
      if (crc & 0x80000000) {
        crc <<= 1;
        crc ^= poly;
      } else {
        crc <<= 1;
      }
      if (data & xbit) crc ^= poly;
      xbit >>= 1;
    }
  }
  return crc;
}

double wall_time() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

fs::path deploy_dir() {
  return fs::path(G1_CPP_SOURCE_DIR).parent_path();
}

fs::path resolve_config_path(const fs::path& g1_dir, const std::string& name) {
  fs::path p(name);
  if (p.is_absolute() || fs::exists(p)) return p;
  if (!p.empty() && *p.begin() == "config") return g1_dir / p;
  return g1_dir / "config" / p;
}

fs::path resolve_policy_path(const fs::path& g1_dir, const std::string& policy_path, const std::string& model_name) {
  fs::path m(model_name);
  if (m.is_absolute() || fs::exists(m)) return m;
  fs::path base = policy_path.empty() ? fs::path("exported_policy") : fs::path(policy_path).parent_path();
  if (base.empty() || base == ".") base = "exported_policy";
  return g1_dir / base / m;
}

template <typename T>
std::vector<T> reorder(const std::vector<T>& v, const std::vector<int>& map) {
  std::vector<T> out(map.size());
  for (size_t i = 0; i < map.size(); ++i) out[i] = v.at(static_cast<size_t>(map[i]));
  return out;
}

std::vector<float> subtract(const std::vector<float>& a, const std::vector<float>& b) {
  std::vector<float> out(a.size());
  for (size_t i = 0; i < a.size(); ++i) out[i] = a[i] - b[i];
  return out;
}

void append(std::vector<float>& dst, const std::vector<float>& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

void append3(std::vector<float>& dst, const Vec3& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}

QuatWxyz quat_inv_wxyz(const QuatWxyz& q) {
  return {q[0], -q[1], -q[2], -q[3]};
}

QuatWxyz quat_mul_wxyz(const QuatWxyz& a, const QuatWxyz& b) {
  return {
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  };
}

Vec3 quat_apply_wxyz(const QuatWxyz& q, const Vec3& v) {
  return quat_apply_xyzw({q[1], q[2], q[3], q[0]}, v);
}

std::array<float, 9> matrix_from_quat_wxyz(const QuatWxyz& q) {
  float w = q[0], x = q[1], y = q[2], z = q[3];
  return {
      1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y - z * w), 2.0f * (x * z + y * w),
      2.0f * (x * y + z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z - x * w),
      2.0f * (x * z - y * w), 2.0f * (y * z + x * w), 1.0f - 2.0f * (x * x + y * y),
  };
}

Range cfg_range(const Config& cfg, const std::string& key, Range fallback) {
  auto it = cfg.command_range.find(key);
  return it == cfg.command_range.end() ? fallback : it->second;
}

class Sim2RealController {
 public:
  explicit Sim2RealController(RealRunOptions options)
      : options_(std::move(options)), g1_dir_(deploy_dir()) {
    is_local_sim_ = options_.net == "lo" || options_.domain_id != 0;
    flat_cfg_ = load_config(resolve_config_path(g1_dir_, options_.flat_config_name), g1_dir_);
    if (options_.kind == RealPolicyKind::Mimic) {
      cfg_ = flat_cfg_;
      main_cfg_path_ = resolve_config_path(g1_dir_, options_.config_name);
      main_model_name_ = options_.model_name;
      load_policy(flat_cfg_, options_.flat_model_name);
      active_mimic_policy_ = false;
    } else {
      cfg_ = load_config(resolve_config_path(g1_dir_, options_.config_name), g1_dir_);
      load_policy(cfg_, options_.model_name);
    }
    configure_policy_state();
    connect_dds();
  }

  ~Sim2RealController() {
    running_ = false;
    if (control_thread_.joinable()) control_thread_.join();
    if (publish_thread_.joinable()) publish_thread_.join();
  }

  void run() {
    wait_for_low_state();
    wait_for_start();
    move_to_default_pos();
    wait_for_control();
    std::cout << "Start Control!\n";
    control_start_time_ = wall_time();
    running_ = true;
    publish_thread_ = std::thread([this] { publish_loop(); });
    control_thread_ = std::thread([this] { control_loop(); });

    if (options_.kind == RealPolicyKind::Mimic) {
      std::cout << "C++ SDK2 mimic starts in flat stabilize policy. Press B to enter mimic / tracking mode; press A to return flat.\n";
    }
    while (running_) {
      if (gamepad_.select.pressed) {
        std::cout << "Select Button detected, Exit!\n";
        running_ = false;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    send_damping();
  }

 private:
  void connect_dds() {
    if (cfg_.msg_type != "hg") {
      throw std::runtime_error("C++ sim2real currently supports msg_type=hg only.");
    }
    ChannelFactory::Instance()->Init(options_.domain_id, options_.net);
    lowcmd_pub_ = std::make_unique<ChannelPublisher<LowCmdHg>>(cfg_.lowcmd_topic);
    lowcmd_pub_->InitChannel();
    lowstate_sub_ = std::make_unique<ChannelSubscriber<LowStateHg>>(cfg_.lowstate_topic);
    lowstate_sub_->InitChannel([this](const void* msg) { low_state_handler(msg); }, 10);
    if (options_.kind == RealPolicyKind::Mimic) {
      sport_sub_ = std::make_unique<ChannelSubscriber<SportModeState>>("rt/sportmodestate");
      sport_sub_->InitChannel([this](const void* msg) { sport_state_handler(msg); }, 10);
    }
  }

  void low_state_handler(const void* message) {
    LowStateHg state = *static_cast<const LowStateHg*>(message);
    uint32_t expected_crc = crc32_core(reinterpret_cast<uint32_t*>(&state), (sizeof(LowStateHg) >> 2) - 1);
    if (state.crc() != expected_crc) {
      if (!is_local_sim_) return;
      if (!crc_warned_) {
        std::cout << "[WARN] LowState CRC mismatch in local sim; accepting bridge packets without CRC check.\n";
        crc_warned_ = true;
      }
    }
    std::lock_guard<std::mutex> lock(state_mutex_);
    low_state_ = state;
    low_state_received_ = true;
    mode_machine_ = state.mode_machine();
    RemotePacket packet{};
    std::memcpy(&packet, state.wireless_remote().data(), std::min(sizeof(packet), state.wireless_remote().size()));
    gamepad_.start.update(packet.btn.start);
    gamepad_.select.update(packet.btn.select);
    gamepad_.a.update(packet.btn.a);
    gamepad_.b.update(packet.btn.b);
    gamepad_.lx = std::fabs(packet.lx) < 0.01f ? 0.0f : packet.lx;
    gamepad_.ly = std::fabs(packet.ly) < 0.01f ? 0.0f : packet.ly;
    gamepad_.rx = std::fabs(packet.rx) < 0.01f ? 0.0f : packet.rx;
  }

  void sport_state_handler(const void* message) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    sport_state_ = *static_cast<const SportModeState*>(message);
    sport_state_received_ = true;
  }

  LowStateHg low_state_copy() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return low_state_;
  }

  SportModeState sport_state_copy(bool* received) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    *received = sport_state_received_;
    return sport_state_;
  }

  void wait_for_low_state() {
    double last_print = 0.0;
    while (!low_state_received_) {
      double now = wall_time();
      if (now - last_print > 1.0) {
        std::cout << "Waiting for LowState on " << cfg_.lowstate_topic
                  << " (net=" << options_.net << ", domain_id=" << options_.domain_id << ")...\n";
        last_print = now;
      }
      std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.control_dt));
    }
    std::cout << "Successfully connected to the robot.\n";
  }

  void wait_for_start() {
    std::cout << "Enter zero torque state.\nWaiting for the start signal to move to default pos...\n";
    while (!gamepad_.start.pressed) {
      if (gamepad_.select.pressed) throw std::runtime_error("Exit requested before start.");
      std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.control_dt));
    }
  }

  void wait_for_control() {
    std::cout << "Enter default pos state.\nWaiting for the Button A signal to Start Control...\n";
    while (!gamepad_.a.pressed) {
      if (gamepad_.select.pressed) throw std::runtime_error("Exit requested before control.");
      std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.control_dt));
    }
  }

  void move_to_default_pos() {
    std::cout << "Moving to default pos.\n";
    LowStateHg state = low_state_copy();
    std::vector<float> init(cfg_.num_actions, 0.0f);
    for (int i = 0; i < cfg_.num_actions; ++i) init[i] = state.motor_state().at(cfg_.sdk2isaac_idx.at(i)).q();
    int steps = std::max(1, static_cast<int>(2.0 / cfg_.control_dt));
    for (int step = 0; step < steps; ++step) {
      float a = static_cast<float>(step) / static_cast<float>(steps);
      std::lock_guard<std::mutex> lock(cmd_mutex_);
      for (int i = 0; i < cfg_.num_actions; ++i) {
        int motor = cfg_.sdk2isaac_idx.at(i);
        low_cmd_.motor_cmd().at(motor).mode() = 1;
        low_cmd_.motor_cmd().at(motor).q() = init[i] * (1.0f - a) + cfg_.default_joint_pos.at(i) * a;
        low_cmd_.motor_cmd().at(motor).dq() = 0.0f;
        low_cmd_.motor_cmd().at(motor).kp() = cfg_.kps.at(i);
        low_cmd_.motor_cmd().at(motor).kd() = cfg_.kds.at(i);
        low_cmd_.motor_cmd().at(motor).tau() = 0.0f;
      }
      std::this_thread::sleep_for(std::chrono::duration<double>(cfg_.control_dt));
    }
  }

  void publish_loop() {
    while (running_) {
      {
        std::lock_guard<std::mutex> lock(cmd_mutex_);
        low_cmd_.mode_pr() = 0;
        low_cmd_.mode_machine() = mode_machine_;
        low_cmd_.crc() = crc32_core(reinterpret_cast<uint32_t*>(&low_cmd_), (sizeof(LowCmdHg) >> 2) - 1);
        lowcmd_pub_->Write(low_cmd_);
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  }

  void control_loop() {
    double start = wall_time();
    int tick = 0;
    while (running_) {
      try {
        control_step();
      } catch (const std::exception& e) {
        std::cerr << "[ERROR] control_step: " << e.what() << "\n";
      }
      ++tick;
      double expected = start + tick * cfg_.control_dt;
      double sleep_s = expected - wall_time();
      if (sleep_s > 0.0) std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
    }
  }

  void control_step() {
    if (options_.kind == RealPolicyKind::Mimic) update_mimic_switch();

    LowStateHg state = low_state_copy();
    std::vector<float> q(cfg_.num_actions, 0.0f), dq(cfg_.num_actions, 0.0f);
    for (int i = 0; i < cfg_.num_actions; ++i) {
      int motor = cfg_.sdk2isaac_idx.at(i);
      q[i] = state.motor_state().at(motor).q();
      dq[i] = state.motor_state().at(motor).dq();
    }
    auto q_rel = subtract(q, cfg_.default_joint_pos);
    const auto& quat_arr = state.imu_state().quaternion();
    QuatWxyz quat{quat_arr[0], quat_arr[1], quat_arr[2], quat_arr[3]};
    const auto& gyro = state.imu_state().gyroscope();
    Vec3 base_ang{gyro[0] * cfg_.ang_vel_scale, gyro[1] * cfg_.ang_vel_scale, gyro[2] * cfg_.ang_vel_scale};

    std::vector<float> obs;
    if (options_.kind == RealPolicyKind::Mimic && active_mimic_policy_) {
      obs = build_mimic_obs(state, q_rel, dq, quat, base_ang);
    } else {
      VelocityCommand command = command_from_gamepad();
      Vec3 gravity = quat_apply_wxyz(quat_inv_wxyz(quat), {0.0f, 0.0f, -1.0f});
      std::vector<float> dq_scaled = dq;
      for (float& v : q_rel) v *= cfg_.dof_pos_scale;
      for (float& v : dq_scaled) v *= cfg_.dof_vel_scale;
      auto frame = build_proprio_obs(base_ang, gravity, command, q_rel, dq_scaled, last_action_,
                                     1.0f, 1.0f, 1.0f, cfg_.last_action_scale, cfg_.command_scale);
      obs_history_.erase(obs_history_.begin());
      obs_history_.push_back(frame);
      obs = stack_history_group_major(obs_history_, cfg_.num_actions);
    }

    std::vector<float> action;
    if (options_.kind == RealPolicyKind::Mimic && active_mimic_policy_) {
      auto outs = policy_->run_named(obs, static_cast<float>(time_step_));
      action = tensor_to_vector(outs.at(0));
      ref_joint_pos_ = tensor_to_vector(outs.at(1));
      ref_joint_vel_ = tensor_to_vector(outs.at(2));
      ref_body_pos_w_ = tensor_to_vector(outs.at(3));
      ref_body_quat_w_ = tensor_to_vector(outs.at(4));
      ++time_step_;
      if (cfg_.motion_total_steps > 0 && time_step_ >= cfg_.motion_total_steps) time_step_ = 0;
    } else {
      action = policy_->run_single(obs);
    }
    apply_action(action);
    if (options_.debug_policy && wall_time() - last_debug_time_ > 0.5) {
      std::cout << "[PolicyDebug/CPP] obs=" << obs.size()
                << " action=[" << *std::min_element(action.begin(), action.end())
                << "," << *std::max_element(action.begin(), action.end()) << "]"
                << " mode=" << (active_mimic_policy_ ? "mimic" : "flat") << "\n";
      last_debug_time_ = wall_time();
    }
  }

  VelocityCommand command_from_gamepad() {
    Range vx = cfg_range(cfg_, "lin_vel_x", {-1.0f, 1.0f});
    Range vy = cfg_range(cfg_, "lin_vel_y", {-0.5f, 0.5f});
    Range wz = cfg_range(cfg_, "ang_vel_z", {-1.0f, 1.0f});
    auto scale_axis = [](float a, Range r) { return a >= 0.0f ? a * r.hi : -(-a) * std::abs(r.lo); };
    VelocityCommand raw{
        std::clamp(scale_axis(gamepad_.ly, vx), vx.lo, vx.hi),
        std::clamp(scale_axis(-gamepad_.lx, vy), vy.lo, vy.hi),
        std::clamp(scale_axis(-gamepad_.rx, wz), wz.lo, wz.hi),
    };
    return raw;
  }

  std::vector<float> build_mimic_obs(const LowStateHg& state,
                                     const std::vector<float>& q_rel,
                                     const std::vector<float>& dq,
                                     const QuatWxyz& base_quat,
                                     const Vec3& base_ang_b) {
    bool sport_ok = false;
    SportModeState sport = sport_state_copy(&sport_ok);
    Vec3 base_lin_w{0.0f, 0.0f, 0.0f};
    Vec3 robot_pos{0.0f, 0.0f, 0.0f};
    QuatWxyz robot_q = base_quat;
    if (sport_ok) {
      base_lin_w = {sport.velocity()[0], sport.velocity()[1], sport.velocity()[2]};
      robot_pos = {sport.position()[0], sport.position()[1], sport.position()[2]};
      const auto& sq = sport.imu_state().quaternion();
      robot_q = {sq[0], sq[1], sq[2], sq[3]};
    }
    Vec3 base_lin_b = quat_apply_wxyz(quat_inv_wxyz(base_quat), base_lin_w);
    int anchor_idx = 0;
    auto it = std::find(cfg_.body_names.begin(), cfg_.body_names.end(), cfg_.anchor_body_name);
    if (it != cfg_.body_names.end()) anchor_idx = static_cast<int>(std::distance(cfg_.body_names.begin(), it));
    size_t pos_i = static_cast<size_t>(anchor_idx * 3);
    size_t quat_i = static_cast<size_t>(anchor_idx * 4);
    Vec3 motion_pos{ref_body_pos_w_.at(pos_i), ref_body_pos_w_.at(pos_i + 1), ref_body_pos_w_.at(pos_i + 2)};
    QuatWxyz motion_q{ref_body_quat_w_.at(quat_i), ref_body_quat_w_.at(quat_i + 1),
                      ref_body_quat_w_.at(quat_i + 2), ref_body_quat_w_.at(quat_i + 3)};
    QuatWxyz inv_robot = quat_inv_wxyz(robot_q);
    Vec3 delta{motion_pos[0] - robot_pos[0], motion_pos[1] - robot_pos[1], motion_pos[2] - robot_pos[2]};
    Vec3 anchor_pos_b = quat_apply_wxyz(inv_robot, delta);
    QuatWxyz anchor_quat_b = quat_mul_wxyz(inv_robot, motion_q);
    auto mat = matrix_from_quat_wxyz(anchor_quat_b);

    std::vector<float> obs;
    append(obs, ref_joint_pos_);
    append(obs, ref_joint_vel_);
    if (cfg_.include_state_estimation) append3(obs, anchor_pos_b);
    obs.insert(obs.end(), {mat[0], mat[1], mat[3], mat[4], mat[6], mat[7]});
    if (cfg_.include_state_estimation) append3(obs, base_lin_b);
    append3(obs, base_ang_b);
    append(obs, q_rel);
    append(obs, dq);
    append(obs, last_action_);
    return obs;
  }

  void apply_action(std::vector<float> action) {
    action.resize(static_cast<size_t>(cfg_.num_actions), 0.0f);
    last_action_ = action;
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    for (int i = 0; i < cfg_.num_actions; ++i) {
      int motor = cfg_.sdk2isaac_idx.at(i);
      low_cmd_.motor_cmd().at(motor).mode() = 1;
      low_cmd_.motor_cmd().at(motor).q() = cfg_.default_joint_pos.at(i) + action.at(i) * cfg_.action_scale.at(i);
      low_cmd_.motor_cmd().at(motor).dq() = 0.0f;
      low_cmd_.motor_cmd().at(motor).kp() = cfg_.kps.at(i);
      low_cmd_.motor_cmd().at(motor).kd() = cfg_.kds.at(i);
      low_cmd_.motor_cmd().at(motor).tau() = 0.0f;
    }
  }

  void update_mimic_switch() {
    if (gamepad_.a.on_press && active_mimic_policy_) {
      std::cout << "[PolicySwitch] A pressed: switching to flat stabilize policy.\n";
      cfg_ = flat_cfg_;
      load_policy(cfg_, options_.flat_model_name);
      configure_policy_state();
      active_mimic_policy_ = false;
    }
    if (gamepad_.b.on_press && !active_mimic_policy_) {
      std::cout << "[PolicySwitch] B pressed: switching to mimic / tracking policy.\n";
      cfg_ = load_config(main_cfg_path_, g1_dir_);
      load_policy(cfg_, main_model_name_);
      configure_policy_state();
      auto outs = policy_->run_named(std::vector<float>(static_cast<size_t>(cfg_.num_obs), 0.0f), 0.0f);
      ref_joint_pos_ = tensor_to_vector(outs.at(1));
      ref_joint_vel_ = tensor_to_vector(outs.at(2));
      ref_body_pos_w_ = tensor_to_vector(outs.at(3));
      ref_body_quat_w_ = tensor_to_vector(outs.at(4));
      active_mimic_policy_ = true;
    }
  }

  void configure_policy_state() {
    if (cfg_.sdk2isaac_idx.empty()) cfg_.sdk2isaac_idx = cfg_.mujoco_to_isaac_map;
    last_action_.assign(static_cast<size_t>(cfg_.num_actions), 0.0f);
    int frame_dim = 9 + 3 * cfg_.num_actions;
    obs_history_.assign(static_cast<size_t>(std::max(1, cfg_.history_length)), std::vector<float>(frame_dim, 0.0f));
    time_step_ = 0;
  }

  void load_policy(const Config& cfg, const std::string& model_name) {
    fs::path policy_path = resolve_policy_path(g1_dir_, cfg.policy_path, model_name);
    std::cout << "Loading ONNX policy: " << policy_path << "\n";
    policy_ = std::make_unique<OnnxPolicy>(policy_path.string());
  }

  void send_damping() {
    std::lock_guard<std::mutex> lock(cmd_mutex_);
    for (auto& cmd : low_cmd_.motor_cmd()) {
      cmd.mode() = 1;
      cmd.q() = 0.0f;
      cmd.dq() = 0.0f;
      cmd.kp() = 0.0f;
      cmd.kd() = 1.0f;
      cmd.tau() = 0.0f;
    }
    low_cmd_.crc() = crc32_core(reinterpret_cast<uint32_t*>(&low_cmd_), (sizeof(LowCmdHg) >> 2) - 1);
    if (lowcmd_pub_) lowcmd_pub_->Write(low_cmd_);
  }

  RealRunOptions options_;
  bool is_local_sim_ = false;
  fs::path g1_dir_;
  fs::path main_cfg_path_;
  std::string main_model_name_;
  Config cfg_;
  Config flat_cfg_;
  std::unique_ptr<OnnxPolicy> policy_;

  std::unique_ptr<ChannelPublisher<LowCmdHg>> lowcmd_pub_;
  std::unique_ptr<ChannelSubscriber<LowStateHg>> lowstate_sub_;
  std::unique_ptr<ChannelSubscriber<SportModeState>> sport_sub_;
  LowCmdHg low_cmd_;
  LowStateHg low_state_;
  SportModeState sport_state_;
  std::mutex state_mutex_;
  std::mutex cmd_mutex_;
  std::atomic<bool> low_state_received_{false};
  bool sport_state_received_ = false;
  std::atomic<bool> running_{false};
  uint8_t mode_machine_ = 0;
  Gamepad gamepad_;

  std::thread publish_thread_;
  std::thread control_thread_;
  std::vector<std::vector<float>> obs_history_;
  std::vector<float> last_action_;
  std::vector<float> ref_joint_pos_;
  std::vector<float> ref_joint_vel_;
  std::vector<float> ref_body_pos_w_;
  std::vector<float> ref_body_quat_w_;
  int time_step_ = 0;
  bool active_mimic_policy_ = false;
  double last_debug_time_ = 0.0;
  double control_start_time_ = 0.0;
  bool crc_warned_ = false;
};

}  // namespace

int run_sim2real_main(int argc, char** argv,
                      RealPolicyKind kind,
                      const char* default_config,
                      const char* default_model) {
  std::cout.setf(std::ios::unitbuf);
  RealRunOptions options;
  options.kind = kind;
  options.config_name = default_config;
  options.model_name = default_model;
  for (int i = 1; i < argc; ++i) {
    std::string a(argv[i]);
    auto need = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("Missing value for " + flag);
      return argv[++i];
    };
    if (a == "--net") options.net = need(a);
    else if (a == "--domain_id") options.domain_id = std::stoi(need(a));
    else if (a == "--config" || a == "--config_path") options.config_name = need(a);
    else if (a == "--model") options.model_name = need(a);
    else if (a == "--flat_config" || a == "--flat_config_path") options.flat_config_name = need(a);
    else if (a == "--flat_model") options.flat_model_name = need(a);
    else if (a == "--debug_policy") options.debug_policy = true;
    else if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0]
                << " --net IFACE [--domain_id N] [--config g1_x.yaml] [--model policy.onnx] [--debug_policy]\n";
      return 0;
    } else {
      throw std::runtime_error("Unknown argument: " + a);
    }
  }
  try {
    Sim2RealController controller(options);
    controller.run();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[ERROR] " << e.what() << "\n";
    return 1;
  }
}

}  // namespace g1

#include "g1_cpp/controllers.hpp"

#include <fcntl.h>
#include <linux/joystick.h>
#include <sys/select.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace g1 {

namespace {

double now_seconds() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

float clamp(float value, Range r) {
  return std::max(r.lo, std::min(r.hi, value));
}

GamepadController::Map gamepad_map_for(const std::string& type) {
  if (type == "xbox") {
    return GamepadController::Map{0, 1, 3, 4, 0, 1, 2, 3, 6, 7};
  }
  return GamepadController::Map{0, 1, 2, 3, 0, 1, 3, 4, 10, 11};
}

float normalize_js_axis(int16_t value) {
  float v = static_cast<float>(value) / 32767.0f;
  return std::fabs(v) < 0.05f ? 0.0f : clamp(v, {-1.0f, 1.0f});
}

}  // namespace

KeyboardController::KeyboardController(Range vx, Range vy, Range vyaw)
    : vx_range_(vx), vy_range_(vy), vyaw_range_(vyaw) {}

KeyboardController::~KeyboardController() {
  stop();
}

void KeyboardController::start() {
  if (!isatty(STDIN_FILENO)) {
    throw std::runtime_error("Keyboard input requires an interactive terminal. Use --input const for headless runs.");
  }
  if (tcgetattr(STDIN_FILENO, &old_termios_) == 0) {
    termios raw = old_termios_;
    raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSANOW, &raw);
    raw_enabled_ = true;
  }
  running_ = true;
  thread_ = std::thread(&KeyboardController::loop, this);
}

void KeyboardController::stop() {
  running_ = false;
  if (thread_.joinable()) thread_.join();
  if (raw_enabled_) {
    tcsetattr(STDIN_FILENO, TCSANOW, &old_termios_);
    raw_enabled_ = false;
  }
}

VelocityCommand KeyboardController::velocity() {
  std::lock_guard<std::mutex> lock(mutex_);
  return cmd_;
}

void KeyboardController::loop() {
  while (running_ && !exit_requested_) {
    fd_set set;
    FD_ZERO(&set);
    FD_SET(STDIN_FILENO, &set);
    timeval tv{0, 20000};
    if (select(STDIN_FILENO + 1, &set, nullptr, nullptr, &tv) > 0) {
      char c = 0;
      if (read(STDIN_FILENO, &c, 1) == 1) apply_key(c);
    }
  }
}

void KeyboardController::apply_key(char key) {
  std::lock_guard<std::mutex> lock(mutex_);
  switch (key) {
    case 'w':
    case 'W':
      cmd_.vx += 0.1f;
      break;
    case 's':
    case 'S':
      cmd_.vx -= 0.1f;
      break;
    case 'a':
    case 'A':
      cmd_.vy += 0.05f;
      break;
    case 'd':
    case 'D':
      cmd_.vy -= 0.05f;
      break;
    case 'q':
    case 'Q':
      cmd_.vyaw += 0.1f;
      break;
    case 'e':
    case 'E':
      cmd_.vyaw -= 0.1f;
      break;
    case ' ':
    case '0':
      cmd_ = {};
      break;
    case '1':
    case '2':
    case '3':
    case '4':
      active_policy_ = key - '0';
      break;
    case 'x':
    case 'X':
    case 27:
      exit_requested_ = true;
      running_ = false;
      break;
    default:
      break;
  }
  cmd_.vx = clamp(cmd_.vx, vx_range_);
  cmd_.vy = clamp(cmd_.vy, vy_range_);
  cmd_.vyaw = clamp(cmd_.vyaw, vyaw_range_);
}

ConstantController::ConstantController(float vx, float vy, float vyaw, float warmup_s, float ramp_s)
    : target_{vx, vy, vyaw}, warmup_s_(warmup_s), ramp_s_(std::max(ramp_s, 1e-3f)) {}

void ConstantController::start() {
  start_time_ = now_seconds();
}

VelocityCommand ConstantController::velocity() {
  double elapsed = now_seconds() - start_time_;
  if (elapsed < warmup_s_) return {};
  float alpha = std::min<float>((elapsed - warmup_s_) / ramp_s_, 1.0f);
  return {target_.vx * alpha, target_.vy * alpha, target_.vyaw * alpha};
}

GamepadController::GamepadController(Range vx, Range vy, Range vyaw, std::string joystick_type)
    : vx_range_(vx),
      vy_range_(vy),
      vyaw_range_(vyaw),
      map_(gamepad_map_for(joystick_type)),
      joystick_type_(std::move(joystick_type)) {}

GamepadController::~GamepadController() {
  stop();
}

void GamepadController::start() {
  const char* env_path = std::getenv("G1_CPP_JOYSTICK");
  std::vector<std::string> candidates;
  if (env_path && env_path[0] != '\0') candidates.emplace_back(env_path);
  candidates.emplace_back("/dev/input/js0");
  candidates.emplace_back("/dev/input/js1");
  for (const auto& path : candidates) {
    fd_ = open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd_ >= 0) {
      std::cout << "[input] gamepad=" << path << " type=" << joystick_type_ << "\n";
      break;
    }
  }
  if (fd_ < 0) {
    throw std::runtime_error("Gamepad input requested but /dev/input/js0 or js1 was not found. Use --input keyboard.");
  }
  running_ = true;
  thread_ = std::thread(&GamepadController::loop, this);
}

void GamepadController::stop() {
  running_ = false;
  if (thread_.joinable()) thread_.join();
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

VelocityCommand GamepadController::velocity() {
  std::lock_guard<std::mutex> lock(mutex_);
  return cmd_;
}

void GamepadController::loop() {
  while (running_ && !exit_requested_) {
    js_event e{};
    bool changed = false;
    while (read(fd_, &e, sizeof(e)) == static_cast<ssize_t>(sizeof(e))) {
      e.type &= ~JS_EVENT_INIT;
      std::lock_guard<std::mutex> lock(mutex_);
      if (e.type == JS_EVENT_AXIS) {
        axes_[e.number] = normalize_js_axis(e.value);
        changed = true;
      } else if (e.type == JS_EVENT_BUTTON) {
        buttons_[e.number] = e.value != 0;
        changed = true;
      }
    }
    if (changed) apply_state();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
}

float GamepadController::axis(int index) const {
  auto it = axes_.find(static_cast<unsigned char>(index));
  return it == axes_.end() ? 0.0f : it->second;
}

bool GamepadController::button(int index) const {
  auto it = buttons_.find(static_cast<unsigned char>(index));
  return it != buttons_.end() && it->second;
}

float GamepadController::scale_axis(float value, Range range) const {
  return value >= 0.0f ? value * range.hi : -(-value) * std::abs(range.lo);
}

void GamepadController::apply_state() {
  std::lock_guard<std::mutex> lock(mutex_);
  cmd_.vx = clamp(scale_axis(-axis(map_.axis_ly), vx_range_), vx_range_);
  cmd_.vy = clamp(scale_axis(-axis(map_.axis_lx), vy_range_), vy_range_);
  cmd_.vyaw = clamp(scale_axis(-axis(map_.axis_rx), vyaw_range_), vyaw_range_);
  if (button(map_.button_a)) active_policy_ = 1;
  if (button(map_.button_b)) active_policy_ = 2;
  if (button(map_.button_x)) active_policy_ = 3;
  if (button(map_.button_y)) active_policy_ = 4;
  if (button(map_.button_select)) {
    exit_requested_ = true;
    running_ = false;
  }
}

}  // namespace g1

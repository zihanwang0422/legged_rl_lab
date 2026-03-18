#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一手柄控制模块

支持三种手柄:
1. Logitech F710 (Linux /dev/input/jsX 原生接口)
2. GameSir 盖世小鸡 (pygame 接口)
3. Unitree 官方手柄 (Linux /dev/input/jsX 原生接口, 与 F710 同协议但按键映射不同)

使用方法:
    from utils.joystick import create_gamepad_controller
    gamepad = create_gamepad_controller("f710", vx_range=(0, 1.0), vy_range=(-0.5, 0.5), vyaw_range=(-1.0, 1.0))
    gamepad.start()
    vx, vy, vyaw = gamepad.get_velocity()
"""

import struct
import threading
import time
import numpy as np


def apply_deadzone(value, deadzone=0.08):
    """应用死区，线性缩放保持平滑过渡"""
    if abs(value) < deadzone:
        return 0.0
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


# ============================================================================
#  底层手柄读取器 (Linux /dev/input/jsX)
# ============================================================================

class LinuxJoystickReader:
    """
    直接读取 Linux joystick 设备 (/dev/input/jsX)
    适用于 Logitech F710 和 Unitree 官方手柄

    事件格式 (8 bytes):
    - timestamp (4 bytes unsigned int)
    - value (2 bytes signed short)
    - type (1 byte unsigned char): 0x01=button, 0x02=axis
    - number (1 byte unsigned char): 按钮/轴编号
    """

    def __init__(self, device_path='/dev/input/js0'):
        self.device_path = device_path
        self.device_file = None
        self.running = False
        self.thread = None
        self.axes = [0] * 8
        self.buttons = [0] * 16
        self.lock = threading.Lock()

        try:
            self.device_file = open(self.device_path, 'rb')
            print(f"✅ Joystick opened: {self.device_path}")
        except Exception as e:
            print(f"❌ Failed to open joystick: {e}")
            raise

    def _read_thread(self):
        while self.running:
            try:
                event_data = self.device_file.read(8)
                if len(event_data) < 8:
                    break
                timestamp, value, event_type, number = struct.unpack('IhBB', event_data)
                event_type_masked = event_type & 0x7F
                with self.lock:
                    if event_type_masked == 0x01 and number < len(self.buttons):
                        self.buttons[number] = value
                    elif event_type_masked == 0x02 and number < len(self.axes):
                        self.axes[number] = value
            except Exception:
                break

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.device_file:
            self.device_file.close()


# ============================================================================
#  GameSir 底层读取器 (pygame 接口)
# ============================================================================

class PygameJoystickReader:
    """
    使用 pygame 读取 GameSir 手柄
    pygame 事件循环在后台线程中运行
    """

    # 轴索引
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 2
    AXIS_RIGHT_Y = 3
    AXIS_RT = 4
    AXIS_LT = 5

    # 按钮索引
    BTN_A = 0
    BTN_B = 1
    BTN_X = 3
    BTN_Y = 4
    BTN_LB = 6
    BTN_RB = 7
    BTN_LT = 8
    BTN_RT = 9
    BTN_SELECT = 10
    BTN_START = 11
    BTN_HOME = 12
    BTN_L3 = 13
    BTN_R3 = 14

    def __init__(self):
        import pygame
        self.pygame = pygame
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError("未检测到手柄，请先连接手柄")

        self.js = pygame.joystick.Joystick(0)
        self.js.init()
        print(f"✅ GameSir joystick opened: {self.js.get_name()}")
        print(f"   Buttons: {self.js.get_numbuttons()}, Axes: {self.js.get_numaxes()}, Hats: {self.js.get_numhats()}")

        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # 归一化后的轴值 [-1, 1]
        self.axes = [0.0] * 6
        self.buttons = [0] * 19  # 包含虚拟 D-pad 按钮
        self.hat = (0, 0)

    def _read_thread(self):
        while self.running:
            try:
                for event in self.pygame.event.get():
                    with self.lock:
                        if event.type == self.pygame.JOYBUTTONDOWN:
                            if event.button < 15:
                                self.buttons[event.button] = 1
                        elif event.type == self.pygame.JOYBUTTONUP:
                            if event.button < 15:
                                self.buttons[event.button] = 0
                        elif event.type == self.pygame.JOYAXISMOTION:
                            if event.axis < len(self.axes):
                                self.axes[event.axis] = event.value
                        elif event.type == self.pygame.JOYHATMOTION:
                            self.hat = event.value
                            # 映射 hat 到虚拟按钮
                            self.buttons[15] = 1 if event.value[1] == 1 else 0   # UP
                            self.buttons[16] = 1 if event.value[1] == -1 else 0  # DOWN
                            self.buttons[17] = 1 if event.value[0] == -1 else 0  # LEFT
                            self.buttons[18] = 1 if event.value[0] == 1 else 0   # RIGHT
                time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._read_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pygame.quit()


# ============================================================================
#  GamepadController 基类
# ============================================================================

class GamepadController:
    """
    手柄速度控制器基类
    所有手柄实现需要提供统一的 get_velocity() / start() / stop() 接口
    """

    def __init__(self, vx_range=(-2.0, 4.0), vy_range=(-1.0, 1.0), vyaw_range=(-1.57, 1.57)):
        self.vx = 0.0
        self.vy = 0.0
        self.vyaw = 0.0
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.vyaw_range = vyaw_range
        self.lock = threading.Lock()
        self.running = True
        self.exit_requested = False
        self.thread = None

        self.deadzone = 0.05
        self.vx_increment = 0.1
        self.dpad_last_state = {'up': False, 'down': False}
        self.walk_requested = False  # RB+A 组合触发，进入 walk policy
        # 当前激活的策略索引: 0=空闲, 1=walk(RB+A), 2=policy1(RB+B), 3=policy2(RB+X), 4=policy3(RB+Y)
        self.active_policy = 0

    def get_velocity(self):
        with self.lock:
            return self.vx, self.vy, self.vyaw

    def set_velocity(self, vx, vy, vyaw):
        with self.lock:
            self.vx = np.clip(vx, self.vx_range[0], self.vx_range[1])
            self.vy = np.clip(vy, self.vy_range[0], self.vy_range[1])
            self.vyaw = np.clip(vyaw, self.vyaw_range[0], self.vyaw_range[1])

    def start(self):
        self.thread = threading.Thread(target=self._control_thread, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self._stop_backend()
        if self.thread:
            self.thread.join(timeout=1.0)

    def _control_thread(self):
        raise NotImplementedError

    def _stop_backend(self):
        raise NotImplementedError


# ============================================================================
#  Logitech F710 手柄控制器
# ============================================================================

class F710GamepadController(GamepadController):
    """Logitech F710 手柄 (Linux /dev/input/jsX 原生接口)"""

    # F710 轴映射 (X模式)
    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 3
    AXIS_RIGHT_Y = 4
    AXIS_DPAD_X = 6
    AXIS_DPAD_Y = 7
    # X模式默认按键: BTN_A=0, BTN_RB=5, BTN_START=7
    # D模式默认按键: BTN_A=1, BTN_RB=7, BTN_START=9  (通过 YAML gamepad_btn_* 覆盖)
    BTN_A = 0
    BTN_RB = 5
    BTN_START = 7

    def __init__(self, vx_range=(-2.0, 4.0), vy_range=(-1.0, 1.0), vyaw_range=(-1.57, 1.57),
                 device_path='/dev/input/js0', btn_start=None, btn_rb=None, btn_a=None):
        super().__init__(vx_range, vy_range, vyaw_range)
        if btn_start is not None: self.BTN_START = btn_start
        if btn_rb   is not None: self.BTN_RB    = btn_rb
        if btn_a    is not None: self.BTN_A     = btn_a
        try:
            self.reader = LinuxJoystickReader(device_path)
            self.reader.start()
            print(f"✅ F710 Gamepad initialized (BTN_A={self.BTN_A}, BTN_RB={self.BTN_RB}, BTN_START={self.BTN_START})")
        except Exception as e:
            print(f"❌ F710 init failed: {e}")
            self.reader = None

    def _control_thread(self):
        if self.reader is None:
            print("Gamepad not available, using zero velocity")
            return

        update_interval = 1.0 / 33.0

        while self.running:
            try:
                loop_start = time.time()

                with self.reader.lock:
                    raw_axes = list(self.reader.axes)
                    raw_buttons = list(self.reader.buttons)

                # 归一化摇杆 [-1, 1]
                left_x = apply_deadzone(raw_axes[self.AXIS_LEFT_X] / 32767.0, self.deadzone)
                left_y = apply_deadzone(raw_axes[self.AXIS_LEFT_Y] / 32767.0, self.deadzone)
                right_x = apply_deadzone(raw_axes[self.AXIS_RIGHT_X] / 32767.0, self.deadzone)

                # D-pad
                dpad_y = raw_axes[self.AXIS_DPAD_Y] if len(raw_axes) > self.AXIS_DPAD_Y else 0
                dpad_up = (dpad_y < -16000)
                dpad_down = (dpad_y > 16000)

                if dpad_up and not self.dpad_last_state['up']:
                    self.vx = min(self.vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] speed step: {self.vx:.1f} m/s")
                if dpad_down and not self.dpad_last_state['down']:
                    self.vx = max(self.vx - self.vx_increment, 0.0)
                    print(f"\n[D-pad DOWN] speed step: {self.vx:.1f} m/s")

                self.dpad_last_state['up'] = dpad_up
                self.dpad_last_state['down'] = dpad_down

                # 摇杆映射到速度
                if abs(left_y) > 0.1:
                    if left_y <= 0:
                        self.vx = (-left_y) * self.vx_range[1]
                    else:
                        self.vx = (-left_y) * abs(self.vx_range[0])

                self.vy = -left_x * self.vy_range[1]
                self.vyaw = -right_x * self.vyaw_range[1]

                self.set_velocity(self.vx, self.vy, self.vyaw)

                # RB+A 进入 walk policy
                if (self.BTN_RB < len(raw_buttons) and raw_buttons[self.BTN_RB] and
                        self.BTN_A < len(raw_buttons) and raw_buttons[self.BTN_A]):
                    if not self.walk_requested:
                        print("\n✅ [RB+A] Walk policy activated!")
                        self.walk_requested = True

                # Start 按钮退出
                if self.BTN_START < len(raw_buttons) and raw_buttons[self.BTN_START]:
                    print("\n✅ Start button pressed - exiting")
                    self.exit_requested = True
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0, update_interval - elapsed))

            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)

    def _stop_backend(self):
        if self.reader:
            self.reader.stop()


# ============================================================================
#  Unitree 官方手柄控制器
# ============================================================================

class UnitreeGamepadController(GamepadController):
    """
    Unitree 官方手柄 (Linux /dev/input/jsX)
    与 F710 使用相同的 Linux joystick 协议，但按键映射可能不同
    轴映射与 F710 一致 (X模式):
      左摇杆: axes[0]=X, axes[1]=Y
      右摇杆: axes[3]=X, axes[4]=Y
    """

    AXIS_LEFT_X = 0
    AXIS_LEFT_Y = 1
    AXIS_RIGHT_X = 3
    AXIS_RIGHT_Y = 4
    AXIS_DPAD_X = 6
    AXIS_DPAD_Y = 7
    BTN_A = 8
    BTN_RB = 5
    BTN_START = 7

    def __init__(self, vx_range=(-2.0, 4.0), vy_range=(-1.0, 1.0), vyaw_range=(-1.57, 1.57),
                 device_path='/dev/input/js0', btn_start=None, btn_rb=None, btn_a=None):
        super().__init__(vx_range, vy_range, vyaw_range)
        if btn_start is not None: self.BTN_START = btn_start
        if btn_rb   is not None: self.BTN_RB    = btn_rb
        if btn_a    is not None: self.BTN_A     = btn_a
        try:
            self.reader = LinuxJoystickReader(device_path)
            self.reader.start()
            print(f"✅ Unitree Gamepad initialized (BTN_A={self.BTN_A}, BTN_RB={self.BTN_RB}, BTN_START={self.BTN_START})")
        except Exception as e:
            print(f"❌ Unitree gamepad init failed: {e}")
            self.reader = None

    def _control_thread(self):
        if self.reader is None:
            print("Gamepad not available, using zero velocity")
            return

        update_interval = 1.0 / 33.0

        while self.running:
            try:
                loop_start = time.time()

                with self.reader.lock:
                    raw_axes = list(self.reader.axes)
                    raw_buttons = list(self.reader.buttons)

                left_x = apply_deadzone(raw_axes[self.AXIS_LEFT_X] / 32767.0, self.deadzone)
                left_y = apply_deadzone(raw_axes[self.AXIS_LEFT_Y] / 32767.0, self.deadzone)
                right_x = apply_deadzone(raw_axes[self.AXIS_RIGHT_X] / 32767.0, self.deadzone)

                # D-pad
                dpad_y = raw_axes[self.AXIS_DPAD_Y] if len(raw_axes) > self.AXIS_DPAD_Y else 0
                dpad_up = (dpad_y < -16000)
                dpad_down = (dpad_y > 16000)

                if dpad_up and not self.dpad_last_state['up']:
                    self.vx = min(self.vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] speed step: {self.vx:.1f} m/s")
                if dpad_down and not self.dpad_last_state['down']:
                    self.vx = max(self.vx - self.vx_increment, 0.0)
                    print(f"\n[D-pad DOWN] speed step: {self.vx:.1f} m/s")

                self.dpad_last_state['up'] = dpad_up
                self.dpad_last_state['down'] = dpad_down

                # 摇杆映射
                if abs(left_y) > 0.1:
                    if left_y <= 0:
                        self.vx = (-left_y) * self.vx_range[1]
                    else:
                        self.vx = (-left_y) * abs(self.vx_range[0])

                self.vy = -left_x * self.vy_range[1]
                self.vyaw = -right_x * self.vyaw_range[1]

                self.set_velocity(self.vx, self.vy, self.vyaw)

                # RB+A 进入 walk policy
                if (self.BTN_RB < len(raw_buttons) and raw_buttons[self.BTN_RB] and
                        self.BTN_A < len(raw_buttons) and raw_buttons[self.BTN_A]):
                    if not self.walk_requested:
                        print("\n✅ [RB+A] Walk policy activated!")
                        self.walk_requested = True

                if self.BTN_START < len(raw_buttons) and raw_buttons[self.BTN_START]:
                    print("\n✅ Start button pressed - exiting")
                    self.exit_requested = True
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0, update_interval - elapsed))

            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)

    def _stop_backend(self):
        if self.reader:
            self.reader.stop()


# ============================================================================
#  GameSir 盖世小鸡手柄控制器
# ============================================================================

class GameSirGamepadController(GamepadController):
    """
    GameSir 盖世小鸡手柄 (pygame 接口)

    轴映射:
      左摇杆: axes[0]=X(-1左,+1右), axes[1]=Y(-1上,+1下)
      右摇杆: axes[2]=X(-1左,+1右), axes[3]=Y(-1上,+1下)
    按钮映射:
      START=11, D-pad 通过 hat 事件 -> 虚拟按钮 15(UP) 16(DOWN)
    """

    def __init__(self, vx_range=(-2.0, 4.0), vy_range=(-1.0, 1.0), vyaw_range=(-1.57, 1.57)):
        super().__init__(vx_range, vy_range, vyaw_range)
        try:
            self.reader = PygameJoystickReader()
            self.reader.start()
            print("✅ GameSir Gamepad initialized")
        except Exception as e:
            print(f"❌ GameSir init failed: {e}")
            self.reader = None

    def _control_thread(self):
        if self.reader is None:
            print("Gamepad not available, using zero velocity")
            return

        update_interval = 1.0 / 33.0
        _prev_combos = {1: False, 2: False, 3: False, 4: False}

        while self.running:
            try:
                loop_start = time.time()

                with self.reader.lock:
                    # pygame axes 已经是 [-1, 1]
                    left_x = self.reader.axes[0]
                    left_y = self.reader.axes[1]
                    right_x = self.reader.axes[2]
                    buttons = list(self.reader.buttons)

                left_x = apply_deadzone(left_x, self.deadzone)
                left_y = apply_deadzone(left_y, self.deadzone)
                right_x = apply_deadzone(right_x, self.deadzone)

                # D-pad (虚拟按钮 15=UP, 16=DOWN)
                dpad_up = bool(buttons[15])
                dpad_down = bool(buttons[16])

                if dpad_up and not self.dpad_last_state['up']:
                    self.vx = min(self.vx + self.vx_increment, self.vx_range[1])
                    print(f"\n[D-pad UP] speed step: {self.vx:.1f} m/s")
                if dpad_down and not self.dpad_last_state['down']:
                    self.vx = max(self.vx - self.vx_increment, 0.0)
                    print(f"\n[D-pad DOWN] speed step: {self.vx:.1f} m/s")

                self.dpad_last_state['up'] = dpad_up
                self.dpad_last_state['down'] = dpad_down

                # 左摇杆: 前后左右速度
                # Y轴: 向上(-1) -> vx正(前进), 向下(+1) -> vx负(后退)
                if abs(left_y) > 0.1:
                    if left_y <= 0:
                        self.vx = (-left_y) * self.vx_range[1]
                    else:
                        self.vx = (-left_y) * abs(self.vx_range[0])
                # X轴: 向左(-1) -> vy正(左移), 向右(+1) -> vy负(右移)
                self.vy = -left_x * self.vy_range[1]

                # 右摇杆: yaw 速度
                self.vyaw = -right_x * self.vyaw_range[1]

                self.set_velocity(self.vx, self.vy, self.vyaw)

                # RB + 面键 组合 → 策略切换 (上升沿触发)
                rb = bool(buttons[PygameJoystickReader.BTN_RB])
                combos = {
                    1: rb and bool(buttons[PygameJoystickReader.BTN_A]),
                    2: rb and bool(buttons[PygameJoystickReader.BTN_B]),
                    3: rb and bool(buttons[PygameJoystickReader.BTN_X]),
                    4: rb and bool(buttons[PygameJoystickReader.BTN_Y]),
                }
                combo_names = {1: 'RB+A (Walk)', 2: 'RB+B (Policy1)',
                               3: 'RB+X (Policy2)', 4: 'RB+Y (Policy3)'}
                for idx, pressed in combos.items():
                    if pressed and not _prev_combos[idx]:
                        self.active_policy = idx
                        if idx == 1:
                            self.walk_requested = True
                        print(f"\n✅ [{combo_names[idx]}] activated!")
                _prev_combos = dict(combos)

                # START 按钮 (index 11) 退出
                if buttons[PygameJoystickReader.BTN_START]:
                    print("\n✅ Start button pressed - exiting")
                    self.exit_requested = True
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0, update_interval - elapsed))

            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)

    def _stop_backend(self):
        if self.reader:
            self.reader.stop()


# ============================================================================
#  Unitree 官方手柄 pygame 控制器 (USB 连接, 用于 sim2sim)
# ============================================================================

class UnitreePygameGamepadController(GamepadController):
    """
    Unitree 官方手柄通过 USB + pygame 读取 (用于 sim2sim)。
    按键映射与 UnitreeSDKGamepadController 完全一致，命令逻辑相同。

    pygame 轴/按键映射 (Unitree 官方手柄, 与 LogicJoystick 相同协议):
      axes[0]=lx, axes[1]=-ly, axes[3]=rx, axes[4]=-ry
      buttons: A=0, B=1, X=2, Y=3, LB=4, RB=5, back=6, start=7
      hat: up/down/left/right
    """

    AXIS_LX = 0
    AXIS_LY = 1   # pygame Y轴向下为正，需取反
    AXIS_RX = 3
    AXIS_RY = 4

    BTN_A = 0
    BTN_B = 1
    BTN_X = 2
    BTN_Y = 3
    BTN_LB = 4
    BTN_RB = 5
    BTN_BACK = 6
    BTN_START = 7

    def __init__(self, vx_range=(-1.0, 1.0), vy_range=(-0.5, 0.5), vyaw_range=(-1.0, 1.0)):
        super().__init__(vx_range, vy_range, vyaw_range)
        try:
            import pygame
            self.pygame = pygame
            pygame.init()
            pygame.joystick.init()
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("未检测到手柄")
            self.js = pygame.joystick.Joystick(0)
            self.js.init()
            print(f"✅ Unitree pygame gamepad: {self.js.get_name()}")
        except Exception as e:
            print(f"❌ UnitreePygame init failed: {e}")
            self.js = None
        self._prev_combos = {1: False, 2: False, 3: False, 4: False}

    def _control_thread(self):
        if self.js is None:
            return
        update_interval = 1.0 / 50.0
        while self.running:
            try:
                loop_start = time.time()
                for event in self.pygame.event.get():
                    pass  # pump events

                lx = apply_deadzone(self.js.get_axis(self.AXIS_LX), self.deadzone)
                ly = apply_deadzone(-self.js.get_axis(self.AXIS_LY), self.deadzone)  # 取反: 前推为正
                rx = apply_deadzone(self.js.get_axis(self.AXIS_RX), self.deadzone)

                vx = ly * self.vx_range[1] if ly >= 0 else ly * abs(self.vx_range[0])
                vy = -lx * self.vy_range[1]
                vyaw = -rx * self.vyaw_range[1]
                self.set_velocity(vx, vy, vyaw)

                rb = bool(self.js.get_button(self.BTN_RB))
                combos = {
                    1: rb and bool(self.js.get_button(self.BTN_A)),
                    2: rb and bool(self.js.get_button(self.BTN_B)),
                    3: rb and bool(self.js.get_button(self.BTN_X)),
                    4: rb and bool(self.js.get_button(self.BTN_Y)),
                }
                combo_names = {1: 'RB+A (Walk)', 2: 'RB+B (Policy1)',
                               3: 'RB+X (Policy2)', 4: 'RB+Y (Policy3)'}
                for idx, pressed in combos.items():
                    if pressed and not self._prev_combos[idx]:
                        self.active_policy = idx
                        if idx == 1:
                            self.walk_requested = True
                        print(f"\n✅ [{combo_names[idx]}] activated!")
                self._prev_combos = dict(combos)

                if self.js.get_button(self.BTN_START):
                    print("\n✅ Start button pressed - exiting")
                    self.exit_requested = True
                    break

                elapsed = time.time() - loop_start
                time.sleep(max(0, update_interval - elapsed))
            except Exception as e:
                print(f"\nGamepad error: {e}")
                time.sleep(0.1)

    def _stop_backend(self):
        if self.js:
            self.pygame.quit()


# ============================================================================
#  Unitree SDK 官方手柄控制器 (从 LowState.wireless_remote 解析)
# ============================================================================

class UnitreeSDKGamepadController(GamepadController):
    """
    Unitree 官方手柄，通过 LowState.wireless_remote (40字节) 解析。
    不需要独立线程，由外部在每个控制循环中调用 update(wireless_remote)。

    按键布局 (wireless_remote 字节协议):
      Byte 2: [0,0, LT, RT, back, start, LB, RB]
      Byte 3: [left, down, right, up, Y, X, B, A]
      Bytes 4-7:   lx (float32 LE)
      Bytes 8-11:  rx (float32 LE)
      Bytes 12-15: ry (float32 LE)
      Bytes 20-23: ly (float32 LE)

    速度映射:
      左摇杆 ly (前后) -> vx,  lx (左右) -> vy
      右摇杆 rx (左右) -> vyaw
      RB+A  -> active_policy=1 (walk)
      RB+B  -> active_policy=2
      RB+X  -> active_policy=3
      RB+Y  -> active_policy=4
      start -> exit_requested
    """

    def __init__(self, vx_range=(-1.0, 1.0), vy_range=(-0.5, 0.5), vyaw_range=(-1.0, 1.0)):
        super().__init__(vx_range, vy_range, vyaw_range)
        self._prev_combos = {1: False, 2: False, 3: False, 4: False}
        self._prev_start = False
        print("✅ UnitreeSDK Gamepad initialized (reads from LowState.wireless_remote)")

    def update(self, wireless_remote):
        """
        解析 40 字节 wireless_remote，更新速度和按键状态。
        在每个控制循环中调用。
        """
        if wireless_remote is None or len(wireless_remote) < 24:
            return

        b2 = int(wireless_remote[2])
        b3 = int(wireless_remote[3])

        rb    = bool((b2 >> 0) & 1)
        lb    = bool((b2 >> 1) & 1)
        start = bool((b2 >> 2) & 1)

        btn_a = bool((b3 >> 0) & 1)
        btn_b = bool((b3 >> 1) & 1)
        btn_x = bool((b3 >> 2) & 1)
        btn_y = bool((b3 >> 3) & 1)

        import struct
        lx = struct.unpack('f', bytes(wireless_remote[4:8]))[0]
        rx = struct.unpack('f', bytes(wireless_remote[8:12]))[0]
        ry = struct.unpack('f', bytes(wireless_remote[12:16]))[0]
        ly = struct.unpack('f', bytes(wireless_remote[20:24]))[0]

        lx = apply_deadzone(lx, self.deadzone)
        ly = apply_deadzone(ly, self.deadzone)
        rx = apply_deadzone(rx, self.deadzone)

        # ly: 向前推为正 -> vx正; lx: 向右推为正 -> vy负
        vx = ly * self.vx_range[1] if ly >= 0 else ly * abs(self.vx_range[0])
        vy = -lx * self.vy_range[1]
        vyaw = -rx * self.vyaw_range[1]
        self.set_velocity(vx, vy, vyaw)

        # RB + 面键 组合 -> 策略切换 (上升沿)
        combos = {
            1: rb and btn_a,
            2: rb and btn_b,
            3: rb and btn_x,
            4: rb and btn_y,
        }
        combo_names = {1: 'RB+A (Walk)', 2: 'RB+B (Policy1)',
                       3: 'RB+X (Policy2)', 4: 'RB+Y (Policy3)'}
        for idx, pressed in combos.items():
            if pressed and not self._prev_combos[idx]:
                self.active_policy = idx
                if idx == 1:
                    self.walk_requested = True
                print(f"\n✅ [{combo_names[idx]}] activated!")
        self._prev_combos = dict(combos)

        # start 退出
        if start and not self._prev_start:
            print("\n✅ Start button pressed - exiting")
            self.exit_requested = True
        self._prev_start = start

    def _control_thread(self):
        # 不使用独立线程，由外部 update() 驱动
        pass

    def _stop_backend(self):
        pass

    def start(self):
        # 无需启动线程
        pass

    def stop(self):
        pass


# ============================================================================
#  工厂函数: 根据类型创建手柄控制器
# ============================================================================

# 支持的手柄类型映射
GAMEPAD_TYPES = {
    'f710': F710GamepadController,
    'logitech': F710GamepadController,
    'unitree': UnitreeGamepadController,
    'unitree_pygame': UnitreePygameGamepadController,
    'unitree_sdk': UnitreeSDKGamepadController,
    'gamesir': GameSirGamepadController,
}


def create_gamepad_controller(gamepad_type, vx_range=(-2.0, 4.0), vy_range=(-1.0, 1.0),
                              vyaw_range=(-1.57, 1.57), device_path='/dev/input/js0',
                              btn_start=None, btn_rb=None, btn_a=None):
    """
    根据手柄类型创建对应的控制器

    Args:
        gamepad_type: 手柄类型 ('f710', 'logitech', 'unitree', 'unitree_sdk', 'gamesir')
        vx_range: 前后速度范围 (m/s)
        vy_range: 左右速度范围 (m/s)
        vyaw_range: 旋转速度范围 (rad/s)
        device_path: Linux 设备路径 (仅 f710/unitree 使用)
        btn_start/btn_rb/btn_a: 覆盖默认按键索引 (f710/unitree)

    Returns:
        GamepadController 子类实例
    """
    gamepad_type = gamepad_type.lower()
    if gamepad_type not in GAMEPAD_TYPES:
        raise ValueError(f"不支持的手柄类型: '{gamepad_type}', 可选: {list(GAMEPAD_TYPES.keys())}")

    cls = GAMEPAD_TYPES[gamepad_type]

    if cls is UnitreeSDKGamepadController:
        return cls(vx_range=vx_range, vy_range=vy_range, vyaw_range=vyaw_range)
    elif cls is UnitreePygameGamepadController:
        return cls(vx_range=vx_range, vy_range=vy_range, vyaw_range=vyaw_range)
    elif cls in (F710GamepadController, UnitreeGamepadController):
        return cls(vx_range=vx_range, vy_range=vy_range, vyaw_range=vyaw_range,
                   device_path=device_path, btn_start=btn_start, btn_rb=btn_rb, btn_a=btn_a)
    else:
        return cls(vx_range=vx_range, vy_range=vy_range, vyaw_range=vyaw_range)


# ============================================================================
#  测试代码
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="手柄测试工具")
    parser.add_argument('--type', type=str, default='f710', choices=list(GAMEPAD_TYPES.keys()),
                        help='手柄类型')
    parser.add_argument('--device', type=str, default='/dev/input/js0',
                        help='Linux 设备路径 (仅 f710/unitree)')
    args = parser.parse_args()

    print("=" * 70)
    print(f"手柄测试 - 类型: {args.type}")
    print("=" * 70)
    print("  左摇杆: 控制 vx (前后) / vy (左右)")
    print("  右摇杆: 控制 vyaw (转向)")
    print("  D-pad 上/下: 步进调速")
    print("  Start: 退出")
    print("=" * 70 + "\n")

    gamepad = create_gamepad_controller(args.type, vx_range=(0, 2.0), vy_range=(-0.5, 0.5),
                                        vyaw_range=(-1.5, 1.5), device_path=args.device)
    gamepad.start()

    try:
        while not gamepad.exit_requested:
            vx, vy, vyaw = gamepad.get_velocity()
            print(f"\rvx={vx:+.2f} m/s | vy={vy:+.2f} m/s | vyaw={vyaw:+.2f} rad/s   ", end='', flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n✅ Ctrl+C")
    finally:
        gamepad.stop()
        print("\n测试结束")

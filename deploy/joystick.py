#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
直接使用 Linux 系统接口读取 Logitech F710 手柄
无需 pygame 依赖

基于 /dev/input/jsX 接口直接读取手柄数据
"""

import struct
import threading
import time
import os


class RemoteController:
    """
    直接读取 Linux joystick 设备
    
    事件格式 (8 bytes):
    - timestamp (4 bytes unsigned int)
    - value (2 bytes signed short)
    - type (1 byte unsigned char): 0x01=button, 0x02=axis
    - number (1 byte unsigned char): 按钮/轴编号
    """
    
    # F710 轴映射 (X模式)
    AXIS_LEFT_X = 0      # 左摇杆 X: -32767(左) ~ +32767(右)
    AXIS_LEFT_Y = 1      # 左摇杆 Y: -32767(上) ~ +32767(下)
    AXIS_RIGHT_X = 3     # 右摇杆 X: -32767(左) ~ +32767(右)
    AXIS_RIGHT_Y = 4     # 右摇杆 Y: -32767(上) ~ +32767(下)
    AXIS_LT = 2          # 左扳机: -32767(未按) ~ +32767(按到底)
    AXIS_RT = 5          # 右扳机: -32767(未按) ~ +32767(按到底)
    
    # 按钮映射
    BTN_A = 0
    BTN_B = 1
    BTN_X = 2
    BTN_Y = 3
    BTN_LB = 4
    BTN_RB = 5
    BTN_BACK = 6
    BTN_START = 7
    BTN_LOGITECH = 8
    BTN_LEFT_STICK = 9
    BTN_RIGHT_STICK = 10
    
    def __init__(self, device_path='/dev/input/js0'):
        self.device_path = device_path
        self.device_file = None
        self.running = False
        self.thread = None
        
        # 当前状态
        self.axes = [0] * 8  # 8个轴
        self.buttons = [0] * 12  # 12个按钮
        self.lock = threading.Lock()
        
        # 打开设备
        try:
            self.device_file = open(self.device_path, 'rb')
            print(f"✅ Gamepad opened: {self.device_path}")
        except Exception as e:
            print(f"❌ Failed to open gamepad: {e}")
            raise
    
    def _read_event(self):
        """读取一个事件 (8 bytes)"""
        try:
            event_data = self.device_file.read(8)
            if len(event_data) < 8:
                return None
            
            # 解析事件: timestamp, value, type, number
            timestamp, value, event_type, number = struct.unpack('IhBB', event_data)
            return {
                'timestamp': timestamp,
                'value': value,
                'type': event_type,
                'number': number
            }
        except Exception:
            return None
    
    def _read_thread(self):
        """后台读取线程"""
        while self.running:
            event = self._read_event()
            if event is None:
                break
            
            with self.lock:
                # 0x01 = button event, 0x02 = axis event
                # 0x80 = init flag (初始化事件,忽略)
                event_type = event['type'] & 0x7F
                
                if event_type == 0x01:  # Button
                    if event['number'] < len(self.buttons):
                        self.buttons[event['number']] = event['value']
                
                elif event_type == 0x02:  # Axis
                    if event['number'] < len(self.axes):
                        self.axes[event['number']] = event['value']
    
    def start(self):
        """启动后台读取线程"""
        self.running = True
        self.thread = threading.Thread(target=self._read_thread, daemon=True)
        self.thread.start()
    
    def stop(self):
        """停止读取"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.device_file:
            self.device_file.close()
    
    def get_left_stick(self, normalize=True):
        """
        获取左摇杆值
        
        Args:
            normalize: True返回[-1.0, 1.0], False返回[-32767, 32767]
        
        Returns:
            (x, y): x正=右, y正=下
        """
        with self.lock:
            x = self.axes[self.AXIS_LEFT_X]
            y = self.axes[self.AXIS_LEFT_Y]
        
        if normalize:
            x = x / 32767.0
            y = y / 32767.0
        
        return x, y
    
    def get_right_stick(self, normalize=True):
        """
        获取右摇杆值
        
        Args:
            normalize: True返回[-1.0, 1.0], False返回[-32767, 32767]
        
        Returns:
            (x, y): x正=右, y正=下
        """
        with self.lock:
            x = self.axes[self.AXIS_RIGHT_X]
            y = self.axes[self.AXIS_RIGHT_Y]
        
        if normalize:
            x = x / 32767.0
            y = y / 32767.0
        
        return x, y
    
    def get_triggers(self, normalize=True):
        """
        获取扳机值
        
        Args:
            normalize: True返回[0.0, 1.0], False返回[-32767, 32767]
        
        Returns:
            (left, right): 0=未按, 1=按到底
        """
        with self.lock:
            lt = self.axes[self.AXIS_LT]
            rt = self.axes[self.AXIS_RT]
        
        if normalize:
            # 扳机从 -32767(未按) 到 +32767(按到底)
            lt = (lt + 32767) / 65534.0
            rt = (rt + 32767) / 65534.0
        
        return lt, rt
    
    def get_buttons(self):
        """
        获取按钮状态
        
        Returns:
            list: 按下的按钮编号列表
        """
        with self.lock:
            pressed = [i for i, state in enumerate(self.buttons) if state]
        return pressed
    
    def is_button_pressed(self, button_num):
        """检查特定按钮是否按下"""
        with self.lock:
            if button_num < len(self.buttons):
                return self.buttons[button_num] != 0
        return False
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# Alias for backward compatibility
F710GamePadLinux = RemoteController


def apply_deadzone(value, deadzone=0.08):
    """
    应用死区
    
    Args:
        value: 归一化值 [-1.0, 1.0]
        deadzone: 死区阈值
    
    Returns:
        处理后的值
    """
    if abs(value) < deadzone:
        return 0.0
    # 线性缩放,保持平滑过渡
    sign = 1 if value > 0 else -1
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


# ========== 测试代码 ==========

if __name__ == '__main__':
    print("="*70)
    print("F710 手柄测试 (直接读取 Linux 设备)")
    print("="*70)
    print("\n控制说明:")
    print("  左摇杆: 轴0(X), 轴1(Y)")
    print("    - 向左: X=-32767, 向右: X=+32767")
    print("    - 向上: Y=-32767, 向下: Y=+32767")
    print("  右摇杆: 轴3(X), 轴4(Y)")
    print("    - 向左: X=-32767, 向右: X=+32767")
    print("    - 向上: Y=-32767, 向下: Y=+32767")
    print("  按 Start(7) 退出\n")
    print("="*70 + "\n")
    
    try:
        with F710GamePadLinux() as gamepad:
            time.sleep(0.5)  # 等待初始化
            
            last_print_time = time.time()
            
            while True:
                # 获取摇杆值 (归一化到 [-1, 1])
                left_x, left_y = gamepad.get_left_stick(normalize=True)
                right_x, right_y = gamepad.get_right_stick(normalize=True)
                
                # 应用死区
                left_x = apply_deadzone(left_x, 0.05)
                left_y = apply_deadzone(left_y, 0.05)
                right_x = apply_deadzone(right_x, 0.05)
                right_y = apply_deadzone(right_y, 0.05)
                
                # 获取按钮
                buttons = gamepad.get_buttons()
                
                # 每 0.1 秒打印一次
                current_time = time.time()
                if current_time - last_print_time > 0.1:
                    print(f"\r"
                          f"左摇杆: X={left_x:+.3f} Y={left_y:+.3f}  "
                          f"右摇杆: X={right_x:+.3f} Y={right_y:+.3f}  "
                          f"按钮: {buttons if buttons else '无'}     ",
                          end='', flush=True)
                    last_print_time = current_time
                
                # Start 按钮退出
                if gamepad.is_button_pressed(gamepad.BTN_START):
                    print("\n\n✅ Start 按钮按下,退出...")
                    break
                
                time.sleep(0.02)  # 50Hz
    
    except KeyboardInterrupt:
        print("\n\n✅ 用户中断 (Ctrl+C)")
    
    except Exception as e:
        print(f"\n\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("测试结束")
    print("="*70)

import datetime
import sys
import os
import json
import time
import threading
import traceback
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

# --- UI / System ---
import ctypes
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal, QRect
from PySide6.QtGui import QColor, QCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLineEdit, QFileDialog, QTextEdit, QSpinBox, QColorDialog, QTabWidget,
    QMessageBox, QCheckBox, QComboBox
)

# --- Screen / Input / Imaging ---
import mss
import pyautogui
from pynput import mouse, keyboard
from pynput.keyboard import GlobalHotKeys
import numpy as np
import cv2

DEBUG = False

# 等待时间宏变量定义
WAIT_REFRESH_ITEM = 0  # 点击货物后的等待时间
WAIT_CLICK_MAX_AMOUNT = 0  # 点击最大数量按钮后的等待时间
WAIT_AFTER_BUY = 0  # 购买后的等待时间
WAIT_AFTER_ESC = 0  # 按ESC后的等待时间

ENABLE_TESSERACT = False
ENABLE_PADDLEOCR = False
ENABLE_EASYOCR = True


# 添加获取系统DPI缩放因子的函数
def get_system_scaling_factor():
    try:
        # 获取系统DPI缩放因子
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        # 获取主显示器的缩放因子
        dpi = user32.GetDpiForSystem()
        # Windows默认DPI是96，计算缩放比例
        scaling_factor = dpi / 96.0
        return scaling_factor
    except:
        # 如果获取失败，默认返回1.0（无缩放）
        return 1.0


# 获取系统缩放因子
system_scaling_factor = get_system_scaling_factor()


# ============================= OCR Manager (GPU first) =============================
class OCRManager:
    """Try Tesseract OCR first, then PaddleOCR (GPU) -> EasyOCR (GPU). Fallback to CPU (not recommended)."""

    def __init__(self, logger):
        self.logger = logger
        self.backend = None  # "tesseract" | "paddle" | "easyocr"
        self.tesseract = None
        self.paddle = None
        self.easy = None
        # 添加调试图像保存目录
        self.debug_images_dir = "debug_ocr_images"
        # 确保目录存在
        if not os.path.exists(self.debug_images_dir):
            os.makedirs(self.debug_images_dir)

        # 初始化OCR引擎
        self._init_ocr()

    def _init_ocr(self):
        # Try Tesseract OCR (最高优先级)
        if ENABLE_TESSERACT:
            try:
                import pytesseract
                # 明确指定Tesseract的安装路径
                pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # 修改为你的实际安装路径

                # 测试Tesseract是否可用
                test_text = pytesseract.image_to_string(np.zeros((10, 10), dtype=np.uint8))
                self.tesseract = pytesseract
                self.backend = "tesseract"
                self.logger("OCR: Tesseract OCR 已启用")
                return
            except Exception as e:
                self.logger(f"OCR: Tesseract OCR 不可用: {e}")
        else:
            self.logger("OCR: Tesseract OCR 已禁用")

        # Try PaddleOCR GPU (次高优先级)
        if ENABLE_PADDLEOCR:
            try:
                from paddleocr import PaddleOCR
                self.paddle = PaddleOCR(use_angle_cls=False, lang='en')
                self.backend = "paddle"
                self.logger("OCR: PaddleOCR(GPU) 已启用")
                return
            except Exception as e:
                self.logger(f"OCR: PaddleOCR(GPU) 不可用: {e}")
        else:
            self.logger("OCR: PaddleOCR(GPU) 已禁用")

        # Try EasyOCR GPU (第三优先级)
        if ENABLE_EASYOCR:
            try:
                import easyocr
                self.easy = easyocr.Reader(['en'], gpu=True)  # loads model once
                self.backend = "easyocr"
                self.logger("OCR: EasyOCR(GPU) 已启用")
                return
            except Exception as e:
                self.logger(f"OCR: EasyOCR(GPU) 不可用: {e}")
        else:
            self.logger("OCR: EasyOCR(GPU) 已禁用")

        # Fallback CPU
        try:
            from paddleocr import PaddleOCR
            self.paddle = PaddleOCR(use_angle_cls=False, lang='en')
            self.backend = "paddle"
            self.logger("⚠️ OCR: GPU 不可用，暂用 PaddleOCR(CPU)")
        except Exception:
            try:
                import easyocr
                self.easy = easyocr.Reader(['en'], gpu=False)
                self.backend = "easyocr"
                self.logger("⚠️ OCR: GPU 不可用，暂用 EasyOCR(CPU)")
            except Exception as e:
                self.logger("❌ OCR 初始化失败，请安装 Tesseract OCR、PaddleOCR 或 EasyOCR")
                raise e

    @staticmethod
    def _preprocess(img: np.ndarray, scale: float = 2.0, binarize: bool = True):
        """
        Preprocess ROI: grayscale -> resize -> (optional) threshold -> morphology
        """
        if img is None or img.size == 0:
            return img
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        """
        if scale != 1.0:
            h, w = gray.shape[:2]
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        if binarize:
            _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((2, 2), np.uint8)
            th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
            return th"""
        return gray

    def _save_debug_image(self, img, prefix="preprocessed"):
        """
        保存调试图像到本地目录
        """
        try:
            filename = "prefix_debug.png"

            # 如果图像是灰度图，转换为BGR以便正确保存
            if len(img.shape) == 2:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = img

            cv2.imwrite(filename, img_bgr)
            print(f"保存调试图像: {filename}")
        except Exception as e:
            print(f"保存调试图像失败: {e}")

    # 修改read_text方法，添加保存调试图像功能
    def read_digital(self, img_bgr: np.ndarray, digits_only: bool = True) -> str:
        """
        Return best numeric text detected in the image.
        """
        roi = self._preprocess(img_bgr, scale=2.0, binarize=True)

        # 保存预处理后的图像用于调试
        global DEBUG
        if DEBUG:
            self._save_debug_image(roi)

        if self.backend == "tesseract" and self.tesseract is not None:
            # 为Tesseract配置自定义选项，优化数字识别
            custom_config = r'--oem 3 --psm 6 outputbase digits'
            text = self.tesseract.image_to_string(roi, config=custom_config)
            # 清理识别结果中的空白字符
            text = text.strip()
        elif self.backend == "paddle" and self.paddle is not None:
            result = self.paddle.ocr(roi, cls=False, det=True, rec=True)
            text_candidates = []
            for line in result:
                for item in line:
                    txt, conf = item[1]
                    text_candidates.append((txt, conf))
            text_candidates.sort(key=lambda x: x[1], reverse=True)
            text = text_candidates[0][0] if text_candidates else ""
        elif self.backend == "easyocr" and self.easy is not None:
            result = self.easy.readtext(roi)
            result.sort(key=lambda x: x[2], reverse=True)
            text = result[0][1] if result else ""
        else:
            text = ""

        if digits_only:
            # 只保留数字字符，移除小数点和逗号
            cleaned = "".join(ch for ch in text if ch.isdigit())
            return cleaned
        return text

    # 修改read_price_value方法的数值转换逻辑
    def read_price_value(self, img_bgr: np.ndarray) -> Optional[int]:
        s = self.read_digital(img_bgr, digits_only=True)
        if not s:
            return None
        try:
            # 修改：直接将清理后的字符串转换为整数
            val = int(s)
            return val
        except Exception:
            return None


# ============================= Screen capture (thread-safe) =============================
class Screen:
    """
    修复 mss 在多线程中的句柄问题：
    - 每个线程首次使用时在该线程内创建 mss 实例
    """
    _tls = threading.local()  # thread-local storage

    def _ensure_ctx(self):
        if not hasattr(self._tls, "sct"):
            self._tls.sct = mss.mss()

    def grab_region(self, region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        region: (x, y, w, h)  -> BGR image
        """
        self._ensure_ctx()
        x, y, w, h = region
        monitor = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
        img = np.array(self._tls.sct.grab(monitor))  # BGRA
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # -> BGR
        return bgr

    @staticmethod
    def click(x: int, y: int, button="left"):
        pyautogui.moveTo(x, y)
        pyautogui.click(button=button)

    @staticmethod
    def press_esc():
        pyautogui.press('esc')

    @staticmethod
    def get_pixel(x: int, y: int) -> float | tuple[int, ...] | None:
        return pyautogui.screenshot().getpixel((x, y))


# ============================= Macro Recorder =============================
@dataclass
class MacroEvent:
    t: float  # timestamp relative to start
    kind: str  # "mouse_click" / "mouse_move" / "key_down" / "key_up"
    data: dict


class MacroRecorder:
    """
    方法2：宏相关

    """

    def __init__(self, logger):
        self.logger = logger
        self.events: List[MacroEvent] = []
        self._recording = False
        self._start_time = 0.0
        self.mouse_listener = None
        self.keyboard_listener = None

    def start(self):
        if self._recording:
            return
        self.logger("开始录制（再次点击停止）...")
        self.events.clear()
        self._recording = True
        self._start_time = time.time()

        def on_click(x, y, button, pressed):
            if not self._recording:
                return False
            t = time.time() - self._start_time
            if pressed:
                self.events.append(
                    MacroEvent(t, "mouse_click", {"x": x, "y": y, "button": str(button), "action": "down"}))
            else:
                self.events.append(
                    MacroEvent(t, "mouse_click", {"x": x, "y": y, "button": str(button), "action": "up"}))
            return True

        def on_move(x, y):
            if not self._recording:
                return False
            t = time.time() - self._start_time
            self.events.append(MacroEvent(t, "mouse_move", {"x": x, "y": y}))
            return True

        # 修改on_press函数
        def on_press(key):
            if not self._recording:
                return None  # 修改为None
            t = time.time() - self._start_time
            try:
                k = key.char if hasattr(key, 'char') and key.char else str(key)
            except:
                k = str(key)
            self.events.append(MacroEvent(t, "key_down", {"key": k}))
            return None  # 修改为None

        # 修改on_release函数
        def on_release(key):
            if not self._recording:
                return None  # 修改为None
            t = time.time() - self._start_time
            try:
                k = key.char if hasattr(key, 'char') and key.char else str(key)
            except:
                k = str(key)
            self.events.append(MacroEvent(t, "key_up", {"key": k}))
            return None  # 修改为None

        self.mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
        self.keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.mouse_listener.start()
        self.keyboard_listener.start()

    def stop(self):
        if not self._recording:
            return
        self._recording = False
        if self.mouse_listener:
            self.mouse_listener.stop()
        if self.keyboard_listener:
            self.keyboard_listener.stop()
        self.logger(f"录制结束，共 {len(self.events)} 个事件。")

    def replay(self, stop_flag_callable=lambda: False):
        """
        Replay recorded events. Respect timing intervals.
        """
        if not self.events:
            self.logger("无可回放的事件")
            return
        self.logger("开始回放宏...")
        base = time.time()
        for ev in self.events:
            if stop_flag_callable():
                self.logger("检测到停止，终止回放。")
                return
            now = time.time()
            delay = (base + ev.t) - now
            if delay > 0:
                time.sleep(delay)
            kind = ev.kind
            data = ev.data
            if kind == "mouse_move":
                pyautogui.moveTo(data["x"], data["y"])
            elif kind == "mouse_click":
                btn = "left"
                if "Button.right" in data.get("button", ""):
                    btn = "right"
                if data.get("action") == "down":
                    pyautogui.mouseDown(x=data["x"], y=data["y"], button=btn)
                else:
                    pyautogui.mouseUp(x=data["x"], y=data["y"], button=btn)
            elif kind == "key_down":
                pyautogui.keyDown(data["key"])
            elif kind == "key_up":
                pyautogui.keyUp(data["key"])
        self.logger("回放完成。")


# ============================= Config =============================
DEFAULT_CONFIG_PATH = "config.json"


@dataclass
class Region:
    x: int
    y: int
    w: int
    h: int


@dataclass
class AppConfig:
    trade_button: Tuple[int, int] = (0, 0)
    main_menu_button: Tuple[int, int] = (0, 0)
    category_button: Tuple[int, int] = (0, 0)
    buy_button: Tuple[int, int] = (0, 0)
    max_amount_button: Tuple[int, int] = (0, 0)

    price1_region: Region = field(default_factory=lambda: Region(0, 0, 0, 0))


    # --- 模式1新逻辑 ---
    mode1_item_click_coord: Tuple[int, int] = (0, 0)  # 每轮先点击该货物
    mode1_refresh_immediate: bool = True  # 不符合后 立即 Esc+再点货物，立刻下一轮
    max_amount_clicks: int = 2  # 购买前点击“最大额度”按钮的次数

    # 模式2 configuration
    mode2_price_coord: Tuple[int, int] = (0, 0)
    mode2_threshold: float = 0.0
    mode2_target_color_coord: Tuple[int, int] = (0, 0)
    mode2_target_color_rgb: Tuple[int, int, int] = (0, 255, 0)

    scan_interval_ms: int = 150  # OCR 周期（毫秒）

    @staticmethod
    def from_json(d: Dict[str, Any]):
        def _tuple(name, default=(0, 0)):
            v = d.get(name, list(default))
            return int(v[0]), int(v[1])

        def _region(name):
            v = d.get(name, [0, 0, 0, 0])
            return Region(int(v[0]), int(v[1]), int(v[2]), int(v[3]))

        def _color_tuple(name, default=(0, 255, 0)):
            v = d.get(name, list(default))
            return int(v[0]), int(v[1]), int(v[2])

        cfg = AppConfig()
        cfg.trade_button = _tuple("trade_button")
        cfg.main_menu_button = _tuple("main_menu_button")
        cfg.category_button = _tuple("category_button")
        cfg.buy_button = _tuple("buy_button")
        cfg.max_amount_button = _tuple("max_amount_button")
        cfg.price1_region = _region("price1_region")
        cfg.price2_region = _region("price2_region")

        # 模式1新增
        cfg.mode1_item_click_coord = _tuple("mode1_item_click_coord")
        cfg.mode1_refresh_immediate = bool(d.get("mode1_refresh_immediate", True))
        cfg.max_amount_clicks = int(d.get("max_amount_clicks", 2))

        cfg.mode2_price_coord = _tuple("mode2_price_coord")
        cfg.mode2_threshold = float(d.get("mode2_threshold", 0.0))
        cfg.mode2_target_color_coord = _tuple("mode2_target_color_coord")
        cfg.mode2_target_color_rgb = _color_tuple("mode2_target_color_rgb")
        cfg.scan_interval_ms = int(d.get("scan_interval_ms", 150))
        return cfg

    def to_json(self) -> Dict[str, Any]:
        return {
            "trade_button": list(self.trade_button),
            "main_menu_button": list(self.main_menu_button),
            "category_button": list(self.category_button),
            "buy_button": list(self.buy_button),
            "max_amount_button": list(self.max_amount_button),
            "price1_region": [self.price1_region.x, self.price1_region.y, self.price1_region.w, self.price1_region.h],
            # 删除价格2区域配置
            # "price2_region": [self.price2_region.x, self.price2_region.y, self.price2_region.w, self.price2_region.h],

            # 模式1新增
            "mode1_item_click_coord": list(self.mode1_item_click_coord),
            "mode1_refresh_immediate": self.mode1_refresh_immediate,
            "max_amount_clicks": self.max_amount_clicks,

            "mode2_price_coord": list(self.mode2_price_coord),
            "mode2_threshold": self.mode2_threshold,
            "mode2_target_color_coord": list(self.mode2_target_color_coord),
            "mode2_target_color_rgb": list(self.mode2_target_color_rgb),
            "scan_interval_ms": self.scan_interval_ms,
        }


class ConfigManager:
    def __init__(self, path=DEFAULT_CONFIG_PATH, logger=lambda s: None):
        self.path = path
        self.logger = logger
        self.config = AppConfig()
        self.all_configs = {DEFAULT_CONFIG_PATH: self.config}  # 存储所有加载的配置文件
        self.current_config_name = DEFAULT_CONFIG_PATH

    def load(self):
        if os.path.exists(self.path):
            try:
                # 检查文件大小，避免空文件
                if os.path.getsize(self.path) == 0:
                    self.logger(f"配置文件为空：{self.path}")
                    # 使用默认配置并保存到文件
                    self.config = AppConfig()
                    # 修改这里，只使用文件名作为键
                    config_name = os.path.basename(self.path)
                    self.all_configs[config_name] = self.config
                    self.current_config_name = config_name
                    # 将默认配置保存到空文件中
                    self.save()
                    return

                with open(self.path, "r", encoding="utf-8") as f:
                    try:
                        d = json.load(f)
                        self.config = AppConfig.from_json(d)
                        # 修改这里，只使用文件名作为键
                        config_name = os.path.basename(self.path)
                        self.all_configs[config_name] = self.config
                        self.current_config_name = config_name
                        self.logger(f"配置已读取：{self.path}")
                    except json.JSONDecodeError as e:
                        self.logger(f"配置文件格式错误：{self.path}，错误：{str(e)}")
                        # 使用默认配置
                        self.config = AppConfig()
                        # 修改这里，只使用文件名作为键
                        config_name = os.path.basename(self.path)
                        self.all_configs[config_name] = self.config
                        self.current_config_name = config_name
                    except Exception as e:
                        self.logger(f"读取配置文件失败：{self.path}，错误：{str(e)}")
                        # 使用默认配置
                        self.config = AppConfig()
                        # 修改这里，只使用文件名作为键
                        config_name = os.path.basename(self.path)
                        self.all_configs[config_name] = self.config
                        self.current_config_name = config_name
            except Exception as e:
                self.logger(f"打开配置文件失败：{self.path}，错误：{str(e)}")
                # 使用默认配置
                self.config = AppConfig()
                # 修改这里，只使用文件名作为键
                config_name = os.path.basename(self.path)
                self.all_configs[config_name] = self.config
                self.current_config_name = config_name
        else:
            self.logger("未发现配置文件，将在配置完成后自动保存。")

    def load_all_configs(self):
        """扫描并加载当前目录下所有的JSON配置文件"""
        # 确保目录存在
        config_dir = os.path.dirname(self.path) if os.path.dirname(self.path) else '.'

        try:
            # 获取目录下所有的JSON文件
            json_files = [f for f in os.listdir(config_dir)
                          if f.endswith('.json') and os.path.isfile(os.path.join(config_dir, f))]

            # 加载每个JSON文件
            for json_file in json_files:
                file_path = os.path.join(config_dir, json_file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        d = json.load(f)
                    config = AppConfig.from_json(d)
                    self.all_configs[json_file] = config
                    self.logger(f"已加载配置文件：{json_file}")
                except Exception as e:
                    self.logger(f"加载配置文件 {json_file} 失败：{str(e)}")

            # 如果有配置文件，默认使用第一个
            if json_files:
                self.current_config_name = json_files[0]
                self.config = self.all_configs[self.current_config_name]
                self.path = os.path.join(config_dir, self.current_config_name)
                self.logger(f"当前使用配置：{self.current_config_name}")
        except Exception as e:
            self.logger(f"扫描配置文件失败：{str(e)}")

    def save(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_json(), f, indent=2, ensure_ascii=False)
        self.logger(f"配置已保存：{self.path}")
        # 更新配置缓存
        self.all_configs[self.current_config_name] = self.config

    def switch_config(self, config_name):
        """切换到指定的配置文件"""
        if config_name in self.all_configs:
            self.current_config_name = config_name
            self.config = self.all_configs[config_name]
            self.path = os.path.join(os.path.dirname(self.path) if os.path.dirname(self.path) else '.', config_name)
            self.logger(f"已切换到配置：{config_name}")
            return True
        return False

    def get_config_names(self):
        """获取所有已加载的配置文件名列表"""
        return list(self.all_configs.keys())


class Mode1Worker(QtCore.QThread):
    log = Signal(str)
    finished = Signal()
    # 修改信号定义，删除价格2参数
    price_signal = Signal(float)  # 只保留price1

    def __init__(self, config: AppConfig, ocr: OCRManager, stop_flag: threading.Event,
                 threshold: float, logger, parent=None):
        super().__init__(parent)
        self.cfg = config
        self.ocr = ocr
        self.stop_flag = stop_flag
        self.threshold = threshold
        self.logger = logger
        self.screen = Screen()

    def _refresh_item(self):
        ix, iy = self.cfg.mode1_item_click_coord
        if ix or iy:
            Screen.click(ix, iy)
            time.sleep(WAIT_REFRESH_ITEM)

    def _click_max_amount(self):
        x, y = self.cfg.max_amount_button
        clicks = max(1, int(self.cfg.max_amount_clicks))
        for _ in range(clicks):
            Screen.click(x, y)
            time.sleep(WAIT_CLICK_MAX_AMOUNT)

    def run(self):
        try:
            self.log.emit("模式1：开始监控...")
            interval = max(30, int(self.cfg.scan_interval_ms)) / 1000.0
            refresh_count = 0  # 添加刷新计数器
            while not self.stop_flag.is_set():
                bought = False

                # 1) 点货物（始终先点）
                self._refresh_item()

                # 2) OCR 价格1 - 添加调试代码
                r1 = self.cfg.price1_region
                img1 = self.screen.grab_region((r1.x, r1.y, r1.w, r1.h))

                # 添加调试代码：保存截图以便查看
                # 全局DEBUG开关
                global DEBUG
                if DEBUG:
                    import cv2
                    cv2.imwrite("price1_debug.png", img1)
                    self.log.emit(f"已保存价格1区域截图到 price1_debug.png")

                    # 添加调试代码：显示区域信息
                    self.log.emit(f"价格1区域: x={r1.x}, y={r1.y}, w={r1.w}, h={r1.h}")

                # 初始OCR识别
                p1 = self.ocr.read_price_value(img1)

                # 如果识别失败，重新截取图片并再次尝试识别
                if p1 is None:
                    self.log.emit("[价格1] 首次识别失败，重新截取图片并再次尝试...")
                    # 重新截取图片
                    img1_retry = self.screen.grab_region((r1.x, r1.y, r1.w, r1.h))
                    # 再次进行OCR识别
                    p1 = self.ocr.read_price_value(img1_retry)

                if p1 is not None:
                    # 修改信号发送，只发送价格1
                    self.price_signal.emit(p1)
                    self.log.emit(f"[价格1] {p1}")
                else:
                    self.log.emit("[价格1] 识别失败")
                    # 添加调试代码：尝试获取原始识别文本
                    raw_text = self.ocr.read_digital(img1, digits_only=False)
                    self.log.emit(f"[调试] 原始OCR文本: '{raw_text}'")

                # 3) 判定 & 购买流程
                if p1 is not None and p1 < self.threshold:
                    # 最大额度（多次点击）
                    self._click_max_amount()
                    bx, by = self.cfg.buy_button
                    Screen.click(bx, by)
                    self.log.emit(f"✅ 触发购买！价格1={p1} 阈值={self.threshold}")
                    bought = True
                    time.sleep(WAIT_AFTER_BUY)
                    refresh_count = 0  # 购买成功后重置计数器

                # 4) 若本轮未买成：按 Esc → 再点货物（立即刷新到下一轮）
                if not bought:
                    Screen.press_esc()
                    time.sleep(WAIT_AFTER_ESC)
                    if self.cfg.mode1_refresh_immediate:
                        # 立即刷新，不等间隔
                        self._refresh_item()
                        refresh_count += 1  # 增加刷新计数
                        # 检查是否需要插入等待
                        if refresh_count % 300 == 0:
                            self.log.emit("已连续刷新300次，插入3秒等待...")
                            time.sleep(3.0)
                        # 直接继续下一轮
                        continue

                # 5) 正常间隔
                slept = 0.0
                while slept < interval:
                    if self.stop_flag.is_set():
                        break
                    t = min(0.02, interval - slept)
                    time.sleep(t)
                    slept += t

            self.log.emit("模式1：已停止。")
        except Exception as e:
            self.log.emit("模式1线程异常：" + str(e))
            self.log.emit(traceback.format_exc())
        finally:
            self.finished.emit()


# ============================= UI =============================
class DragPickButton(QPushButton):
    """
    拖到屏幕任意位置松开以采集坐标；coordPicked(x, y)
    """
    coordPicked = Signal(int, int)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        super().mousePressEvent(e)
        self.setCursor(Qt.ClosedHandCursor)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        super().mouseReleaseEvent(e)
        self.setCursor(Qt.ArrowCursor)
        pos = QCursor.pos()
        # 应用缩放因子（使用四舍五入）
        scaled_x = round(pos.x() * system_scaling_factor)
        scaled_y = round(pos.y() * system_scaling_factor)
        self.coordPicked.emit(scaled_x, scaled_y)


class RegionPickerOverlay(QWidget):
    """
    全屏半透明覆盖层，框选矩形区域；regionSelected(x,y,w,h)
    """
    regionSelected = Signal(int, int, int, int)

    def __init__(self):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.start = None
        self.end = None
        self.setMouseTracking(True)
        self.showFullScreen()

    def paintEvent(self, e):
        if self.start and self.end:
            p = QtGui.QPainter(self)
            p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            rect = QRect(self.start, self.end).normalized()
            p.setPen(QtGui.QPen(QtGui.QColor(0, 255, 0, 200), 2))
            p.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 0, 50)))
            p.drawRect(rect)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        self.start = e.globalPosition().toPoint()
        self.end = self.start
        self.update()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        self.end = e.globalPosition().toPoint()
        self.update()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        self.end = e.globalPosition().toPoint()
        rect = QRect(self.start, self.end).normalized()
        # 应用缩放因子到区域坐标（使用四舍五入）
        scaled_x = round(rect.x() * system_scaling_factor)
        scaled_y = round(rect.y() * system_scaling_factor)
        scaled_w = round(rect.width() * system_scaling_factor)
        scaled_h = round(rect.height() * system_scaling_factor)
        self.regionSelected.emit(scaled_x, scaled_y, scaled_w, scaled_h)
        self.close()


# ============================= Main Window =============================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("游戏商城自动抢货工具（GPU OCR）")
        self.resize(1024, 720)

        self.log_box = QTextEdit(readOnly=True)
        self.log_box.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)

        self.cfg_mgr = ConfigManager(logger=self._log)
        self.cfg_mgr.load()
        self.ocr = OCRManager(logger=self._log)
        self.stop_flag = threading.Event()

        self.mode1_thread: Optional[Mode1Worker] = None

        # Macros for Mode 2
        self.macro1 = MacroRecorder(self._log)

        # Global hotkeys
        self._gh_listener = None
        self._setup_global_hotkeys()

        self._build_ui()
        self._log(
            "提示：窗口内 F2 捕捉点、F3 框选、F8 开始、F9 停止；同时支持**全局热键**：F8(模式1)、Shift+F8(模式2)、F9(停止)。")

    # ---------- Global Hotkeys ----------
    def _setup_global_hotkeys(self):
        def start_mode1():
            self._start_mode1()

        def start_mode2():
            self._start_mode2()

        def stop_all():
            self._stop_all()

        # 全局监听（需要管理员权限）
        try:
            self._gh_listener = GlobalHotKeys({
                '<f8>': start_mode1,
                '<shift>+<f8>': start_mode2,
                '<f9>': stop_all
            })
            self._gh_listener.start()
            self._log("全局热键已注册：F8=模式1，Shift+F8=模式2，F9=停止。若无效请用管理员运行。")
        except Exception as e:
            self._log(f"⚠️ 全局热键注册失败：{e}（可用窗口内快捷键代替）")

    def closeEvent(self, e: QtGui.QCloseEvent):
        try:
            if self._gh_listener:
                self._gh_listener.stop()
        except:
            pass
        return super().closeEvent(e)

    # ---------- UI Tabs ----------
    def _build_ui(self):
        self.tabs = QTabWidget()

        self.tabs.addTab(self._build_config_tab(), "配置/坐标")
        self.tabs.addTab(self._build_mode1_tab(), "模式1：扫货")
        self.tabs.addTab(self._build_log_tab(), "日志")

        # Global controls
        topbar = QHBoxLayout()

        # 配置文件选择下拉框
        self.config_selector = QComboBox()
        self.config_selector.addItems(self.cfg_mgr.get_config_names())
        # 增加下拉框的最小宽度，防止文件名显示不全
        self.config_selector.setMinimumWidth(200)  # 设置为合适的宽度值
        if self.cfg_mgr.current_config_name in self.cfg_mgr.get_config_names():
            self.config_selector.setCurrentText(self.cfg_mgr.current_config_name)
        self.config_selector.currentTextChanged.connect(self._on_config_selected)
        topbar.addWidget(QLabel("配置文件："))
        topbar.addWidget(self.config_selector)

        self.btn_load = QPushButton("打开配置文件")
        self.btn_save = QPushButton("保存配置")
        self.btn_save_as = QPushButton("配置另存为")
        self.btn_load.clicked.connect(self._on_load)
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_as.clicked.connect(self._on_save_as)
        topbar.addWidget(self.btn_load)
        topbar.addWidget(self.btn_save)
        topbar.addWidget(self.btn_save_as)
        topbar.addStretch()

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(topbar)
        layout.addWidget(self.tabs)
        self.setCentralWidget(container)

        self.installEventFilter(self)

    def _build_log_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self.log_box)
        return w

    def _build_config_tab(self):
        w = QWidget()
        grid = QGridLayout(w)

        # 修改 coord_row 函数
        # 在_build_config_tab方法中修改coord_row函数
        def coord_row(label_text, getter, setter):
            row = QWidget()
            h = QHBoxLayout(row)
            h.addWidget(QLabel(label_text))
            x = QLineEdit()
            x.setPlaceholderText("x")
            y = QLineEdit()
            y.setPlaceholderText("y")
            pb = DragPickButton("拖我到目标地址（松开即记录）")
            pb.setFixedWidth(220)

            def on_pick(px, py):
                x.setText(str(px))
                y.setText(str(py))
                save_vals()

            pb.coordPicked.connect(on_pick)

            def load_vals():
                vx, vy = getter()
                x.setText(str(vx))
                y.setText(str(vy))

            def save_vals():
                try:
                    setter((int(x.text()), int(y.text())))
                    self._log(f"{label_text} 坐标设置为 {x.text()},{y.text()}")
                except:
                    pass

            # 添加这两行实现即时保存
            x.editingFinished.connect(save_vals)
            y.editingFinished.connect(save_vals)
            h.addWidget(x)
            h.addWidget(y)
            h.addWidget(pb)

            # 自动加载配置值到输入框
            load_vals()

            return row

        # 修改 region_row 函数
        # 在_build_config_tab方法中修改region_row函数
        def region_row(label_text, getter, setter):
            row = QWidget()
            h = QHBoxLayout(row)
            h.addWidget(QLabel(label_text))
            ex = QLineEdit()
            ex.setPlaceholderText("x")
            ey = QLineEdit()
            ey.setPlaceholderText("y")
            ew = QLineEdit()
            ew.setPlaceholderText("w")
            eh = QLineEdit()
            eh.setPlaceholderText("h")
            btn_pick = QPushButton("框选区域(F3)")

            def pick_region():
                overlay = RegionPickerOverlay()
                overlay.regionSelected.connect(lambda x, y, w, h: (
                    ex.setText(str(x)), ey.setText(str(y)), ew.setText(str(w)), eh.setText(str(h)), apply_region()
                ))
                overlay.show()

            def apply_region():
                try:
                    setter(Region(int(ex.text()), int(ey.text()), int(ew.text()), int(eh.text())))
                    self._log(f"{label_text} 设置为 ({ex.text()},{ey.text()},{ew.text()},{eh.text()})")
                except:
                    pass

            def load_vals():
                r = getter()
                ex.setText(str(r.x))
                ey.setText(str(r.y))
                ew.setText(str(r.w))
                eh.setText(str(r.h))

            btn_pick.clicked.connect(pick_region)
            # 添加这几行实现即时保存
            for edit in [ex, ey, ew, eh]:
                edit.editingFinished.connect(apply_region)
            h.addWidget(ex)
            h.addWidget(ey)
            h.addWidget(ew)
            h.addWidget(eh)
            h.addWidget(btn_pick)

            # 自动加载配置值到输入框
            load_vals()
            return row

        # 基础坐标
        grid.addWidget(coord_row("交易行按钮", lambda: self.cfg_mgr.config.trade_button, self._set_trade), 0, 0)
        grid.addWidget(coord_row("主界面按钮", lambda: self.cfg_mgr.config.main_menu_button, self._set_main), 1, 0)
        grid.addWidget(coord_row("装备分类按钮", lambda: self.cfg_mgr.config.category_button, self._set_category), 2, 0)
        grid.addWidget(coord_row("购买按钮", lambda: self.cfg_mgr.config.buy_button, self._set_buy), 3, 0)
        grid.addWidget(coord_row("最大额度按钮", lambda: self.cfg_mgr.config.max_amount_button, self._set_max), 4, 0)
        grid.addWidget(region_row("价格1区域", lambda: self.cfg_mgr.config.price1_region, self._set_price1), 5, 0)
        # 删除价格2区域配置行
        # grid.addWidget(region_row("价格2区域", lambda: self.cfg_mgr.config.price2_region, self._set_price2), 6, 0)

        # （模式1）货物点击坐标 + 立即刷新开关 + 最大额度点击次数
        grid.addWidget(coord_row("（模式1）货物点击坐标",
                                 lambda: self.cfg_mgr.config.mode1_item_click_coord,
                                 lambda xy: setattr(self.cfg_mgr.config, "mode1_item_click_coord", xy)
                                 ), 7, 0)

        h2 = QHBoxLayout()
        self.cb_refresh_immediate = QCheckBox("不符合立即 Esc+再点货物刷新（推荐）")
        self.cb_refresh_immediate.setChecked(self.cfg_mgr.config.mode1_refresh_immediate)
        self.cb_refresh_immediate.stateChanged.connect(
            lambda s: setattr(self.cfg_mgr.config, "mode1_refresh_immediate", bool(s)))
        h2.addWidget(self.cb_refresh_immediate)

        h2.addWidget(QLabel("最大额度点击次数："))
        self.spin_max_clicks = QSpinBox()
        self.spin_max_clicks.setRange(0, 5)
        self.spin_max_clicks.setValue(self.cfg_mgr.config.max_amount_clicks)
        self.spin_max_clicks.valueChanged.connect(lambda v: setattr(self.cfg_mgr.config, "max_amount_clicks", int(v)))
        h2.addWidget(self.spin_max_clicks)
        h2.addStretch()
        grid.addLayout(h2, 8, 0)

        # Interval
        h = QHBoxLayout()
        h.addWidget(QLabel("OCR间隔(ms)："))
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(30, 5000)
        self.spin_interval.setValue(self.cfg_mgr.config.scan_interval_ms)
        self.spin_interval.valueChanged.connect(lambda v: setattr(self.cfg_mgr.config, "scan_interval_ms", int(v)))
        h.addWidget(self.spin_interval)
        h.addStretch()
        grid.addLayout(h, 9, 0)

        tips = QLabel(
            "提示：拖拽按钮到目标位置（松开即记录）；窗口内也支持 F2/F3/F8/F9。若切到游戏，用全局热键 F8/Shift+F8/F9。")
        tips.setWordWrap(True)
        grid.addWidget(tips, 10, 0)

        return w

    def _build_mode1_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        h = QHBoxLayout()
        h.addWidget(QLabel("扫货最低价阈值："))
        self.edit_threshold = QLineEdit()
        self.edit_threshold.setPlaceholderText("例如 1234.56")
        h.addWidget(self.edit_threshold)

        self.btn_mode1_start = QPushButton("开始模式1（F8 / 全局F8）")
        self.btn_mode1_stop = QPushButton("停止（F9 / 全局F9）")
        self.btn_mode1_start.clicked.connect(self._start_mode1)
        self.btn_mode1_stop.clicked.connect(self._stop_all)
        h.addWidget(self.btn_mode1_start)
        h.addWidget(self.btn_mode1_stop)
        h.addStretch()
        v.addLayout(h)

        self.lbl_price1 = QLabel("价格1：-")
        # 删除价格2标签
        # self.lbl_price2 = QLabel("价格2：-")
        v.addWidget(self.lbl_price1)
        # v.addWidget(self.lbl_price2)

        return w

    # ---------------- Event Filter for window-level hotkeys ----------------
    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key == Qt.Key_F2:
                pos = QCursor.pos()
                # 应用缩放因子，将Qt逻辑坐标转换为物理像素坐标（使用四舍五入）
                scaled_x = round(pos.x() * system_scaling_factor)
                scaled_y = round(pos.y() * system_scaling_factor)
                self._log(f"F2 捕捉坐标：{scaled_x},{scaled_y} (缩放因子: {system_scaling_factor})")
                self._fill_focused_coord(scaled_x, scaled_y)
                return True
            elif key == Qt.Key_F3:
                self._log("F3：框选区域")
                overlay = RegionPickerOverlay()
                overlay.regionSelected.connect(lambda x, y, w, h: self._log(f"区域：{x},{y},{w},{h}"))
                overlay.show()
                return True
            elif key == Qt.Key_F8:
                self._start_mode1()
                return True
            elif key == Qt.Key_F9:
                self._stop_all()
                return True
        return super().eventFilter(obj, event)

    def _fill_focused_coord(self, x, y):
        w = QApplication.focusWidget()
        if isinstance(w, QLineEdit):
            w.setText(str(x))
            # 尝试给下一个焦点控件写入 y
            QApplication.sendEvent(self, QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, Qt.Key_Tab, Qt.NoModifier))
            w2 = QApplication.focusWidget()
            if isinstance(w2, QLineEdit):
                w2.setText(str(y))

    # ---------------- Config setters ----------------
    def _set_trade(self, xy):
        self.cfg_mgr.config.trade_button = xy

    def _set_main(self, xy):
        self.cfg_mgr.config.main_menu_button = xy

    def _set_category(self, xy):
        self.cfg_mgr.config.category_button = xy

    def _set_buy(self, xy):
        self.cfg_mgr.config.buy_button = xy

    def _set_max(self, xy):
        self.cfg_mgr.config.max_amount_button = xy

    def _set_price1(self, r: 'Region'):
        self.cfg_mgr.config.price1_region = r


    # ---------------- Buttons ----------------
    def _on_load(self):
        try:
            path, _ = QFileDialog.getOpenFileName(self, "选择配置", ".", "JSON (*.json)")
            if path:
                self.cfg_mgr.path = path
                self.cfg_mgr.load()

                # 1. 更新ConfigManager的all_configs字典，将新配置添加进去
                config_name = os.path.basename(path)
                self.cfg_mgr.all_configs[config_name] = self.cfg_mgr.config
                self.cfg_mgr.current_config_name = config_name

                # 2. 清空下拉框并重新填充所有配置名称
                self.config_selector.clear()
                self.config_selector.addItems(self.cfg_mgr.get_config_names())

                # 3. 选中当前加载的配置
                self.config_selector.setCurrentText(config_name)

                # refresh UI reflect critical fields
                self.spin_interval.setValue(self.cfg_mgr.config.scan_interval_ms)
                # 模式1新增字段
                self.cb_refresh_immediate.setChecked(self.cfg_mgr.config.mode1_refresh_immediate)
                self.spin_max_clicks.setValue(self.cfg_mgr.config.max_amount_clicks)

                self._log(f"已成功加载配置文件：{path}")
        except Exception as e:
            self._log("读取失败：" + str(e))
            self._log(traceback.format_exc())

    def _on_config_selected(self, config_name):
        try:
            # 使用switch_config方法来正确切换配置
            if self.cfg_mgr.switch_config(config_name):
                # 刷新UI以反映配置更改
                self.spin_interval.setValue(self.cfg_mgr.config.scan_interval_ms)
                # 模式1相关配置
                self.cb_refresh_immediate.setChecked(self.cfg_mgr.config.mode1_refresh_immediate)
                self.spin_max_clicks.setValue(self.cfg_mgr.config.max_amount_clicks)

                # 刷新配置标签页以自动加载坐标和区域值
                # 首先获取当前选中的标签页索引
                current_tab_index = self.tabs.currentIndex()
                # 移除并重新添加配置标签页
                self.tabs.removeTab(0)  # 配置标签页通常是第一个
                self.tabs.insertTab(0, self._build_config_tab(), "配置/坐标")
                # 如果之前选中的是配置标签页，则重新选中它
                if current_tab_index == 0:
                    self.tabs.setCurrentIndex(0)

            self._log(f"已加载配置：{config_name}")
        except Exception as e:
            self._log(f"加载配置失败：{e}")
            self._log(traceback.format_exc())

    def _on_save(self):
        try:
            try:
                self.cfg_mgr.config.scan_interval_ms = int(self.spin_interval.value())
            except:
                pass
            self.cfg_mgr.save()
        except Exception as e:
            self._log("保存失败：" + str(e))
            self._log(traceback.format_exc())

    def _on_save_as(self):
        try:
            # 更新配置对象中的值
            try:
                self.cfg_mgr.config.scan_interval_ms = int(self.spin_interval.value())
            except:
                pass

            # 打开文件保存对话框
            # 默认保存到当前目录，并使用当前配置文件名作为默认文件名
            current_dir = os.path.dirname(self.cfg_mgr.path) if os.path.dirname(self.cfg_mgr.path) else '.'
            default_filename = os.path.basename(self.cfg_mgr.path) if self.cfg_mgr.path else "config.json"

            path, _ = QFileDialog.getSaveFileName(
                self, "配置另存为",
                os.path.join(current_dir, default_filename),
                "JSON Files (*.json);;All Files (*)"
            )

            if path:
                # 确保文件扩展名为.json
                if not path.endswith('.json'):
                    path += '.json'

                # 保存配置到新路径
                original_path = self.cfg_mgr.path
                self.cfg_mgr.path = path
                self.cfg_mgr.save()

                # 更新配置管理器
                config_name = os.path.basename(path)
                self.cfg_mgr.all_configs[config_name] = self.cfg_mgr.config
                self.cfg_mgr.current_config_name = config_name

                # 刷新配置选择下拉框
                self.config_selector.clear()
                self.config_selector.addItems(self.cfg_mgr.get_config_names())
                self.config_selector.setCurrentText(config_name)

                self._log(f"配置已另存为：{path}")
        except Exception as e:
            self._log(f"配置另存为失败：{e}")
            self._log(traceback.format_exc())

    def _start_mode1(self):
        if self.mode1_thread and self.mode1_thread.isRunning():
            self._log("模式1已在运行。")
            return
        try:
            th = float(self.edit_threshold.text())
        except:
            # 将警告对话框替换为日志记录
            self._log("⚠️ 错误：请先输入有效的扫货最低价阈值")
            return
        self.stop_flag.clear()
        self.mode1_thread = Mode1Worker(self.cfg_mgr.config, self.ocr, self.stop_flag, th, logger=self._log)
        self.mode1_thread.log.connect(self._log)
        self.mode1_thread.price_signal.connect(self._on_price_update)
        self.mode1_thread.finished.connect(lambda: self._log("模式1线程结束"))
        self.mode1_thread.start()
        self._log("模式1启动。")

    def _stop_all(self):
        self.stop_flag.set()
        self._log("已请求停止（F9 / 全局F9）。")

    def _on_price_update(self, p1):
        # 修改价格更新方法，只处理价格1
        if p1 >= 0:
            self.lbl_price1.setText(f"价格1：{p1}")
        # 删除价格2更新代码
        # if p2 >= 0:
        #     self.lbl_price2.setText(f"价格2：{p2}")

    def _log(self, s: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)

        self.log_box.append(f"[{ts}] {s}")
        self.log_box.moveCursor(QtGui.QTextCursor.MoveOperation.End)


# ============================= main =============================
def main():
    pyautogui.FAILSAFE = True  # 鼠标移到左上角可紧急终止
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()

    # “几秒后没有配置则提示配置”
    QtCore.QTimer.singleShot(3500, lambda: (
        None if (win.cfg_mgr.config.price1_region.w > 0)
        # 删除价格2区域检查
        # else QMessageBox.information(win, "提示", "未检测到价格区域配置。\n请使用F3框选价格1/价格2区域，并保存配置。")
        else QMessageBox.information(win, "提示", "未检测到价格区域配置。\n请使用F3框选价格1区域，并保存配置。")
    ))
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
"""
MG400 + Webcam (Eye-to-hand) : จับหมากสีส้มและสีดำ

ใช้งาน:
  python mg400_vision.py calibrate   # สอบเทียบกล้อง -> หุ่น (ทำครั้งเดียว / ทำใหม่เมื่อกล้องขยับ)
  python mg400_vision.py camfocus    # ปรับโฟกัสกล้องด้วยแถบเลื่อน
  python mg400_vision.py setclear    # บันทึกจุดหลีกกล้อง (ทำครั้งเดียว)
  python mg400_vision.py pumptest    # ทดสอบปั๊ม (ไม่ขยับแขน)
  python mg400_vision.py pose        # อ่านตำแหน่งปัจจุบัน (ใช้วัด PICK_Z)
  python mg400_vision.py run         # ตรวจจับวัตถุ -> ไปเหนือวัตถุ -> ค้าง -> ถอยไปจุดหลีกกล้อง
"""
import os
import re
import socket
import sys
import time

import cv2
import numpy as np

# ==============================
# CONFIG
# ==============================
ROBOT_IP = "192.168.1.6"
DASHBOARD_PORT = 29999
MOVE_PORT = 30003
CAM_INDEX = 0
ROTATE_180 = True           # หมุนภาพกล้อง 180 องศา (ถ้าเปลี่ยนค่านี้ ต้องสอบเทียบใหม่)
CAM_FOCUS = None            # None = autofocus / 0-255 = ล็อกโฟกัสด้วยมือ (หาค่าด้วยโหมด camfocus)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(BASE_DIR, "calib.npz")
CLEAR_FILE = os.path.join(BASE_DIR, "clear_pos.npz")   # จุดหลีกกล้อง (ตั้งด้วยโหมด setclear)

ALLOW_MOVE = True           # True = หุ่นขยับจริง / False = ดูอย่างเดียวไม่สั่งหุ่น
SPEED = 20                   # % ความเร็ว
HOVER_ABOVE = 50.0          # mm เหนือระนาบที่สอบเทียบ
HOLD_SECONDS = 0.5          # ค้างหลังยกวัตถุขึ้น ก่อนถอยไปจุดหลีกกล้อง

# ---- ปั๊มสุญญากาศ ----
USE_PUMP = True             # True = ลงดูดวัตถุด้วยปั๊ม / False = แค่ไปเหนือวัตถุเหมือนเดิม
VAC_DO = 2                  # DO2 = ดูดสุญญากาศ (vacuum)
BLOW_DO = 1                 # DO1 = เป่าลม (pressure) ใช้ปล่อยวัตถุ
BLOW_SECONDS = 0.5          # เวลาเป่าลมตอนปล่อยวัตถุ
PUMP_SETTLE = 1.0           # วินาทีที่รอให้แรงดูดจับวัตถุหลังเปิดปั๊ม
PICK_Z = -149               # Z (mm) ที่หัวดูดแตะผิวบนของวัตถุ -> ต้องวัดเอง (ใช้โหมด pose)
# ขอบเขตพื้นที่ทำงานที่ปลอดภัย (mm) -> แก้ให้ตรงกับโต๊ะของคุณ
X_RANGE = (225.0, 420.0)
Y_RANGE = (-92.0, 95.0)
Z_RANGE = (-150.0, 0.0)

# ช่วงสี HSV ของวัตถุ -> ปรับตามวัตถุจริง (กด 'm' ตอนรัน เพื่อดูภาพ mask ช่วยปรับค่า)
#   H: 0-179, S: 0-255, V: 0-255
#   "circ" = ความกลมขั้นต่ำ (0 = ไม่กรอง) ใช้กรองเงา/เส้นขอบสีเข้มที่ไม่ใช่หมาก
COLORS = {
    "orange": {"low": (5, 120, 120), "high": (22, 255, 255), "circ": 0.0},
    "black":  {"low": (0, 0, 0),     "high": (179, 255, 110), "circ": 0.6},  # ดำ = V ต่ำ (หมากจริงวัดได้ V ~75-95)
}
TAG = {"orange": "O", "black": "B"}                       # ตัวอักษรต่อท้ายหมายเลขบนภาพ
DRAW_COLOR = {"orange": (0, 255, 0), "black": (255, 0, 255)}  # สีวงกลม (BGR) ที่ใช้วาด
MIN_AREA = 400
MAX_AREA = 15000            # ตัดวัตถุที่ใหญ่เกินหมาก (เช่น เงาพื้นที่กว้าง, ตัวหุ่น)
SHOW_OUT_MARGIN = 40.0     # แสดงวัตถุนอกพื้นที่ปลอดภัยเฉพาะที่อยู่ห่างขอบไม่เกินกี่ mm (ซ่อนมือ/ชิ้นส่วนหุ่นที่อยู่ไกล)
MAX_OBJECTS = 24            # จำนวนหมากสูงสุดที่แสดงหมายเลข (กดเลขได้ 1-9, ที่เหลือใช้คลิกเมาส์)


# ==============================
# ROBOT
# ==============================
class MG400:
    def __init__(self, ip):
        self.dash = self._open(ip, DASHBOARD_PORT, 5)
        self.move = self._open(ip, MOVE_PORT, 60)  # Sync() อาจรอนาน

    @staticmethod
    def _open(ip, port, timeout):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        return s

    @staticmethod
    def _txrx(sock, cmd):
        sock.sendall(cmd.encode())
        buf = b""
        while not buf.endswith(b";"):
            chunk = sock.recv(1024)
            if not chunk:
                raise ConnectionError("Robot closed connection")
            buf += chunk
        resp = buf.decode().strip()
        print(f">> {cmd}   << {resp}")
        err = int(resp.split(",")[0])
        if err != 0:
            raise RuntimeError(f"Robot error {err}: {resp}")
        return resp

    def d(self, cmd):
        return self._txrx(self.dash, cmd)

    def m(self, cmd):
        return self._txrx(self.move, cmd)

    def pose(self):
        resp = self.d("GetPose()")
        vals = re.search(r"\{(.*?)\}", resp).group(1).split(",")
        return [float(v) for v in vals[:4]]  # x, y, z, r

    def enable(self):
        self.d("ClearError()")
        self.d("EnableRobot()")
        for _ in range(30):
            if "{5}" in self.d("RobotMode()"):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Robot did not reach enabled state")
        self.d(f"SpeedFactor({SPEED})")

    # --- ปั๊ม: ไม่เปิดดูดกับเป่าพร้อมกันเด็ดขาด ---
    def vacuum_on(self):
        self.d(f"DO({BLOW_DO},0)")
        self.d(f"DO({VAC_DO},1)")

    def pump_off(self):
        """ปิดทั้งดูดและเป่า (ใช้ตอนเริ่ม / เกิด error / ออกโปรแกรม)"""
        self.d(f"DO({VAC_DO},0)")
        self.d(f"DO({BLOW_DO},0)")

    def release(self):
        """ปล่อยวัตถุ: ปิดดูด -> เป่าลมสั้น ๆ -> ปิดเป่า"""
        self.d(f"DO({VAC_DO},0)")
        self.d(f"DO({BLOW_DO},1)")
        time.sleep(BLOW_SECONDS)
        self.d(f"DO({BLOW_DO},0)")

    def close(self):
        for s in (self.dash, self.move):
            try:
                s.close()
            except Exception:
                pass


# ==============================
# CAMERA
# ==============================
def read_frame(cap):
    """อ่านภาพจากกล้อง และหมุน 180 องศาถ้าตั้ง ROTATE_180"""
    ok, frame = cap.read()
    if ok and ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return ok, frame


def apply_focus(cap):
    """ตั้งโฟกัส: CAM_FOCUS=None -> autofocus, ไม่งั้นปิด autofocus แล้วล็อกค่าโฟกัส"""
    if CAM_FOCUS is None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, float(CAM_FOCUS))


def open_camera():
    t0 = time.time()
    print("กำลังเปิดกล้อง ...")
    # Windows: DirectShow เปิดกล้อง USB เร็วกว่า backend เริ่มต้น (MSMF) มาก
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAM_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # BRIO ส่งภาพเร็วขึ้นเมื่อใช้ MJPG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    apply_focus(cap)
    for _ in range(10):  # ทิ้งเฟรมแรก ๆ ให้ exposure นิ่ง
        cap.read()
    apply_focus(cap)     # ตั้งซ้ำหลังกล้องเริ่มส่งภาพ (บางรุ่นไม่รับค่าตอนยังไม่มีเฟรม)
    for _ in range(5):
        cap.read()
    print(f"เปิดกล้องเสร็จ ({time.time() - t0:.1f} วินาที)")
    return cap


def camfocus():
    """
    ปรับโฟกัสด้วยแถบเลื่อน: วางวัตถุ/กระดาษที่มีตัวหนังสือบนโต๊ะ (ระดับเดียวกับที่วัตถุจะวาง)
    เลื่อนจนภาพคมที่สุด (ตัวเลข Sharpness สูงสุด) แล้วกด q จะแสดงค่าให้ไปใส่ CAM_FOCUS
    """
    cap = open_camera()
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    start = int(CAM_FOCUS) if CAM_FOCUS is not None else 100
    win = "camfocus"
    cv2.namedWindow(win)
    state = {"focus": start}

    def on_change(v):
        state["focus"] = v
        cap.set(cv2.CAP_PROP_FOCUS, float(v))

    cv2.createTrackbar("Focus", win, start, 255, on_change)
    cap.set(cv2.CAP_PROP_FOCUS, float(start))
    print("เลื่อนแถบ Focus จนภาพคมที่สุด แล้วกด 'q'")
    try:
        while True:
            ok, frame = read_frame(cap)
            if not ok:
                break
            h, w = frame.shape[:2]
            roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]  # วัดความคมที่กลางภาพ
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 255), 1)
            cv2.putText(frame, f"Focus={state['focus']}  Sharpness={sharp:.0f}  (Q = done)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, frame)
            if cv2.waitKey(30) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"\nใส่ค่านี้ใน CONFIG:  CAM_FOCUS = {state['focus']}")
    print("หลังเปลี่ยนโฟกัส ควรสอบเทียบ (calibrate) ใหม่")


def build_masks(frame):
    """สร้าง mask แยกตามสีที่กำหนดใน COLORS"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    masks = {}
    for name, cfg in COLORS.items():
        m = cv2.inRange(hsv, cfg["low"], cfg["high"])
        masks[name] = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    return masks


def find_objects(frame):
    """คืนรายการ (u, v, ชื่อสี) ของหมากทุกชิ้นที่ใหญ่พอและรูปร่างผ่านเกณฑ์"""
    pts = []
    for name, mask in build_masks(frame).items():
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < MIN_AREA or area > MAX_AREA:
                continue
            per = cv2.arcLength(c, True)
            if per == 0 or 4 * np.pi * area / (per * per) < COLORS[name]["circ"]:
                continue
            M = cv2.moments(c)
            pts.append((M["m10"] / M["m00"], M["m01"] / M["m00"], name))
    return pts


# ==============================
# CALIBRATE
# ==============================
def calibrate():
    """
    1) วางสติกเกอร์ 6-9 จุดบนโต๊ะ กระจายทั่วพื้นที่ทำงาน (ที่ระนาบเดียวกับวัตถุ)
    2) คลิกแต่ละจุดในภาพ แล้วกด 's'
    3) สคริปต์จะบอกให้ย้ายปลายหุ่นไปแตะจุดนั้นทีละจุด (ใช้ปุ่ม drag/jog) แล้วกด Enter
    """
    cap = open_camera()
    ok, frame = read_frame(cap)
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read frame")

    clicks = []
    view = frame.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, str(len(clicks)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    print("คลิกจุดอ้างอิงตามลำดับ แล้วกด 's' (ต้องมีอย่างน้อย 4 จุด, 'q' ยกเลิก)")
    while True:
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return
        if k == ord("s") and len(clicks) >= 4:
            break
    cv2.destroyAllWindows()

    print(f"กำลังเชื่อมต่อหุ่น {ROBOT_IP} ...")
    robot = MG400(ROBOT_IP)
    try:
        robot.d("ClearError()")
        robot_pts = []
        for i, _ in enumerate(clicks, 1):
            input(f"ย้ายปลายหุ่นไปแตะจุดที่ {i} แล้วกด Enter ...")
            x, y, z, _r = robot.pose()
            robot_pts.append((x, y, z))
            print(f"  จุด {i}: x={x:.1f} y={y:.1f} z={z:.1f}")
    finally:
        robot.close()

    px = np.array(clicks, dtype=np.float32)
    rb = np.array(robot_pts, dtype=np.float32)
    H, _ = cv2.findHomography(px, rb[:, :2], cv2.RANSAC, 5.0)

    # ตรวจความคลาดเคลื่อน
    proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - rb[:, :2], axis=1)
    print(f"\nReprojection error: mean={err.mean():.2f} mm, max={err.max():.2f} mm")
    if err.max() > 5:
        print("คำเตือน: error สูง ลองสอบเทียบใหม่ / เพิ่มจำนวนจุด")

    np.savez(CALIB_FILE, H=H, z_plane=float(rb[:, 2].mean()), rot180=ROTATE_180)
    print(f"บันทึกแล้วที่: {CALIB_FILE}")
    print(f"z ระนาบ = {rb[:, 2].mean():.1f} mm")
    mn, mx = rb[:, :2].min(axis=0), rb[:, :2].max(axis=0)
    print("\nจุดสอบเทียบครอบคลุม:")
    print(f"  X {mn[0]:.1f} ถึง {mx[0]:.1f}   Y {mn[1]:.1f} ถึง {mx[1]:.1f}")
    print("ค่าแนะนำสำหรับ CONFIG (เผื่อ 10 mm ตรวจเช็กว่าหุ่นเอื้อมถึงและปลอดภัยก่อนใช้):")
    print(f"  X_RANGE = ({mn[0] - 10:.0f}, {mx[0] + 10:.0f})")
    print(f"  Y_RANGE = ({mn[1] - 10:.0f}, {mx[1] + 10:.0f})")


# ==============================
# SET CLEAR POSITION (จุดหลีกกล้อง)
# ==============================
def setclear():
    """
    ย้ายปลายหุ่นไปจุดที่แขนไม่บังกล้อง (และไม่ชนอะไร) แล้วกด Enter เพื่อบันทึก
    แนะนำให้อยู่สูงกว่าความสูง hover ของวัตถุ เพื่อไม่ให้ชนวัตถุตอนถอย
    """
    robot = MG400(ROBOT_IP)
    try:
        robot.d("ClearError()")
        input("ย้ายปลายหุ่นไปจุดหลีกกล้อง แล้วกด Enter ...")
        x, y, z, r = robot.pose()
    finally:
        robot.close()
    np.savez(CLEAR_FILE, pos=np.array([x, y, z, r]))
    print(f"บันทึกจุดหลีกกล้อง: x={x:.1f} y={y:.1f} z={z:.1f} r={r:.1f}")
    print(f"ไฟล์: {CLEAR_FILE}")


# ==============================
# PUMP TEST / POSE
# ==============================
def pumptest():
    """ทดสอบปั๊มโดยไม่ขยับแขน: ดูด 3 วินาที แล้วปล่อย (เป่าลม)"""
    robot = MG400(ROBOT_IP)
    try:
        robot.enable()
        input(f"จะเปิดดูด (DO{VAC_DO}) 3 วินาที แล้วเป่าปล่อย (DO{BLOW_DO}) กด Enter เพื่อเริ่ม ...")
        robot.vacuum_on()
        time.sleep(3)
        robot.release()
        print("ปล่อยแล้ว")
    finally:
        try:
            robot.pump_off()
            print("ปิดปั๊มทั้งหมดแล้ว")
        finally:
            robot.close()


def show_pose():
    """อ่านตำแหน่งปัจจุบัน ใช้วัดค่า PICK_Z (เลื่อนหัวดูดไปแตะผิวบนวัตถุ แล้วอ่านค่า z)"""
    robot = MG400(ROBOT_IP)
    try:
        x, y, z, r = robot.pose()
        print(f"x={x:.1f} y={y:.1f} z={z:.1f} r={r:.1f}")
    finally:
        robot.close()


# ==============================
# RUN
# ==============================
def in_range(x, y):
    return X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]


def margin_outside(x, y):
    """ระยะ (mm) ที่อยู่นอกพื้นที่ปลอดภัย (0 = อยู่ข้างใน)"""
    dx = max(X_RANGE[0] - x, 0.0, x - X_RANGE[1])
    dy = max(Y_RANGE[0] - y, 0.0, y - Y_RANGE[1])
    return float(np.hypot(dx, dy))


def annotate(frame, objs):
    """
    วาดหมายเลขบนภาพ และรายการพิกัดเป็นแผงข้อความมุมซ้ายบน (ไม่ให้ตัวหนังสือทับกัน)
    คืนรายการ (x, y, u, v) ของหมากที่สั่งได้ เรียงตามหมายเลข (x,y = พิกัดหุ่น, u,v = พิกเซล)
    """
    numbered = []
    lines = []
    for (x, y, u, v, name) in objs:
        pt = (int(u), int(v))
        if in_range(x, y) and len(numbered) < MAX_OBJECTS:
            numbered.append((x, y, u, v))
            n = len(numbered)
            cv2.circle(frame, pt, 9, (255, 255, 255), 2)  # ขอบขาวให้เห็นชัดบนหมากสีเข้ม
            cv2.circle(frame, pt, 7, DRAW_COLOR[name], -1)
            cv2.putText(frame, f"{n}{TAG[name]}", (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
            cv2.putText(frame, f"{n}{TAG[name]}", (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, DRAW_COLOR[name], 2)
            lines.append((f"{n}{TAG[name]}  X={x:.0f} Y={y:.0f}", DRAW_COLOR[name]))
        elif margin_outside(x, y) <= SHOW_OUT_MARGIN:
            cv2.circle(frame, pt, 8, (0, 0, 255), 2)
            cv2.putText(frame, "OUT", (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            lines.append((f"OUT{TAG[name]} X={x:.0f} Y={y:.0f} (+{margin_outside(x, y):.0f}mm)", (0, 0, 255)))
        # ไกลจากพื้นที่ทำงานมาก (มือ/ตัวหุ่น/อื่น ๆ) ไม่วาดอะไร

    lines = lines[:24]
    if lines:
        panel_h = 14 + 22 * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 40), (300, 40 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (8, 62 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return numbered


def confirm_target(cap, to_robot, tx, ty):
    """อ่านภาพใหม่ 5 เฟรม ยืนยันว่าวัตถุที่เลือกยังอยู่ที่เดิมและนิ่ง"""
    samples = []
    for _ in range(5):
        ok, f = read_frame(cap)
        if not ok:
            return None
        cands = [to_robot(p[0], p[1]) for p in find_objects(f)]
        near = [c for c in cands if np.hypot(c[0] - tx, c[1] - ty) <= 5.0]
        if not near:
            return None
        samples.append(min(near, key=lambda c: np.hypot(c[0] - tx, c[1] - ty)))
    arr = np.array(samples)
    if np.ptp(arr, axis=0).max() > 3.0:
        return None
    return arr.mean(axis=0)


def pick_sequence(robot, x, y, z_hover, clear_pos):
    """
    เหนือวัตถุ -> ลงแตะ -> เปิดดูด -> ยกขึ้น -> ค้าง -> ถอยไปจุดหลีกกล้อง -> ปล่อยวัตถุ
    คืน True ถ้ายังถือวัตถุอยู่ (ปกติจะเป็น False เพราะปล่อยที่จุดหลีกกล้อง)
    """
    def go(px, py, pz, pr=0.0):
        robot.m(f"MovL({px:.1f},{py:.1f},{pz:.1f},{pr:.1f})")
        robot.m("Sync()")

    holding = False
    try:
        go(x, y, z_hover)
        if USE_PUMP:
            go(x, y, PICK_Z)
            robot.vacuum_on()
            holding = True
            time.sleep(PUMP_SETTLE)
            go(x, y, z_hover)
        print(f"ค้างไว้ {HOLD_SECONDS:.1f} วินาที ...")
        time.sleep(HOLD_SECONDS)
        cx, cy, cz, cr = clear_pos
        print("ถอยไปจุดหลีกกล้อง")
        go(cx, cy, cz, cr)
        if USE_PUMP:
            print("ปล่อยวัตถุที่จุดหลีกกล้อง")
            robot.release()
            holding = False
    except Exception:
        # เกิดปัญหากลางทาง: ปิดปั๊มเพื่อความปลอดภัย แล้วส่งต่อ error
        if USE_PUMP:
            try:
                robot.pump_off()
            except Exception:
                pass
        raise
    return holding


def run():
    data = np.load(CALIB_FILE)
    H, z_plane = data["H"], float(data["z_plane"])
    z_target = z_plane + HOVER_ABOVE
    calib_rot = bool(data["rot180"]) if "rot180" in data.files else False
    if calib_rot != ROTATE_180:
        raise RuntimeError(
            f"ค่า ROTATE_180 ตอนสอบเทียบ ({calib_rot}) ไม่ตรงกับตอนนี้ ({ROTATE_180}) "
            "ให้ตั้งให้ตรงกัน หรือสอบเทียบ (calibrate) ใหม่")

    def to_robot(u, v):
        p = H @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    if not (Z_RANGE[0] <= z_target <= Z_RANGE[1]):
        raise RuntimeError(f"Z เป้าหมาย {z_target:.1f} อยู่นอก Z_RANGE {Z_RANGE}")

    clear_pos = None
    if os.path.exists(CLEAR_FILE):
        clear_pos = [float(v) for v in np.load(CLEAR_FILE)["pos"]]
        print(f"จุดหลีกกล้อง: {clear_pos[:3]}")
        if clear_pos[2] < z_target:
            print("คำเตือน: จุดหลีกกล้องต่ำกว่าความสูง hover อาจชนวัตถุตอนถอย")
    elif ALLOW_MOVE:
        raise RuntimeError("ยังไม่ได้ตั้งจุดหลีกกล้อง ให้รันโหมด setclear ก่อน")

    if ALLOW_MOVE and USE_PUMP:
        if PICK_Z is None:
            raise RuntimeError("ยังไม่ได้ตั้ง PICK_Z (ใช้โหมด pose วัด z ตอนหัวดูดแตะผิวบนวัตถุ)")
        if not (Z_RANGE[0] <= PICK_Z <= z_target):
            raise RuntimeError(f"PICK_Z={PICK_Z} ต้องอยู่ระหว่าง {Z_RANGE[0]} ถึง {z_target:.1f}")

    robot = MG400(ROBOT_IP) if ALLOW_MOVE else None
    if robot:
        robot.enable()
        if USE_PUMP:
            robot.pump_off()  # เริ่มต้นด้วยปั๊มปิด

    cap = open_camera()
    holding = False
    show_mask = False
    click = {"pt": None}

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (mx, my)

    cv2.namedWindow("run")
    cv2.setMouseCallback("run", on_mouse)
    print("กดเลข 1-9 หรือ คลิกที่หมาก เพื่อให้หุ่นหยิบหมากหมายเลขนั้นไปปล่อยที่จุดหลีกกล้อง, "
          "'o' ปล่อยด้วยมือ, 'm' ดู mask, 'q' ออก")
    try:
        while True:
            ok, frame = read_frame(cap)
            if not ok:
                break
            frame_raw = frame.copy()

            # แปลงทุกชิ้นเป็นพิกัดหุ่น แล้วเรียงตาม X, Y ให้เลขคงที่
            objs = []
            for (u, v, name) in find_objects(frame):
                x, y = to_robot(u, v)
                objs.append((x, y, u, v, name))
            objs.sort(key=lambda o: (round(o[0] / 15), o[1]))

            numbered = annotate(frame, objs)

            cv2.putText(frame, f"Objects: {len(numbered)}  press 1-9 or CLICK a piece, Q quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("run", frame)
            if show_mask:
                masks = build_masks(frame_raw)
                combined = np.zeros(frame_raw.shape[:2], np.uint8)
                for m in masks.values():
                    combined = cv2.bitwise_or(combined, m)
                cv2.imshow("mask", combined)

            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), ord("Q")):
                break
            if k in (ord("m"), ord("M")):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("mask")
                continue
            if k in (ord("o"), ord("O")):
                if robot and USE_PUMP:
                    robot.release()
                    holding = False
                    print("ปล่อยวัตถุแล้ว (ปิดดูด + เป่าลม)")
                continue
            idx = None
            if ord("1") <= k <= ord("9"):
                idx = k - ord("1")
            elif click["pt"] is not None:
                cx_, cy_ = click["pt"]
                click["pt"] = None
                if numbered:
                    d = [np.hypot(p[2] - cx_, p[3] - cy_) for p in numbered]
                    j = int(np.argmin(d))
                    if d[j] <= 45:  # คลิกใกล้หมากไม่เกิน 45 พิกเซล
                        idx = j
            if idx is not None:
                if holding:
                    print("กำลังถือวัตถุอยู่ กด 'o' เพื่อปล่อยก่อน")
                    continue
                if idx >= len(numbered):
                    print(f"ไม่มีวัตถุหมายเลข {idx + 1}")
                    continue
                tx, ty = numbered[idx][:2]
                pos = confirm_target(cap, to_robot, tx, ty)
                if pos is None:
                    print("วัตถุไม่นิ่ง / หายไป ลองใหม่อีกครั้ง")
                    continue
                x, y = pos
                print(f"เป้าหมาย #{idx + 1}: X={x:.1f} Y={y:.1f} Z={z_target:.1f}")
                if robot:
                    holding = pick_sequence(robot, x, y, z_target, clear_pos)
                    for _ in range(5):  # ทิ้งเฟรมเก่าในบัฟเฟอร์
                        cap.read()
                else:
                    print("(ALLOW_MOVE=False จึงไม่สั่งหุ่น)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot:
            if USE_PUMP:
                try:
                    robot.pump_off()  # ออกโปรแกรมแล้วปิดปั๊มเสมอ (ของที่ถืออยู่จะหล่น)
                except Exception:
                    pass
            robot.close()


# ถ้ากด Run เฉย ๆ จะใช้โหมดนี้ ("calibrate", "camfocus", "setclear", "pumptest", "pose" หรือ "run")
DEFAULT_MODE = "run"

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE
    if mode == "calibrate":
        calibrate()
    elif mode == "setclear":
        setclear()
    elif mode == "camfocus":
        camfocus()
    elif mode == "pumptest":
        pumptest()
    elif mode == "pose":
        show_pose()
    elif mode == "run":
        run()
    else:
        print(__doc__)

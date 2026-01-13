"""
模板创建工具 - 支持拉伸和旋转矩形（只用OpenCV + NumPy）

操作方式：
1. 左键拖拽空白处：框选新区域
2. 左键拖拽角点：拉伸矩形
3. 左键拖拽边：调整宽度/高度
4. 右键拖拽：旋转矩形框
5. 按1/2/3保存模板
6. 按ESC退出
"""

import cv2
import numpy as np


class TemplateCreator:
    """模板创建器 - 支持拉伸和旋转"""
    
    # 固定保存路径
    SAVE_PATHS = {
        'flat_washer': 'D:/1',
        'spring_washer': 'D:/2',
        'red_fiber': 'D:/3'
    }
    
    # 拖拽模式
    MODE_NONE = 0
    MODE_DRAW = 1      # 绘制新矩形
    MODE_ROTATE = 2    # 旋转
    MODE_RESIZE_TL = 3 # 拉伸左上角
    MODE_RESIZE_TR = 4 # 拉伸右上角
    MODE_RESIZE_BL = 5 # 拉伸左下角
    MODE_RESIZE_BR = 6 # 拉伸右下角
    MODE_RESIZE_T = 7  # 拉伸上边
    MODE_RESIZE_B = 8  # 拉伸下边
    MODE_RESIZE_L = 9  # 拉伸左边
    MODE_RESIZE_R = 10 # 拉伸右边
    MODE_MOVE = 11     # 移动矩形
    
    def __init__(self):
        self.mode = self.MODE_NONE
        self.start_point = None
        self.current_roi = None  # (x1, y1, x2, y2)
        self.current_angle = 0
        self.rotate_start_angle = 0
        self.drag_offset = (0, 0)
        self.handle_size = 25  # 拖拽点大小（更大更容易点击）
    
    def _get_handles(self, roi):
        """获取8个拖拽点位置（角点和边中点）"""
        if roi is None:
            return {}
        
        x1, y1, x2, y2 = roi
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        return {
            'tl': (x1, y1),      # 左上角
            'tr': (x2, y1),      # 右上角
            'bl': (x1, y2),      # 左下角
            'br': (x2, y2),      # 右下角
            't': (cx, y1),       # 上边中点
            'b': (cx, y2),       # 下边中点
            'l': (x1, cy),       # 左边中点
            'r': (x2, cy),       # 右边中点
            'center': (cx, cy)   # 中心点
        }
    
    def _hit_test(self, x, y, roi):
        """检测鼠标点击位置"""
        if roi is None:
            return self.MODE_DRAW
        
        handles = self._get_handles(roi)
        hs = self.handle_size
        
        # 检测角点
        if abs(x - handles['tl'][0]) < hs and abs(y - handles['tl'][1]) < hs:
            return self.MODE_RESIZE_TL
        if abs(x - handles['tr'][0]) < hs and abs(y - handles['tr'][1]) < hs:
            return self.MODE_RESIZE_TR
        if abs(x - handles['bl'][0]) < hs and abs(y - handles['bl'][1]) < hs:
            return self.MODE_RESIZE_BL
        if abs(x - handles['br'][0]) < hs and abs(y - handles['br'][1]) < hs:
            return self.MODE_RESIZE_BR
        
        # 检测边中点
        if abs(x - handles['t'][0]) < hs and abs(y - handles['t'][1]) < hs:
            return self.MODE_RESIZE_T
        if abs(x - handles['b'][0]) < hs and abs(y - handles['b'][1]) < hs:
            return self.MODE_RESIZE_B
        if abs(x - handles['l'][0]) < hs and abs(y - handles['l'][1]) < hs:
            return self.MODE_RESIZE_L
        if abs(x - handles['r'][0]) < hs and abs(y - handles['r'][1]) < hs:
            return self.MODE_RESIZE_R
        
        # 检测是否在矩形内部（移动）
        x1, y1, x2, y2 = roi
        if x1 < x < x2 and y1 < y < y2:
            return self.MODE_MOVE
        
        # 点击外部，绘制新矩形
        return self.MODE_DRAW
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        
        # 左键按下
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mode = self._hit_test(x, y, self.current_roi)
            self.start_point = (x, y)
            
            if self.mode == self.MODE_DRAW:
                self.current_roi = None
                self.current_angle = 0
            elif self.mode == self.MODE_MOVE and self.current_roi:
                x1, y1, x2, y2 = self.current_roi
                self.drag_offset = (x - x1, y - y1)
        
        # 左键移动
        elif event == cv2.EVENT_MOUSEMOVE and self.mode != self.MODE_NONE and self.mode != self.MODE_ROTATE:
            if self.mode == self.MODE_DRAW:
                # 绘制新矩形
                sx, sy = self.start_point
                x1 = min(sx, x)
                y1 = min(sy, y)
                x2 = max(sx, x)
                y2 = max(sy, y)
                if x2 - x1 > 5 and y2 - y1 > 5:
                    self.current_roi = (x1, y1, x2, y2)
            
            elif self.current_roi:
                x1, y1, x2, y2 = self.current_roi
                
                if self.mode == self.MODE_MOVE:
                    # 移动矩形
                    w, h = x2 - x1, y2 - y1
                    new_x1 = x - self.drag_offset[0]
                    new_y1 = y - self.drag_offset[1]
                    self.current_roi = (new_x1, new_y1, new_x1 + w, new_y1 + h)
                
                elif self.mode == self.MODE_RESIZE_TL:
                    self.current_roi = (x, y, x2, y2)
                elif self.mode == self.MODE_RESIZE_TR:
                    self.current_roi = (x1, y, x, y2)
                elif self.mode == self.MODE_RESIZE_BL:
                    self.current_roi = (x, y1, x2, y)
                elif self.mode == self.MODE_RESIZE_BR:
                    self.current_roi = (x1, y1, x, y)
                elif self.mode == self.MODE_RESIZE_T:
                    self.current_roi = (x1, y, x2, y2)
                elif self.mode == self.MODE_RESIZE_B:
                    self.current_roi = (x1, y1, x2, y)
                elif self.mode == self.MODE_RESIZE_L:
                    self.current_roi = (x, y1, x2, y2)
                elif self.mode == self.MODE_RESIZE_R:
                    self.current_roi = (x1, y1, x, y2)
        
        # 左键释放
        elif event == cv2.EVENT_LBUTTONUP:
            # 确保坐标有效
            if self.current_roi:
                x1, y1, x2, y2 = self.current_roi
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if x2 - x1 > 10 and y2 - y1 > 10:
                    self.current_roi = (x1, y1, x2, y2)
                else:
                    self.current_roi = None
            self.mode = self.MODE_NONE
        
        # 右键按下：开始旋转
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.current_roi:
                self.mode = self.MODE_ROTATE
                x1, y1, x2, y2 = self.current_roi
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                self.rotate_start_angle = np.arctan2(y - cy, x - cx) * 180 / np.pi - self.current_angle
        
        # 右键移动：旋转
        elif event == cv2.EVENT_MOUSEMOVE and self.mode == self.MODE_ROTATE:
            if self.current_roi:
                x1, y1, x2, y2 = self.current_roi
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                mouse_angle = np.arctan2(y - cy, x - cx) * 180 / np.pi
                self.current_angle = mouse_angle - self.rotate_start_angle
        
        # 右键释放
        elif event == cv2.EVENT_RBUTTONUP:
            self.current_angle = round(self.current_angle / 5) * 5
            self.mode = self.MODE_NONE
    
    def _draw_handles(self, img, roi, color=(0, 255, 255)):
        """绘制拖拽点"""
        handles = self._get_handles(roi)
        hs = self.handle_size
        
        for name, pos in handles.items():
            if name == 'center':
                cv2.circle(img, pos, 5, (0, 0, 255), -1)
            else:
                x, y = pos
                cv2.rectangle(img, (x - hs//2, y - hs//2), (x + hs//2, y + hs//2), color, -1)
    
    def _get_rotated_box(self, roi, angle):
        """获取旋转矩形的四个角点"""
        x1, y1, x2, y2 = roi
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        w = x2 - x1
        h = y2 - y1
        
        rect = (center, (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        return box, center, (w, h)
    
    def _extract_rotated_roi(self, img, roi, angle):
        """提取旋转矩形区域"""
        x1, y1, x2, y2 = roi
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        w = x2 - x1
        h = y2 - y1
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        img_h, img_w = img.shape[:2]
        rotated_img = cv2.warpAffine(img, M, (img_w, img_h))
        
        cx, cy = int(center[0]), int(center[1])
        half_w, half_h = int(w / 2), int(h / 2)
        
        new_x1 = max(0, cx - half_w)
        new_y1 = max(0, cy - half_h)
        new_x2 = min(img_w, cx + half_w)
        new_y2 = min(img_h, cy + half_h)
        
        template = rotated_img[new_y1:new_y2, new_x1:new_x2]
        
        return template
    
    def _count_templates(self, part_name):
        """统计已有模板数量"""
        save_dir = self.SAVE_PATHS[part_name]
        count = 0
        for i in range(1, 100):
            test_path = f"{save_dir}/template{i}.png"
            test_img = cv2.imread(test_path)
            if test_img is not None:
                count = i
            else:
                break
        return count
    
    def interactive_create(self, image_path):
        """
        交互式创建模板
        
        操作：
        - 左键拖拽空白处：框选新区域
        - 左键拖拽角点：拉伸矩形
        - 左键拖拽边中点：调整宽度/高度
        - 左键拖拽内部：移动矩形
        - 右键拖拽：旋转矩形
        - 1/2/3：保存到 D:/1, D:/2, D:/3
        - ESC：退出
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return
        
        window_name = "Template Creator"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("Template Creator - Resize & Rotate Support")
        print("=" * 60)
        print("Mouse:")
        print("  Left drag (empty)  - Draw new rectangle")
        print("  Left drag (corner) - Resize corner")
        print("  Left drag (edge)   - Resize edge")
        print("  Left drag (inside) - Move rectangle")
        print("  Right drag         - Rotate rectangle")
        print("Keyboard:")
        print("  1 - Save to D:/1 (flat_washer)")
        print("  2 - Save to D:/2 (spring_washer)")
        print("  3 - Save to D:/3 (red_fiber)")
        print("  ESC - Exit")
        print("=" * 60)
        
        part_names = {
            ord('1'): 'flat_washer',
            ord('2'): 'spring_washer',
            ord('3'): 'red_fiber'
        }
        
        while True:
            display = img.copy()
            
            # 绘制矩形
            if self.current_roi:
                x1, y1, x2, y2 = self.current_roi
                
                if abs(self.current_angle) < 1:
                    # 正矩形
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    # 绘制拖拽点
                    self._draw_handles(display, self.current_roi)
                else:
                    # 旋转矩形
                    box, center, size = self._get_rotated_box(self.current_roi, self.current_angle)
                    cv2.drawContours(display, [box], 0, (0, 255, 255), 2)
                    cv2.circle(display, (int(center[0]), int(center[1])), 5, (0, 0, 255), -1)
                
                # 显示信息
                info = f"Size: {x2-x1}x{y2-y1}, Angle: {self.current_angle:.0f}"
                cv2.putText(display, info, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 显示帮助
            cv2.putText(display, "Left:Draw/Resize/Move  Right:Rotate  1/2/3:Save", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            # 保存模板
            if key in part_names and self.current_roi:
                part_name = part_names[key]
                
                if abs(self.current_angle) < 1:
                    x1, y1, x2, y2 = self.current_roi
                    template = img[y1:y2, x1:x2]
                else:
                    template = self._extract_rotated_roi(img, self.current_roi, self.current_angle)
                
                save_dir = self.SAVE_PATHS[part_name]
                count = self._count_templates(part_name) + 1
                save_path = f"{save_dir}/template{count}.png"
                
                cv2.imwrite(save_path, template)
                h, w = template.shape[:2]
                print(f"Saved: {save_path} ({w}x{h}, angle={self.current_angle:.0f})")
                
                self.current_roi = None
                self.current_angle = 0
        
        cv2.destroyAllWindows()
        print("\nTemplate creation completed.")


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("""
Template Creator - Resize & Rotate Support

Usage:
  python template_creator.py <image>
  
Mouse Operations:
  Left drag (empty area)  - Draw new rectangle
  Left drag (corner)      - Resize by corner
  Left drag (edge)        - Resize by edge
  Left drag (inside)      - Move rectangle
  Right drag              - Rotate rectangle

Keyboard:
  1/2/3 - Save to D:/1, D:/2, D:/3
  ESC   - Exit
""")
        return
    
    image_path = sys.argv[1]
    creator = TemplateCreator()
    creator.interactive_create(image_path)


if __name__ == "__main__":
    main()

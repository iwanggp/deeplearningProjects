"""
模板创建工具 - 支持旋转矩形（只用OpenCV + NumPy）

操作方式：
1. 鼠标拖拽框选区域
2. 按R/T调整旋转角度（R减少，T增加）
3. 按1/2/3保存对应零件模板
4. 按ESC退出
"""

import cv2
import numpy as np


class TemplateCreator:
    """模板创建器 - 支持旋转矩形"""
    
    # 固定保存路径
    SAVE_PATHS = {
        'flat_washer': 'D:/1',
        'spring_washer': 'D:/2',
        'red_fiber': 'D:/3'
    }
    
    def __init__(self):
        # 鼠标交互状态
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_roi = None
        self.current_angle = 0  # 当前旋转角度
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            self.current_angle = 0  # 重置角度
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # 计算ROI
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.current_roi = (x1, y1, x2, y2)
    
    def _get_rotated_box(self, roi, angle):
        """获取旋转矩形的四个角点"""
        x1, y1, x2, y2 = roi
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        w = x2 - x1
        h = y2 - y1
        
        # 创建旋转矩形
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
        
        # 获取旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转整个图像
        img_h, img_w = img.shape[:2]
        rotated_img = cv2.warpAffine(img, M, (img_w, img_h))
        
        # 计算旋转后的ROI位置（中心点不变）
        cx, cy = int(center[0]), int(center[1])
        half_w, half_h = int(w / 2), int(h / 2)
        
        # 提取ROI
        new_x1 = max(0, cx - half_w)
        new_y1 = max(0, cy - half_h)
        new_x2 = min(img_w, cx + half_w)
        new_y2 = min(img_h, cy + half_h)
        
        template = rotated_img[new_y1:new_y2, new_x1:new_x2]
        
        return template
    
    def _count_templates(self, part_name):
        """统计已有模板数量"""
        save_dir = self.SAVE_PATHS[part_name]
        try:
            count = 0
            # 简单计数，不使用os模块
            for i in range(1, 100):
                test_path = f"{save_dir}/template{i}.png"
                test_img = cv2.imread(test_path)
                if test_img is not None:
                    count = i
                else:
                    break
            return count
        except:
            return 0
    
    def interactive_create(self, image_path):
        """
        交互式创建模板 - 支持旋转矩形
        
        操作：
        - 鼠标拖拽框选区域
        - R键：逆时针旋转5度
        - T键：顺时针旋转5度
        - 1键：保存为平垫圈 (D:/1)
        - 2键：保存为弹簧垫圈 (D:/2)
        - 3键：保存为红钢纸 (D:/3)
        - ESC：退出
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return
        
        window_name = "Template Creator (Rotate: R/T, Save: 1/2/3, Exit: ESC)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("Template Creator - Rotate Support")
        print("=" * 60)
        print("1. Drag mouse to select region")
        print("2. Press R/T to rotate:")
        print("   R - Rotate -5 degrees")
        print("   T - Rotate +5 degrees")
        print("3. Press key to save:")
        print("   1 - Save to D:/1 (flat_washer)")
        print("   2 - Save to D:/2 (spring_washer)")
        print("   3 - Save to D:/3 (red_fiber)")
        print("   ESC - Exit")
        print("=" * 60)
        
        part_names = {
            ord('1'): 'flat_washer',
            ord('2'): 'spring_washer',
            ord('3'): 'red_fiber'
        }
        
        while True:
            display = img.copy()
            
            # 绘制当前选择框（绘制中）
            if self.drawing and self.start_point and self.end_point:
                cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # 绘制旋转矩形
            if self.current_roi:
                if self.current_angle == 0:
                    # 正矩形
                    x1, y1, x2, y2 = self.current_roi
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                else:
                    # 旋转矩形
                    box, center, size = self._get_rotated_box(self.current_roi, self.current_angle)
                    cv2.drawContours(display, [box], 0, (0, 255, 255), 2)
                
                # 显示信息
                x1, y1, x2, y2 = self.current_roi
                info = f"Size: {x2-x1}x{y2-y1}, Angle: {self.current_angle}"
                cv2.putText(display, info, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 显示帮助信息
            cv2.putText(display, "R/T:Rotate  1:D:/1  2:D:/2  3:D:/3  ESC:Exit", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display, f"Current Angle: {self.current_angle}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            # 旋转角度调整
            if key == ord('r') or key == ord('R'):
                self.current_angle -= 5
                print(f"Angle: {self.current_angle}")
            
            if key == ord('t') or key == ord('T'):
                self.current_angle += 5
                print(f"Angle: {self.current_angle}")
            
            # 保存模板
            if key in part_names and self.current_roi:
                part_name = part_names[key]
                
                # 提取模板（支持旋转）
                if self.current_angle == 0:
                    x1, y1, x2, y2 = self.current_roi
                    template = img[y1:y2, x1:x2]
                else:
                    template = self._extract_rotated_roi(img, self.current_roi, self.current_angle)
                
                # 保存
                save_dir = self.SAVE_PATHS[part_name]
                count = self._count_templates(part_name) + 1
                save_path = f"{save_dir}/template{count}.png"
                
                cv2.imwrite(save_path, template)
                h, w = template.shape[:2]
                print(f"Saved: {save_path} ({w}x{h}, angle={self.current_angle})")
                
                self.current_roi = None
                self.current_angle = 0
        
        cv2.destroyAllWindows()
        print("\nTemplate creation completed.")
    
    def create_from_coords(self, image_path, part_name, x, y, w, h, angle=0):
        """
        从坐标创建模板（支持旋转）
        
        Args:
            image_path: 图片路径
            part_name: 零件名称 (flat_washer/spring_washer/red_fiber)
            x, y, w, h: ROI坐标和大小
            angle: 旋转角度（度）
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None
        
        if angle == 0:
            template = img[y:y+h, x:x+w]
        else:
            roi = (x, y, x + w, y + h)
            template = self._extract_rotated_roi(img, roi, angle)
        
        # 保存
        save_dir = self.SAVE_PATHS[part_name]
        count = self._count_templates(part_name) + 1
        save_path = f"{save_dir}/template{count}.png"
        
        cv2.imwrite(save_path, template)
        th, tw = template.shape[:2]
        print(f"Created: {save_path} ({tw}x{th}, angle={angle})")
        
        return save_path


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("""
Template Creator - Rotate Support

Usage:
  python template_creator.py <image>
  python template_creator.py <image> coords <part> <x> <y> <w> <h> [angle]
  
Interactive Mode:
  - Drag mouse to select region
  - R/T: Rotate -5/+5 degrees
  - 1/2/3: Save to D:/1, D:/2, D:/3
  - ESC: Exit

Coords Mode:
  python template_creator.py D:/test.png coords flat_washer 130 133 340 86
  python template_creator.py D:/test.png coords flat_washer 130 133 340 86 45
""")
        return
    
    image_path = sys.argv[1]
    
    creator = TemplateCreator()
    
    if len(sys.argv) > 2 and sys.argv[2] == "coords":
        if len(sys.argv) < 8:
            print("Usage: ... coords <part_name> <x> <y> <w> <h> [angle]")
            return
        part_name = sys.argv[3]
        x, y, w, h = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        angle = int(sys.argv[8]) if len(sys.argv) > 8 else 0
        creator.create_from_coords(image_path, part_name, x, y, w, h, angle)
    else:
        creator.interactive_create(image_path)


if __name__ == "__main__":
    main()

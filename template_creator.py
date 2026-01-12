"""
模板创建工具 - 快速建立模板（只用OpenCV + NumPy）

三种方式：
1. 鼠标框选 - 交互式选择ROI
2. 坐标裁剪 - 指定坐标提取
3. 自动提取 - 基于特征自动检测
"""

import cv2
import numpy as np
import os


class TemplateCreator:
    """模板创建器"""
    
    def __init__(self, template_dir="templates"):
        self.template_dir = template_dir
        self._ensure_dirs()
        
        # 鼠标交互状态
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_roi = None
    
    def _ensure_dirs(self):
        """确保模板目录存在"""
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)
        
        # 创建子目录
        for part_name in ['flat_washer', 'spring_washer', 'red_fiber']:
            part_dir = os.path.join(self.template_dir, part_name)
            if not os.path.exists(part_dir):
                os.makedirs(part_dir)
    
    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
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
    
    def interactive_create(self, image_path):
        """
        交互式创建模板 - 鼠标框选
        
        操作：
        - 鼠标拖拽框选区域
        - 按1/2/3选择零件类型并保存
        - 按ESC退出
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return
        
        window_name = "Template Creator - Drag to select, 1/2/3 to save, ESC to exit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        print("\n" + "=" * 60)
        print("Template Creator - Interactive Mode")
        print("=" * 60)
        print("1. Drag mouse to select region")
        print("2. Press key to save:")
        print("   1 - Save as flat_washer")
        print("   2 - Save as spring_washer")
        print("   3 - Save as red_fiber")
        print("   ESC - Exit")
        print("=" * 60)
        
        part_names = {
            ord('1'): 'flat_washer',
            ord('2'): 'spring_washer',
            ord('3'): 'red_fiber'
        }
        
        template_counts = {name: self._count_templates(name) for name in part_names.values()}
        
        while True:
            display = img.copy()
            
            # 绘制当前选择框
            if self.start_point and self.end_point:
                cv2.rectangle(display, self.start_point, self.end_point, (0, 255, 0), 2)
            
            # 绘制当前ROI
            if self.current_roi:
                x1, y1, x2, y2 = self.current_roi
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(display, f"ROI: {x2-x1}x{y2-y1}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # 显示帮助信息
            cv2.putText(display, "1:flat_washer 2:spring_washer 3:red_fiber ESC:exit", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                break
            
            if key in part_names and self.current_roi:
                part_name = part_names[key]
                x1, y1, x2, y2 = self.current_roi
                
                # 提取并保存模板
                template = img[y1:y2, x1:x2]
                template_counts[part_name] += 1
                
                # 保存到子目录
                save_path = os.path.join(self.template_dir, part_name, 
                                        f"template{template_counts[part_name]}.png")
                cv2.imwrite(save_path, template)
                print(f"Saved: {save_path} ({x2-x1}x{y2-y1})")
                
                self.current_roi = None
        
        cv2.destroyAllWindows()
        print("\nTemplate creation completed.")
    
    def _count_templates(self, part_name):
        """统计已有模板数量"""
        part_dir = os.path.join(self.template_dir, part_name)
        if not os.path.exists(part_dir):
            return 0
        return len([f for f in os.listdir(part_dir) if f.endswith(('.png', '.jpg'))])
    
    def create_from_coords(self, image_path, part_name, x, y, w, h, template_name=None):
        """
        从坐标创建模板
        
        Args:
            image_path: 图片路径
            part_name: 零件名称 (flat_washer/spring_washer/red_fiber)
            x, y, w, h: ROI坐标和大小
            template_name: 模板名称（可选）
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None
        
        # 提取模板
        template = img[y:y+h, x:x+w]
        
        # 确定保存路径
        if template_name is None:
            count = self._count_templates(part_name) + 1
            template_name = f"template{count}.png"
        
        # 保存
        part_dir = os.path.join(self.template_dir, part_name)
        if not os.path.exists(part_dir):
            os.makedirs(part_dir)
        
        save_path = os.path.join(part_dir, template_name)
        cv2.imwrite(save_path, template)
        print(f"Created template: {save_path} ({w}x{h})")
        
        return save_path
    
    def create_from_detection(self, image_path, detection_result):
        """
        从检测结果创建模板
        
        Args:
            image_path: 图片路径
            detection_result: 检测结果字典
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return
        
        for part_name, data in detection_result.items():
            x, y, w, h = data['bbox']
            self.create_from_coords(image_path, part_name, x, y, w, h)
    
    def batch_create(self, image_path, rois):
        """
        批量创建模板
        
        Args:
            image_path: 图片路径
            rois: ROI列表，格式 [(part_name, x, y, w, h), ...]
        """
        for part_name, x, y, w, h in rois:
            self.create_from_coords(image_path, part_name, x, y, w, h)
    
    def auto_extract_by_color(self, image_path, save=False):
        """
        根据颜色自动提取模板（适用于红钢纸等颜色明显的零件）
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return []
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 红色范围
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2
        
        # 形态学处理
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(contour)
                results.append(('red_fiber', x, y, w, h))
                
                if save:
                    self.create_from_coords(image_path, 'red_fiber', x, y, w, h)
        
        return results


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("""
Template Creator

Usage:
  python template_creator.py <image> [mode]
  
Modes:
  interactive  - Mouse drag to select (default)
  coords       - Create from coordinates
  auto         - Auto extract by color
  
Examples:
  python template_creator.py D:/test.png
  python template_creator.py D:/test.png interactive
  python template_creator.py D:/test.png coords flat_washer 130 133 340 86
""")
        return
    
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "interactive"
    
    creator = TemplateCreator()
    
    if mode == "interactive":
        creator.interactive_create(image_path)
    
    elif mode == "coords":
        if len(sys.argv) < 8:
            print("Usage: python template_creator.py <image> coords <part_name> <x> <y> <w> <h>")
            return
        part_name = sys.argv[3]
        x, y, w, h = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
        creator.create_from_coords(image_path, part_name, x, y, w, h)
    
    elif mode == "auto":
        results = creator.auto_extract_by_color(image_path, save=True)
        print(f"\nAuto extracted {len(results)} template(s)")
    
    else:
        print(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()

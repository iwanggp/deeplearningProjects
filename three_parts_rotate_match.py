"""
三零件检测器 - 支持旋转的模板匹配（只用OpenCV + NumPy）

特点：
1. 支持旋转矩形检测
2. 多角度匹配（0°, 15°, 30°, ... 345°）
3. 返回旋转角度和旋转矩形
"""

import cv2
import numpy as np


class RotateTemplateDetector:
    """支持旋转的模板检测器"""
    
    # 固定模板文件路径
    TEMPLATE_PATHS = {
        'flat_washer': [
            'D:/1/template1.png',
            'D:/1/template2.png',
            'D:/1/template3.png',
        ],
        'spring_washer': [
            'D:/2/template1.png',
            'D:/2/template2.png',
            'D:/2/template3.png',
        ],
        'red_fiber': [
            'D:/3/template1.png',
            'D:/3/template2.png',
            'D:/3/template3.png',
        ]
    }
    
    def __init__(self, angle_step=15):
        """
        Args:
            angle_step: 旋转角度步长（度），默认15度
        """
        self.templates = {}
        self.part_names = ['flat_washer', 'spring_washer', 'red_fiber']
        self.angle_step = angle_step
        self._load_templates()
    
    def _load_templates(self):
        """加载模板"""
        print("Loading templates:")
        
        for part_name in self.part_names:
            self.templates[part_name] = []
            
            for template_path in self.TEMPLATE_PATHS[part_name]:
                img = cv2.imread(template_path)
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.templates[part_name].append({
                        'image': gray,
                        'path': template_path
                    })
                    h, w = gray.shape
                    print(f"Loaded: {template_path} ({w}x{h})")
        
        print(f"\nTemplate summary:")
        for part_name in self.part_names:
            count = len(self.templates[part_name])
            print(f"  {part_name}: {count} template(s)")
    
    def _rotate_image(self, image, angle):
        """
        旋转图像
        
        Args:
            image: 输入图像
            angle: 旋转角度（度）
        
        Returns:
            rotated: 旋转后的图像
            M: 旋转矩阵
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算旋转矩阵
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 计算旋转后的边界框大小
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # 调整旋转中心
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        # 执行旋转
        rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                 borderValue=0)
        
        return rotated, M
    
    def _match_with_rotation(self, gray, template, scales, threshold, angles):
        """
        带旋转的模板匹配
        
        Args:
            gray: 灰度图像
            template: 模板
            scales: 缩放比例列表
            threshold: 置信度阈值
            angles: 旋转角度列表
        
        Returns:
            best_match: 最佳匹配结果（包含角度）
        """
        h, w = gray.shape
        th, tw = template.shape
        
        best_match = None
        best_val = -1
        
        for angle in angles:
            # 旋转模板
            if angle != 0:
                rotated_template, _ = self._rotate_image(template, angle)
            else:
                rotated_template = template
            
            rth, rtw = rotated_template.shape
            
            for scale in scales:
                scaled_tw = int(rtw * scale)
                scaled_th = int(rth * scale)
                
                if scaled_tw < 10 or scaled_th < 10:
                    continue
                if scaled_tw > w or scaled_th > h:
                    continue
                
                scaled_template = cv2.resize(rotated_template, (scaled_tw, scaled_th),
                                            interpolation=cv2.INTER_LINEAR)
                
                # 模板匹配
                result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val > best_val and max_val >= threshold:
                    best_val = max_val
                    best_match = {
                        'loc': max_loc,
                        'size': (scaled_tw, scaled_th),
                        'confidence': float(max_val),
                        'scale': scale,
                        'angle': angle
                    }
        
        return best_match
    
    def detect(self, image_path, scales=None, confidence_threshold=0.6, 
               use_rotation=True):
        """
        检测零件
        
        Args:
            image_path: 图片路径
            scales: 缩放比例列表
            confidence_threshold: 置信度阈值
            use_rotation: 是否使用旋转匹配
        """
        if scales is None:
            scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        # 生成旋转角度列表
        if use_rotation:
            angles = list(range(0, 360, self.angle_step))
        else:
            angles = [0]
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        print(f"\nImage: {w}x{h}")
        print(f"Scales: {scales}")
        print(f"Angles: {angles}")
        print(f"Threshold: {confidence_threshold}")
        
        results = {}
        
        for part_name in self.part_names:
            if not self.templates[part_name]:
                continue
            
            print(f"\nDetecting: {part_name}")
            
            best_match = None
            best_conf = -1
            best_template_path = ""
            
            for template_data in self.templates[part_name]:
                template = template_data['image']
                match = self._match_with_rotation(gray, template, scales, 
                                                  confidence_threshold, angles)
                
                if match and match['confidence'] > best_conf:
                    best_conf = match['confidence']
                    best_match = match
                    best_template_path = template_data['path']
            
            if best_match:
                x, y = best_match['loc']
                bw, bh = best_match['size']
                conf = best_match['confidence']
                angle = best_match['angle']
                
                results[part_name] = {
                    'bbox': (x, y, bw, bh),
                    'confidence': conf,
                    'scale': best_match['scale'],
                    'angle': angle,
                    'template': best_template_path
                }
                
                print(f"  Found: loc=({x},{y}), size={bw}x{bh}, angle={angle}, conf={conf:.3f}")
            else:
                print(f"  Not found")
        
        result_img = self._draw_results(img, results)
        self._print_report(results)
        
        return result_img
    
    def _draw_rotated_rect(self, img, center, size, angle, color, thickness=2):
        """绘制旋转矩形"""
        rect = (center, size, angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, color, thickness)
        return box
    
    def _draw_results(self, img, results):
        """绘制检测结果"""
        result = img.copy()
        
        colors = {
            'flat_washer': (0, 255, 0),
            'spring_washer': (0, 255, 255),
            'red_fiber': (255, 0, 255)
        }
        
        labels = {
            'flat_washer': 'Flat Washer',
            'spring_washer': 'Spring Washer',
            'red_fiber': 'Red Fiber'
        }
        
        for part_name, data in results.items():
            x, y, bw, bh = data['bbox']
            angle = data['angle']
            color = colors.get(part_name, (255, 255, 255))
            
            if angle == 0:
                # 正矩形
                cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
            else:
                # 旋转矩形
                center = (x + bw // 2, y + bh // 2)
                self._draw_rotated_rect(result, center, (bw, bh), -angle, color, 2)
            
            conf = data['confidence']
            label = f"{labels[part_name]} ({conf:.2f}, {angle}deg)"
            cv2.putText(result, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        found = len(results)
        status = "OK" if found == 3 else f"Missing {3-found}"
        status_color = (0, 255, 0) if found == 3 else (0, 0, 255)
        cv2.putText(result, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        return result
    
    def _print_report(self, results):
        """打印检测报告"""
        print("\n" + "=" * 60)
        print("DETECTION RESULT (Rotation Support)")
        print("=" * 60)
        
        labels = {
            'flat_washer': 'Flat Washer',
            'spring_washer': 'Spring Washer',
            'red_fiber': 'Red Fiber'
        }
        
        for part_name in self.part_names:
            if part_name in results:
                data = results[part_name]
                conf = data['confidence']
                x, y, w, h = data['bbox']
                angle = data['angle']
                print(f"[OK] {labels[part_name]}: "
                      f"bbox=({x},{y},{w},{h}), angle={angle}, conf={conf:.3f}")
            else:
                print(f"[X]  {labels[part_name]}: NOT FOUND")
        
        found = len(results)
        print("=" * 60)
        if found == 3:
            print("STATUS: OK - All parts detected")
        else:
            print(f"STATUS: ERROR - {3-found} part(s) missing")
        print("=" * 60)


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("""
Rotate Template Detector

Usage:
  python three_parts_rotate_match.py <image> [angle_step]
  
Args:
  image      - Image path
  angle_step - Rotation step in degrees (default: 15)
  
Template Paths:
  D:/1  ->  flat_washer
  D:/2  ->  spring_washer
  D:/3  ->  red_fiber
""")
        return
    
    image_path = sys.argv[1]
    angle_step = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    
    detector = RotateTemplateDetector(angle_step=angle_step)
    
    result = detector.detect(image_path)
    
    if result is not None:
        cv2.imwrite("result_rotate.png", result)
        print("\nSaved: result_rotate.png")


if __name__ == "__main__":
    main()

"""
三零件检测器 - 优化版（只用OpenCV + NumPy）

特点：
1. 多尺度匹配
2. 小角度旋转（±15度）
3. 速度优化
"""

import cv2
import numpy as np


class FastTemplateDetector:
    """模板检测器"""
    
    TEMPLATE_DIRS = {
        'flat_washer': 'D:/1',
        'spring_washer': 'D:/2',
        'red_fiber': 'D:/3'
    }
    
    def __init__(self):
        self.templates = {}
        self.part_names = ['flat_washer', 'spring_washer', 'red_fiber']
        self._load_templates()
    
    def _load_templates(self):
        """加载模板"""
        print("Loading templates...")
        
        for part_name in self.part_names:
            self.templates[part_name] = []
            base_path = self.TEMPLATE_DIRS[part_name]
            
            i = 1
            while True:
                path = f"{base_path}/template{i}.png"
                img = cv2.imread(path)
                if img is None:
                    path = f"{base_path}/template{i}.jpg"
                    img = cv2.imread(path)
                if img is None:
                    break
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.templates[part_name].append({'image': gray, 'path': path})
                h, w = gray.shape
                print(f"  {path} ({w}x{h})")
                i += 1
        
        for part_name in self.part_names:
            print(f"  {part_name}: {len(self.templates[part_name])} template(s)")
    
    def _rotate_template(self, template, angle):
        """旋转模板"""
        h, w = template.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        return cv2.warpAffine(template, M, (new_w, new_h), borderValue=0)
    
    def detect(self, image_path, threshold=0.5):
        """检测零件"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        print(f"\nImage: {w}x{h}, Threshold: {threshold}")
        
        results = {}
        
        for part_name in self.part_names:
            templates = self.templates[part_name]
            if not templates:
                print(f"[X] {part_name}: no template")
                continue
            
            best_match = None
            best_conf = -1
            best_angle = 0
            
            for template_data in templates:
                template = template_data['image']
                th, tw = template.shape
                
                # 第一步：只用scale=1.0和angle=0快速匹配
                if tw <= w and th <= h:
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_conf:
                        best_conf = max_val
                        best_match = {'loc': max_loc, 'size': (tw, th), 'confidence': float(max_val)}
                        best_angle = 0
                
                # 如果置信度>0.9，直接返回
                if best_conf > 0.9:
                    break
                
                # 第二步：尝试不同尺度
                for scale in [0.8, 0.9, 1.1, 1.2]:
                    stw = int(tw * scale)
                    sth = int(th * scale)
                    if stw < 10 or sth < 10 or stw > w or sth > h:
                        continue
                    
                    scaled = cv2.resize(template, (stw, sth))
                    result = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_conf:
                        best_conf = max_val
                        best_match = {'loc': max_loc, 'size': (stw, sth), 'confidence': float(max_val)}
                        best_angle = 0
                
                if best_conf > 0.9:
                    break
                
                # 第三步：尝试小角度旋转（只在置信度低时）
                if best_conf < 0.7:
                    for angle in [-15, 15]:
                        rotated = self._rotate_template(template, angle)
                        rh, rw = rotated.shape
                        
                        if rw > w or rh > h:
                            continue
                        
                        result = cv2.matchTemplate(gray, rotated, cv2.TM_CCOEFF_NORMED)
                        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                        
                        if max_val > best_conf:
                            best_conf = max_val
                            best_match = {'loc': max_loc, 'size': (rw, rh), 'confidence': float(max_val)}
                            best_angle = angle
            
            if best_match and best_conf >= threshold:
                x, y = best_match['loc']
                bw, bh = best_match['size']
                results[part_name] = {
                    'bbox': (x, y, bw, bh),
                    'confidence': best_conf,
                    'angle': best_angle
                }
                print(f"[OK] {part_name}: ({x},{y}) {bw}x{bh} angle={best_angle} conf={best_conf:.3f}")
            else:
                max_conf = best_conf if best_conf > 0 else 0
                print(f"[X]  {part_name}: not found (max_conf={max_conf:.3f})")
        
        # 绘制结果
        result_img = img.copy()
        colors = {
            'flat_washer': (0, 255, 0),
            'spring_washer': (0, 255, 255),
            'red_fiber': (255, 0, 255)
        }
        
        for part_name, data in results.items():
            x, y, bw, bh = data['bbox']
            angle = data['angle']
            color = colors.get(part_name, (255, 255, 255))
            
            if angle == 0:
                cv2.rectangle(result_img, (x, y), (x + bw, y + bh), color, 2)
            else:
                center = (x + bw // 2, y + bh // 2)
                rect = (center, (bw, bh), -angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(result_img, [box], 0, color, 2)
            
            label = f"{part_name} ({data['confidence']:.2f})"
            if angle != 0:
                label += f" {angle}deg"
            cv2.putText(result_img, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        found = len(results)
        status = "OK" if found == 3 else f"Missing {3-found}"
        color = (0, 255, 0) if found == 3 else (0, 0, 255)
        cv2.putText(result_img, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        print(f"\nStatus: {status}")
        
        return result_img


def main():
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("""
Template Detector

Usage:
  python three_parts_opencv_fast.py <image> [threshold]
  
Template Paths:
  D:/1/template1.png -> flat_washer
  D:/2/template1.png -> spring_washer
  D:/3/template1.png -> red_fiber
""")
        return
    
    image_path = sys.argv[1]
    threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    
    detector = FastTemplateDetector()
    
    start = time.time()
    result = detector.detect(image_path, threshold=threshold)
    elapsed = time.time() - start
    
    print(f"\nTime: {elapsed*1000:.1f} ms")
    
    if result is not None:
        cv2.imwrite("result.png", result)
        print("Saved: result.png")


if __name__ == "__main__":
    main()

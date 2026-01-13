"""
三零件检测器 - 极速版（只用OpenCV + NumPy）

特点：
1. 支持多模板（数量不限）
2. 图像金字塔加速
3. 自动检测模板数量
"""

import cv2
import numpy as np


class FastTemplateDetector:
    """极速模板检测器 - 支持多模板"""
    
    # 模板文件夹路径
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
        """自动加载所有模板（数量不限）"""
        print("Loading templates...")
        
        for part_name in self.part_names:
            self.templates[part_name] = []
            base_path = self.TEMPLATE_DIRS[part_name]
            
            # 尝试加载 template1.png, template2.png, ... 直到找不到为止
            i = 1
            while True:
                path = f"{base_path}/template{i}.png"
                img = cv2.imread(path)
                
                if img is None:
                    # 也尝试 .jpg 格式
                    path = f"{base_path}/template{i}.jpg"
                    img = cv2.imread(path)
                
                if img is None:
                    break  # 没有更多模板了
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.templates[part_name].append({
                    'image': gray,
                    'path': path
                })
                h, w = gray.shape
                print(f"  {path} ({w}x{h})")
                i += 1
        
        print()
        for part_name in self.part_names:
            count = len(self.templates[part_name])
            print(f"  {part_name}: {count} template(s)")
        
        total = sum(len(t) for t in self.templates.values())
        print(f"\nTotal: {total} template(s)")
    
    def _fast_match(self, gray, template, threshold=0.6):
        """极速模板匹配（图像金字塔）"""
        h, w = gray.shape
        th, tw = template.shape
        
        # 第一步：缩小图快速搜索
        scale = 0.25
        small_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_template = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        sth, stw = small_template.shape
        if sth < 5 or stw < 5:
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            if max_val >= threshold:
                return {'loc': max_loc, 'size': (tw, th), 'confidence': float(max_val)}
            return None
        
        result = cv2.matchTemplate(small_gray, small_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < threshold * 0.8:
            return None
        
        # 第二步：局部精细搜索
        rough_x = int(max_loc[0] / scale)
        rough_y = int(max_loc[1] / scale)
        
        margin = max(tw, th) // 2
        search_x1 = max(0, rough_x - margin)
        search_y1 = max(0, rough_y - margin)
        search_x2 = min(w, rough_x + tw + margin)
        search_y2 = min(h, rough_y + th + margin)
        
        roi = gray[search_y1:search_y2, search_x1:search_x2]
        
        if roi.shape[0] < th or roi.shape[1] < tw:
            return {
                'loc': (rough_x, rough_y),
                'size': (tw, th),
                'confidence': float(max_val)
            }
        
        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val >= threshold:
            final_x = search_x1 + max_loc[0]
            final_y = search_y1 + max_loc[1]
            return {
                'loc': (final_x, final_y),
                'size': (tw, th),
                'confidence': float(max_val)
            }
        
        return None
    
    def detect(self, image_path, threshold=0.6):
        """检测零件"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        print(f"\nImage: {w}x{h}")
        
        results = {}
        
        for part_name in self.part_names:
            templates = self.templates[part_name]
            if not templates:
                print(f"[X]  {part_name}: no template")
                continue
            
            best_match = None
            best_conf = -1
            best_path = ""
            
            for template_data in templates:
                template = template_data['image']
                match = self._fast_match(gray, template, threshold)
                
                if match and match['confidence'] > best_conf:
                    best_conf = match['confidence']
                    best_match = match
                    best_path = template_data['path']
                
                # 提前终止
                if best_conf > 0.95:
                    break
            
            if best_match:
                x, y = best_match['loc']
                bw, bh = best_match['size']
                results[part_name] = {
                    'bbox': (x, y, bw, bh),
                    'confidence': best_conf,
                    'template': best_path
                }
                print(f"[OK] {part_name}: ({x},{y}) {bw}x{bh} conf={best_conf:.3f}")
            else:
                print(f"[X]  {part_name}: not found")
        
        # 绘制结果
        result_img = img.copy()
        colors = {
            'flat_washer': (0, 255, 0),
            'spring_washer': (0, 255, 255),
            'red_fiber': (255, 0, 255)
        }
        
        for part_name, data in results.items():
            x, y, bw, bh = data['bbox']
            color = colors.get(part_name, (255, 255, 255))
            cv2.rectangle(result_img, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(result_img, f"{part_name} ({data['confidence']:.2f})", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
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
Fast Template Detector (Multi-Template)

Usage:
  python three_parts_opencv_fast.py <image>
  
Template Paths (auto-detect count):
  D:/1/template1.png, template2.png, ...  -> flat_washer
  D:/2/template1.png, template2.png, ...  -> spring_washer
  D:/3/template1.png, template2.png, ...  -> red_fiber

Supports .png and .jpg formats.
No limit on template count.
""")
        return
    
    image_path = sys.argv[1]
    
    detector = FastTemplateDetector()
    
    start = time.time()
    result = detector.detect(image_path)
    elapsed = time.time() - start
    
    print(f"\nTime: {elapsed*1000:.1f} ms")
    
    if result is not None:
        cv2.imwrite("result.png", result)
        print("Saved: result.png")


if __name__ == "__main__":
    main()

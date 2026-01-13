"""
三零件检测器 - 极速版（图像金字塔加速）
"""

import cv2
import numpy as np


class FastTemplateDetector:
    """极速模板检测器"""
    
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
                self.templates[part_name].append(gray)
                i += 1
        
        total = sum(len(t) for t in self.templates.values())
        print(f"Loaded {total} templates")
    
    def detect(self, image_path, threshold=0.5):
        """极速检测（金字塔加速）"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 缩小图用于快速搜索
        scale = 0.25
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        
        results = {}
        
        for part_name in self.part_names:
            templates = self.templates[part_name]
            if not templates:
                continue
            
            best_conf = -1
            best_loc = None
            best_size = None
            
            for template in templates:
                th, tw = template.shape
                
                # 缩小模板
                small_t = cv2.resize(template, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                sth, stw = small_t.shape
                
                if sth < 5 or stw < 5:
                    # 模板太小，直接在原图匹配
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_conf:
                        best_conf = max_val
                        best_loc = max_loc
                        best_size = (tw, th)
                    continue
                
                # 在缩小图上匹配
                result = cv2.matchTemplate(small, small_t, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val < threshold * 0.7:
                    continue
                
                # 在原图局部区域精细匹配
                rx = int(max_loc[0] / scale)
                ry = int(max_loc[1] / scale)
                margin = max(tw, th) // 2
                
                x1 = max(0, rx - margin)
                y1 = max(0, ry - margin)
                x2 = min(w, rx + tw + margin)
                y2 = min(h, ry + th + margin)
                
                roi = gray[y1:y2, x1:x2]
                
                if roi.shape[0] >= th and roi.shape[1] >= tw:
                    result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    
                    if max_val > best_conf:
                        best_conf = max_val
                        best_loc = (x1 + max_loc[0], y1 + max_loc[1])
                        best_size = (tw, th)
            
            if best_conf >= threshold and best_loc:
                results[part_name] = {
                    'bbox': (best_loc[0], best_loc[1], best_size[0], best_size[1]),
                    'confidence': best_conf
                }
        
        # 绘制
        for part_name, data in results.items():
            x, y, bw, bh = data['bbox']
            cv2.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
        
        return img, results


def main():
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("Usage: python three_parts_opencv_fast.py <image>")
        return
    
    detector = FastTemplateDetector()
    
    start = time.time()
    result = detector.detect(sys.argv[1])
    elapsed = (time.time() - start) * 1000
    
    if result:
        img, results = result
        for name, data in results.items():
            print(f"[OK] {name}: conf={data['confidence']:.3f}")
        print(f"Time: {elapsed:.1f} ms")
        cv2.imwrite("result.png", img)


if __name__ == "__main__":
    main()

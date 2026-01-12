"""
三零件检测器 - OpenCV高效实现（只用OpenCV + NumPy）

特点：
1. 使用OpenCV的matchTemplate（C++优化，高性能）
2. 全图搜索，位置无关
3. 多尺度匹配
4. 固定模板路径
"""

import cv2
import numpy as np


class FastTemplateDetector:
    """高效模板检测器（OpenCV实现）"""
    
    # 固定模板文件路径（写死）
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
    
    def __init__(self):
        self.templates = {}
        self.part_names = ['flat_washer', 'spring_washer', 'red_fiber']
        self._load_templates()
    
    def _load_templates(self):
        """从固定路径加载模板"""
        print("Loading templates:")
        print("  D:/1 -> flat_washer")
        print("  D:/2 -> spring_washer")
        print("  D:/3 -> red_fiber")
        print()
        
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
    
    def _match_template(self, gray, template, scales, threshold):
        """使用OpenCV的matchTemplate进行模板匹配"""
        th, tw = template.shape
        h, w = gray.shape
        
        best_match = None
        best_val = -1
        
        for scale in scales:
            scaled_tw = int(tw * scale)
            scaled_th = int(th * scale)
            
            if scaled_tw < 10 or scaled_th < 10:
                continue
            if scaled_tw > w or scaled_th > h:
                continue
            
            scaled_template = cv2.resize(template, (scaled_tw, scaled_th), 
                                        interpolation=cv2.INTER_LINEAR)
            
            # OpenCV高效模板匹配（全图搜索）
            result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > best_val and max_val >= threshold:
                best_val = max_val
                best_match = {
                    'loc': max_loc,
                    'size': (scaled_tw, scaled_th),
                    'confidence': float(max_val),
                    'scale': scale
                }
        
        return best_match
    
    def detect(self, image_path, scales=None, confidence_threshold=0.6):
        """
        检测零件
        
        Args:
            image_path: 图片路径
            scales: 缩放比例列表（默认0.7-1.3）
            confidence_threshold: 置信度阈值
        """
        if scales is None:
            scales = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        print(f"\nImage: {w}x{h}")
        print(f"Scales: {scales}")
        print(f"Threshold: {confidence_threshold}")
        
        results = {}
        
        for part_name in self.part_names:
            if not self.templates[part_name]:
                continue
            
            print(f"\nDetecting: {part_name} ({len(self.templates[part_name])} templates)")
            
            best_match = None
            best_conf = -1
            best_template_path = ""
            
            for template_data in self.templates[part_name]:
                template = template_data['image']
                match = self._match_template(gray, template, scales, confidence_threshold)
                
                if match and match['confidence'] > best_conf:
                    best_conf = match['confidence']
                    best_match = match
                    best_template_path = template_data['path']
            
            if best_match:
                x, y = best_match['loc']
                bw, bh = best_match['size']
                conf = best_match['confidence']
                
                results[part_name] = {
                    'bbox': (x, y, bw, bh),
                    'confidence': conf,
                    'scale': best_match['scale'],
                    'template': best_template_path
                }
                
                print(f"  Found: loc=({x},{y}), size={bw}x{bh}, conf={conf:.3f}")
            else:
                print(f"  Not found")
        
        result_img = self._draw_results(img, results)
        self._print_report(results)
        
        return result_img
    
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
            color = colors.get(part_name, (255, 255, 255))
            
            cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
            
            conf = data['confidence']
            label = f"{labels[part_name]} ({conf:.2f})"
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
        print("DETECTION RESULT")
        print("=" * 60)
        
        labels = {
            'flat_washer': 'Flat Washer (D:/1)',
            'spring_washer': 'Spring Washer (D:/2)',
            'red_fiber': 'Red Fiber (D:/3)'
        }
        
        for part_name in self.part_names:
            if part_name in results:
                data = results[part_name]
                conf = data['confidence']
                x, y, w, h = data['bbox']
                print(f"[OK] {labels[part_name]}: "
                      f"bbox=({x},{y},{w},{h}), conf={conf:.3f}")
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
OpenCV Fast Template Detector

Usage:
  python three_parts_opencv_fast.py <image>
  
Template Paths (Fixed):
  D:/1/template1.png, template2.png, template3.png  ->  flat_washer
  D:/2/template1.png, template2.png, template3.png  ->  spring_washer
  D:/3/template1.png, template2.png, template3.png  ->  red_fiber
""")
        return
    
    image_path = sys.argv[1]
    
    detector = FastTemplateDetector()
    
    result = detector.detect(image_path)
    
    if result is not None:
        cv2.imwrite("result.png", result)
        print("\nSaved: result.png")


if __name__ == "__main__":
    main()

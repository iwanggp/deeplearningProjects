"""
三零件检测器 - 完整版（检测缺少/多装）
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
    
    # 标准数量
    STANDARD_COUNT = {
        'flat_washer': 1,
        'spring_washer': 1,
        'red_fiber': 1
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
        """检测零件（金字塔加速）"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 金字塔
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
                    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                    if max_val > best_conf:
                        best_conf = max_val
                        best_loc = max_loc
                        best_size = (tw, th)
                    continue
                
                # 缩小图匹配
                result = cv2.matchTemplate(small, small_t, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                
                if max_val < threshold * 0.7:
                    continue
                
                # 精细匹配
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
        
        # 绘制结果
        result_img = self._draw_results(img, results)
        
        # 检查装配状态
        status = self._check_status(results)
        
        return result_img, results, status
    
    def _draw_results(self, img, results):
        """绘制结果"""
        result = img.copy()
        
        colors = {
            'flat_washer': (0, 255, 0),      # 绿色
            'spring_washer': (0, 255, 255),  # 黄色
            'red_fiber': (255, 0, 255)       # 粉色
        }
        
        labels = {
            'flat_washer': 'Flat Washer',
            'spring_washer': 'Spring Washer',
            'red_fiber': 'Red Fiber'
        }
        
        for part_name, data in results.items():
            x, y, bw, bh = data['bbox']
            color = colors.get(part_name, (255, 255, 255))
            conf = data['confidence']
            
            # 绘制矩形
            cv2.rectangle(result, (x, y), (x+bw, y+bh), color, 2)
            
            # 绘制标签
            label = f"{labels[part_name]} ({conf:.2f})"
            cv2.putText(result, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return result
    
    def _check_status(self, results):
        """检查装配状态"""
        status = {
            'ok': True,
            'missing': [],
            'extra': [],
            'message': ''
        }
        
        for part_name in self.part_names:
            standard = self.STANDARD_COUNT[part_name]
            detected = 1 if part_name in results else 0
            
            if detected < standard:
                status['ok'] = False
                status['missing'].append(part_name)
            elif detected > standard:
                status['ok'] = False
                status['extra'].append(part_name)
        
        # 生成消息
        if status['ok']:
            status['message'] = 'OK - All parts present'
        else:
            msg = []
            if status['missing']:
                missing_names = [p.replace('_', ' ').title() for p in status['missing']]
                msg.append(f"Missing: {', '.join(missing_names)}")
            if status['extra']:
                extra_names = [p.replace('_', ' ').title() for p in status['extra']]
                msg.append(f"Extra: {', '.join(extra_names)}")
            status['message'] = ' | '.join(msg)
        
        return status


def main():
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("""
Template Detector - Check Missing/Extra Parts

Usage:
  python three_parts_opencv_fast.py <image>

Standard Configuration:
  - 1x Flat Washer
  - 1x Spring Washer
  - 1x Red Fiber

Template Paths:
  D:/1/template1.png -> flat_washer
  D:/2/template1.png -> spring_washer
  D:/3/template1.png -> red_fiber
""")
        return
    
    detector = FastTemplateDetector()
    
    start = time.time()
    result = detector.detect(sys.argv[1])
    elapsed = (time.time() - start) * 1000
    
    if result:
        img, results, status = result
        
        print("\n" + "=" * 60)
        print("DETECTION RESULT")
        print("=" * 60)
        
        labels = {
            'flat_washer': 'Flat Washer',
            'spring_washer': 'Spring Washer',
            'red_fiber': 'Red Fiber'
        }
        
        for part_name in detector.part_names:
            standard = detector.STANDARD_COUNT[part_name]
            if part_name in results:
                conf = results[part_name]['confidence']
                print(f"[OK] {labels[part_name]}: Found (conf={conf:.3f})")
            else:
                print(f"[X]  {labels[part_name]}: Missing")
        
        print("=" * 60)
        print(f"Status: {status['message']}")
        print("=" * 60)
        print(f"Time: {elapsed:.1f} ms\n")
        
        # 在图片上显示状态
        if status['ok']:
            cv2.putText(img, "OK", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(img, "ERROR", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            
            # 显示错误详情
            y_offset = 80
            for line in status['message'].split('|'):
                cv2.putText(img, line.strip(), (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                y_offset += 30
        
        cv2.imwrite("result.png", img)
        print("Saved: result.png")


if __name__ == "__main__":
    main()

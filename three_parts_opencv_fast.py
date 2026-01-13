"""
三零件检测器 - 极速版（支持旋转）

特点：
1. 支持多模板
2. 支持旋转匹配
3. 图像金字塔加速
4. 多尺度匹配
"""

import cv2
import numpy as np


class FastTemplateDetector:
    """极速模板检测器 - 支持旋转"""
    
    # 模板文件夹路径
    TEMPLATE_DIRS = {
        'flat_washer': 'D:/1',
        'spring_washer': 'D:/2',
        'red_fiber': 'D:/3'
    }
    
    def __init__(self, use_rotation=True, angle_step=30):
        """
        Args:
            use_rotation: 是否启用旋转匹配
            angle_step: 旋转角度步长（度）
        """
        self.templates = {}
        self.part_names = ['flat_washer', 'spring_washer', 'red_fiber']
        self.use_rotation = use_rotation
        self.angle_step = angle_step
        self._load_templates()
    
    def _load_templates(self):
        """加载所有模板"""
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
        print(f"Rotation: {self.use_rotation} (step={self.angle_step})")
    
    def _rotate_image(self, image, angle):
        """旋转图像"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        
        rotated = cv2.warpAffine(image, M, (new_w, new_h), borderValue=0)
        return rotated
    
    def _match_single(self, gray, template, threshold=0.5):
        """单次模板匹配（金字塔加速）"""
        h, w = gray.shape
        th, tw = template.shape
        
        if tw > w or th > h:
            return None
        
        # 缩小图快速搜索
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
        
        if stw > small_gray.shape[1] or sth > small_gray.shape[0]:
            return None
        
        result = cv2.matchTemplate(small_gray, small_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val < threshold * 0.7:
            return None
        
        # 局部精细搜索
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
    
    def _match_with_rotation(self, gray, template, threshold=0.5):
        """带旋转和缩放的匹配"""
        best_match = None
        best_val = -1
        best_angle = 0
        best_scale = 1.0
        
        # 缩放比例
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        # 旋转角度
        if self.use_rotation:
            angles = list(range(0, 360, self.angle_step))
        else:
            angles = [0]
        
        # 先用0度匹配
        for s in scales:
            if s != 1.0:
                scaled_template = cv2.resize(template, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            else:
                scaled_template = template
            
            match = self._match_single(gray, scaled_template, threshold)
            
            if match and match['confidence'] > best_val:
                best_val = match['confidence']
                best_match = match
                best_angle = 0
                best_scale = s
        
        # 如果0度匹配够好，直接返回
        if best_val > 0.9:
            if best_match:
                best_match['angle'] = 0
                best_match['scale'] = best_scale
            return best_match
        
        # 尝试旋转匹配
        if self.use_rotation:
            for angle in angles:
                if angle == 0:
                    continue
                
                rotated_template = self._rotate_image(template, angle)
                
                # 只用scale=1.0快速搜索角度
                match = self._match_single(gray, rotated_template, threshold)
                
                if match and match['confidence'] > best_val:
                    best_val = match['confidence']
                    best_match = match
                    best_angle = angle
                    best_scale = 1.0
                
                # 提前终止
                if best_val > 0.9:
                    break
        
        if best_match:
            best_match['angle'] = best_angle
            best_match['scale'] = best_scale
        
        return best_match
    
    def detect(self, image_path, threshold=0.5):
        """检测零件"""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Cannot read {image_path}")
            return None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        print(f"\nImage: {w}x{h}")
        print(f"Threshold: {threshold}")
        
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
                match = self._match_with_rotation(gray, template, threshold)
                
                if match and match['confidence'] > best_conf:
                    best_conf = match['confidence']
                    best_match = match
                    best_path = template_data['path']
                
                if best_conf > 0.9:
                    break
            
            if best_match:
                x, y = best_match['loc']
                bw, bh = best_match['size']
                angle = best_match.get('angle', 0)
                results[part_name] = {
                    'bbox': (x, y, bw, bh),
                    'confidence': best_conf,
                    'angle': angle,
                    'template': best_path
                }
                print(f"[OK] {part_name}: ({x},{y}) {bw}x{bh} angle={angle} conf={best_conf:.3f}")
            else:
                print(f"[X]  {part_name}: not found")
        
        # 绘制结果
        result_img = self._draw_results(img, results)
        
        found = len(results)
        status = "OK" if found == 3 else f"Missing {3-found}"
        print(f"\nStatus: {status}")
        
        return result_img
    
    def _draw_rotated_rect(self, img, x, y, w, h, angle, color):
        """绘制旋转矩形"""
        center = (x + w // 2, y + h // 2)
        rect = (center, (w, h), -angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, color, 2)
    
    def _draw_results(self, img, results):
        """绘制结果"""
        result = img.copy()
        colors = {
            'flat_washer': (0, 255, 0),
            'spring_washer': (0, 255, 255),
            'red_fiber': (255, 0, 255)
        }
        
        for part_name, data in results.items():
            x, y, bw, bh = data['bbox']
            angle = data.get('angle', 0)
            color = colors.get(part_name, (255, 255, 255))
            
            if angle == 0:
                cv2.rectangle(result, (x, y), (x + bw, y + bh), color, 2)
            else:
                self._draw_rotated_rect(result, x, y, bw, bh, angle, color)
            
            label = f"{part_name} ({data['confidence']:.2f})"
            if angle != 0:
                label += f" {angle}deg"
            cv2.putText(result, label, (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        found = len(results)
        status = "OK" if found == 3 else f"Missing {3-found}"
        color = (0, 255, 0) if found == 3 else (0, 0, 255)
        cv2.putText(result, f"Status: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        return result


def main():
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("""
Fast Template Detector (Rotation + Multi-Scale)

Usage:
  python three_parts_opencv_fast.py <image> [--no-rotate] [--step N]
  
Options:
  --no-rotate  Disable rotation matching
  --step N     Rotation step (default: 30)
  
Template Paths:
  D:/1/template1.png, template2.png, ...  -> flat_washer
  D:/2/template1.png, template2.png, ...  -> spring_washer
  D:/3/template1.png, template2.png, ...  -> red_fiber
""")
        return
    
    image_path = sys.argv[1]
    use_rotation = '--no-rotate' not in sys.argv
    angle_step = 30
    
    if '--step' in sys.argv:
        idx = sys.argv.index('--step')
        if idx + 1 < len(sys.argv):
            angle_step = int(sys.argv[idx + 1])
    
    detector = FastTemplateDetector(use_rotation=use_rotation, angle_step=angle_step)
    
    start = time.time()
    result = detector.detect(image_path)
    elapsed = time.time() - start
    
    print(f"\nTime: {elapsed*1000:.1f} ms")
    
    if result is not None:
        cv2.imwrite("result.png", result)
        print("Saved: result.png")


if __name__ == "__main__":
    main()

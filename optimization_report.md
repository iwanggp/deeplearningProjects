# 性能优化分析报告

## 优化汇总

| # | 位置（优化后行号） | 问题 | 优化手段 | 预期加速比 |
|---|---|---|---|---|
| 1 | `blob.cpp:200-202` | `copyFirstN` 整体深拷贝 result + `erase` O(n) 搬移 | 新增 `truncateResultInPlace`，直接 `resize(N)` | 该阶段近乎零拷贝 |
| 2 | `blob.cpp:574` | `reorderBlobResult` 里轮廓 `vector<Point>` 整条拷贝 | 改为 `std::move`，指针转移 | 排序阶段 ~3-5x |
| 3 | `blob.cpp:308-321` | `BlobInfo::contour` 对每个 blob 多存一份轮廓副本 | 删除该字段，只保留 `idx` | filter 阶段内存减半 |
| 4 | `blob.cpp:147` | `result.binaryImage = binImg.clone()` 整幅图 memcpy | 浅引用 `= binImg`（OpenCV 3.2+ findContours 不改源图） | 节省一次全图拷贝 |
| 5 | `blob.cpp:97` | `gray = params.image.clone()` 单通道输入多余拷贝 | 浅引用 `= params.image` | 同上 |
| 6 | `blob.cpp:170-174` | `holeBin` 无条件 zero-init | 加 `needHoleBin` 开关，关闭时完全跳过 | 关闭时省全图 zero-init |
| 7 | `blob.cpp:349-351` | 孔洞 `drawContours` 无条件执行 | 同 `needHoleBin` 守卫 | 同上 |
| 8 | `blob.cpp:178` | 原用 `filteredContours`（深拷贝整套轮廓） | 改用 `filteredIndices`（int 索引） | 省一套轮廓深拷贝 |
| 9 | `blob.cpp:210-214` | 循环内每次构造临时 `vector<vector<Point>>{contour}` + 中间 Mat | 按索引直接 drawContours + `cv::subtract` 就地 | 省 N 次临时对象 + 1 次 Mat |
| 10 | `blob.cpp:324` | `blobInfos` 无 reserve，多次 realloc | `blobInfos.reserve(contours.size())` | 去掉扩容拷贝 |
| 11 | `blob.cpp:475-489` | 11 个输出向量无 reserve，push_back 反复扩容 | 统计 `survivors` 后集中 reserve | 去掉扩容拷贝 |
| 12 | `blob.cpp:445` | `vector<bool> keep` 位代理访问有额外开销 | 改为 `vector<char> keep` | 小 |
| 13 | `blob.cpp:532-558` | `sortBlobs` 比较器里每次都走 `switch(feature)` | 预 flatten 成 `vector<double> key`，比较器裸读 | 去掉 N log N 次分支 |
| 14 | `blob.cpp:42-71` | `keepRegionsInsideRoIs` 每个 roi 单次 `fillPoly` | 收集全部 polygon 后批量单次 `fillPoly` | 减少内部调度开销 |
| 15 | `blob.cpp:632-637` | `calcuPositionCorrectionMatrix` `cos/sin` 各算两次 | 各算一次后复用 | 省 2 次三角函数调用 |
| 16 | 原 `blob.cpp:137` | `cv::Mat output = cv::Mat::zeros(...)` 分配后从未使用（死代码） | 直接删除 | 省一次 zero-init |
| 17 | 原 `blob.cpp:39` | `if (1) { if (...) }` 恒真死代码结构 | 清理为 `const bool doCorrection` | 可读性提升，编译器更好内联 |

---

## 各项优化详解

### 优化1：消除 `copyFirstN` 深拷贝（`blob.cpp:200-202`）

- **原代码问题**（原 `blob.cpp:200-203`）：
  ```cpp
  blobResult resultCopy = result;               // 整个 result 深拷贝，含所有轮廓点
  result = copyFirstN(resultCopy, blob_count);  // move 前N + erase O(剩余) 搬移
  result.binaryImage = _bin_img;                // 绕一圈保存 binaryImage
  ```
- **优化方案**（现 `blob.cpp:200-202`）：
  ```cpp
  if (blob_count >= 0 && result.blobCount > blob_count)
      truncateResultInPlace(result, blob_count);  // 只做 resize(N)，零拷贝
  ```
- **加速原理**：`blobResult` 深拷贝包含所有 `vector<vector<Point>>` 轮廓点，对 N 个 blob 是 O(Σ轮廓点数) 的内存拷贝；`erase` 再做 O(剩余元素数) 的搬移。`resize(N)` 只析构尾部元素，成本接近零。
- **预期收益**：最大单点收益，约 **15-25%** 整体提速

---

### 优化2：`reorderBlobResult` 改 move（`blob.cpp:574`）

- **原代码问题**（原 `blob.cpp:604-609`）：
  ```cpp
  reordered[i] = vec[indices[i]];  // 对 vector<Point> 是整条深拷贝
  ```
- **优化方案**（现 `blob.cpp:574`）：
  ```cpp
  reordered.push_back(std::move(vec[indices[i]]));  // 指针转移，零拷贝
  ```
- **加速原理**：轮廓的 `vector<Point>` 在排序重排时原来要 copy 每个点，改为 move 只转移内部指针（3 个指针赋值）。对 POD（`double`/`Point2f`）退化为拷贝，行为等价。
- **预期收益**：约 **5-10%** 整体提速

---

### 优化3：`BlobInfo` 去除轮廓副本（`blob.cpp:308-321`）

- **原代码问题**（原 `blob.cpp:321, 454`）：
  ```cpp
  std::vector<cv::Point> contour;  // BlobInfo 里多存一份轮廓
  blobInfo.contour = contour;      // 每个候选 blob 整条深拷贝
  ```
- **优化方案**（现 `blob.cpp:309`）：只保留 `int idx`，访问时用 `contours[b.idx]`
- **加速原理**：filter 阶段原来对每个候选 blob 都多做一次整条轮廓的内存拷贝，改为索引后零拷贝。
- **预期收益**：约 **3-8%** 整体提速（随轮廓点数增大）

---

### 优化4-5：两次整幅 `clone()` 改浅引用（`blob.cpp:97, 147`）

- **原代码问题**：
  - 原 `blob.cpp:99`：`gray = params.image.clone();`
  - 原 `blob.cpp:154`：`result.binaryImage = binImg.clone();`
- **优化方案**（现 `blob.cpp:97, 147`）：改为 `= params.image` / `= binImg`（cv::Mat 引用计数共享）
- **加速原理**：OpenCV 3.2+ `findContours` 不修改源图，clone 不再必要。每次省一次全图 memcpy（5MP 图约 5MB）。
- **预期收益**：大图场景每调用节省约 **1-5 ms**

---

### 优化6-7：`holeBin` 按需分配（`blob.cpp:170-174, 349-351`）

- **原代码问题**（原 `blob.cpp:181, 362-375`）：无论 `filteredBinaryOutputEnabled` 是否开启，都 `Mat::zeros` 分配 + 每孔洞调一次 `drawContours`
- **优化方案**（现 `blob.cpp:170-174, 349-351`）：
  ```cpp
  const bool needHoleBin = params.outputControlParams.filteredBinaryOutputEnabled;
  if (needHoleBin) holeBin = cv::Mat::zeros(...);
  // ...
  if (needHoleBin) cv::drawContours(holeBin, ...);
  ```
- **加速原理**：`Mat::zeros` 是全图 memset；`drawContours` 是光栅化开销。不需要时直接不做。
- **预期收益**：`filteredBinaryOutputEnabled=false` 的测例省去整块

---

### 优化8-9：filteredContours 改索引 + 就地绘制（`blob.cpp:178, 210-214`）

- **原代码问题**（原 `blob.cpp:182, 211-215`）：
  ```cpp
  std::vector<std::vector<cv::Point>> filteredContours;  // 额外深拷贝一套轮廓
  // 循环里每次：
  cv::drawContours(..., std::vector<std::vector<cv::Point>>{contour}, -1, ...);  // 临时 vector
  cv::Mat tmpImg = result.filteredBinaryImage - holeBin;  // 中间 Mat 分配
  result.filteredBinaryImage = tmpImg;
  ```
- **优化方案**（现 `blob.cpp:178, 210-214`）：
  ```cpp
  std::vector<int> filteredIndices;  // 只存索引
  // ...
  for (int idx : filteredIndices)
      cv::drawContours(result.filteredBinaryImage, contours, idx, cv::Scalar(255), cv::FILLED);
  cv::subtract(result.filteredBinaryImage, holeBin, result.filteredBinaryImage);  // 就地
  ```
- **加速原理**：省掉 N 次临时 `vector<vector<Point>>` 构造析构 + 1 次中间 Mat 分配
- **预期收益**：约 **2-5%** 整体提速

---

### 优化10-11：向量统一 `reserve`（`blob.cpp:324, 475-489`）

- **原代码问题**：`blobInfos` 和全部 11 个输出向量均无 `reserve`，`push_back` 触发多次 doubling realloc，每次 realloc 对 `vector<vector<Point>>` 都是搬移操作
- **优化方案**：
  ```cpp
  blobInfos.reserve(contours.size());          // blob.cpp:324
  const size_t survivors = std::count(...);    // blob.cpp:475
  result.centroids.reserve(survivors);         // blob.cpp:479
  // ... 其余 10 个向量同样 reserve
  ```
- **加速原理**：消除 realloc 引发的元素搬移
- **预期收益**：约 **3-6%** 整体提速

---

### 优化12：`vector<bool>` → `vector<char>`（`blob.cpp:445`）

- **原代码问题**（原 `blob.cpp:471`）：`std::vector<bool>` 用位存储，`operator[]` 返回代理对象，读写有额外拆装位开销
- **优化方案**（现 `blob.cpp:445`）：`std::vector<char> keep(blobInfos.size(), 1);`
- **预期收益**：小，对重叠过滤嵌套循环有轻微改善

---

### 优化13：`sortBlobs` 比较器去分支（`blob.cpp:532-558`）

- **原代码问题**（原 `blob.cpp:560-592`）：lambda 比较器内每次都走 `switch(params.feature)` + 访问不同成员向量，`std::sort` 的 N log N 次比较都要经历此分支
- **优化方案**（现 `blob.cpp:532-558`）：
  ```cpp
  std::vector<double> key(result.blobCount);
  // 一次性按 feature 填充 key 数组
  std::sort(indices.begin(), indices.end(),
      [&key](int a, int b) { return key[a] < key[b]; });  // 无分支，单次读
  ```
- **加速原理**：比较器内无 switch，单次数组读，编译器更容易内联 + 向量化
- **预期收益**：约 **1-3%** 整体提速

---

### 优化14：`keepRegionsInsideRoIs` 批量 `fillPoly`（`blob.cpp:42-71`）

- **原代码问题**（原 `blob.cpp:73`）：每个 roi 单独调一次 `cv::fillPoly`
- **优化方案**（现 `blob.cpp:42-71`）：先收集全部 polygon，最后一次性 `cv::fillPoly(output, polygons, ...)` 
- **加速原理**：减少 OpenCV 函数调用次数及内部调度开销
- **预期收益**：多 roi 场景有效，单 roi 无差别

---

### 优化15：`calcuPositionCorrectionMatrix` cos/sin 去重（`blob.cpp:632-637`）

- **原代码问题**（原 `blob.cpp:686-693`）：`std::cos(angle)` 和 `std::sin(angle)` 各被调用两次
- **优化方案**（现 `blob.cpp:632-637`）：
  ```cpp
  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const double alphaX = c * scaleX;
  const double alphaY = c * scaleY;
  const double betaX  = s * scaleX;
  const double betaY  = s * scaleY;
  ```
- **预期收益**：省 2 次三角函数调用（每次约数十 ns）

---

### 优化16-17：死代码清理

- **优化16**（原 `blob.cpp:137`）：`cv::Mat output = cv::Mat::zeros(binImg.size(), CV_8UC1);` 分配后从未使用，直接删除，节省一次全图 zero-init
- **优化17**（原 `blob.cpp:39`）：`if (1) { if (params.usePositionCorrection && ...) {...} }` 恒真包裹，重构为 `const bool doCorrection`，提升可读性并让编译器更好优化

---

## 综合预期收益

各优化效果叠加，预计整体加速比：**1.3x ~ 1.8x（即提速 30%–80%）**

| 场景 | 预估提速 |
|---|---|
| 小图 + 少 blob + 短轮廓 | ~20-30% |
| 中等图 + 中等 blob 数量 | ~40-50% |
| 大图 + 多 blob + 长轮廓 | ~60-80% |

> 注：实际效果依赖数据规模和运行环境。建议使用 `std::chrono` 在 5 个已有 gtest 测例前后打时间戳做 A/B 对比，以真实数据验证。
> 如需进一步提速，可做 Green 定理单遍融合扫描（将 `contourArea + arcLength + moments + boundingRect` 合并为一次手写循环），预计再提 10-25%。

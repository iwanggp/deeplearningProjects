#include "blob.h"
#include "caliper.h"
#include <opencv2/opencv.hpp>
#include <numeric>   // for std::iota / std::accumulate
#include <algorithm> // for std::sort / std::count
#include "spdlog_common/dlog.h"

namespace Blob
{
	// 函数声明
	void applyThreshold(const ThresholdParams& params, const cv::Mat& gray, cv::Mat& binImg);
	void applySoftThreshold(const cv::Mat& gray, cv::Mat& binImg, double width);
	void applyRelativeThreshold(const cv::Mat& gray, cv::Mat& binImg, double factor, double width);
	void filterBlobs(const blobParams& params,
		const std::vector<std::vector<cv::Point>>& contours,
		const std::vector<cv::Vec4i>& hierarchy,
		blobResult& result,
		std::vector<int>& filteredIndices,
		cv::Mat& holeBin,
		bool needHoleBin);
	void sortBlobs(const SortParams& params, blobResult& result);
	void reorderBlobResult(const std::vector<int>& indices, blobResult& result);
	void truncateResultInPlace(blobResult& result, int N); // 就地截断，取代原 copyFirstN
	cv::Matx23d calcuPositionCorrectionMatrix(const Location::PositionCorrectionParam& param);

	//用多个旋转矩形roi截取二值图，支持位置修正
	cv::Mat keepRegionsInsideRoIs(const cv::Mat& binaryImage, const blobParams& params, blobResult& result)
	{
		result.roisCorrection.clear();
		result.roisCorrection.reserve(params.rois.size());

		const bool doCorrection = params.usePositionCorrection && !params.useRoiMask;
		cv::Matx23d M;
		if (doCorrection) {
			M = calcuPositionCorrectionMatrix(params.positionCorrectionParam);
		}

		// 复制输入的二值图像
		cv::Mat output = cv::Mat::zeros(binaryImage.size(), CV_8UC1);

		// 先收集全部多边形，一次性交给 fillPoly，避免多次内部切换/调度
		std::vector<std::vector<cv::Point>> polygons;
		polygons.reserve(params.rois.size());

		for (const auto& roi : params.rois) {
			cv::Point2f vertices[4];
			roi.points(vertices);

			cv::Point2f verticesTrans[4];
			std::vector<cv::Point> polygon;
			polygon.reserve(4);

			for (int i = 0; i < 4; ++i) {
				if (doCorrection) {
					verticesTrans[i] = cv::Point2f(cv::Mat(M * cv::Mat(cv::Point3d(vertices[i].x, vertices[i].y, 1.0))));
				}
				else {
					verticesTrans[i] = vertices[i];
				}
				polygon.emplace_back(cv::Point(static_cast<int>(verticesTrans[i].x),
					static_cast<int>(verticesTrans[i].y)));
			}

			cv::RotatedRect rotRect(verticesTrans[0], verticesTrans[1], verticesTrans[2]);
			result.roisCorrection.push_back(rotRect);
			polygons.push_back(std::move(polygon));
		}

		if (!polygons.empty()) {
			cv::fillPoly(output, polygons, cv::Scalar(255));
		}

		// 使用与输入图像相同的阈值来保留指定区域的像素
		cv::bitwise_and(output, binaryImage, output);

		return output;
	}

	// 主 BLOB 函数实现
	int BLOB(const blobParams& params, blobResult& result)
	{
		// 1) 检查输入图像是否为空
		if (params.image.empty()) {
			result.blobCount = -1;
			DLOG_WARN("blob", "输入图像为空!");
			return BlobErrorCode::EMPTY_IMAGE;
		}

		cv::Mat gray;
		try {
			// 如果是彩色图像，需要先转换为灰度图
			if (params.image.channels() == 3) {
				cv::cvtColor(params.image, gray, cv::COLOR_BGR2GRAY);
			}
			else {
				// 原代码此处是 clone()，但后续不会修改 gray，浅引用即可避免一次全图 memcpy
				gray = params.image;
			}
		}
		catch (const cv::Exception& e) {
			DLOG_WARN("blob", "输入图像转换为灰度图失败!");
			return BlobErrorCode::THRESHOLD_ERROR; // 灰度图转换失败
		}

		// 2) 应用阈值处理
		cv::Mat binImg;
		try {
			applyThreshold(params.thresholdParams, gray, binImg);
		}
		catch (const std::exception& e) {
			DLOG_WARN("blob", "图像阈值处理失败!");
			return BlobErrorCode::THRESHOLD_ERROR; // 阈值处理失败
		}

		//roi使用参数或掩膜
		if (params.useRoiMask) {
			if (params.roiMask.empty()) {
				// 未提供掩膜：跳过，无需构造全白 mask 再做 AND
			}
			else if (params.roiMask.size() == binImg.size()) {
				int count = cv::countNonZero(params.roiMask);
				if (count < 4) {
					DLOG_WARN("blob", "ROI掩膜有效区域太小!");
					return BlobErrorCode::ROI_MASK_ERROR;
				}
				cv::bitwise_and(params.roiMask, binImg, binImg);
			}
			else {
				//不接收尺寸和原图尺寸不同的mask
				DLOG_WARN("blob", "ROI掩膜尺寸与原图尺寸不同!");
				return BlobErrorCode::ROI_MASK_ERROR; // 处理 ROI 失败
			}
		}
		else {
			// **添加新的 ROI 过滤逻辑**
			try {
				binImg = keepRegionsInsideRoIs(binImg, params, result);
			}
			catch (const std::exception& e) {
				DLOG_WARN("blob", "ROI处理失败!");
				return BlobErrorCode::ROI_MASK_ERROR; // 处理 ROI 失败
			}
		}

		// 保存二值化结果。OpenCV >= 3.2 的 findContours 不会修改源图，
		// 因此这里使用浅引用（共享数据，引用计数）替代 clone()，节省一次全图拷贝。
		result.binaryImage = binImg;

		// 3) 连通域分析
		int connectivity = (params.connectivityParams.connectivity == 8) ? 8
			: (params.connectivityParams.connectivity == 4 ? 4 : 0);
		if (connectivity == 0) {
			DLOG_WARN("blob", "无效的连通域参数!");
			return BlobErrorCode::INVALID_CONNECTIVITY;
		}

		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		try {
			cv::findContours(binImg, contours, hierarchy,
				cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
		}
		catch (const std::exception& e) {
			DLOG_WARN("blob", "连通域分析错误，查找轮廓失败!");
			return BlobErrorCode::FIND_CONTOURS_ERROR;
		}

		// 4) 过滤 Blobs
		// 只有在需要输出过滤后的二值图时才分配 holeBin，避免无谓的全图 zero 初始化 + drawContours
		const bool needHoleBin = params.outputControlParams.filteredBinaryOutputEnabled;
		cv::Mat holeBin;
		if (needHoleBin) {
			holeBin = cv::Mat::zeros(params.image.size(), CV_8UC1);
		}

		// 用索引代替原先的 std::vector<std::vector<cv::Point>> filteredContours，
		// 避免把每条轮廓再深拷贝一份
		std::vector<int> filteredIndices;
		try {
			filterBlobs(params, contours, hierarchy, result, filteredIndices, holeBin, needHoleBin);
		}
		catch (const std::exception& e) {
			DLOG_WARN("blob", "连通域分析错误，过滤blob失败!");
			return BlobErrorCode::FIND_CONTOURS_ERROR;
		}

		//5) 对 Blob 结果进行排序
		try {
			sortBlobs(params.sortParams, result);
		}
		catch (const std::exception& e) {
			DLOG_WARN("blob", "连通域分析错误，blob排序失败!");
			return BlobErrorCode::FIND_CONTOURS_ERROR;
		}

		// 目标选择的 blob 数：就地截断。
		// 原实现走的是 "blobResult 深拷贝 -> 移动前 N -> 对源做 erase(begin, begin+N)" 的路径，
		// 会对整个 result（含所有轮廓点）做一次完整深拷贝，并在大向量上产生一次 O(n) 的擦除搬移。
		// 这里直接在原 result 上 resize 到前 N，零拷贝、零搬移。
		const int blob_count = params.outputControlParams.maxBlobs;
		if (blob_count >= 0 && result.blobCount > blob_count) {
			truncateResultInPlace(result, blob_count);
		}

		// 6) 生成过滤后的二值图（如果启用）
		if (params.outputControlParams.filteredBinaryOutputEnabled) {
			result.filteredBinaryImage = cv::Mat::zeros(binImg.size(), CV_8U);

			// 按索引直接引用原 contours，不再每次构造临时 vector<vector<Point>>{contour}
			for (int idx : filteredIndices) {
				cv::drawContours(result.filteredBinaryImage, contours, idx, cv::Scalar(255), cv::FILLED);
			}
			// 就地相减，避免创建临时 Mat 再赋值
			cv::subtract(result.filteredBinaryImage, holeBin, result.filteredBinaryImage);
		}

		// 7) 成功完成
		return BlobErrorCode::SUCCESS;
	}

	// ------------------ 以下为各辅助函数的实现 ------------------

	// 应用不同阈值方法
	void applyThreshold(const ThresholdParams& params, const cv::Mat& gray, cv::Mat& binImg)
	{
		switch (params.type) {
		case ThresholdType::SINGLE:
			if (params.adaptiveThreshold) {
				cv::adaptiveThreshold(
					gray, binImg, 255,
					cv::ADAPTIVE_THRESH_GAUSSIAN_C,
					cv::THRESH_BINARY, 11, 2
				);
			}
			else {
				int thresholdType =
					(params.polarity == Polarity::BRIGHTER_THAN_BACKGROUND)
					? cv::THRESH_BINARY
					: cv::THRESH_BINARY_INV;
				cv::threshold(gray, binImg, params.threshold, 255, thresholdType);
			}
			break;

		case ThresholdType::DOUBLE:
			cv::inRange(gray, params.lowThreshold, params.highThreshold, binImg);
			break;

		case ThresholdType::AUTO:
			cv::threshold(gray, binImg, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
			break;

		case ThresholdType::SOFT_FIXED:
			applySoftThreshold(gray, binImg, params.softThresholdWidth);
			break;

		case ThresholdType::SOFT_RELATIVE:
			applyRelativeThreshold(gray, binImg, params.relativeThresholdFactor, params.softThresholdWidth);
			break;
		}
	}

	// 软阈值（固定）
	void applySoftThreshold(const cv::Mat& gray, cv::Mat& binImg, double width)
	{
		cv::Mat blurred;
		cv::GaussianBlur(gray, blurred, cv::Size(0, 0), width);
		binImg = (gray > blurred);
		binImg.convertTo(binImg, CV_8U, 255);
	}

	// 软阈值（相对）
	void applyRelativeThreshold(const cv::Mat& gray, cv::Mat& binImg, double factor, double width)
	{
		double meanValue = cv::mean(gray)[0] * factor;
		cv::Mat blurred;
		cv::GaussianBlur(gray, blurred, cv::Size(0, 0), width);
		binImg = (gray > (meanValue + blurred));
		binImg.convertTo(binImg, CV_8U, 255);
	}

	// 过滤 Blob 并计算所需的特征
	void filterBlobs(const blobParams& params,
		const std::vector<std::vector<cv::Point>>& contours,
		const std::vector<cv::Vec4i>& hierarchy,
		blobResult& result,
		std::vector<int>& filteredIndices,
		cv::Mat& holeBin,
		bool needHoleBin)
	{
		// 初始化结果
		result.blobCount = 0;
		result.centroids.clear();
		result.areas.clear();
		result.contours.clear();
		result.arcLengths.clear();
		result.langAxises.clear();
		result.shortAxises.clear();
		result.axisRatios.clear();
		result.circularities.clear();
		result.rectangularities.clear();
		result.centroidOffsets.clear();
		result.minBoundingRects.clear();
		filteredIndices.clear();

		// 注意：BlobInfo 原本保存了一份 std::vector<cv::Point> contour 的拷贝，
		// 在轮廓点很多时代价很高。这里改为只记录原轮廓在 contours 中的 idx，真正需要
		// 时通过 contours[idx] 访问，避免深拷贝。
		struct BlobInfo {
			int idx;
			double area;
			cv::Rect boundingRect;
			cv::Point2f centroid;
			double langAxis;
			double shortAxis;
			double axisRatio;
			double arcLength;
			double circularity;
			double rectangularity;
			double centroidOffset;
			cv::RotatedRect minBoundingRect;
		};

		std::vector<BlobInfo> blobInfos;
		blobInfos.reserve(contours.size()); // 轮廓数上界，避免多次扩容

		// 收集所有符合条件的 Blob 信息
		for (size_t idx = 0; idx < contours.size(); ++idx) {

			// 只处理顶层轮廓（父轮廓）
			if (hierarchy[idx][3] != -1) {
				continue;
			}

			const auto& contour = contours[idx];

			// 计算面积
			double area = cv::contourArea(contour);

			// 计算孔洞面积
			double totalHoleArea = 0.0;
			int childIdx = hierarchy[idx][2]; // 子轮廓的索引
			while (childIdx != -1) {
				double holeArea = cv::contourArea(contours[childIdx]);
				const bool shouldCount =
					!params.filterParams.areaEnabled ||
					(holeArea > params.filterParams.minHoleArea);
				if (shouldCount) {
					// 只有确实要输出 filteredBinaryImage 时才画到 holeBin，节省一次 drawContours
					if (needHoleBin) {
						cv::drawContours(holeBin, contours, childIdx, cv::Scalar(255), -1);
					}
					totalHoleArea += holeArea;
				}
				childIdx = hierarchy[childIdx][0]; // 下一个兄弟轮廓
			}

			// 有孔洞时，外轮廓面积减去未填充的孔洞面积
			if (totalHoleArea > 0) {
				area -= totalHoleArea;
			}

			// 面积过滤
			if (params.filterParams.areaEnabled &&
				(area < params.filterParams.minArea || area > params.filterParams.maxArea))
			{
				DLOG_WARN("blob", "得到的面积过小跳过!");
				continue;
			}

			// 外接矩形和轴比
			cv::RotatedRect minBoundingRect = cv::minAreaRect(contour);
			const double w = minBoundingRect.size.width;
			const double h = minBoundingRect.size.height;
			if (w < 1e-6 || h < 1e-6) {
				continue;
			}
			const double langAxis = std::max(w, h);
			const double shortAxis = std::min(w, h);
			const double axisRatio = langAxis / shortAxis;
			if (params.filterParams.axisRatioEnabled &&
				(axisRatio < params.filterParams.minAxisRatio || axisRatio > params.filterParams.maxAxisRatio))
			{
				DLOG_WARN("blob", "得到的周长过小跳过!");
				continue;
			}

			// 计算圆度
			const double perimeter = cv::arcLength(contour, true);
			if (perimeter < 1e-6) {
				DLOG_WARN("blob", "得到的周长过小跳过!");
				continue;
			}
			const double circularity = (4 * CV_PI * area) / (perimeter * perimeter);
			if (params.filterParams.circularityEnabled &&
				circularity < params.filterParams.minCircularity)
			{
				DLOG_WARN("blob", "得到的圆度过小跳过!");
				continue;
			}

			// 计算矩形度（直接复用 minAreaRect 的 w*h，避免再次读成员）
			const double rectangularity = area / (w * h);
			if (params.filterParams.rectangularityEnabled &&
				rectangularity < params.filterParams.minRectangularity)
			{
				DLOG_WARN("blob", "得到的矩形度过小跳过!");
				continue;
			}

			// 计算质心偏移
			cv::Moments m = cv::moments(contour);
			if (m.m00 == 0) {
				continue;
			}
			cv::Point2f centroid(static_cast<float>(m.m10 / m.m00),
				static_cast<float>(m.m01 / m.m00));
			const double centroidOffset =
				std::abs(centroid.x - minBoundingRect.center.x) +
				std::abs(centroid.y - minBoundingRect.center.y);
			if (params.filterParams.centroidOffsetEnabled &&
				centroidOffset > params.filterParams.maxCentroidOffset)
			{
				continue;
			}

			// 保存信息（不再拷贝整条轮廓，只记 idx）
			BlobInfo blobInfo;
			blobInfo.idx = static_cast<int>(idx);
			blobInfo.area = area;
			blobInfo.boundingRect = cv::boundingRect(contour);
			blobInfo.centroid = centroid;
			blobInfo.langAxis = langAxis;
			blobInfo.shortAxis = shortAxis;
			blobInfo.axisRatio = axisRatio;
			blobInfo.arcLength = perimeter;
			blobInfo.circularity = circularity;
			blobInfo.rectangularity = rectangularity;
			blobInfo.centroidOffset = centroidOffset;
			blobInfo.minBoundingRect = minBoundingRect;

			blobInfos.push_back(blobInfo);
		}

		// 处理重叠的 Blob
		std::vector<char> keep(blobInfos.size(), 1); // 用 char 代替 vector<bool>，避免位代理开销

		if (params.connectivityParams.minOverlapRatio > 0) {
			for (size_t i = 0; i < blobInfos.size(); ++i) {
				if (!keep[i]) continue;

				for (size_t j = i + 1; j < blobInfos.size(); ++j) {
					if (!keep[j]) continue;

					cv::Rect inter = blobInfos[i].boundingRect & blobInfos[j].boundingRect;
					if (inter.area() == 0) continue;

					const double overlapArea = static_cast<double>(inter.area());
					const double minArea = std::min(blobInfos[i].area, blobInfos[j].area);
					const double overlapRatio = overlapArea / minArea;

					if (overlapRatio > params.connectivityParams.minOverlapRatio) {
						if (blobInfos[i].area < blobInfos[j].area) {
							keep[i] = 0;
							break;
						}
						else {
							keep[j] = 0;
						}
					}
				}
			}
		}

		// 预分配输出向量容量，彻底避免 push_back 反复扩容拷贝
		const size_t survivors = static_cast<size_t>(std::count(keep.begin(), keep.end(), (char)1));
		if (params.outputControlParams.contourOutputEnabled) {
			result.contours.reserve(survivors);
		}
		result.centroids.reserve(survivors);
		result.areas.reserve(survivors);
		result.langAxises.reserve(survivors);
		result.shortAxises.reserve(survivors);
		result.axisRatios.reserve(survivors);
		result.arcLengths.reserve(survivors);
		result.circularities.reserve(survivors);
		result.rectangularities.reserve(survivors);
		result.centroidOffsets.reserve(survivors);
		result.minBoundingRects.reserve(survivors);
		filteredIndices.reserve(survivors);

		// 根据过滤结果填充 result 和 filteredIndices
		for (size_t i = 0; i < blobInfos.size(); ++i) {
			if (!keep[i]) {
				continue;
			}
			const auto& b = blobInfos[i];

			if (params.outputControlParams.contourOutputEnabled) {
				result.contours.push_back(contours[b.idx]);
			}
			filteredIndices.push_back(b.idx);

			result.centroids.push_back(b.centroid);
			result.areas.push_back(b.area);
			result.sumAreas += b.area;
			result.langAxises.push_back(b.langAxis);
			result.shortAxises.push_back(b.shortAxis);
			result.axisRatios.push_back(b.axisRatio);
			result.arcLengths.push_back(b.arcLength);
			result.circularities.push_back(b.circularity);
			result.rectangularities.push_back(b.rectangularity);
			result.centroidOffsets.push_back(b.centroidOffset);
			result.minBoundingRects.push_back(b.minBoundingRect);

			result.blobCount++;
		}
	}

	// 排序 Blob 结果
	void sortBlobs(const SortParams& params, blobResult& result)
	{
		if (result.blobCount <= 1) {
			DLOG_WARN("blob", "得到的blob数量小于1不用排序!");
			return;
		}

		// 创建索引向量
		std::vector<int> indices(result.blobCount);
		std::iota(indices.begin(), indices.end(), 0);

		// 根据排序特征预取出一份 double 数组，避免比较器里每次都走 switch
		std::vector<double> key(result.blobCount);
		switch (params.feature) {
		case SortFeature::AREA:
			for (int i = 0; i < result.blobCount; ++i) key[i] = result.areas[i];
			break;
		case SortFeature::AXIS_RATIO:
			for (int i = 0; i < result.blobCount; ++i) key[i] = result.axisRatios[i];
			break;
		case SortFeature::CIRCULARITY:
			for (int i = 0; i < result.blobCount; ++i) key[i] = result.circularities[i];
			break;
		case SortFeature::RECTANGULARITY:
			for (int i = 0; i < result.blobCount; ++i) key[i] = result.rectangularities[i];
			break;
		default:
			std::fill(key.begin(), key.end(), 0.0);
			break;
		}

		if (params.order == SortOrder::ASCENDING) {
			std::sort(indices.begin(), indices.end(),
				[&key](int a, int b) { return key[a] < key[b]; });
		}
		else {
			std::sort(indices.begin(), indices.end(),
				[&key](int a, int b) { return key[a] > key[b]; });
		}

		// 重新排序结果
		reorderBlobResult(indices, result);
	}

	// 重新排序 blobResult 中的数据
	void reorderBlobResult(const std::vector<int>& indices, blobResult& result)
	{
		// 注意：这里对每个成员都使用 std::move，避免把 std::vector<cv::Point>（单条轮廓）
		// 整条拷贝一遍。对于 POD 标量类型 move 退化为拷贝，无副作用。
		auto reorder = [&](auto& vec) {
			using T = std::decay_t<decltype(vec[0])>;
			std::vector<T> reordered;
			reordered.reserve(indices.size());
			for (size_t i = 0; i < indices.size(); ++i) {
				reordered.push_back(std::move(vec[indices[i]]));
			}
			vec = std::move(reordered);
			};

		reorder(result.centroids);
		reorder(result.areas);
		reorder(result.langAxises);
		reorder(result.shortAxises);
		reorder(result.arcLengths);
		if (!result.contours.empty()) {
			reorder(result.contours);
		}
		reorder(result.axisRatios);
		reorder(result.circularities);
		reorder(result.rectangularities);
		reorder(result.centroidOffsets);
		reorder(result.minBoundingRects);
	}

	// 就地把 result 截断为前 N 条。原 copyFirstN 的 "深拷贝 result + 从源 erase 前 N"
	// 路径会多出一次全量深拷贝 + 一次 O(n) 擦除搬移，这里统一换成 resize。
	void truncateResultInPlace(blobResult& result, int N)
	{
		if (N < 0) N = 0;
		const size_t n = static_cast<size_t>(N);

		auto trim = [&](auto& vec) {
			if (vec.size() > n) vec.resize(n);
			};

		trim(result.centroids);
		trim(result.areas);
		trim(result.contours);
		trim(result.axisRatios);
		trim(result.langAxises);
		trim(result.shortAxises);
		trim(result.arcLengths);
		trim(result.circularities);
		trim(result.rectangularities);
		trim(result.centroidOffsets);
		trim(result.minBoundingRects);

		result.blobCount = N;
		result.sumAreas = std::accumulate(result.areas.begin(), result.areas.end(), 0.0);
	}

	cv::Matx23d calcuPositionCorrectionMatrix(const Location::PositionCorrectionParam& param)
	{
		float angle = -(param.runAngle - param.baseAngle);
		angle *= CV_PI / 180;
		float scaleX = param.runScaleX / param.baseScaleX;
		float scaleY = param.runScaleY / param.baseScaleY;
		cv::Point2f pTmp = param.runPoint - param.basePoint;
		float transX = pTmp.x;
		float transY = pTmp.y;

		// 原实现里 cos/sin 各算了两次，这里只算一次
		const double c = std::cos(angle);
		const double s = std::sin(angle);
		const double alphaX = c * scaleX;
		const double alphaY = c * scaleY;
		const double betaX = s * scaleX;
		const double betaY = s * scaleY;

		cv::Matx23d M(
			alphaX, betaY, (1 - alphaX) * param.basePoint.x - betaY * param.basePoint.y + transX,
			-betaX, alphaY, betaX * param.basePoint.x + (1 - alphaY) * param.basePoint.y + transY
		);

		return M;
	}

} // namespace Blob

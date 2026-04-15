#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <map>
#include <set>

namespace fs = std::filesystem;

// Marker configuration
static constexpr int MEASUREMENT_ID = 0;
static const std::set<int> ANCHOR_IDS = {1, 2, 3, 4};
static constexpr int REQUIRED_ANCHOR_COUNT = 4;


class HomographyStabilizer {
public:
    struct Metrics {
        int anchor_markers_found = 0;
        bool all_anchors_found = false;
        bool homography_valid = false;
        bool used_fallback = false;
        double reprojection_error_px = 0.0;
        // Decomposed compensation
        double tx = 0.0;
        double ty = 0.0;
        double rotation_deg = 0.0;
        double scale = 1.0;
        // Centre marker (ID 0) tracking in stabilized frame
        bool aruco_detected = false;
        cv::Point2f aruco_center = cv::Point2f(0, 0);
    };

    HomographyStabilizer() {
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

        det_params_.adaptiveThreshWinSizeMin = 3;
        det_params_.adaptiveThreshWinSizeMax = 53;
        det_params_.adaptiveThreshWinSizeStep = 4;
        det_params_.adaptiveThreshConstant = 7;
        det_params_.minMarkerPerimeterRate = 0.01;
        det_params_.maxMarkerPerimeterRate = 4.0;
        det_params_.polygonalApproxAccuracyRate = 0.05;
        det_params_.minCornerDistanceRate = 0.02;
        det_params_.minDistanceToBorder = 1;
        det_params_.minOtsuStdDev = 3.0;
        det_params_.perspectiveRemovePixelPerCell = 8;
        det_params_.perspectiveRemoveIgnoredMarginPerCell = 0.2;
        det_params_.maxErroneousBitsInBorderRate = 0.5;
        det_params_.errorCorrectionRate = 0.8;
        det_params_.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        det_params_.cornerRefinementWinSize = 5;
        det_params_.cornerRefinementMaxIterations = 50;
        det_params_.cornerRefinementMinAccuracy = 0.01;

        detector_ = cv::aruco::ArucoDetector(aruco_dict_, det_params_);
    }

    cv::Mat stabilize(const cv::Mat& frame, Metrics& m) {
        std::vector<cv::Point2f> current_corners = detectAnchorMarkers(frame, m);
        
        // Initialize reference on first successful detection
        if (!initialized_) {
            if (!m.all_anchors_found) {
                return frame.clone();
            }
            reference_corners_ = current_corners;
            initialized_ = true;
            return frame.clone();
        }

        // Compute homography
        cv::Mat H;
        if (m.all_anchors_found) {
            H = computeHomography(current_corners, reference_corners_, m);
            if (m.homography_valid) {
                last_valid_H_ = H.clone();
            }
        }

        // Use fallback if needed
        if (!m.homography_valid) {
            if (last_valid_H_.empty()) {
                return frame.clone();
            }
            H = last_valid_H_;
            m.used_fallback = true;
        }

        // Decompose H
        m.tx           = H.at<double>(0, 2);
        m.ty           = H.at<double>(1, 2);
        m.scale        = std::sqrt(H.at<double>(0,0)*H.at<double>(0,0) +
                                   H.at<double>(1,0)*H.at<double>(1,0));
        m.rotation_deg = std::atan2(H.at<double>(1,0),
                                    H.at<double>(0,0)) * 180.0 / CV_PI;

        // Apply stabilization (GPU)
        cv::Mat stabilized;
        cv::cuda::GpuMat gpu_frame, gpu_stabilized;
        gpu_frame.upload(frame);
        cv::cuda::warpPerspective(gpu_frame, gpu_stabilized, H, frame.size(),
                                  cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        gpu_stabilized.download(stabilized);

        return stabilized;
    }

    cv::Point2f detectCenterMarker(const cv::Mat& frame, Metrics& m) {
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        detector_.detectMarkers(frame, corners, ids);

        cv::Point2f center(0, 0);
        m.aruco_detected = false;

        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == MEASUREMENT_ID) {
                for (const auto& c : corners[i]) center += c;
                center /= 4.0f;
                m.aruco_detected = true;
                m.aruco_center = center;
                break;
            }
        }
        return center;
    }

    void reset() {
        initialized_ = false;
        reference_corners_.clear();
        last_valid_H_ = cv::Mat();
    }

private:

    std::vector<cv::Point2f> detectAnchorMarkers(const cv::Mat& frame, Metrics& m) {
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<std::vector<cv::Point2f>> rejected;

        detector_.detectMarkers(frame, corners, ids, rejected);

        std::set<int> found_anchor_ids;
        std::map<int, std::vector<cv::Point2f>> anchor_corners_by_id;

        for (size_t i = 0; i < ids.size(); ++i) {
            int id = ids[i];
            if (ANCHOR_IDS.count(id)) {
                found_anchor_ids.insert(id);
                anchor_corners_by_id[id] = corners[i];
            }
        }

        m.anchor_markers_found = static_cast<int>(found_anchor_ids.size());
        m.all_anchors_found = (m.anchor_markers_found == REQUIRED_ANCHOR_COUNT);

        // Build ordered corner list: ID1, ID2, ID3, ID4
        std::vector<cv::Point2f> ordered_corners;
        if (m.all_anchors_found) {
            ordered_corners.reserve(16); // 4 markers × 4 corners
            for (int aid : {1, 2, 3, 4}) {
                for (const auto& pt : anchor_corners_by_id[aid]) {
                    ordered_corners.push_back(pt);
                }
            }
        }

        return ordered_corners;
    }

    cv::Mat computeHomography(const std::vector<cv::Point2f>& src,
                             const std::vector<cv::Point2f>& dst,
                             Metrics& m) {
        if (src.size() != 16 || dst.size() != 16) {
            return cv::Mat();
        }

        std::vector<uchar> mask;
        cv::Mat H = cv::findHomography(src, dst, cv::RANSAC, 2.0, mask, 2000, 0.999);

        if (H.empty()) {
            return cv::Mat();
        }

        // Basic sanity check
        double det = cv::determinant(H);
        if (std::abs(det) < 0.1 || std::abs(det) > 10.0) {
            return cv::Mat();
        }

        // Calculate reprojection error
        std::vector<cv::Point2f> proj;
        cv::perspectiveTransform(src, proj, H);

        double err = 0.0;
        int inliers = 0;
        for (size_t i = 0; i < dst.size() && i < mask.size(); ++i) {
            if (mask[i]) {
                err += cv::norm(dst[i] - proj[i]);
                inliers++;
            }
        }

        if (inliers > 0) {
            m.reprojection_error_px = err / inliers;
            m.homography_valid = true;
        }

        return H;
    }


    // State
    cv::aruco::Dictionary aruco_dict_;
    cv::aruco::DetectorParameters det_params_;
    cv::aruco::ArucoDetector detector_;
    bool initialized_ = false;
    std::vector<cv::Point2f> reference_corners_;
    cv::Mat last_valid_H_;
};

void processDataset(const std::string& input_dir, const std::string& output_dir) {
    HomographyStabilizer stabilizer;
    fs::create_directories(output_dir + "/frames");

    // Collect frame files
    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(input_dir + "/frames")) {
        if (e.path().extension() == ".png") {
            files.push_back(e.path().filename().string());
        }
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No PNG frames found in " << input_dir << "/frames\n";
        return;
    }

    std::cout << "Processing " << files.size() << " frames...\n";

    // Statistics
    int valid_homographies = 0;
    int fallback_used = 0;
    int all_anchors_detected = 0;
    double total_error = 0.0;

    for (size_t i = 0; i < files.size(); ++i) {
        if (i % 100 == 0) {
            std::cout << "Frame " << i << "/" << files.size() << std::endl;
        }

        cv::Mat frame = cv::imread(input_dir + "/frames/" + files[i]);
        if (frame.empty()) {
            std::cerr << "Failed to load " << files[i] << std::endl;
            continue;
        }

        HomographyStabilizer::Metrics m;
        cv::Mat stabilized = stabilizer.stabilize(frame, m);
        cv::imwrite(output_dir + "/frames/" + files[i], stabilized);

        // Update statistics
        if (m.homography_valid) valid_homographies++;
        if (m.used_fallback) fallback_used++;
        if (m.all_anchors_found) all_anchors_detected++;
        total_error += m.reprojection_error_px;
    }

    // Summary
    size_t n = files.size();
    std::cout << "\n=== Summary ===\n";
    std::cout << "Frames processed: " << n << "\n";
    std::cout << "All anchors detected: " << all_anchors_detected << " (" << (100.0*all_anchors_detected/n) << "%)\n";
    std::cout << "Valid homographies: " << valid_homographies << " (" << (100.0*valid_homographies/n) << "%)\n";
    std::cout << "Fallback used: " << fallback_used << " (" << (100.0*fallback_used/n) << "%)\n";
    std::cout << "Avg reprojection error: " << (total_error/n) << " px\n";
}


// ─────────────────────────────────────────────────────────────────────────────
//  Video input mode
// ─────────────────────────────────────────────────────────────────────────────

void processVideo(const std::string& video_path, const std::string& output_dir, int max_frames = 0) {
    HomographyStabilizer stabilizer;
    fs::create_directories(output_dir);

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Cannot open video: " << video_path << "\n";
        return;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps       = cap.get(cv::CAP_PROP_FPS);
    int width        = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height       = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    std::cout << "ArUco Homography Stabilizer — Video Mode\n";
    std::cout << "Input  : " << video_path << "\n";
    std::cout << "Frames : " << total_frames << " @ " << fps << " fps, "
              << width << "x" << height << "\n\n";

    std::ofstream csv(output_dir + "/compensation.csv");
    csv << "frame_id,tx_px,ty_px,rotation_deg,scale,"
           "anchor_markers_found,all_anchors_found,homography_valid,used_fallback,"
           "reprojection_error_px,aruco_detected,aruco_center_x,aruco_center_y\n";

    cv::VideoWriter writer(
        output_dir + "/stabilized.avi",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps, cv::Size(width, height));

    int valid_homographies = 0, fallback_used = 0, all_anchors_detected = 0;
    int frame_idx = 0;
    cv::Mat frame;

    int limit = (max_frames > 0) ? max_frames : total_frames;
    while (cap.read(frame) && frame_idx < limit) {
        if (frame_idx % 100 == 0)
            std::cout << "Frame " << frame_idx << "/" << limit << "\n";

        HomographyStabilizer::Metrics m;
        cv::Mat stabilized = stabilizer.stabilize(frame, m);
        writer.write(stabilized);

        // Detect centre marker in stabilized frame
        stabilizer.detectCenterMarker(stabilized, m);

        csv << frame_idx << ","
            << m.tx << "," << m.ty << ","
            << m.rotation_deg << "," << m.scale << ","
            << m.anchor_markers_found << "," << m.all_anchors_found << ","
            << m.homography_valid << "," << m.used_fallback << ","
            << m.reprojection_error_px << ","
            << m.aruco_detected << ","
            << m.aruco_center.x << "," << m.aruco_center.y << "\n";

        if (m.homography_valid) valid_homographies++;
        if (m.used_fallback) fallback_used++;
        if (m.all_anchors_found) all_anchors_detected++;
        frame_idx++;
    }

    cap.release();
    writer.release();

    std::cout << "\n=== Summary ===\n";
    std::cout << "Frames processed: " << frame_idx << "\n";
    std::cout << "All anchors detected: " << all_anchors_detected
              << " (" << (100.0 * all_anchors_detected / frame_idx) << "%)\n";
    std::cout << "Valid homographies: " << valid_homographies
              << " (" << (100.0 * valid_homographies / frame_idx) << "%)\n";
    std::cout << "Stabilized video → " << output_dir << "/stabilized.avi\n";
    std::cout << "Compensation CSV  → " << output_dir << "/compensation.csv\n";
}


static bool isVideoFile(const std::string& path) {
    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input> <output_dir> [--frames N]\n";
        std::cerr << "  input     : video file (.mp4/.avi/.mov) OR directory with frames/\n";
        std::cerr << "  --frames N: process only first N frames (default: all)\n";
        return 1;
    }

    std::string in  = argv[1];
    std::string out = argv[2];

    int max_frames = 0;
    for (int i = 3; i < argc - 1; i++) {
        if (std::string(argv[i]) == "--frames")
            max_frames = std::stoi(argv[i + 1]);
    }

    if (isVideoFile(in)) {
        if (!fs::exists(in)) {
            std::cerr << "Video not found: " << in << "\n";
            return 1;
        }
        processVideo(in, out, max_frames);
    } else {
        if (!fs::exists(in + "/frames")) {
            std::cerr << "Frames directory not found: " << in << "/frames\n";
            return 1;
        }
        processDataset(in, out);
    }

    return 0;
}
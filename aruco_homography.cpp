#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
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
    };

    HomographyStabilizer() {
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        det_params_ = cv::aruco::DetectorParameters::create();
        
        // detection parameters
        det_params_->adaptiveThreshWinSizeMin = 5;
        det_params_->adaptiveThreshWinSizeMax = 23;
        det_params_->minMarkerPerimeterRate = 0.02;
        det_params_->maxMarkerPerimeterRate = 4.0;
        det_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        det_params_->cornerRefinementWinSize = 5;
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

        // Apply stabilization
        cv::Mat stabilized;
        cv::warpPerspective(frame, stabilized, H, frame.size(),
                          cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        
        return stabilized;
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

        cv::aruco::detectMarkers(frame, aruco_dict_, corners, ids, det_params_, rejected);

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
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> det_params_;
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


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>\n";
        std::cerr << "  input_dir  : directory containing frames/ subfolder\n";
        std::cerr << "  output_dir : stabilised frames written here\n";
        return 1;
    }

    std::string in  = argv[1];
    std::string out = argv[2];

    if (!fs::exists(in + "/frames")) {
        std::cerr << "Frames directory not found: " << in << "/frames\n";
        return 1;
    }

    processDataset(in, out);
    return 0;
}
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ─────────────────────────────────────────────────────────────────────────────
//  RANSAC Stabilizer with ArUco Displacement Tracking
//
//  1. Uses ORB+RANSAC to stabilize background motion
//  2. Tracks ArUco marker (ID 0) displacement after stabilization
//  3. Compares measured displacement to ground truth structural vibration
// ─────────────────────────────────────────────────────────────────────────────

// ── Frame geometry ────────────────────────────────────────────────────────────
static constexpr int FRAME_W = 1440;
static constexpr int FRAME_H = 1080;
static constexpr int ARUCO_SIZE = 150;
static constexpr int EXCLUSION_MARGIN = 40;

// ArUco exclusion region for ORB detection
static const cv::Rect ARUCO_EXCLUSION(
    FRAME_W / 2 - ARUCO_SIZE / 2 - EXCLUSION_MARGIN,
    FRAME_H / 2 - ARUCO_SIZE / 2 - EXCLUSION_MARGIN,
    ARUCO_SIZE + 2 * EXCLUSION_MARGIN,
    ARUCO_SIZE + 2 * EXCLUSION_MARGIN
);

// ── Refined Parameters for Sub-pixel Accuracy ───────────────────────────────
static constexpr int   ORB_FEATURES   = 5000;   // More features for stability
static constexpr float LOWE_RATIO     = 0.65f;  // Tighter ratio test (was 0.7)
static constexpr double RANSAC_THRESH = 1.0;    // Tighter RANSAC threshold
static constexpr int   MIN_MATCHES    = 25;     // More matches required
static constexpr double HOMOGRAPHY_CONFIDENCE = 0.995;  // Higher confidence
// Reference refresh: update reference when inlier ratio drops below this
// to prevent match degradation as the scene drifts from frame 0.
static constexpr double MIN_INLIER_RATIO = 0.10; // 10% of good matches must be inliers
static constexpr int    REFRESH_INLIER_THRESHOLD = 60; // refresh if inliers < this


struct DisplacementData {
    int frame_id;
    cv::Point2f measured_displacement;
    cv::Point2f ground_truth_displacement;
    double error_magnitude;
};

class RansacStabilizer {
public:
    struct Metrics {
        double  stabilization_ms = 0.0;
        double  aruco_detection_ms = 0.0;
        int     keypoints_found  = 0;
        int     good_matches     = 0;
        int     inliers          = 0;
        bool    homography_valid = false;
        bool    aruco_detected   = false;
        cv::Point2f aruco_center = cv::Point2f(0, 0);
        // Decomposed compensation applied to this frame
        double tx = 0.0;
        double ty = 0.0;
        double rotation_deg = 0.0;
        double scale = 1.0;
    };

    RansacStabilizer()
        : orb_(cv::ORB::create(ORB_FEATURES)),
          aruco_dict_(cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250)),
          aruco_params_(cv::aruco::DetectorParameters::create())
    {
        // Configure ORB for better feature quality
        orb_->setScaleFactor(1.2f);     // Smaller scale factor for more levels
        orb_->setNLevels(12);           // More pyramid levels
        orb_->setEdgeThreshold(15);     // Smaller edge threshold
        orb_->setFirstLevel(0);         // Start from original resolution
        orb_->setWTA_K(2);              // Use 2-point sampling
        orb_->setScoreType(cv::ORB::HARRIS_SCORE);  // Better corner response
        
        // Configure ArUco detection for sub-pixel accuracy
        aruco_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        aruco_params_->cornerRefinementWinSize = 5;
        aruco_params_->cornerRefinementMaxIterations = 30;
        aruco_params_->cornerRefinementMinAccuracy = 0.01;
        
        // Build feature mask - exclude ArUco region from ORB detection
        feature_mask_ = cv::Mat::ones(FRAME_H, FRAME_W, CV_8U) * 255;
        feature_mask_(ARUCO_EXCLUSION) = 0;
    }

    cv::Mat stabilize(const cv::Mat& frame, Metrics& m) {
        auto t_start = cv::getTickCount();
        
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // ── 1. ORB feature detection and matching ────────────────────────
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        orb_->detectAndCompute(gray, feature_mask_, kps, desc);
        m.keypoints_found = static_cast<int>(kps.size());

        // Initialize reference frame
        if (!initialized_) {
            if (kps.size() < MIN_MATCHES) {
                return frame.clone();
            }
            ref_kps_ = kps;
            ref_desc_ = desc.clone();
            initialized_ = true;
            return frame.clone();
        }

        // ── 2. Match and estimate homography ──────────────────────────────
        cv::Mat H;
        if (kps.size() >= MIN_MATCHES) {
            cv::BFMatcher matcher(cv::NORM_HAMMING);
            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(desc, ref_desc_, knn_matches, 2);

            // Lowe ratio test
            std::vector<cv::DMatch> good_matches;
            for (const auto& pair : knn_matches) {
                if (pair.size() == 2 && pair[0].distance < LOWE_RATIO * pair[1].distance) {
                    good_matches.push_back(pair[0]);
                }
            }
            m.good_matches = static_cast<int>(good_matches.size());

            if (good_matches.size() >= MIN_MATCHES) {
                std::vector<cv::Point2f> src_pts, dst_pts;
                for (const auto& dm : good_matches) {
                    src_pts.push_back(kps[dm.queryIdx].pt);
                    dst_pts.push_back(ref_kps_[dm.trainIdx].pt);
                }

                std::vector<uchar> mask;
                H = cv::findHomography(src_pts, dst_pts, cv::RANSAC, RANSAC_THRESH, mask, 
                                     2000, HOMOGRAPHY_CONFIDENCE);
                
                if (!H.empty()) {
                    m.inliers = cv::countNonZero(mask);
                    double inlier_ratio = static_cast<double>(m.inliers) /
                                         static_cast<double>(good_matches.size());
                    m.homography_valid = m.inliers >= MIN_MATCHES &&
                                        inlier_ratio >= MIN_INLIER_RATIO;
                    if (m.homography_valid) {
                        last_valid_H_ = H.clone();
                        // Adaptive reference refresh: if inliers are dropping, update
                        // the reference to the current stabilized frame so that future
                        // frames match against a more recent (and thus more similar)
                        // scene view rather than drifting far from frame 0.
                        if (m.inliers < REFRESH_INLIER_THRESHOLD) {
                            ref_kps_   = kps;
                            ref_desc_  = desc.clone();
                            // Reset last_valid_H_ so next frame computes fresh
                            last_valid_H_ = cv::Mat();
                        }
                    }
                }
            }
        }

        // ── 3. Apply stabilization with sub-pixel interpolation ──────────────
        // Decompose the H that will actually be applied
        {
            const cv::Mat& Huse = (m.homography_valid && !H.empty()) ? H : last_valid_H_;
            if (!Huse.empty()) {
                m.tx           = Huse.at<double>(0, 2);
                m.ty           = Huse.at<double>(1, 2);
                m.scale        = std::sqrt(Huse.at<double>(0,0)*Huse.at<double>(0,0) + Huse.at<double>(1,0)*Huse.at<double>(1,0));
                m.rotation_deg = std::atan2(Huse.at<double>(1,0), Huse.at<double>(0,0)) * 180.0 / CV_PI;
            }
        }

        cv::Mat stabilized;
        if (m.homography_valid) {
            cv::warpPerspective(frame, stabilized, H, frame.size(),
                              cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        } else if (!last_valid_H_.empty()) {
            cv::warpPerspective(frame, stabilized, last_valid_H_, frame.size(),
                              cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
        } else {
            stabilized = frame.clone();
        }

        m.stabilization_ms = tickMs(cv::getTickCount() - t_start);
        return stabilized;
    }

    cv::Point2f detectArUcoCenter(const cv::Mat& frame, Metrics& m) {
        auto t_start = cv::getTickCount();
        
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(frame, aruco_dict_, corners, ids, aruco_params_);
        
        cv::Point2f center(0, 0);
        m.aruco_detected = false;
        
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] == 0) {  // Target marker ID
                // Sub-pixel corner refinement for higher accuracy
                cv::Mat gray;
                if (frame.channels() == 3) {
                    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                } else {
                    gray = frame;
                }
                
                // Refine corners to sub-pixel accuracy
                cv::cornerSubPix(gray, corners[i], cv::Size(5, 5), cv::Size(-1, -1),
                    cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
                
                // Calculate center from refined corners
                for (const auto& corner : corners[i]) {
                    center += corner;
                }
                center /= 4.0f;
                m.aruco_detected = true;
                m.aruco_center = center;
                break;
            }
        }
        
        m.aruco_detection_ms = tickMs(cv::getTickCount() - t_start);
        return center;
    }

    void reset() {
        initialized_ = false;
        ref_kps_.clear();
        ref_desc_ = cv::Mat();
        last_valid_H_ = cv::Mat();
        reference_aruco_center_ = cv::Point2f(0, 0);
    }

private:
    static double tickMs(int64 ticks) {
        return static_cast<double>(ticks) / cv::getTickFrequency() * 1000.0;
    }

    cv::Ptr<cv::ORB> orb_;
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat feature_mask_;

    bool initialized_ = false;
    std::vector<cv::KeyPoint> ref_kps_;
    cv::Mat ref_desc_;
    cv::Mat last_valid_H_;
    cv::Point2f reference_aruco_center_;
};


// ─────────────────────────────────────────────────────────────────────────────

json loadGroundTruth(const std::string& input_dir) {
    std::ifstream file(input_dir + "/ground_truth.json");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ground_truth.json");
    }
    json gt;
    file >> gt;
    return gt;
}

void saveDisplacementData(const std::vector<DisplacementData>& data, const std::string& output_file) {
    json output;
    output["displacement_analysis"] = json::array();
    
    for (const auto& d : data) {
        json entry;
        entry["frame_id"] = d.frame_id;
        entry["measured_displacement"] = {d.measured_displacement.x, d.measured_displacement.y};
        entry["ground_truth_displacement"] = {d.ground_truth_displacement.x, d.ground_truth_displacement.y};
        entry["error_magnitude"] = d.error_magnitude;
        output["displacement_analysis"].push_back(entry);
    }
    
    std::ofstream file(output_file);
    file << output.dump(2);
}

void processDataset(const std::string& input_dir, const std::string& output_dir) {
    RansacStabilizer stabilizer;
    fs::create_directories(output_dir + "/frames");

    std::ofstream csv(output_dir + "/compensation.csv");
    csv << "frame_id,tx_px,ty_px,rotation_deg,scale,"
           "keypoints_found,good_matches,inliers,homography_valid,"
           "gt_sway_x,gt_sway_y,gt_rotation_deg\n";

    // Load ground truth data
    json ground_truth;
    try {
        ground_truth = loadGroundTruth(input_dir);
    } catch (const std::exception& e) {
        std::cerr << "Warning: " << e.what() << "\n";
    }

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(input_dir + "/frames")) {
        if (e.path().extension() == ".png")
            files.push_back(e.path().filename().string());
    }
    std::sort(files.begin(), files.end());

    if (files.empty()) {
        std::cerr << "No PNG frames found in " << input_dir << "/frames\n";
        return;
    }

    std::cout << "RANSAC Stabilizer with ArUco Displacement Tracking\n";
    std::cout << "Processing " << files.size() << " frames...\n\n";

    std::vector<DisplacementData> displacement_data;
    cv::Point2f reference_center(0, 0);
    bool reference_set = false;

    for (size_t i = 0; i < files.size(); ++i) {
        if (i % 50 == 0) {
            std::cout << "Frame " << i << "/" << files.size() << "\n";
        }

        cv::Mat frame = cv::imread(input_dir + "/frames/" + files[i]);
        if (frame.empty()) continue;

        // Stabilize frame
        RansacStabilizer::Metrics m;
        cv::Mat stabilized = stabilizer.stabilize(frame, m);
        cv::imwrite(output_dir + "/frames/" + files[i], stabilized);

        // Detect ArUco in stabilized frame
        cv::Point2f current_center = stabilizer.detectArUcoCenter(stabilized, m);

        // Ground truth camera sway for this frame (CSV)
        cv::Point2f gt_sway(0, 0);
        if (!ground_truth.empty() && ground_truth.contains("camera_sway")) {
            auto& cs = ground_truth["camera_sway"];
            if (cs.contains("displacement_x") && cs.contains("displacement_y")) {
                auto& sx = cs["displacement_x"];
                auto& sy = cs["displacement_y"];
                if (i < sx.size() && i < sy.size()) {
                    gt_sway.x = sx[i];
                    gt_sway.y = sy[i];
                }
            }
        }

        // Ground truth rotation for this frame (CSV)
        double gt_rot_deg = 0.0;
        if (!ground_truth.empty() && ground_truth.contains("rotation")) {
            auto& ra = ground_truth["rotation"]["angles"];
            if (i < ra.size()) {
                gt_rot_deg = ra[i];
            }
        }

        // Ground truth structural vibration (displacement_analysis.json only)
        cv::Point2f gt_disp(0, 0);
        if (!ground_truth.empty() && ground_truth.contains("structural_vibration")) {
            auto& sv = ground_truth["structural_vibration"];
            if (sv.contains("displacement_x") && sv.contains("displacement_y")) {
                auto& dx = sv["displacement_x"];
                auto& dy = sv["displacement_y"];
                if (i < dx.size() && i < dy.size()) {
                    gt_disp.x = dx[i];
                    gt_disp.y = dy[i];
                }
            }
        }

        cv::Point2f measured_disp(0, 0);
        double error_px = 0.0;
        if (m.aruco_detected && reference_set) {
            measured_disp = current_center - reference_center;
            error_px = cv::norm(measured_disp - gt_disp);
        }

        csv << i << ","
            << m.tx << "," << m.ty << ","
            << m.rotation_deg << "," << m.scale << ","
            << m.keypoints_found << "," << m.good_matches << "," << m.inliers << ","
            << m.homography_valid << ","
            << gt_sway.x << "," << gt_sway.y << ","
            << gt_rot_deg << "\n";
        
        if (m.aruco_detected) {
            if (!reference_set) {
                reference_center = current_center;
                reference_set = true;
            } else {
                displacement_data.push_back({static_cast<int>(i), measured_disp, gt_disp, error_px});
            }
        }
    }

    // Save displacement analysis
    if (!displacement_data.empty()) {
        saveDisplacementData(displacement_data, output_dir + "/displacement_analysis.json");
        
        // Calculate statistics
        double mean_error = 0.0;
        for (const auto& d : displacement_data) {
            mean_error += d.error_magnitude;
        }
        mean_error /= displacement_data.size();
        
        std::cout << "\n=== Displacement Analysis ===\n";
        std::cout << "Frames with ArUco detected: " << displacement_data.size() << "/" << files.size() << "\n";
        std::cout << "Mean displacement error: " << mean_error << " px\n";
        std::cout << "Results saved to: " << output_dir << "/displacement_analysis.json\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>\n";
        std::cerr << "  input_dir  : directory with frames/ and ground_truth.json\n";
        std::cerr << "  output_dir : stabilized frames and displacement analysis\n";
        return 1;
    }
    
    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    
    if (!fs::exists(input_dir + "/frames")) {
        std::cerr << "Frames directory not found: " << input_dir << "/frames\n";
        return 1;
    }
    
    try {
        processDataset(input_dir, output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
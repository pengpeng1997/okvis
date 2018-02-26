/*
 * okvis_library.cpp
 *
 *  Created on: 10 Mar 2017
 *      Author: toky
 */

#include<SLAMBenchAPI.h>


#include <stdint.h>
#include <vector>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <csignal>
#include <io/sensor/CameraSensor.h>
#include <io/sensor/CameraSensorFinder.h>
#include <io/sensor/IMUSensor.h>
#include "io/SLAMFrame.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <sstream>
#include <iomanip>

#include <okvis/VioParametersReader.hpp>
#include <timings.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <okvis/ThreadedKFVio.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/VioVisualizer.hpp>
#include <vector>



//=========================================================================
// SLAMBENCH SENSORS
//=========================================================================

static slambench::io::CameraSensor *grey_sensor_one = nullptr;
static slambench::io::CameraSensor *grey_sensor_two = nullptr;
static slambench::io::IMUSensor *IMU_sensor = nullptr;


//=========================================================================
// OKVIS INPUTS AND ESTIMATOR
//=========================================================================

okvis::Time t_one;
okvis::Time t_two;

cv::Mat* img_one;
cv::Mat* img_two;

static std::vector<okvis::Time>   tim_data;
static std::vector<Eigen::Vector3d>      gyr_data;
static std::vector<Eigen::Vector3d>      acc_data;

static bool one_ok = false;
static bool two_ok = false;

okvis::VioParameters parameters;
okvis::ThreadedKFVio* okvis_estimator;




//=========================================================================
// Object used to collect current pose via callbacks
//=========================================================================


class PoseViewer
{
public:

	Eigen::Matrix4d callback_position;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	constexpr static const double imageSize = 500.0;
	PoseViewer()
	{
	}
	// this we can register as a callback
	void publishFullStateAsCallback(
			const okvis::Time & /*t*/, const okvis::kinematics::Transformation & T_WS,
			const Eigen::Matrix<double, 9, 1> & speedAndBiases,
			const Eigen::Matrix<double, 3, 1> & /*omega_S*/)
	{

		callback_position = T_WS.T();
	}
} poseViewer;


//=========================================================================
// SLAMBench output values
//=========================================================================


slambench::outputs::Output *pose_output;

slambench::outputs::Output *left_frame_output = nullptr;
slambench::outputs::Output *right_frame_output = nullptr;


//=========================================================================
// Algorithmic parameters and default_values
//=========================================================================


bool blockingEstimator;
int numKeyframes;
int numImuFrames;

double detectionThreshold;  ///< Keypoint detection threshold.
int detectionOctaves;     ///< Number of keypoint detection octaves.
int maxNoKeypoints;       ///< Restrict to a maximum of this many keypoints per image (strongest ones).
int max_iterations; ///< Maximum iterations the optimization should perform.
int min_iterations; ///< Minimum iterations the optimization should perform.
double timeLimitForMatchingAndOptimization; ///< The time limit for both matching and optimization. [s]
double frameTimestampTolerance; ///< Time tolerance between frames to accept them as stereo frames. [s]


bool default_blockingEstimator = true;
int default_max_iterations = 10; ///< Maximum iterations the optimization should perform.
int default_min_iterations = 3; ///< Minimum iterations the optimization should perform.
double default_timeLimitForMatchingAndOptimization = 0.035; ///< The time limit for both matching and optimization. [s]
double default_detectionThreshold = 40.0;  ///< Keypoint detection threshold.
int default_detectionOctaves = 0;     ///< Number of keypoint detection octaves.
int default_maxNoKeypoints = 400;       ///< Restrict to a maximum of this many keypoints per image (strongest ones).
int default_numKeyframes = 5; ///< Number of keyframes.
int default_numImuFrames = 3; ///< Number of IMU frames.
double default_frameTimestampTolerance = 0.005; ///< Time tolerance between frames to accept them as stereo frames. [s]



//=========================================================================
// IMU parameters are hard-coded TODO !!
//=========================================================================

/// Those values are NOT available in the dataset
//------------------------------------------------------

double tau= 3600.0 ; // # reversion time constant, currently not in use [s]
double g= 9.81007 ; // # Earth's acceleration due to gravity [m/s^2]
Eigen::Vector3d a0 = {0.0, 0.0, 0.0};// # Accelerometer bias [m/s^2]


//=========================================================================
// What are those ? TODO
//=========================================================================


int publishRate = 200;  ///< Maximum publishing rate. [Hz] # rate at which odometry updates are published only works properly if imu_rate/publish_rate is an integer!!
bool publishLandmarks = true; ///< Select, if you want to publish landmarks at all.
float landmarkQualityThreshold = 1.0e-2; ///< Quality threshold under which landmarks are not published. Between 0 and 1.
float maxLandmarkQuality = 0.05; ///< Quality above which landmarks are assumed to be of the best quality. Between 0 and 1.
size_t maxPathLength = 20 ; ///< Maximum length of ros::nav_mgsgs::Path to be published.
bool publishImuPropagatedState = true; ///< Should the state that is propagated with IMU messages be published? Or just the optimized ones?
okvis::kinematics::Transformation T_Wc_W = okvis::kinematics::Transformation::Identity(); ///< Provide custom World frame Wc
okvis::FrameName trackedBodyFrame = okvis::FrameName::B; ///< B or S, the frame of reference that will be expressed relative to the selected worldFrame Wc
okvis::FrameName velocitiesFrame = okvis::FrameName::Wc; ///< B or S,  the frames in which the velocities of the selected trackedBodyFrame will be expressed in



//=========================================================================
// QUICK FIX TODO
//=========================================================================

bool visualization_fixed = false;


//=========================================================================
// SLAMBENCH API FUNCTIONS
//=========================================================================




bool sb_new_slam_configuration(SLAMBenchLibraryHelper * slam_settings) {


	slam_settings->addParameter(DiscretParameter<int>({1, 2, 3, 5, 10}, "nkf", "numKeyframes", "Number of key Frames", &numKeyframes, &default_numKeyframes));
	slam_settings->addParameter(DiscretParameter<int>({1, 2, 3, 5, 10}, "nif", "numImuframes", "Number of IMU Frames", &numImuFrames, &default_numImuFrames));

	slam_settings->addParameter(TypedParameter<int>("do", "detectionOctaves", "Number of keypoint detection octaves", &detectionOctaves, &default_detectionOctaves));
	slam_settings->addParameter(TypedParameter<int>("mnk", "maxNoKeypoints", "Restrict to a maximum of this many keypoints per image (strongest ones)", &maxNoKeypoints, &default_maxNoKeypoints));

	slam_settings->addParameter(TypedParameter<double>("dt", "detectionThreshold", "Keypoint detection threshold", &detectionThreshold, &default_detectionThreshold));
	slam_settings->addParameter(TypedParameter<double>("tm", "timeLimitForMatchingAndOptimization", "The time limit for both matching and optimization. [s]", &timeLimitForMatchingAndOptimization, &default_timeLimitForMatchingAndOptimization));

	slam_settings->addParameter(TypedParameter<int>("maxi", "max-iterations", "Maximum iterations the optimization should perform.", &max_iterations, &default_max_iterations));
	slam_settings->addParameter(TypedParameter<int>("mini", "min-iterations", "Minimum iterations the optimization should perform.", &min_iterations, &default_min_iterations));

	slam_settings->addParameter(TypedParameter<double>("tst", "frameTimestampTolerance", "Time tolerance between frames to accept them as stereo frames. [s]", &frameTimestampTolerance, &default_frameTimestampTolerance));

	slam_settings->addParameter(TypedParameter<bool>("be", "blockingEstimator", "The estimator has a blocking feature.", &blockingEstimator, &default_blockingEstimator));



	return true;
}

bool sb_init_slam_system(SLAMBenchLibraryHelper * slam_settings) {


	//=========================================================================
	//OKVIS uses glog
	//=========================================================================
	google::InitGoogleLogging("");
	FLAGS_stderrthreshold = 0;  // INFO: 0, WARNING: 1, ERROR: 2, FATAL: 3
	FLAGS_colorlogtostderr = 1;
	//=========================================================================


	//=========================================================================
	// We retrieve two grey sensors + IMU
	//=========================================================================

	slambench::io::CameraSensorFinder sensor_finder;
	auto grey_sensors = sensor_finder.Find(slam_settings->get_sensors(), {{"camera_type", "grey"}});
	if(grey_sensors.size() < 2) {
		std::cout << "Init failed, did not found enough cameras." << std::endl;
		return false;
	}

	grey_sensor_one = grey_sensors.at(0);
	grey_sensor_two = grey_sensors.at(1);
	img_one = new cv::Mat ( grey_sensor_one->Height ,  grey_sensor_one->Width, CV_8UC1);
	img_two = new cv::Mat ( grey_sensor_two->Height ,  grey_sensor_two->Width, CV_8UC1);

	IMU_sensor = (slambench::io::IMUSensor*)slam_settings->get_sensors().GetSensor(slambench::io::IMUSensor::kIMUType);
	if(IMU_sensor == nullptr) {
		std::cout << "Init failed, did not found IMU." << std::endl;
		return false;
	}

	if (grey_sensor_one->Rate != grey_sensor_two->Rate) {
		std::cout << "Camera sensors have a different rate, this is not supported." << std::endl;
		return false;
	}

	//=========================================================================
	// Prepare okvis parameters
	//=========================================================================


	parameters.optimization.numKeyframes = numKeyframes;
	parameters.optimization.numImuFrames = numImuFrames;

	parameters.optimization.max_iterations = max_iterations;
	parameters.optimization.min_iterations = min_iterations;
	parameters.optimization.timeLimitForMatchingAndOptimization = timeLimitForMatchingAndOptimization;

	parameters.optimization.detectionThreshold = detectionThreshold;
	parameters.optimization.detectionOctaves = detectionOctaves;
	parameters.optimization.maxNoKeypoints = maxNoKeypoints;

	parameters.sensors_information.cameraRate = grey_sensor_one->Rate;     ///< Camera rate in Hz.
	parameters.sensors_information.frameTimestampTolerance = frameTimestampTolerance; ///< Time tolerance between frames to accept them as stereo frames. [s]

	parameters.sensors_information.imageDelay = 0.0;  ///< Camera image delay. [s]
	parameters.sensors_information.imuIdx = 0;         ///< IMU index. Anything other than 0 will probably not work.




	okvis::kinematics::Transformation T_BS = okvis::kinematics::Transformation(IMU_sensor->Pose.cast<double>()); //  # tranform Body-Sensor (IMU)




	/// Those values are available in the dataset
	//------------------------------------------------------


	parameters.imu.sigma_g_c  = IMU_sensor->GyroscopeNoiseDensity;
	parameters.imu.sigma_gw_c = IMU_sensor->GyroscopeDriftNoiseDensity;
	parameters.imu.sigma_bg  = IMU_sensor->GyroscopeBiasDiffusion;
	parameters.imu.g_max  = IMU_sensor->GyroscopeSaturation;

	parameters.imu.sigma_a_c  = IMU_sensor->AcceleratorNoiseDensity;
	parameters.imu.sigma_aw_c  = IMU_sensor->AcceleratorDriftNoiseDensity;
	parameters.imu.sigma_ba  = IMU_sensor->AcceleratorBiasDiffusion;
	parameters.imu.a_max  =IMU_sensor->AcceleratorSaturation;


	parameters.imu.tau  = tau;
	parameters.imu.g  = g;
	parameters.imu.a0  = a0;

	parameters.imu.rate = IMU_sensor->Rate;
	parameters.imu.T_BS = T_BS;

	///< ************** Wind parameters.

	//// double priorStdev;  ///< Prior of wind. [m/s]
	//// double sigma_c;     ///< Drift noise density. [m/s/sqrt(s)]
	//// double tau;         ///< Reversion time constant. [s];
	//// double updateFrequency; ///< Related state estimates are inserted at this frequency. [Hz]
	////
	//// parameters.wind.priorStdev = priorStdev;
	//// parameters.wind.sigma_c = sigma_c;
	//// parameters.wind.tau = tau;
	//// parameters.wind.updateFrequency = updateFrequency;

	///< ****************** Camera extrinsic estimation parameters.

	// absolute (prior) w.r.t frame S
	double sigma_absolute_translation = 0.0; ///< Absolute translation stdev. [m]
	double sigma_absolute_orientation= 0.0; ///< Absolute orientation stdev. [rad]

	// relative (temporal)
	double sigma_c_relative_translation= 0.0; ///< Relative translation noise density. [m/sqrt(Hz)]
	double sigma_c_relative_orientation= 0.0; ///< Relative orientation noise density. [rad/sqrt(Hz)]

	parameters.camera_extrinsics.sigma_absolute_translation = sigma_absolute_translation;
	parameters.camera_extrinsics.sigma_absolute_orientation = sigma_absolute_orientation;
	parameters.camera_extrinsics.sigma_c_relative_translation = sigma_c_relative_translation;
	parameters.camera_extrinsics.sigma_c_relative_orientation = sigma_c_relative_orientation;

	///< ********** Publishing parameters.


	parameters.publishing.publishRate = publishRate ;
	parameters.publishing.publishLandmarks = publishLandmarks ;
	parameters.publishing.landmarkQualityThreshold = landmarkQualityThreshold ;
	parameters.publishing.maxLandmarkQuality = maxLandmarkQuality ;
	parameters.publishing.maxPathLength = maxPathLength ;
	parameters.publishing.publishImuPropagatedState = publishImuPropagatedState ;
	parameters.publishing.T_Wc_W = T_Wc_W ;
	parameters.publishing.trackedBodyFrame = trackedBodyFrame ;
	parameters.publishing.velocitiesFrame = velocitiesFrame ;


	///< Camera configuration.

	if (parameters.nCameraSystem.numCameras() != 0) {
		std::cout << "Unsupported case of configuration" << std::endl;
		exit(1);
	}


	std::cout << "Add cameras, may take a wee while..." << std::endl;

	for (auto sensor : grey_sensors) {
		Eigen::Matrix4d CAM_TSC = sensor->Pose.cast<double>();
		Eigen::Vector2d CAM_dimension = {sensor->Width,sensor->Height};
		Eigen::Vector2f CAM_focalLength    = {sensor->Intrinsics[0] * sensor->Width, sensor->Intrinsics[1] * sensor->Height};
		Eigen::Vector2f CAM_principalPoint = {sensor->Intrinsics[2] * sensor->Width, sensor->Intrinsics[3] * sensor->Height};

		if (sensor->DistortionType != slambench::io::CameraSensor::distortion_type_t::RadialTangential) {
			std::cerr << "Unsupported sensor with no distortion" << std::endl;
			exit(1);
		}

		Eigen::Vector4f CAM_distortionCoefficients = { sensor->RadialTangentialDistortion[0],
				sensor->RadialTangentialDistortion[1],
				sensor->RadialTangentialDistortion[2],
				sensor->RadialTangentialDistortion[3]};
		okvis::cameras::NCameraSystem::DistortionType CAM_distortion_type =  okvis::cameras::NCameraSystem::RadialTangential;

		parameters.nCameraSystem.addCamera(
				std::shared_ptr<const okvis::kinematics::Transformation>(new okvis::kinematics::Transformation (CAM_TSC)),
				std::shared_ptr<const okvis::cameras::CameraBase>(
						new okvis::cameras::PinholeCamera<
						okvis::cameras::RadialTangentialDistortion>(
								CAM_dimension[0],
								CAM_dimension[1],
								CAM_focalLength[0],
								CAM_focalLength[1],
								CAM_principalPoint[0],
								CAM_principalPoint[1],
								okvis::cameras::RadialTangentialDistortion(
										CAM_distortionCoefficients[0],
										CAM_distortionCoefficients[1],
										CAM_distortionCoefficients[2],
										CAM_distortionCoefficients[3])/*, id ?*/)),
										CAM_distortion_type/*, computeOverlaps ?*/);

	}
	std::cout << "Done !" << std::endl;



	/// Those parameters are not supported yet....

	///  parameters.magnetometer;  ///< Magnetometer parameters.
	///  parameters.position;  ///< Position sensor parameters.
	///  parameters.gps; ///< GPS parameters
	///  parameters.magnetic_enu_z;  ///< Dynamics of magnetic ENU z component variation.
	///  parameters.barometer;  ///< Barometer parameters.
	///  parameters.qff;  ///< QFF parameters.
	///  parameters.differential; ///< Differential pressure sensor parameters.
	///


	//=========================================================================
	// Start the visualization (only if rendering will happen)
	//=========================================================================

	parameters.visualization.displayImages = true;
	if (visualization_fixed) parameters.visualization.displayImages = false;


	//=========================================================================
	// START ESTIMATOR
	//=========================================================================

	okvis_estimator = new okvis::ThreadedKFVio(parameters);


	okvis_estimator->setFullStateCallback(
			std::bind(&PoseViewer::publishFullStateAsCallback, &poseViewer,
					std::placeholders::_1, std::placeholders::_2,
					std::placeholders::_3, std::placeholders::_4));

	okvis_estimator->setBlocking(blockingEstimator);



	//=========================================================================
	// DECLARE OUTPTUS
	//=========================================================================


	pose_output = new slambench::outputs::Output("Pose", slambench::values::VT_POSE, true);
	slam_settings->GetOutputManager().RegisterOutput(pose_output);

	left_frame_output = new slambench::outputs::Output("Left Frame", slambench::values::VT_FRAME);
	right_frame_output = new slambench::outputs::Output("Right Frame", slambench::values::VT_FRAME);
	left_frame_output->SetKeepOnlyMostRecent(true);
	right_frame_output->SetKeepOnlyMostRecent(true);


	slam_settings->GetOutputManager().RegisterOutput(left_frame_output);
	slam_settings->GetOutputManager().RegisterOutput(right_frame_output);

	pose_output->SetActive(true);
	left_frame_output->SetActive(true);
	right_frame_output->SetActive(true);


	//=========================================================================
	// END OF CONFIGURATION
	//=========================================================================


	return (grey_sensor_one && grey_sensor_two && IMU_sensor);
}


bool sb_update_frame (SLAMBenchLibraryHelper * slam_settings, slambench::io::SLAMFrame* s) {


	//=====================
	// SKIP UNKNOWNS
	//=====================


	if (s->FrameSensor->GetType() != slambench::io::CameraSensor::kCameraType and s->FrameSensor->GetType() != slambench::io::IMUSensor::kIMUType) {
		// Skip Unsupported Frames
		return false;
	}


	//=====================
	// CRY BAD DATA MANAGEMENT
	//=====================


	if (one_ok && two_ok) {
		std::cout << "ALREADY HAVE REQURIED SENSORS, SHOULD PROCESS !" << std::endl;
		exit(1);
	}

	assert(s != nullptr);



	//=====================
	// GET S1
	//=====================


	if(s->FrameSensor == grey_sensor_one) {

		if (one_ok) {
			std::cout << "BUG: SENSOR ONE SEEN TWICE" << std::endl;
			exit(1);
		}

		one_ok = true;
		t_one = okvis::Time(s->Timestamp.S, s->Timestamp.Ns);
		memcpy(img_one->data, s->GetData(), s->GetSize());



	}


	//=====================
	// GET S2
	//=====================


	if(s->FrameSensor == grey_sensor_two) {

		if (two_ok) {
			std::cout << "BUG: SENSOR TWO SEEn TWICE" << std::endl;
			exit(1);
		}

		two_ok = true;
		t_two = okvis::Time(s->Timestamp.S, s->Timestamp.Ns);
		memcpy(img_two->data, s->GetData(), s->GetSize());

	}


	//=====================
	// GET IMU
	//=====================

	if(s->FrameSensor == IMU_sensor) {

		float* frame_data = (float*)s->GetData();
		gyr_data.push_back(Eigen::Vector3d(frame_data[0],frame_data[1],frame_data[2]));
		acc_data.push_back(Eigen::Vector3d(frame_data[3],frame_data[4],frame_data[5]));
		tim_data.push_back (okvis::Time(s->Timestamp.S, s->Timestamp.Ns));

	}

	return two_ok && one_ok;

}

bool sb_process_once (SLAMBenchLibraryHelper * slam_settings) {

	// First Trail of IMU before Image One

	for (int i =0;i<tim_data.size();i++) {

		// std::cout << "addImuMeasurement:" << tim_data[i] << std::endl << acc_data[i] << std::endl  << gyr_data[i]<< std::endl;
		okvis_estimator->addImuMeasurement(tim_data[i], acc_data[i], gyr_data[i]);

	}

	// Image One
	//std::cout << "addImageOne:" << t_two << std::endl;
	okvis_estimator->addImage(t_one, 0, *img_one);

	// Image Two
	//std::cout << "addImageTwo:" << t_two << std::endl;
	okvis_estimator->addImage(t_two, 1, *img_two);



	if (one_ok && two_ok) {
		one_ok = false;
		two_ok = false;
		tim_data.clear();
		acc_data.clear();
		gyr_data.clear();
	} else {
		std::cerr << "Processed while not having right sensor data." << std::endl;
		exit(1);
	}



	return true;
}

bool       sb_get_tracked  (bool * tracked) {
	*tracked = (poseViewer.callback_position(3,3) != 1);
	return true;
}

bool sb_clean_slam_system(){
	return true;
}


bool sb_update_outputs(SLAMBenchLibraryHelper *lib, const slambench::TimeStamp *latest_output) {
	(void)lib;

	slambench::TimeStamp ts = *latest_output;

	if(pose_output->IsActive()) {

		Eigen::Matrix4f mat = poseViewer.callback_position.cast<float>();
		if (poseViewer.callback_position(3,3) != 1)  {
			mat = Eigen::Matrix4f::Identity();
		}

		std::lock_guard<FastLock> lock (lib->GetOutputManager().GetLock());
		pose_output->AddPoint(ts, new slambench::values::PoseValue(mat));
	}


	if(left_frame_output->IsActive() or right_frame_output->IsActive()) {

		std::vector<cv::Mat> out_images;
		okvis_estimator->get_cameras(out_images);


		if (out_images.size() == 0) {
			std::cout << "vizualizer failed" << std::endl;
		}


		if(left_frame_output->IsActive() and out_images.size() > 0) {
			cv::Mat& image = out_images[0];

			left_frame_output->AddPoint(*latest_output,
					new slambench::values::FrameValue(image.size().width, image.size().height,
							slambench::io::pixelformat::EPixelFormat::RGB_III_888 , image.data));
		}

		if(right_frame_output->IsActive() and out_images.size() > 1) {
			cv::Mat& image = out_images[0];

			right_frame_output->AddPoint(*latest_output,
					new slambench::values::FrameValue(image.size().width, image.size().height,
							slambench::io::pixelformat::EPixelFormat::RGB_III_888 , image.data));
		}

	}





	return true;
}



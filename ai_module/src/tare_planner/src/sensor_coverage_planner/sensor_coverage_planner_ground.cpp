/**
 * @file sensor_coverage_planner_ground.cpp
 * @author Chao Cao (ccao1@andrew.cmu.edu)
 * @brief Class that does the job of exploration
 * @version 0.1
 * @date 2020-06-03
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "sensor_coverage_planner/sensor_coverage_planner_ground.h"
#include "graph/graph.h"
#include <memory>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace std::chrono_literals;

namespace sensor_coverage_planner_3d_ns {
// PlannerParameters::PlannerParameters()
// {
// }

// bool PlannerParameters::ReadParameters(rclcpp::Node::SharedPtr node_)
void SensorCoveragePlanner3D::ReadParameters() {
  this->declare_parameter<std::string>("sub_start_exploration_topic_",
                                       "/exploration_start");
  this->declare_parameter<std::string>("sub_state_estimation_topic_",
                                       "/state_estimation_at_scan");
  this->declare_parameter<std::string>("sub_registered_scan_topic_",
                                       "/registered_scan");
  this->declare_parameter<std::string>("sub_terrain_map_topic_",
                                       "/terrain_map");
  this->declare_parameter<std::string>("sub_terrain_map_ext_topic_",
                                       "/terrain_map_ext");
  this->declare_parameter<std::string>("sub_coverage_boundary_topic_",
                                       "/coverage_boundary");
  this->declare_parameter<std::string>("sub_viewpoint_boundary_topic_",
                                       "/navigation_boundary");
  this->declare_parameter<std::string>("sub_nogo_boundary_topic_",
                                       "/nogo_boundary");
  this->declare_parameter<std::string>("sub_joystick_topic_", "/joy");
  this->declare_parameter<std::string>("sub_reset_waypoint_topic_",
                                       "/reset_waypoint");
  this->declare_parameter<std::string>("sub_target_viewpoints_topic_",
                                       "/exploration/target_viewpoints");
  this->declare_parameter<std::string>("pub_target_feedback_topic_",
                                       "/exploration/target_viewpoint_feedback");
  this->declare_parameter<std::string>("sub_target_preempt_topic_",
                                       "/exploration/target_preempt");
  this->declare_parameter<std::string>("pub_exploration_finish_topic_",
                                       "exploration_finish");
  this->declare_parameter<std::string>("pub_runtime_breakdown_topic_",
                                       "runtime_breakdown");
  this->declare_parameter<std::string>("pub_runtime_topic_", "/runtime");
  this->declare_parameter<std::string>("pub_waypoint_topic_",
                                       "/way_point_with_heading");
  this->declare_parameter<std::string>("pub_momentum_activation_count_topic_",
                                       "momentum_activation_count");

  // Bool
  this->declare_parameter<bool>("kAutoStart", false);
  this->declare_parameter<bool>("kRushHome", false);
  this->declare_parameter<bool>("kUseTerrainHeight", true);
  this->declare_parameter<bool>("kCheckTerrainCollision", true);
  this->declare_parameter<bool>("kExtendWayPoint", true);
  this->declare_parameter<bool>("kUseLineOfSightLookAheadPoint", true);
  this->declare_parameter<bool>("kNoExplorationReturnHome", true);
  this->declare_parameter<bool>("kUseMomentum", false);
  this->declare_parameter<bool>("kUseTargetViewPoints", false);
  this->declare_parameter<bool>("kUseWaypointStallWatchdog", false);

  // Double
  this->declare_parameter<double>("kKeyposeCloudDwzFilterLeafSize", 0.2);
  this->declare_parameter<double>("kRushHomeDist", 10.0);
  this->declare_parameter<double>("kAtHomeDistThreshold", 0.5);
  this->declare_parameter<double>("kTerrainCollisionThreshold", 0.5);
  this->declare_parameter<double>("kLookAheadDistance", 5.0);
  this->declare_parameter<double>("kExtendWayPointDistanceBig", 8.0);
  this->declare_parameter<double>("kExtendWayPointDistanceSmall", 3.0);
  this->declare_parameter<double>("kTargetViewPointTimeout", 5.0);
  // Matches cmu_challenge.yaml, which is the only scenario that sets it. It used to read 1.5
  // here and 0.5 there -- a 3x divergence on a TUNING value (not an on/off fallback like the
  // kUse* flags above), so anyone reading this file got the wrong snap radius. Keep the two
  // equal: the yaml is authoritative, this is what applies when a scenario omits it.
  this->declare_parameter<double>("kTargetViewPointSnapMaxDist", 0.5);
  this->declare_parameter<double>("kWaypointStallTimeout", 3.0);
  // Must stay at least one execute() period above kWaypointStallTimeout; ReadParameters
  // clamps it if a scenario yaml ever sets them the other way round. Matches
  // cmu_challenge.yaml, the only scenario that sets it.
  this->declare_parameter<double>("kWaypointStallEscalateTimeout", 5.0);
  this->declare_parameter<double>("kStallProgressDist", 0.3);
  this->declare_parameter<double>("kStallBlacklistRadius", 0.5);

  // Int
  this->declare_parameter<int>("kDirectionChangeCounterThr", 4);
  this->declare_parameter<int>("kDirectionNoChangeCounterThr", 5);
  this->declare_parameter<int>("kResetWaypointJoystickAxesID", 0);
  this->declare_parameter<int>("kMaxTargetViewPointNum", 8);
  this->declare_parameter<int>("kMaxStallBlacklistNum", 32);

  // grid_world
  this->declare_parameter<int>("kGridWorldXNum", 121);
  this->declare_parameter<int>("kGridWorldYNum", 121);
  this->declare_parameter<int>("kGridWorldZNum", 12);
  this->declare_parameter<double>("kGridWorldCellHeight", 8.0);
  this->declare_parameter<int>("kGridWorldNearbyGridNum", 5);
  this->declare_parameter<int>("kMinAddPointNumSmall", 60);
  this->declare_parameter<int>("kMinAddPointNumBig", 100);
  this->declare_parameter<int>("kMinAddFrontierPointNum", 30);
  this->declare_parameter<int>("kCellExploringToCoveredThr", 1);
  this->declare_parameter<int>("kCellCoveredToExploringThr", 10);
  this->declare_parameter<int>("kCellExploringToAlmostCoveredThr", 10);
  this->declare_parameter<int>("kCellAlmostCoveredToExploringThr", 20);
  this->declare_parameter<int>("kCellUnknownToExploringThr", 1);

  // keypose_graph
  this->declare_parameter<double>("keypose_graph/kAddNodeMinDist", 0.5);
  this->declare_parameter<double>("keypose_graph/kAddNonKeyposeNodeMinDist",
                                  0.5);
  this->declare_parameter<double>("keypose_graph/kAddEdgeConnectDistThr", 0.5);
  this->declare_parameter<double>("keypose_graph/kAddEdgeToLastKeyposeDistThr",
                                  0.5);
  this->declare_parameter<double>("keypose_graph/kAddEdgeVerticalThreshold",
                                  0.5);
  this->declare_parameter<double>(
      "keypose_graph/kAddEdgeCollisionCheckResolution", 0.5);
  this->declare_parameter<double>("keypose_graph/kAddEdgeCollisionCheckRadius",
                                  0.5);
  this->declare_parameter<int>(
      "keypose_graph/kAddEdgeCollisionCheckPointNumThr", 1);

  // local_coverage_planner
  this->declare_parameter<int>("kGreedyViewPointSampleRange", 5);
  this->declare_parameter<int>("kLocalPathOptimizationItrMax", 10);

  // planning_env
  this->declare_parameter<bool>("kUseFrontier", true);
  this->declare_parameter<double>("kSurfaceCloudDwzLeafSize", 0.2);
  this->declare_parameter<double>("kCollisionCloudDwzLeafSize", 0.2);
  this->declare_parameter<int>("kKeyposeCloudStackNum", 5);
  this->declare_parameter<int>("kPointCloudRowNum", 20);
  this->declare_parameter<int>("kPointCloudColNum", 20);
  this->declare_parameter<int>("kPointCloudLevelNum", 10);
  this->declare_parameter<int>("kMaxCellPointNum", 100000);
  this->declare_parameter<double>("kPointCloudCellSize", 24.0);
  this->declare_parameter<double>("kPointCloudCellHeight", 3.0);
  this->declare_parameter<int>("kPointCloudManagerNeighborCellNum", 5);
  this->declare_parameter<double>("kCoverCloudZSqueezeRatio", 2.0);
  this->declare_parameter<double>("kFrontierClusterTolerance", 1.0);
  this->declare_parameter<int>("kFrontierClusterMinSize", 30);
  this->declare_parameter<bool>("kUseCoverageBoundaryOnFrontier", false);
  this->declare_parameter<bool>("kUseCoverageBoundaryOnObjectSurface", false);

  // rolling_occupancy_grid
  this->declare_parameter<double>("rolling_occupancy_grid/resolution_x", 0.3);
  this->declare_parameter<double>("rolling_occupancy_grid/resolution_y", 0.3);
  this->declare_parameter<double>("rolling_occupancy_grid/resolution_z", 0.3);

  // viewpoint_manager
  this->declare_parameter<int>("viewpoint_manager/number_x", 80);
  this->declare_parameter<int>("viewpoint_manager/number_y", 80);
  this->declare_parameter<int>("viewpoint_manager/number_z", 40);
  this->declare_parameter<double>("viewpoint_manager/resolution_x", 0.5);
  this->declare_parameter<double>("viewpoint_manager/resolution_y", 0.5);
  this->declare_parameter<double>("viewpoint_manager/resolution_z", 0.5);
  this->declare_parameter<double>("kConnectivityHeightDiffThr", 0.25);
  this->declare_parameter<double>("kViewPointCollisionMargin", 0.5);
  this->declare_parameter<double>("kViewPointCollisionMarginZPlus", 0.5);
  this->declare_parameter<double>("kViewPointCollisionMarginZMinus", 0.5);
  this->declare_parameter<double>("kCollisionGridZScale", 2.0);
  this->declare_parameter<double>("kCollisionGridResolutionX", 0.5);
  this->declare_parameter<double>("kCollisionGridResolutionY", 0.5);
  this->declare_parameter<double>("kCollisionGridResolutionZ", 0.5);
  this->declare_parameter<bool>("kLineOfSightStopAtNearestObstacle", true);
  this->declare_parameter<bool>("kCheckDynamicObstacleCollision", true);
  this->declare_parameter<int>("kCollisionFrameCountMax", 3);
  this->declare_parameter<double>("kViewPointHeightFromTerrain", 0.75);
  this->declare_parameter<double>("kViewPointHeightFromTerrainChangeThreshold",
                                  0.6);
  this->declare_parameter<int>("kCollisionPointThr", 3);
  this->declare_parameter<double>("kCoverageOcclusionThr", 1.0);
  this->declare_parameter<double>("kCoverageDilationRadius", 1.0);
  this->declare_parameter<double>("kCoveragePointCloudResolution", 1.0);
  this->declare_parameter<double>("kSensorRange", 10.0);
  this->declare_parameter<double>("kNeighborRange", 3.0);

  // tare_visualizer
  this->declare_parameter<bool>("kExploringSubspaceMarkerColorGradientAlpha",
                                true);
  this->declare_parameter<double>("kExploringSubspaceMarkerColorMaxAlpha", 1.0);
  this->declare_parameter<double>("kExploringSubspaceMarkerColorR", 0.0);
  this->declare_parameter<double>("kExploringSubspaceMarkerColorG", 1.0);
  this->declare_parameter<double>("kExploringSubspaceMarkerColorB", 0.0);
  this->declare_parameter<double>("kExploringSubspaceMarkerColorA", 1.0);
  this->declare_parameter<double>("kLocalPlanningHorizonMarkerColorR", 0.0);
  this->declare_parameter<double>("kLocalPlanningHorizonMarkerColorG", 1.0);
  this->declare_parameter<double>("kLocalPlanningHorizonMarkerColorB", 0.0);
  this->declare_parameter<double>("kLocalPlanningHorizonMarkerColorA", 1.0);
  this->declare_parameter<double>("kLocalPlanningHorizonMarkerWidth", 0.3);
  this->declare_parameter<double>("kLocalPlanningHorizonHeight", 3.0);

  bool got_parameter = true;
  got_parameter &= this->get_parameter("sub_start_exploration_topic_",
                                       sub_start_exploration_topic_);
  if (!got_parameter) {
    std::cout << "Failed to get parameter sub_start_exploration_topic_"
              << std::endl;
  }
  this->get_parameter("sub_state_estimation_topic_",
                      sub_state_estimation_topic_);
  this->get_parameter("sub_registered_scan_topic_", sub_registered_scan_topic_);
  this->get_parameter("sub_terrain_map_topic_", sub_terrain_map_topic_);
  this->get_parameter("sub_terrain_map_ext_topic_", sub_terrain_map_ext_topic_);
  this->get_parameter("sub_coverage_boundary_topic_",
                      sub_coverage_boundary_topic_);
  this->get_parameter("sub_viewpoint_boundary_topic_",
                      sub_viewpoint_boundary_topic_);
  this->get_parameter("sub_nogo_boundary_topic_", sub_nogo_boundary_topic_);
  this->get_parameter("sub_joystick_topic_", sub_joystick_topic_);
  this->get_parameter("sub_reset_waypoint_topic_", sub_reset_waypoint_topic_);
  this->get_parameter("sub_target_viewpoints_topic_",
                      sub_target_viewpoints_topic_);
  this->get_parameter("pub_target_feedback_topic_", pub_target_feedback_topic_);
  this->get_parameter("sub_target_preempt_topic_", sub_target_preempt_topic_);
  this->get_parameter("pub_exploration_finish_topic_",
                      pub_exploration_finish_topic_);
  this->get_parameter("pub_runtime_breakdown_topic_",
                      pub_runtime_breakdown_topic_);
  this->get_parameter("pub_runtime_topic_", pub_runtime_topic_);
  this->get_parameter("pub_waypoint_topic_", pub_waypoint_topic_);
  this->get_parameter("pub_momentum_activation_count_topic_",
                      pub_momentum_activation_count_topic_);

  this->get_parameter("kAutoStart", kAutoStart);

  std::cout << "parameter kAutoStart: " << kAutoStart << std::endl;

  this->get_parameter("kRushHome", kRushHome);
  this->get_parameter("kUseTerrainHeight", kUseTerrainHeight);
  this->get_parameter("kCheckTerrainCollision", kCheckTerrainCollision);
  this->get_parameter("kExtendWayPoint", kExtendWayPoint);
  this->get_parameter("kUseLineOfSightLookAheadPoint",
                      kUseLineOfSightLookAheadPoint);
  this->get_parameter("kNoExplorationReturnHome", kNoExplorationReturnHome);
  this->get_parameter("kUseMomentum", kUseMomentum);
  this->get_parameter("kUseTargetViewPoints", kUseTargetViewPoints);
  this->get_parameter("kUseWaypointStallWatchdog", kUseWaypointStallWatchdog);

  this->get_parameter("kKeyposeCloudDwzFilterLeafSize",
                      kKeyposeCloudDwzFilterLeafSize);
  this->get_parameter("kRushHomeDist", kRushHomeDist);
  this->get_parameter("kAtHomeDistThreshold", kAtHomeDistThreshold);
  this->get_parameter("kTerrainCollisionThreshold", kTerrainCollisionThreshold);
  this->get_parameter("kLookAheadDistance", kLookAheadDistance);
  this->get_parameter("kExtendWayPointDistanceBig", kExtendWayPointDistanceBig);
  this->get_parameter("kExtendWayPointDistanceSmall",
                      kExtendWayPointDistanceSmall);

  this->get_parameter("kDirectionChangeCounterThr", kDirectionChangeCounterThr);
  this->get_parameter("kDirectionNoChangeCounterThr",
                      kDirectionNoChangeCounterThr);
  this->get_parameter("kResetWaypointJoystickAxesID",
                      kResetWaypointJoystickAxesID);
  this->get_parameter("kTargetViewPointTimeout", kTargetViewPointTimeout);
  this->get_parameter("kTargetViewPointSnapMaxDist", kTargetViewPointSnapMaxDist);
  this->get_parameter("kMaxTargetViewPointNum", kMaxTargetViewPointNum);
  this->get_parameter("kWaypointStallTimeout", kWaypointStallTimeout);
  this->get_parameter("kWaypointStallEscalateTimeout",
                      kWaypointStallEscalateTimeout);
  this->get_parameter("kStallProgressDist", kStallProgressDist);
  this->get_parameter("kStallBlacklistRadius", kStallBlacklistRadius);
  this->get_parameter("kMaxStallBlacklistNum", kMaxStallBlacklistNum);

  // The two stall thresholds are a ladder, and the rungs must not touch. Level 1 re-sends the
  // waypoint at kWaypointStallTimeout; level 2 retires the viewpoint at the escalate value.
  // Set them equal and both land on the same execute() tick, so the viewpoint is blacklisted
  // before the re-send has had a single tick to work -- the recovery step becomes dead code
  // and the planner throws away viewpoints it never actually retried. Nothing about that is
  // visible in a log, which is why it is clamped here rather than left to the yaml.
  const double kMinStallLadderGap = 1.0;   // one execute() period; execution_timer_ is 1000ms
  if (kWaypointStallEscalateTimeout < kWaypointStallTimeout + kMinStallLadderGap) {
    double clamped = kWaypointStallTimeout + kMinStallLadderGap;
    RCLCPP_WARN(this->get_logger(),
                "kWaypointStallEscalateTimeout (%.1f) must be at least %.1fs above "
                "kWaypointStallTimeout (%.1f) or the re-send never gets a tick -- using %.1f",
                kWaypointStallEscalateTimeout, kMinStallLadderGap, kWaypointStallTimeout,
                clamped);
    kWaypointStallEscalateTimeout = clamped;
  }
}

// PlannerData::PlannerData()
// {
// }

// void PlannerData::Initialize(rclcpp::Node::SharedPtr node_)
void SensorCoveragePlanner3D::InitializeData() {
  keypose_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<PlannerCloudPointType>>(
          shared_from_this(), "keypose_cloud", kWorldFrameID);
  registered_scan_stack_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZ>>(
          shared_from_this(), "registered_scan_stack", kWorldFrameID);
  registered_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "registered_cloud", kWorldFrameID);
  large_terrain_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "terrain_cloud_large", kWorldFrameID);
  terrain_collision_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "terrain_collision_cloud", kWorldFrameID);
  terrain_ext_collision_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "terrain_ext_collision_cloud", kWorldFrameID);
  viewpoint_vis_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "viewpoint_vis_cloud", kWorldFrameID);
  grid_world_vis_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "grid_world_vis_cloud", kWorldFrameID);
  exploration_path_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "bspline_path_cloud", kWorldFrameID);

  selected_viewpoint_vis_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "selected_viewpoint_vis_cloud", kWorldFrameID);
  exploring_cell_vis_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "exploring_cell_vis_cloud", kWorldFrameID);
  collision_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "collision_cloud", kWorldFrameID);
  lookahead_point_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "lookahead_point_cloud", kWorldFrameID);
  keypose_graph_vis_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "keypose_graph_cloud", kWorldFrameID);
  viewpoint_in_collision_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "viewpoint_in_collision_cloud_", kWorldFrameID);
  point_cloud_manager_neighbor_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "pointcloud_manager_cloud", kWorldFrameID);
  reordered_global_subspace_cloud_ =
      std::make_shared<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>(
          shared_from_this(), "reordered_global_subspace_cloud", kWorldFrameID);

  viewpoint_manager_ = std::make_shared<viewpoint_manager_ns::ViewPointManager>(
      shared_from_this());
  keypose_graph_ =
      std::make_shared<keypose_graph_ns::KeyposeGraph>(shared_from_this());
  planning_env_ =
      std::make_shared<planning_env_ns::PlanningEnv>(shared_from_this());
  grid_world_ = std::make_shared<grid_world_ns::GridWorld>(shared_from_this());
  grid_world_->SetUseKeyposeGraph(true);
  local_coverage_planner_ =
      std::make_shared<local_coverage_planner_ns::LocalCoveragePlanner>(
          shared_from_this());
  local_coverage_planner_->SetViewPointManager(viewpoint_manager_);

  visualizer_ =
      std::make_shared<tare_visualizer_ns::TAREVisualizer>(shared_from_this());

  initial_position_.x() = 0.0;
  initial_position_.y() = 0.0;
  initial_position_.z() = 0.0;

  cur_keypose_node_ind_ = 0;

  keypose_graph_node_marker_ = std::make_shared<misc_utils_ns::Marker>(
      shared_from_this(), "keypose_graph_node_marker", kWorldFrameID);
  keypose_graph_node_marker_->SetType(visualization_msgs::msg::Marker::POINTS);
  keypose_graph_node_marker_->SetScale(0.4, 0.4, 0.1);
  keypose_graph_node_marker_->SetColorRGBA(1.0, 0.0, 0.0, 1.0);
  keypose_graph_edge_marker_ = std::make_shared<misc_utils_ns::Marker>(
      shared_from_this(), "keypose_graph_edge_marker", kWorldFrameID);
  keypose_graph_edge_marker_->SetType(
      visualization_msgs::msg::Marker::LINE_LIST);
  keypose_graph_edge_marker_->SetScale(0.05, 0.0, 0.0);
  keypose_graph_edge_marker_->SetColorRGBA(1.0, 1.0, 0.0, 0.9);

  nogo_boundary_marker_ = std::make_shared<misc_utils_ns::Marker>(
      shared_from_this(), "nogo_boundary_marker", kWorldFrameID);
  nogo_boundary_marker_->SetType(visualization_msgs::msg::Marker::LINE_LIST);
  nogo_boundary_marker_->SetScale(0.05, 0.0, 0.0);
  nogo_boundary_marker_->SetColorRGBA(1.0, 0.0, 0.0, 0.8);

  grid_world_marker_ = std::make_shared<misc_utils_ns::Marker>(
      shared_from_this(), "grid_world_marker", kWorldFrameID);
  grid_world_marker_->SetType(visualization_msgs::msg::Marker::CUBE_LIST);
  grid_world_marker_->SetScale(1.0, 1.0, 1.0);
  grid_world_marker_->SetColorRGBA(1.0, 0.0, 0.0, 0.8);

  robot_yaw_ = 0.0;
  lookahead_point_direction_ = Eigen::Vector3d(1.0, 0.0, 0.0);
  moving_direction_ = Eigen::Vector3d(1.0, 0.0, 0.0);
  moving_forward_ = true;

  Eigen::Vector3d viewpoint_resolution = viewpoint_manager_->GetResolution();
  double add_non_keypose_node_min_dist =
      std::min(viewpoint_resolution.x(), viewpoint_resolution.y()) / 2;
  keypose_graph_->SetAddNonKeyposeNodeMinDist() = add_non_keypose_node_min_dist;

  robot_position_.x = 0;
  robot_position_.y = 0;
  robot_position_.z = 0;

  last_robot_position_ = robot_position_;
}

SensorCoveragePlanner3D::SensorCoveragePlanner3D()
    : Node("tare_planner_node"), keypose_cloud_update_(false),
      initialized_(false), has_registered_scan_(false),
      lookahead_point_update_(false), relocation_(false),
      start_exploration_(false), exploration_finished_(false),
      near_home_(false), at_home_(false), stopped_(false),
      test_point_update_(false), viewpoint_ind_update_(false), step_(false),
      use_momentum_(false), lookahead_point_in_line_of_sight_(true),
      reset_waypoint_(false), target_viewpoints_receive_time_(-1.0),
      stall_reference_position_(Eigen::Vector3d::Zero()),
      stall_reference_time_(-1.0), last_waypoint_publish_time_(-1.0),
      waypoint_refresh_count_(0),
      lookahead_point_valid_(false),
      target_preempt_(false), target_preempt_receive_time_(-1.0),
      preempt_target_position_(Eigen::Vector3d::Zero()),
      preempt_target_valid_(false),
      registered_cloud_count_(0), keypose_count_(0),
      direction_change_count_(0), direction_no_change_count_(0),
      momentum_activation_count_(0), reset_waypoint_joystick_axis_value_(-1.0) {
  std::cout << "finished constructor" << std::endl;
}

bool SensorCoveragePlanner3D::initialize() {
  ReadParameters();
  // if (!ReadParameters(shared_from_this()))
  // {
  //   RCLCPP_ERROR(this->get_logger(), "Read parameters failed");
  //   return false;
  // }

  // Initialize(shared_from_this());
  InitializeData();

  keypose_graph_->SetAllowVerticalEdge(false);

  lidar_model_ns::LiDARModel::setCloudDWZResol(
      planning_env_->GetPlannerCloudResolution());

  execution_timer_ = this->create_wall_timer(
      1000ms, std::bind(&SensorCoveragePlanner3D::execute, this));

  exploration_start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      sub_start_exploration_topic_, 5,
      std::bind(&SensorCoveragePlanner3D::ExplorationStartCallback, this,
                std::placeholders::_1));
  registered_scan_sub_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
          sub_registered_scan_topic_, 5,
          std::bind(&SensorCoveragePlanner3D::RegisteredScanCallback, this,
                    std::placeholders::_1));
  terrain_map_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      sub_terrain_map_topic_, 5,
      std::bind(&SensorCoveragePlanner3D::TerrainMapCallback, this,
                std::placeholders::_1));
  terrain_map_ext_sub_ =
      this->create_subscription<sensor_msgs::msg::PointCloud2>(
          sub_terrain_map_ext_topic_, 5,
          std::bind(&SensorCoveragePlanner3D::TerrainMapExtCallback, this,
                    std::placeholders::_1));
  state_estimation_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      sub_state_estimation_topic_, 5,
      std::bind(&SensorCoveragePlanner3D::StateEstimationCallback, this,
                std::placeholders::_1));
  coverage_boundary_sub_ =
      this->create_subscription<geometry_msgs::msg::PolygonStamped>(
          sub_coverage_boundary_topic_, 5,
          std::bind(&SensorCoveragePlanner3D::CoverageBoundaryCallback, this,
                    std::placeholders::_1));
  viewpoint_boundary_sub_ =
      this->create_subscription<geometry_msgs::msg::PolygonStamped>(
          sub_viewpoint_boundary_topic_, 5,
          std::bind(&SensorCoveragePlanner3D::ViewPointBoundaryCallback, this,
                    std::placeholders::_1));
  nogo_boundary_sub_ =
      this->create_subscription<geometry_msgs::msg::PolygonStamped>(
          sub_nogo_boundary_topic_, 5,
          std::bind(&SensorCoveragePlanner3D::NogoBoundaryCallback, this,
                    std::placeholders::_1));
  joystick_sub_ = this->create_subscription<sensor_msgs::msg::Joy>(
      sub_joystick_topic_, 5,
      std::bind(&SensorCoveragePlanner3D::JoystickCallback, this,
                std::placeholders::_1));
  reset_waypoint_sub_ = this->create_subscription<std_msgs::msg::Empty>(
      sub_reset_waypoint_topic_, 1,
      std::bind(&SensorCoveragePlanner3D::ResetWaypointCallback, this,
                std::placeholders::_1));
  target_viewpoints_sub_ =
      this->create_subscription<geometry_msgs::msg::PoseArray>(
          sub_target_viewpoints_topic_, 2,
          std::bind(&SensorCoveragePlanner3D::TargetViewPointsCallback, this,
                    std::placeholders::_1));
  target_preempt_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      sub_target_preempt_topic_, 2,
      std::bind(&SensorCoveragePlanner3D::TargetPreemptCallback, this,
                std::placeholders::_1));

  global_path_full_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>("global_path_full", 1);
  global_path_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>("global_path", 1);
  old_global_path_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>("old_global_path", 1);
  to_nearest_global_subspace_path_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>(
          "to_nearest_global_subspace_path", 1);
  local_tsp_path_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>("local_path", 1);
  exploration_path_publisher_ =
      this->create_publisher<nav_msgs::msg::Path>("exploration_path", 1);
  waypoint_pub_ = this->create_publisher<geometry_msgs::msg::Pose2D>(
      pub_waypoint_topic_, 2);
  target_feedback_pub_ =
      this->create_publisher<std_msgs::msg::String>(pub_target_feedback_topic_, 2);
  exploration_finish_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      pub_exploration_finish_topic_, 2);
  runtime_breakdown_pub_ =
      this->create_publisher<std_msgs::msg::Int32MultiArray>(
          pub_runtime_breakdown_topic_, 2);
  runtime_pub_ =
      this->create_publisher<std_msgs::msg::Float32>(pub_runtime_topic_, 2);
  momentum_activation_count_pub_ = this->create_publisher<std_msgs::msg::Int32>(
      pub_momentum_activation_count_topic_, 2);
  // Debug
  pointcloud_manager_neighbor_cells_origin_pub_ =
      this->create_publisher<geometry_msgs::msg::PointStamped>(
          "pointcloud_manager_neighbor_cells_origin", 1);

  PrintExplorationStatus("Exploration Started", false);
  return true;
}

void SensorCoveragePlanner3D::ExplorationStartCallback(
    const std_msgs::msg::Bool::ConstSharedPtr start_msg) {
  start_exploration_ = start_msg->data;
}

void SensorCoveragePlanner3D::StateEstimationCallback(
    const nav_msgs::msg::Odometry::ConstSharedPtr state_estimation_msg) {
  robot_position_ = state_estimation_msg->pose.pose.position;
  // Todo: use a boolean
  if (std::abs(initial_position_.x()) < 0.01 &&
      std::abs(initial_position_.y()) < 0.01 &&
      std::abs(initial_position_.z()) < 0.01) {
    initial_position_.x() = robot_position_.x;
    initial_position_.y() = robot_position_.y;
    initial_position_.z() = robot_position_.z;
  }
  double roll, pitch, yaw;
  geometry_msgs::msg::Quaternion geo_quat =
      state_estimation_msg->pose.pose.orientation;
  tf2::Matrix3x3(
      tf2::Quaternion(geo_quat.x, geo_quat.y, geo_quat.z, geo_quat.w))
      .getRPY(roll, pitch, yaw);

  robot_yaw_ = yaw;

  if (state_estimation_msg->twist.twist.linear.x > 0.4) {
    moving_forward_ = true;
  } else if (state_estimation_msg->twist.twist.linear.x < -0.4) {
    moving_forward_ = false;
  }
  // initialized_ = true;
}

void SensorCoveragePlanner3D::RegisteredScanCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr registered_scan_msg) {
  if (!initialized_) {
    return;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr registered_scan_tmp(
      new pcl::PointCloud<pcl::PointXYZ>());
  pcl::fromROSMsg(*registered_scan_msg, *registered_scan_tmp);
  if (registered_scan_tmp->points.empty()) {
    return;
  }
  has_registered_scan_ = true;
  *(registered_scan_stack_->cloud_) += *(registered_scan_tmp);
  pointcloud_downsizer_.Downsize(
      registered_scan_tmp, kKeyposeCloudDwzFilterLeafSize,
      kKeyposeCloudDwzFilterLeafSize, kKeyposeCloudDwzFilterLeafSize);
  registered_cloud_->cloud_->clear();
  pcl::copyPointCloud(*registered_scan_tmp, *(registered_cloud_->cloud_));

  planning_env_->UpdateRobotPosition(robot_position_);
  planning_env_->UpdateRegisteredCloud<pcl::PointXYZI>(
      registered_cloud_->cloud_);

  registered_cloud_count_ = (registered_cloud_count_ + 1) % 5;
  if (registered_cloud_count_ == 0) {
    // initialized_ = true;
    keypose_.pose.pose.position = robot_position_;
    keypose_.pose.covariance[0] = keypose_count_++;
    cur_keypose_node_ind_ =
        keypose_graph_->AddKeyposeNode(keypose_, *(planning_env_));

    pointcloud_downsizer_.Downsize(
        registered_scan_stack_->cloud_, kKeyposeCloudDwzFilterLeafSize,
        kKeyposeCloudDwzFilterLeafSize, kKeyposeCloudDwzFilterLeafSize);

    keypose_cloud_->cloud_->clear();
    pcl::copyPointCloud(*(registered_scan_stack_->cloud_),
                        *(keypose_cloud_->cloud_));
    // keypose_cloud_->Publish();
    registered_scan_stack_->cloud_->clear();
    keypose_cloud_update_ = true;
  }
}

void SensorCoveragePlanner3D::TerrainMapCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrain_map_msg) {
  if (kCheckTerrainCollision) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr terrain_map_tmp(
        new pcl::PointCloud<pcl::PointXYZI>());
    pcl::fromROSMsg<pcl::PointXYZI>(*terrain_map_msg, *terrain_map_tmp);
    terrain_collision_cloud_->cloud_->clear();
    for (auto &point : terrain_map_tmp->points) {
      if (point.intensity > kTerrainCollisionThreshold) {
        terrain_collision_cloud_->cloud_->points.push_back(point);
      }
    }
  }
}

void SensorCoveragePlanner3D::TerrainMapExtCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrain_map_ext_msg) {
  if (kUseTerrainHeight) {
    pcl::fromROSMsg<pcl::PointXYZI>(*terrain_map_ext_msg,
                                    *(large_terrain_cloud_->cloud_));
  }
  if (kCheckTerrainCollision) {
    pcl::fromROSMsg<pcl::PointXYZI>(*terrain_map_ext_msg,
                                    *(large_terrain_cloud_->cloud_));
    terrain_ext_collision_cloud_->cloud_->clear();
    for (auto &point : large_terrain_cloud_->cloud_->points) {
      if (point.intensity > kTerrainCollisionThreshold) {
        terrain_ext_collision_cloud_->cloud_->points.push_back(point);
      }
    }
  }
}

void SensorCoveragePlanner3D::CoverageBoundaryCallback(
    const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg) {
  planning_env_->UpdateCoverageBoundary((*polygon_msg).polygon);
}

void SensorCoveragePlanner3D::ViewPointBoundaryCallback(
    const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg) {
  viewpoint_manager_->UpdateViewPointBoundary((*polygon_msg).polygon);
}

void SensorCoveragePlanner3D::NogoBoundaryCallback(
    const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg) {
  if (polygon_msg->polygon.points.empty()) {
    return;
  }
  double polygon_id = polygon_msg->polygon.points[0].z;
  int polygon_point_size = polygon_msg->polygon.points.size();
  std::vector<geometry_msgs::msg::Polygon> nogo_boundary;
  geometry_msgs::msg::Polygon polygon;
  for (int i = 0; i < polygon_point_size; i++) {
    if (polygon_msg->polygon.points[i].z == polygon_id) {
      polygon.points.push_back(polygon_msg->polygon.points[i]);
    } else {
      nogo_boundary.push_back(polygon);
      polygon.points.clear();
      polygon_id = polygon_msg->polygon.points[i].z;
      polygon.points.push_back(polygon_msg->polygon.points[i]);
    }
  }
  nogo_boundary.push_back(polygon);
  viewpoint_manager_->UpdateNogoBoundary(nogo_boundary);

  geometry_msgs::msg::Point point;
  for (int i = 0; i < nogo_boundary.size(); i++) {
    for (int j = 0; j < nogo_boundary[i].points.size() - 1; j++) {
      point.x = nogo_boundary[i].points[j].x;
      point.y = nogo_boundary[i].points[j].y;
      point.z = nogo_boundary[i].points[j].z;
      nogo_boundary_marker_->marker_.points.push_back(point);
      point.x = nogo_boundary[i].points[j + 1].x;
      point.y = nogo_boundary[i].points[j + 1].y;
      point.z = nogo_boundary[i].points[j + 1].z;
      nogo_boundary_marker_->marker_.points.push_back(point);
    }
    point.x = nogo_boundary[i].points.back().x;
    point.y = nogo_boundary[i].points.back().y;
    point.z = nogo_boundary[i].points.back().z;
    nogo_boundary_marker_->marker_.points.push_back(point);
    point.x = nogo_boundary[i].points.front().x;
    point.y = nogo_boundary[i].points.front().y;
    point.z = nogo_boundary[i].points.front().z;
    nogo_boundary_marker_->marker_.points.push_back(point);
  }
  nogo_boundary_marker_->Publish();
}

void SensorCoveragePlanner3D::JoystickCallback(
    const sensor_msgs::msg::Joy::ConstSharedPtr joy_msg) {
  if (kResetWaypointJoystickAxesID >= 0 &&
      kResetWaypointJoystickAxesID < joy_msg->axes.size()) {
    if (reset_waypoint_joystick_axis_value_ > -0.1 &&
        joy_msg->axes[kResetWaypointJoystickAxesID] < -0.1) {
      reset_waypoint_ = true;

      // Set waypoint to the current robot position to stop the robot in place
      SendWaypoint(robot_position_.x, robot_position_.y);
      std::cout << "reset waypoint" << std::endl;
    }
    reset_waypoint_joystick_axis_value_ =
        joy_msg->axes[kResetWaypointJoystickAxesID];
  }
}

void SensorCoveragePlanner3D::ResetWaypointCallback(
    const std_msgs::msg::Empty::ConstSharedPtr empty_msg) {
  reset_waypoint_ = true;

  // Set waypoint to the current robot position to stop the robot in place
  SendWaypoint(robot_position_.x, robot_position_.y);
  std::cout << "reset waypoint" << std::endl;
}

namespace {
/** Append "[x, y]" to a comma-separated JSON array body, tracking how many are in it.
 *
 * std::to_string, not a stream: an ostringstream defaults to six SIGNIFICANT digits, which at
 * room-scale coordinates is fine but degrades with magnitude. The consumer matches a verdict
 * back to the pose that asked for it by nearest-point within a centimetre, so the round trip
 * has to stay well inside that. std::to_string gives six DECIMAL places regardless of scale.
 */
void AppendPoint(std::string &out, int &count, double x, double y) {
  if (count > 0) {
    out += ", ";
  }
  out += "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
  count++;
}
}  // namespace

void SensorCoveragePlanner3D::TargetViewPointsCallback(
    const geometry_msgs::msg::PoseArray::ConstSharedPtr pose_array_msg) {
  // Positions only. The challenge camera is a 360-degree equirectangular panorama, so which
  // way the robot faces does not change what it sees -- orientation carries no information
  // here and is deliberately ignored rather than turned into a heading.
  target_viewpoint_positions_.clear();
  for (const auto &pose : pose_array_msg->poses) {
    target_viewpoint_positions_.push_back(
        Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z));
  }
  target_viewpoints_receive_time_ = this->now().seconds();
}

void SensorCoveragePlanner3D::TargetPreemptCallback(
    const std_msgs::msg::Bool::ConstSharedPtr preempt_msg) {
  target_preempt_ = preempt_msg->data;
  target_preempt_receive_time_ = this->now().seconds();
}

bool SensorCoveragePlanner3D::TargetPreemptActive() const {
  // Shares kTargetViewPointTimeout with the viewpoints themselves: a target_explorer that
  // died must not leave TARE permanently locked out of frontier exploration.
  return kUseTargetViewPoints && target_preempt_ &&
         target_preempt_receive_time_ >= 0.0 &&
         (this->now().seconds() - target_preempt_receive_time_) <=
             kTargetViewPointTimeout;
}

bool SensorCoveragePlanner3D::TargetWorkOutstanding() const {
  // Same freshness rule as the viewpoints and the preempt flag: a target_explorer that died
  // must not pin TARE in exploring forever.
  return kUseTargetViewPoints && !target_viewpoint_positions_.empty() &&
         target_viewpoints_receive_time_ >= 0.0 &&
         (this->now().seconds() - target_viewpoints_receive_time_) <=
             kTargetViewPointTimeout;
}

int SensorCoveragePlanner3D::PreferredTargetViewPointInd() const {
  // Deliberately NOT the nearest one. Nearest is recomputed every cycle from a moving robot,
  // so with two roughly equidistant objects it oscillates: drive at A, B becomes nearer,
  // switch to B, A becomes nearer, switch back -- observed flipping targets within 1 s and
  // covering neither. Distance is also the wrong authority: target_explorer already decided
  // which object to commit to and sends its poses first, and that commitment (its dwell, its
  // stall rotation) is invisible here.
  //
  // 1) Stay on the target we are already driving at for as long as it is still accepted, so
  //    re-ordering *within* a goal cannot make us switch mid-approach.
  if (preempt_target_valid_) {
    for (const auto &viewpoint_ind : accepted_target_viewpoint_indices_) {
      geometry_msgs::msg::Point position =
          viewpoint_manager_->GetViewPointPosition(viewpoint_ind);
      double dx = position.x - preempt_target_position_.x();
      double dy = position.y - preempt_target_position_.y();
      if (dx * dx + dy * dy <=
          kTargetViewPointSnapMaxDist * kTargetViewPointSnapMaxDist) {
        return viewpoint_ind;
      }
    }
  }
  // 2) Otherwise take the highest-priority reachable one. The list is built by walking
  //    target_viewpoint_positions_ in the order target_explorer sent them -- active goal
  //    first -- so front() is its choice, not ours.
  return accepted_target_viewpoint_indices_.empty()
             ? -1
             : accepted_target_viewpoint_indices_.front();
}

void SensorCoveragePlanner3D::UpdateTargetViewPoints() {
  // Two ways to end up with nothing to do, and both must leave TARE bit-for-bit stock:
  // the feature switched off, and a target_explorer that has died or gone quiet. Clearing
  // unconditionally first is what guarantees the second one.
  std::vector<int> target_viewpoint_indices;
  grid_world_->ClearPriorityCells();
  grid_world_->SetTargetPreempt(TargetPreemptActive());
  accepted_target_viewpoint_indices_.clear();
  if (!kUseTargetViewPoints || target_viewpoints_receive_time_ < 0.0 ||
      this->now().seconds() - target_viewpoints_receive_time_ >
          kTargetViewPointTimeout) {
    local_coverage_planner_->SetTargetViewPointIndices(target_viewpoint_indices);
    return;
  }

  // Verdict per request, echoed back so the semantic side can tell a direction it has merely
  // not walked to yet from one where nothing can physically stand. Only the latter is a
  // statement about the world; "far" says nothing and is deliberately not reported.
  std::string accepted_json;
  std::string unreachable_json;
  int accepted_count = 0;
  int unreachable_count = 0;
  int far_count = 0;
  int unstandable_count = 0;

  for (const auto &position : target_viewpoint_positions_) {
    if (static_cast<int>(target_viewpoint_indices.size()) >=
        kMaxTargetViewPointNum) {
      break;
    }
    if (!viewpoint_manager_->InLocalPlanningHorizon(position)) {
      // InLocalPlanningHorizon conflates two verdicts that deserve opposite answers, and
      // reporting both as `far` is what left wall-mounted objects outstanding for whole runs.
      //
      //   outside the lattice   -- a distance statement. Driving closer fixes it, so say
      //                            nothing and pin the subspace EXPLORING (below).
      //   inside, not a candidate -- the robot is already within 4.5 m and STILL cannot stand
      //                            there: no line of sight, not graph-connected, or solid.
      //                            Two of a wall-mounted object's four sectors are behind the
      //                            wall and land here every single cycle. Time cannot fix it,
      //                            and reporting `far` meant target_explorer never heard, so
      //                            the sector stayed OPEN, the goal never closed, coverage
      //                            never completed, and `preempt` (which needs every pending
      //                            goal in-horizon) stayed off for the whole run.
      //
      // Measured over the 13-scene sweep, acceptance rate tracks exactly this: japanese_room's
      // free-standing vases 55%, livingroom_4's wall pictures 3%, hotel_room_1's window 1%.
      //
      // `unreachable` is the honest verdict for the second, and it is not a death sentence:
      // target_explorer answers it by retrying the sector from further out, and only writes
      // the sector off once those retries are spent. A transient (the lattice still filling in
      // at startup) therefore costs one wider retry, not a lost sector.
      if (viewpoint_manager_->InRange(viewpoint_manager_->GetViewPointInd(position))) {
        unstandable_count++;
        AppendPoint(unreachable_json, unreachable_count, position.x(), position.y());
        continue;
      }
      far_count++;
      // Too far to stand on this cycle. Pin the subspace EXPLORING instead, so the global
      // TSP routes there over its own keypose graph and the local layer can pick it up on
      // arrival -- rather than the cell going COVERED on a drive-past and never coming back.
      grid_world_->AddPriorityCell(
          grid_world_->GetCellInd(position.x(), position.y(), position.z()));
      continue;
    }
    int viewpoint_ind =
        viewpoint_manager_->GetNearestCandidateViewPointInd(position);
    if (!viewpoint_manager_->InRange(viewpoint_ind)) {
      AppendPoint(unreachable_json, unreachable_count, position.x(), position.y());
      continue;
    }
    // A candidate is already collision-free, in line of sight and graph-connected, so the
    // only question left is whether one lies where we were actually asked to stand. If the
    // nearest is far away, the request was unreachable (a viewing direction that points into
    // the wall behind the object) and honouring it would send the robot somewhere useless.
    geometry_msgs::msg::Point viewpoint_position =
        viewpoint_manager_->GetViewPointPosition(viewpoint_ind);
    double dx = viewpoint_position.x - position.x();
    double dy = viewpoint_position.y - position.y();
    if (dx * dx + dy * dy >
        kTargetViewPointSnapMaxDist * kTargetViewPointSnapMaxDist) {
      AppendPoint(unreachable_json, unreachable_count, position.x(), position.y());
      continue;
    }
    // The watchdog retires viewpoints the robot demonstrably could not reach. Target
    // injection clears `visited` every cycle, so without this check the two would fight:
    // the watchdog would blacklist a viewpoint and this would hand it straight back.
    if (IsStallBlacklisted(Eigen::Vector3d(viewpoint_position.x,
                                           viewpoint_position.y,
                                           viewpoint_position.z))) {
      AppendPoint(unreachable_json, unreachable_count, position.x(), position.y());
      continue;
    }
    // Without this an object the robot drove past earlier can never be re-inspected:
    // EnqueueViewpointCandidates skips every visited viewpoint, and visited is permanent.
    viewpoint_manager_->SetViewPointVisited(viewpoint_ind, false);
    target_viewpoint_indices.push_back(viewpoint_ind);
    AppendPoint(accepted_json, accepted_count, position.x(), position.y());
  }
  local_coverage_planner_->SetTargetViewPointIndices(target_viewpoint_indices);
  accepted_target_viewpoint_indices_ = target_viewpoint_indices;

  std_msgs::msg::String feedback;
  feedback.data = "{\"stamp\": " + std::to_string(this->now().seconds()) +
                  ", \"accepted\": [" + accepted_json +
                  "], \"unreachable\": [" + unreachable_json + "]}";
  target_feedback_pub_->publish(feedback);

  RCLCPP_INFO_THROTTLE(
      this->get_logger(), *this->get_clock(), 5000,
      "target viewpoints: %d requested -> %d accepted, %d unreachable "
      "(%d of them in-reach but unstandable), %d far",
      static_cast<int>(target_viewpoint_positions_.size()), accepted_count,
      unreachable_count, unstandable_count, far_count);
}

bool SensorCoveragePlanner3D::IsStallBlacklisted(
    const Eigen::Vector3d &position) const {
  for (const auto &blacklisted : stall_blacklist_) {
    double dx = position.x() - blacklisted.x();
    double dy = position.y() - blacklisted.y();
    if (dx * dx + dy * dy < kStallBlacklistRadius * kStallBlacklistRadius) {
      return true;
    }
  }
  return false;
}

void SensorCoveragePlanner3D::CheckWaypointStall() {
  // `stopped_` normally means the mission is over and a parked robot is correct, so watching
  // it would be pure noise. While target work is outstanding it means the opposite -- the
  // robot is parked with places left to visit, which is the single failure this watchdog
  // exists to catch -- so the mute does not apply.
  if (!kUseWaypointStallWatchdog || (stopped_ && !TargetWorkOutstanding())) {
    return;
  }
  double now_s = this->now().seconds();

  // ---- Freshness: is the TOPIC stale? Deliberately independent of robot motion. ----
  //
  // PublishWaypoint() sits at the very end of the `if (keypose_cloud_update_)` block, so a
  // tick that never enters it (the flag is set on every 5th /registered_scan while execute()
  // ticks at 1 Hz) or one that returns early at "Cannot get candidate viewpoints" emits
  // nothing at all. waypoint_converter ships yawConfig -1 ("reach waypoint and stop"), so the
  // robot drives to whatever it last received and parks there.
  //
  // The check below used to be the ONLY one, and it measures robot motion -- so a robot still
  // coasting toward a stale goal reset the alarm on every tick and the waypoint aged without
  // limit. That is the bug: a motion watchdog cannot see a stale topic, because the two are
  // only correlated after the robot has already arrived and stopped.
  //
  // Re-sending is not a no-op even when lookahead_point_ has not changed: PublishWaypoint()
  // re-extends it by kExtendWayPointDistance* from the robot's CURRENT position, so the carrot
  // moves forward as the robot advances.
  if (lookahead_point_valid_ &&
      now_s - last_waypoint_publish_time_ > kWaypointStallTimeout) {
    double stale_s = now_s - last_waypoint_publish_time_;
    keypose_cloud_update_ = true;   // give this same tick a chance at a fresh plan
    PublishWaypoint();              // but do not depend on that round succeeding
    // Say so. This half used to re-send in complete silence, which made a sweep's logs unable
    // to answer "did the waypoint topic ever go stale, and how often" -- the exact question a
    // stale-waypoint watchdog exists to answer. Throttled, because a genuinely wedged planning
    // round trips it on every tick.
    waypoint_refresh_count_++;
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                         "/way_point stale %.1fs -- re-sent (%d so far this run)",
                         stale_s, waypoint_refresh_count_);
  }

  // ---- Progress: is the ROBOT stuck? Only this half may retire a viewpoint. ----
  Eigen::Vector3d robot(robot_position_.x, robot_position_.y, robot_position_.z);
  if (stall_reference_time_ < 0.0) {
    stall_reference_position_ = robot;
    stall_reference_time_ = now_s;
    return;
  }
  // Any real movement clears the alarm. kStallProgressDist has to sit above odometry
  // jitter, or a stationary robot looks like it is creeping and never trips the watchdog.
  double moved_x = robot.x() - stall_reference_position_.x();
  double moved_y = robot.y() - stall_reference_position_.y();
  if (moved_x * moved_x + moved_y * moved_y >
      kStallProgressDist * kStallProgressDist) {
    stall_reference_position_ = robot;
    stall_reference_time_ = now_s;
    return;
  }
  double stalled_s = now_s - stall_reference_time_;
  if (stalled_s < kWaypointStallTimeout) {
    return;
  }

  // Level 1 -- the robot is not moving. Force a planning round: the waypoint it is sitting
  // on may be fine and the PLAN stale. Re-sending is handled by the freshness check above,
  // which fires on the same kWaypointStallTimeout and covers this case too.
  keypose_cloud_update_ = true;

  if (stalled_s < kWaypointStallEscalateTimeout) {
    return;
  }
  if (!lookahead_point_valid_) {
    // No round has ever produced a lookahead, so lookahead_point_ is uninitialised Eigen
    // memory and blacklisting it would poison a random spot. Forcing the round is the whole
    // of what is useful here.
    return;
  }

  // Level 2 -- still nowhere after twice the timeout, so the waypoint is not merely late,
  // it is unreachable. Retire the viewpoint behind it so the planner has to pick something
  // else, and remember the place: the lattice rolls with the robot, so a viewpoint index
  // means nothing a few metres later, but the obstacle that blocked it does not move.
  int viewpoint_ind =
      viewpoint_manager_->GetNearestCandidateViewPointInd(lookahead_point_);
  if (viewpoint_manager_->InRange(viewpoint_ind)) {
    viewpoint_manager_->SetViewPointVisited(viewpoint_ind, true);
  }
  if (!IsStallBlacklisted(lookahead_point_) &&
      static_cast<int>(stall_blacklist_.size()) < kMaxStallBlacklistNum) {
    stall_blacklist_.push_back(lookahead_point_);
    RCLCPP_WARN(this->get_logger(),
                "waypoint stalled %.1fs at (%.2f, %.2f) -- abandoning it",
                stalled_s, lookahead_point_.x(), lookahead_point_.y());
  }
  // Retire the TARGET too, not just the waypoint in front of it. The waypoint is the extended
  // lookahead a few metres from the robot; the thing pinning us is the target itself, and
  // blacklisting only the former leaves the target "accepted" so it is re-adopted next cycle.
  // That is what cost ~60 s of one run on a single unreachable viewpoint. Once blacklisted,
  // UpdateTargetViewPoints reports it `unreachable`, target_explorer retries that sector from
  // further out, and the goal moves on by itself.
  if (preempt_target_valid_ && !IsStallBlacklisted(preempt_target_position_) &&
      static_cast<int>(stall_blacklist_.size()) < kMaxStallBlacklistNum) {
    stall_blacklist_.push_back(preempt_target_position_);
    RCLCPP_WARN(this->get_logger(),
                "target viewpoint (%.2f, %.2f) unreachable after %.1fs -- retiring it",
                preempt_target_position_.x(), preempt_target_position_.y(), stalled_s);
    preempt_target_valid_ = false;
  }
  // Re-arm rather than latch, so a second blocking viewpoint gets retired too.
  stall_reference_time_ = now_s;
}

void SensorCoveragePlanner3D::SendInitialWaypoint() {
  // send waypoint ahead
  double lx = 12.0;
  double ly = 0.0;
  double dx = cos(robot_yaw_) * lx - sin(robot_yaw_) * ly;
  double dy = sin(robot_yaw_) * lx + cos(robot_yaw_) * ly;

  SendWaypoint(robot_position_.x + dx, robot_position_.y + dy);
}

void SensorCoveragePlanner3D::UpdateKeyposeGraph() {
  misc_utils_ns::Timer update_keypose_graph_timer("update keypose graph");
  update_keypose_graph_timer.Start();

  keypose_graph_->GetMarker(keypose_graph_node_marker_->marker_,
                            keypose_graph_edge_marker_->marker_);
  // keypose_graph_node_marker_->Publish();
  keypose_graph_edge_marker_->Publish();
  keypose_graph_vis_cloud_->cloud_->clear();
  keypose_graph_->CheckLocalCollision(robot_position_, viewpoint_manager_);
  keypose_graph_->CheckConnectivity(robot_position_);
  keypose_graph_->GetVisualizationCloud(keypose_graph_vis_cloud_->cloud_);
  keypose_graph_vis_cloud_->Publish();

  update_keypose_graph_timer.Stop(false);
}

int SensorCoveragePlanner3D::UpdateViewPoints() {
  misc_utils_ns::Timer collision_cloud_timer("update collision cloud");
  collision_cloud_timer.Start();
  collision_cloud_->cloud_ = planning_env_->GetCollisionCloud();
  collision_cloud_timer.Stop(false);

  misc_utils_ns::Timer viewpoint_manager_update_timer(
      "update viewpoint manager");
  viewpoint_manager_update_timer.Start();
  if (kUseTerrainHeight) {
    viewpoint_manager_->SetViewPointHeightWithTerrain(
        large_terrain_cloud_->cloud_);
  }
  if (kCheckTerrainCollision) {
    *(collision_cloud_->cloud_) += *(terrain_collision_cloud_->cloud_);
    *(collision_cloud_->cloud_) += *(terrain_ext_collision_cloud_->cloud_);
  }
  viewpoint_manager_->CheckViewPointCollision(collision_cloud_->cloud_);
  viewpoint_manager_->CheckViewPointLineOfSight();
  viewpoint_manager_->CheckViewPointConnectivity();
  int viewpoint_candidate_count = viewpoint_manager_->GetViewPointCandidate();

  UpdateVisitedPositions();
  viewpoint_manager_->UpdateViewPointVisited(visited_positions_);
  viewpoint_manager_->UpdateViewPointVisited(grid_world_);

  // For visualization
  collision_cloud_->Publish();
  // collision_grid_cloud_->Publish();
  viewpoint_manager_->GetCollisionViewPointVisCloud(
      viewpoint_in_collision_cloud_->cloud_);
  viewpoint_in_collision_cloud_->Publish();

  viewpoint_manager_update_timer.Stop(false);
  return viewpoint_candidate_count;
}

void SensorCoveragePlanner3D::UpdateViewPointCoverage() {
  // Update viewpoint coverage
  misc_utils_ns::Timer update_coverage_timer("update viewpoint coverage");
  update_coverage_timer.Start();
  viewpoint_manager_->UpdateViewPointCoverage<PlannerCloudPointType>(
      planning_env_->GetDiffCloud());
  viewpoint_manager_->UpdateRolledOverViewPointCoverage<PlannerCloudPointType>(
      planning_env_->GetStackedCloud());
  // Update robot coverage
  robot_viewpoint_.ResetCoverage();
  geometry_msgs::msg::Pose robot_pose;
  robot_pose.position = robot_position_;
  robot_viewpoint_.setPose(robot_pose);
  UpdateRobotViewPointCoverage();
  update_coverage_timer.Stop(false);
}

void SensorCoveragePlanner3D::UpdateRobotViewPointCoverage() {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud =
      planning_env_->GetCollisionCloud();
  for (const auto &point : cloud->points) {
    if (viewpoint_manager_->InFOVAndRange(
            Eigen::Vector3d(point.x, point.y, point.z),
            Eigen::Vector3d(robot_position_.x, robot_position_.y,
                            robot_position_.z))) {
      robot_viewpoint_.UpdateCoverage<pcl::PointXYZI>(point);
    }
  }
}

void SensorCoveragePlanner3D::UpdateCoveredAreas(
    int &uncovered_point_num, int &uncovered_frontier_point_num) {
  // Update covered area
  misc_utils_ns::Timer update_coverage_area_timer("update covered area");
  update_coverage_area_timer.Start();
  planning_env_->UpdateCoveredArea(robot_viewpoint_, viewpoint_manager_);

  update_coverage_area_timer.Stop(false);
  misc_utils_ns::Timer get_uncovered_area_timer("get uncovered area");
  get_uncovered_area_timer.Start();
  planning_env_->GetUncoveredArea(viewpoint_manager_, uncovered_point_num,
                                  uncovered_frontier_point_num);

  get_uncovered_area_timer.Stop(false);
  planning_env_->PublishUncoveredCloud();
  planning_env_->PublishUncoveredFrontierCloud();
}

void SensorCoveragePlanner3D::UpdateVisitedPositions() {
  Eigen::Vector3d robot_current_position(robot_position_.x, robot_position_.y,
                                         robot_position_.z);
  bool existing = false;
  for (int i = 0; i < visited_positions_.size(); i++) {
    // TODO: parameterize this
    if ((robot_current_position - visited_positions_[i]).norm() < 1) {
      existing = true;
      break;
    }
  }
  if (!existing) {
    visited_positions_.push_back(robot_current_position);
  }
}

void SensorCoveragePlanner3D::UpdateGlobalRepresentation() {
  local_coverage_planner_->SetRobotPosition(
      Eigen::Vector3d(robot_position_.x, robot_position_.y, robot_position_.z));
  bool viewpoint_rollover = viewpoint_manager_->UpdateRobotPosition(
      Eigen::Vector3d(robot_position_.x, robot_position_.y, robot_position_.z));
  if (!grid_world_->Initialized() || viewpoint_rollover) {
    grid_world_->UpdateNeighborCells(robot_position_);
  }

  planning_env_->UpdateRobotPosition(robot_position_);
  planning_env_->GetVisualizationPointCloud(
      point_cloud_manager_neighbor_cloud_->cloud_);
  point_cloud_manager_neighbor_cloud_->Publish();

  // DEBUG
  Eigen::Vector3d pointcloud_manager_neighbor_cells_origin =
      planning_env_->GetPointCloudManagerNeighborCellsOrigin();
  geometry_msgs::msg::PointStamped
      pointcloud_manager_neighbor_cells_origin_point;
  pointcloud_manager_neighbor_cells_origin_point.header.frame_id = "map";
  pointcloud_manager_neighbor_cells_origin_point.header.stamp = this->now();
  pointcloud_manager_neighbor_cells_origin_point.point.x =
      pointcloud_manager_neighbor_cells_origin.x();
  pointcloud_manager_neighbor_cells_origin_point.point.y =
      pointcloud_manager_neighbor_cells_origin.y();
  pointcloud_manager_neighbor_cells_origin_point.point.z =
      pointcloud_manager_neighbor_cells_origin.z();
  pointcloud_manager_neighbor_cells_origin_pub_->publish(
      pointcloud_manager_neighbor_cells_origin_point);

  if (exploration_finished_ && kNoExplorationReturnHome) {
    planning_env_->SetUseFrontier(false);
  }
  planning_env_->UpdateKeyposeCloud<PlannerCloudPointType>(
      keypose_cloud_->cloud_);

  int closest_node_ind = keypose_graph_->GetClosestNodeInd(robot_position_);
  geometry_msgs::msg::Point closest_node_position =
      keypose_graph_->GetClosestNodePosition(robot_position_);
  grid_world_->SetCurKeyposeGraphNodeInd(closest_node_ind);
  grid_world_->SetCurKeyposeGraphNodePosition(closest_node_position);

  grid_world_->UpdateRobotPosition(robot_position_);
  if (!grid_world_->HomeSet()) {
    grid_world_->SetHomePosition(initial_position_);
  }
}

void SensorCoveragePlanner3D::GlobalPlanning(
    std::vector<int> &global_cell_tsp_order,
    exploration_path_ns::ExplorationPath &global_path) {
  misc_utils_ns::Timer global_tsp_timer("Global planning");
  global_tsp_timer.Start();

  grid_world_->UpdateCellStatus(viewpoint_manager_);
  grid_world_->UpdateCellKeyposeGraphNodes(keypose_graph_);
  grid_world_->AddPathsInBetweenCells(viewpoint_manager_, keypose_graph_);

  viewpoint_manager_->UpdateCandidateViewPointCellStatus(grid_world_);

  global_path = grid_world_->SolveGlobalTSP(
      viewpoint_manager_, global_cell_tsp_order, keypose_graph_);

  global_tsp_timer.Stop(false);
  global_planning_runtime_ = global_tsp_timer.GetDuration("ms");
}

void SensorCoveragePlanner3D::PublishGlobalPlanningVisualization(
    const exploration_path_ns::ExplorationPath &global_path,
    const exploration_path_ns::ExplorationPath &local_path) {
  nav_msgs::msg::Path global_path_full = global_path.GetPath();
  global_path_full.header.frame_id = "map";
  global_path_full.header.stamp = this->now();
  global_path_full_publisher_->publish(global_path_full);
  // Get the part that connects with the local path

  int start_index = 0;
  for (int i = 0; i < global_path.nodes_.size(); i++) {
    if (global_path.nodes_[i].type_ ==
            exploration_path_ns::NodeType::GLOBAL_VIEWPOINT ||
        global_path.nodes_[i].type_ == exploration_path_ns::NodeType::HOME ||
        !viewpoint_manager_->InLocalPlanningHorizon(
            global_path.nodes_[i].position_)) {
      break;
    }
    start_index = i;
  }

  int end_index = global_path.nodes_.size() - 1;
  for (int i = global_path.nodes_.size() - 1; i >= 0; i--) {
    if (global_path.nodes_[i].type_ ==
            exploration_path_ns::NodeType::GLOBAL_VIEWPOINT ||
        global_path.nodes_[i].type_ == exploration_path_ns::NodeType::HOME ||
        !viewpoint_manager_->InLocalPlanningHorizon(
            global_path.nodes_[i].position_)) {
      break;
    }
    end_index = i;
  }

  nav_msgs::msg::Path global_path_trim;
  if (local_path.nodes_.size() >= 2) {
    geometry_msgs::msg::PoseStamped first_pose;
    first_pose.pose.position.x = local_path.nodes_.front().position_.x();
    first_pose.pose.position.y = local_path.nodes_.front().position_.y();
    first_pose.pose.position.z = local_path.nodes_.front().position_.z();
    global_path_trim.poses.push_back(first_pose);
  }

  for (int i = start_index; i <= end_index; i++) {
    geometry_msgs::msg::PoseStamped pose;
    pose.pose.position.x = global_path.nodes_[i].position_.x();
    pose.pose.position.y = global_path.nodes_[i].position_.y();
    pose.pose.position.z = global_path.nodes_[i].position_.z();
    global_path_trim.poses.push_back(pose);
  }
  if (local_path.nodes_.size() >= 2) {
    geometry_msgs::msg::PoseStamped last_pose;
    last_pose.pose.position.x = local_path.nodes_.back().position_.x();
    last_pose.pose.position.y = local_path.nodes_.back().position_.y();
    last_pose.pose.position.z = local_path.nodes_.back().position_.z();
    global_path_trim.poses.push_back(last_pose);
  }
  global_path_trim.header.frame_id = "map";
  global_path_trim.header.stamp = this->now();
  global_path_publisher_->publish(global_path_trim);

  grid_world_->GetVisualizationCloud(grid_world_vis_cloud_->cloud_);
  grid_world_vis_cloud_->Publish();
  grid_world_->GetMarker(grid_world_marker_->marker_);
  grid_world_marker_->Publish();
  nav_msgs::msg::Path full_path = exploration_path_.GetPath();
  full_path.header.frame_id = "map";
  full_path.header.stamp = this->now();
  // exploration_path_publisher_->publish(full_path);
  exploration_path_.GetVisualizationCloud(exploration_path_cloud_->cloud_);
  exploration_path_cloud_->Publish();
  // planning_env_->PublishStackedCloud();
}

void SensorCoveragePlanner3D::LocalPlanning(
    int uncovered_point_num, int uncovered_frontier_point_num,
    const exploration_path_ns::ExplorationPath &global_path,
    exploration_path_ns::ExplorationPath &local_path) {
  misc_utils_ns::Timer local_tsp_timer("Local planning");
  local_tsp_timer.Start();
  // While targets preempt, aim the lookahead at the nearest snapped target BEFORE the local
  // planner reads it. GetNavigationViewPointIndices then resolves lookahead_viewpoint_ind_ to
  // that target, and SolveTSP's existing robot/lookahead dummy -- cost 0 to the robot and the
  // lookahead, 9999 to everything else -- forces the target to sit immediately after the robot
  // once the dummies are stripped. Being a must-visit node only guaranteed the target was
  // somewhere in the tour; this is what puts it first.
  int preempt_target_ind =
      TargetPreemptActive() ? PreferredTargetViewPointInd() : -1;
  if (preempt_target_ind >= 0) {
    geometry_msgs::msg::Point target_position =
        viewpoint_manager_->GetViewPointPosition(preempt_target_ind);
    Eigen::Vector3d target(target_position.x, target_position.y,
                           target_position.z);
    // Compared against the previously ADOPTED target, not against lookahead_point_:
    // GetLookAheadPoint overwrites the lookahead every cycle, so testing that made this look
    // like a new adoption on every single tick and logged the same target eleven times.
    if (!preempt_target_valid_ ||
        (target - preempt_target_position_).norm() > kTargetViewPointSnapMaxDist) {
      // Adopting a different target: re-aim the direction hysteresis AT it. GetLookAheadPoint
      // scores candidates by dot(lookahead_point_direction_, unit vector to candidate), and
      // the momentum latch reuses that score, so leaving the old heading in place makes both
      // actively resist turning toward the new target. Resetting to the robot's current
      // heading (what /reset_waypoint does) is not enough here -- it is neutral, not helpful.
      Eigen::Vector3d to_target(target.x() - robot_position_.x,
                                target.y() - robot_position_.y, 0.0);
      if (to_target.norm() > 1e-3) {
        lookahead_point_direction_ = to_target.normalized();
      }
      RCLCPP_INFO(this->get_logger(),
                  "target preempt: driving to target viewpoint (%.2f, %.2f)",
                  target.x(), target.y());
    }
    preempt_target_position_ = target;
    preempt_target_valid_ = true;
    lookahead_point_ = target;
    lookahead_point_update_ = true;
    lookahead_point_valid_ = true;
  } else {
    preempt_target_valid_ = false;
  }
  if (lookahead_point_update_) {
    local_coverage_planner_->SetLookAheadPoint(lookahead_point_);
  }
  local_path = local_coverage_planner_->SolveLocalCoverageProblem(
      global_path, uncovered_point_num, uncovered_frontier_point_num);
  local_tsp_timer.Stop(false);
}

void SensorCoveragePlanner3D::PublishLocalPlanningVisualization(
    const exploration_path_ns::ExplorationPath &local_path) {
  viewpoint_manager_->GetVisualizationCloud(viewpoint_vis_cloud_->cloud_);
  viewpoint_vis_cloud_->Publish();
  lookahead_point_cloud_->Publish();
  nav_msgs::msg::Path local_tsp_path = local_path.GetPath();
  local_tsp_path.header.frame_id = "map";
  local_tsp_path.header.stamp = this->now();
  local_tsp_path_publisher_->publish(local_tsp_path);
  local_coverage_planner_->GetSelectedViewPointVisCloud(
      selected_viewpoint_vis_cloud_->cloud_);
  selected_viewpoint_vis_cloud_->Publish();

  // Visualize local planning horizon box
}

exploration_path_ns::ExplorationPath
SensorCoveragePlanner3D::ConcatenateGlobalLocalPath(
    const exploration_path_ns::ExplorationPath &global_path,
    const exploration_path_ns::ExplorationPath &local_path) {
  exploration_path_ns::ExplorationPath full_path;
  if (exploration_finished_ && near_home_ && kRushHome) {
    exploration_path_ns::Node node;
    node.position_.x() = robot_position_.x;
    node.position_.y() = robot_position_.y;
    node.position_.z() = robot_position_.z;
    node.type_ = exploration_path_ns::NodeType::ROBOT;
    full_path.nodes_.push_back(node);
    node.position_ = initial_position_;
    node.type_ = exploration_path_ns::NodeType::HOME;
    full_path.nodes_.push_back(node);
    return full_path;
  }

  double global_path_length = global_path.GetLength();
  double local_path_length = local_path.GetLength();
  if (global_path_length < 3 && local_path_length < 5) {
    return full_path;
  } else {
    full_path = local_path;
    if (local_path.nodes_.front().type_ ==
            exploration_path_ns::NodeType::LOCAL_PATH_END &&
        local_path.nodes_.back().type_ ==
            exploration_path_ns::NodeType::LOCAL_PATH_START) {
      full_path.Reverse();
    } else if (local_path.nodes_.front().type_ ==
                   exploration_path_ns::NodeType::LOCAL_PATH_START &&
               local_path.nodes_.back() == local_path.nodes_.front()) {
      full_path.nodes_.back().type_ =
          exploration_path_ns::NodeType::LOCAL_PATH_END;
    } else if (local_path.nodes_.front().type_ ==
                   exploration_path_ns::NodeType::LOCAL_PATH_END &&
               local_path.nodes_.back() == local_path.nodes_.front()) {
      full_path.nodes_.front().type_ =
          exploration_path_ns::NodeType::LOCAL_PATH_START;
    }
  }

  return full_path;
}

bool SensorCoveragePlanner3D::GetLookAheadPoint(
    const exploration_path_ns::ExplorationPath &local_path,
    const exploration_path_ns::ExplorationPath &global_path,
    Eigen::Vector3d &lookahead_point) {
  Eigen::Vector3d robot_position(robot_position_.x, robot_position_.y,
                                 robot_position_.z);

  // Determine which direction to follow on the global path
  double dist_from_start = 0.0;
  for (int i = 1; i < global_path.nodes_.size(); i++) {
    dist_from_start +=
        (global_path.nodes_[i - 1].position_ - global_path.nodes_[i].position_)
            .norm();
    if (global_path.nodes_[i].type_ ==
        exploration_path_ns::NodeType::GLOBAL_VIEWPOINT) {
      break;
    }
  }

  double dist_from_end = 0.0;
  for (int i = global_path.nodes_.size() - 2; i > 0; i--) {
    dist_from_end +=
        (global_path.nodes_[i + 1].position_ - global_path.nodes_[i].position_)
            .norm();
    if (global_path.nodes_[i].type_ ==
        exploration_path_ns::NodeType::GLOBAL_VIEWPOINT) {
      break;
    }
  }

  bool local_path_too_short = true;
  for (int i = 0; i < local_path.nodes_.size(); i++) {
    double dist_to_robot =
        (robot_position - local_path.nodes_[i].position_).norm();
    if (dist_to_robot > kLookAheadDistance / 5) {
      local_path_too_short = false;
      break;
    }
  }
  if (local_path.GetNodeNum() < 1 || local_path_too_short) {
    if (dist_from_start < dist_from_end) {
      double dist_from_robot = 0.0;
      for (int i = 1; i < global_path.nodes_.size(); i++) {
        dist_from_robot += (global_path.nodes_[i - 1].position_ -
                            global_path.nodes_[i].position_)
                               .norm();
        if (dist_from_robot > kLookAheadDistance / 2) {
          lookahead_point = global_path.nodes_[i].position_;
          break;
        }
      }
    } else {
      double dist_from_robot = 0.0;
      for (int i = global_path.nodes_.size() - 2; i > 0; i--) {
        dist_from_robot += (global_path.nodes_[i + 1].position_ -
                            global_path.nodes_[i].position_)
                               .norm();
        if (dist_from_robot > kLookAheadDistance / 2) {
          lookahead_point = global_path.nodes_[i].position_;
          break;
        }
      }
    }
    return false;
  }

  bool has_lookahead = false;
  bool dir = true;
  int robot_i = 0;
  int lookahead_i = 0;
  for (int i = 0; i < local_path.nodes_.size(); i++) {
    if (local_path.nodes_[i].type_ == exploration_path_ns::NodeType::ROBOT) {
      robot_i = i;
    }
    if (local_path.nodes_[i].type_ ==
        exploration_path_ns::NodeType::LOOKAHEAD_POINT) {
      has_lookahead = true;
      lookahead_i = i;
    }
  }

  if (reset_waypoint_) {
    has_lookahead = false;
  }

  int forward_viewpoint_count = 0;
  int backward_viewpoint_count = 0;

  bool local_loop = false;
  if (local_path.nodes_.front() == local_path.nodes_.back() &&
      local_path.nodes_.front().type_ == exploration_path_ns::NodeType::ROBOT) {
    local_loop = true;
  }

  if (local_loop) {
    robot_i = 0;
  }
  for (int i = robot_i + 1; i < local_path.GetNodeNum(); i++) {
    if (local_path.nodes_[i].type_ ==
        exploration_path_ns::NodeType::LOCAL_VIEWPOINT) {
      forward_viewpoint_count++;
    }
  }
  if (local_loop) {
    robot_i = local_path.nodes_.size() - 1;
  }
  for (int i = robot_i - 1; i >= 0; i--) {
    if (local_path.nodes_[i].type_ ==
        exploration_path_ns::NodeType::LOCAL_VIEWPOINT) {
      backward_viewpoint_count++;
    }
  }

  Eigen::Vector3d forward_lookahead_point = robot_position;
  Eigen::Vector3d backward_lookahead_point = robot_position;

  bool has_forward = false;
  bool has_backward = false;

  if (local_loop) {
    robot_i = 0;
  }
  bool forward_lookahead_point_in_los = true;
  bool backward_lookahead_point_in_los = true;
  double length_from_robot = 0.0;
  for (int i = robot_i + 1; i < local_path.GetNodeNum(); i++) {
    length_from_robot +=
        (local_path.nodes_[i].position_ - local_path.nodes_[i - 1].position_)
            .norm();
    double dist_to_robot =
        (local_path.nodes_[i].position_ - robot_position).norm();
    bool in_line_of_sight = true;
    if (i < local_path.GetNodeNum() - 1) {
      in_line_of_sight = viewpoint_manager_->InCurrentFrameLineOfSight(
          local_path.nodes_[i + 1].position_);
    }
    if ((length_from_robot > kLookAheadDistance ||
         (kUseLineOfSightLookAheadPoint && !in_line_of_sight) ||
         local_path.nodes_[i].type_ ==
             exploration_path_ns::NodeType::LOCAL_VIEWPOINT ||
         local_path.nodes_[i].type_ ==
             exploration_path_ns::NodeType::LOCAL_PATH_START ||
         local_path.nodes_[i].type_ ==
             exploration_path_ns::NodeType::LOCAL_PATH_END ||
         i == local_path.GetNodeNum() - 1))

    {
      if (kUseLineOfSightLookAheadPoint && !in_line_of_sight) {
        forward_lookahead_point_in_los = false;
      }
      forward_lookahead_point = local_path.nodes_[i].position_;
      has_forward = true;
      break;
    }
  }
  if (local_loop) {
    robot_i = local_path.nodes_.size() - 1;
  }
  length_from_robot = 0.0;
  for (int i = robot_i - 1; i >= 0; i--) {
    length_from_robot +=
        (local_path.nodes_[i].position_ - local_path.nodes_[i + 1].position_)
            .norm();
    double dist_to_robot =
        (local_path.nodes_[i].position_ - robot_position).norm();
    bool in_line_of_sight = true;
    if (i > 0) {
      in_line_of_sight = viewpoint_manager_->InCurrentFrameLineOfSight(
          local_path.nodes_[i - 1].position_);
    }
    if ((length_from_robot > kLookAheadDistance ||
         (kUseLineOfSightLookAheadPoint && !in_line_of_sight) ||
         local_path.nodes_[i].type_ ==
             exploration_path_ns::NodeType::LOCAL_VIEWPOINT ||
         local_path.nodes_[i].type_ ==
             exploration_path_ns::NodeType::LOCAL_PATH_START ||
         local_path.nodes_[i].type_ ==
             exploration_path_ns::NodeType::LOCAL_PATH_END ||
         i == 0))

    {
      if (kUseLineOfSightLookAheadPoint && !in_line_of_sight) {
        backward_lookahead_point_in_los = false;
      }
      backward_lookahead_point = local_path.nodes_[i].position_;
      has_backward = true;
      break;
    }
  }

  if (forward_viewpoint_count > 0 && !has_forward) {
    std::cout << "forward viewpoint count > 0 but does not have forward "
                 "lookahead point"
              << std::endl;
    exit(1);
  }
  if (backward_viewpoint_count > 0 && !has_backward) {
    std::cout << "backward viewpoint count > 0 but does not have backward "
                 "lookahead point"
              << std::endl;
    exit(1);
  }

  double dx = lookahead_point_direction_.x();
  double dy = lookahead_point_direction_.y();

  if (reset_waypoint_) {
    reset_waypoint_ = false;
    double lx = 1.0;
    double ly = 0.0;

    dx = cos(robot_yaw_) * lx - sin(robot_yaw_) * ly;
    dy = sin(robot_yaw_) * lx + cos(robot_yaw_) * ly;
  }

  double forward_angle_score = -2;
  double backward_angle_score = -2;
  double lookahead_angle_score = -2;

  double dist_robot_to_lookahead = 0.0;
  if (has_forward) {
    Eigen::Vector3d forward_diff = forward_lookahead_point - robot_position;
    forward_diff.z() = 0.0;
    forward_diff = forward_diff.normalized();
    forward_angle_score = dx * forward_diff.x() + dy * forward_diff.y();
  }
  if (has_backward) {
    Eigen::Vector3d backward_diff = backward_lookahead_point - robot_position;
    backward_diff.z() = 0.0;
    backward_diff = backward_diff.normalized();
    backward_angle_score = dx * backward_diff.x() + dy * backward_diff.y();
  }
  if (has_lookahead) {
    Eigen::Vector3d prev_lookahead_point =
        local_path.nodes_[lookahead_i].position_;
    dist_robot_to_lookahead = (robot_position - prev_lookahead_point).norm();
    Eigen::Vector3d diff = prev_lookahead_point - robot_position;
    diff.z() = 0.0;
    diff = diff.normalized();
    lookahead_angle_score = dx * diff.x() + dy * diff.y();
  }

  lookahead_point_cloud_->cloud_->clear();

  if (forward_viewpoint_count == 0 && backward_viewpoint_count == 0) {
    relocation_ = true;
  } else {
    relocation_ = false;
  }
  if (relocation_) {
    if (use_momentum_ && kUseMomentum) {
      if (forward_angle_score > backward_angle_score) {
        lookahead_point = forward_lookahead_point;
      } else {
        lookahead_point = backward_lookahead_point;
      }
    } else {
      // follow the shorter distance one
      if (dist_from_start < dist_from_end &&
          local_path.nodes_.front().type_ !=
              exploration_path_ns::NodeType::ROBOT) {
        lookahead_point = backward_lookahead_point;
      } else if (dist_from_end < dist_from_start &&
                 local_path.nodes_.back().type_ !=
                     exploration_path_ns::NodeType::ROBOT) {
        lookahead_point = forward_lookahead_point;
      } else {
        lookahead_point = forward_angle_score > backward_angle_score
                              ? forward_lookahead_point
                              : backward_lookahead_point;
      }
    }
  } else if (has_lookahead && lookahead_angle_score > 0 &&
             dist_robot_to_lookahead > kLookAheadDistance / 2 &&
             viewpoint_manager_->InLocalPlanningHorizon(
                 local_path.nodes_[lookahead_i].position_))

  {
    lookahead_point = local_path.nodes_[lookahead_i].position_;
  } else {
    if (forward_angle_score > backward_angle_score) {
      if (forward_viewpoint_count > 0) {
        lookahead_point = forward_lookahead_point;
      } else {
        lookahead_point = backward_lookahead_point;
      }
    } else {
      if (backward_viewpoint_count > 0) {
        lookahead_point = backward_lookahead_point;
      } else {
        lookahead_point = forward_lookahead_point;
      }
    }
  }

  if ((lookahead_point == forward_lookahead_point &&
       !forward_lookahead_point_in_los) ||
      (lookahead_point == backward_lookahead_point &&
       !backward_lookahead_point_in_los)) {
    lookahead_point_in_line_of_sight_ = false;
  } else {
    lookahead_point_in_line_of_sight_ = true;
  }

  lookahead_point_direction_ = lookahead_point - robot_position;
  lookahead_point_direction_.z() = 0.0;
  lookahead_point_direction_.normalize();

  pcl::PointXYZI point;
  point.x = lookahead_point.x();
  point.y = lookahead_point.y();
  point.z = lookahead_point.z();
  point.intensity = 1.0;
  lookahead_point_cloud_->cloud_->points.push_back(point);

  if (has_lookahead) {
    point.x = local_path.nodes_[lookahead_i].position_.x();
    point.y = local_path.nodes_[lookahead_i].position_.y();
    point.z = local_path.nodes_[lookahead_i].position_.z();
    point.intensity = 0;
    lookahead_point_cloud_->cloud_->points.push_back(point);
  }
  return true;
}

void SensorCoveragePlanner3D::SendWaypoint(double x, double y) {
  // geometry_msgs/Pose2D on /way_point_with_heading — the challenge's waypoint
  // interface (README "System Inputs"). It carries no header, so it cannot go
  // through misc_utils_ns::Publish, which stamps one; the frame is implicitly
  // kWorldFrameID ("map"), the same frame every other output here uses.
  //
  // Heading is nominally ignored this year, and the system's waypoint_converter
  // ships yawConfig=-1 ("reach waypoint and stop"), which overwrites theta with the
  // vehicle's current yaw. Filling in the bearing to the waypoint anyway keeps the
  // message self-consistent and is what visualization_tools renders.
  geometry_msgs::msg::Pose2D waypoint;
  waypoint.x = x;
  waypoint.y = y;
  waypoint.theta = atan2(y - robot_position_.y, x - robot_position_.x);
  waypoint_pub_->publish(waypoint);
  // Every waypoint in the node goes out through here -- the planning round, the stall
  // watchdog and SendInitialWaypoint alike -- so this is the one place that can record the
  // topic's true age. Stamping it at any caller would drift the moment a new one is added.
  last_waypoint_publish_time_ = this->now().seconds();
}

void SensorCoveragePlanner3D::PublishWaypoint() {
  double waypoint_x, waypoint_y;
  if (exploration_finished_ && near_home_ && kRushHome) {
    waypoint_x = initial_position_.x();
    waypoint_y = initial_position_.y();
  } else {
    double dx = lookahead_point_.x() - robot_position_.x;
    double dy = lookahead_point_.y() - robot_position_.y;
    double r = sqrt(dx * dx + dy * dy);
    double extend_dist = lookahead_point_in_line_of_sight_
                             ? kExtendWayPointDistanceBig
                             : kExtendWayPointDistanceSmall;
    // The guard on r matters now that the stall watchdog re-sends on a timer: the robot can
    // be sitting ON its lookahead, where r -> 0 makes dx/r either NaN (r == 0 exactly) or a
    // full extend_dist throw in whatever direction odometry noise happened to point. Below
    // the threshold there is no meaningful bearing to extend along, so keep the robot's
    // heading and push the carrot out that way instead.
    if (r < extend_dist && kExtendWayPoint) {
      if (r > 1e-3) {
        dx = dx / r * extend_dist;
        dy = dy / r * extend_dist;
      } else {
        dx = cos(robot_yaw_) * extend_dist;
        dy = sin(robot_yaw_) * extend_dist;
      }
    }
    waypoint_x = dx + robot_position_.x;
    waypoint_y = dy + robot_position_.y;
  }
  // The z the upstream planner carried here (lookahead_point_.z()) is dropped by
  // Pose2D. Nothing consumed it: waypoint_converter reads x, y and theta only.
  SendWaypoint(waypoint_x, waypoint_y);
}

void SensorCoveragePlanner3D::PublishRuntime() {
  local_viewpoint_sampling_runtime_ =
      local_coverage_planner_->GetViewPointSamplingRuntime() / 1000;
  local_path_finding_runtime_ = (local_coverage_planner_->GetFindPathRuntime() +
                                 local_coverage_planner_->GetTSPRuntime()) /
                                1000;

  std_msgs::msg::Int32MultiArray runtime_breakdown_msg;
  runtime_breakdown_msg.data.clear();
  runtime_breakdown_msg.data.push_back(update_representation_runtime_);
  runtime_breakdown_msg.data.push_back(local_viewpoint_sampling_runtime_);
  runtime_breakdown_msg.data.push_back(local_path_finding_runtime_);
  runtime_breakdown_msg.data.push_back(global_planning_runtime_);
  runtime_breakdown_msg.data.push_back(trajectory_optimization_runtime_);
  runtime_breakdown_msg.data.push_back(overall_runtime_);
  runtime_breakdown_pub_->publish(runtime_breakdown_msg);

  float runtime = 0;
  if (!exploration_finished_ && kNoExplorationReturnHome) {
    for (int i = 0; i < runtime_breakdown_msg.data.size() - 1; i++) {
      runtime += runtime_breakdown_msg.data[i];
    }
  }

  std_msgs::msg::Float32 runtime_msg;
  runtime_msg.data = runtime / 1000.0;
  runtime_pub_->publish(runtime_msg);
}

double SensorCoveragePlanner3D::GetRobotToHomeDistance() {
  Eigen::Vector3d robot_position(robot_position_.x, robot_position_.y,
                                 robot_position_.z);
  return (robot_position - initial_position_).norm();
}

void SensorCoveragePlanner3D::PublishExplorationState() {
  std_msgs::msg::Bool exploration_finished_msg;
  exploration_finished_msg.data = exploration_finished_;
  exploration_finish_pub_->publish(exploration_finished_msg);
}

void SensorCoveragePlanner3D::PrintExplorationStatus(std::string status,
                                                     bool clear_last_line) {
  if (clear_last_line) {
    printf(cursup);
    printf(cursclean);
    printf(cursup);
    printf(cursclean);
  }
  std::cout << std::endl << "\033[1;32m" << status << "\033[0m" << std::endl;
}

void SensorCoveragePlanner3D::CountDirectionChange() {
  Eigen::Vector3d current_moving_direction_ =
      Eigen::Vector3d(robot_position_.x, robot_position_.y, robot_position_.z) -
      Eigen::Vector3d(last_robot_position_.x, last_robot_position_.y,
                      last_robot_position_.z);

  if (current_moving_direction_.norm() > 0.5) {
    if (moving_direction_.dot(current_moving_direction_) < 0) {
      direction_change_count_++;
      direction_no_change_count_ = 0;
      if (direction_change_count_ > kDirectionChangeCounterThr) {
        if (!use_momentum_) {
          momentum_activation_count_++;
        }
        use_momentum_ = true;
      }
    } else {
      direction_no_change_count_++;
      if (direction_no_change_count_ > kDirectionNoChangeCounterThr) {
        direction_change_count_ = 0;
        use_momentum_ = false;
      }
    }
    moving_direction_ = current_moving_direction_;
  }
  last_robot_position_ = robot_position_;

  std_msgs::msg::Int32 momentum_activation_count_msg;
  momentum_activation_count_msg.data = momentum_activation_count_;
  momentum_activation_count_pub_->publish(momentum_activation_count_msg);
}

void SensorCoveragePlanner3D::execute() {
  if (!kAutoStart && !start_exploration_) {
    RCLCPP_INFO(this->get_logger(), "Waiting for start signal");
    return;
  }
  Timer overall_processing_timer("overall processing");
  update_representation_runtime_ = 0;
  local_viewpoint_sampling_runtime_ = 0;
  local_path_finding_runtime_ = 0;
  global_planning_runtime_ = 0;
  trajectory_optimization_runtime_ = 0;
  overall_runtime_ = 0;

  if (!initialized_) {
    SendInitialWaypoint();
    start_time_ = this->now().seconds();
    if(start_time_ == 0.0){
      RCLCPP_ERROR(this->get_logger(), "Start time is zero, time source (use_time_time) not set correctly. Exiting...");
      exit(1);
    }
    global_direction_switch_time_ = this->now().seconds();
    initialized_ = true;
    return;
  }

  if (!has_registered_scan_) {
    RCLCPP_INFO_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "tare_planner: waiting for /registered_scan point cloud data...");
    start_time_ = this->now().seconds();
    stall_reference_time_ = -1.0;
    return;
  }

  // Before the planning block, so it still runs on the ticks that never reach
  // PublishWaypoint -- which are exactly the ticks that strand the robot.
  CheckWaypointStall();

  overall_processing_timer.Start();
  if (keypose_cloud_update_) {
    keypose_cloud_update_ = false;

    CountDirectionChange();

    misc_utils_ns::Timer update_representation_timer("update representation");
    update_representation_timer.Start();

    // Update grid world
    UpdateGlobalRepresentation();

    int viewpoint_candidate_count = UpdateViewPoints();
    if (viewpoint_candidate_count == 0) {
      RCLCPP_WARN(rclcpp::get_logger("standalone_logger"),
                  "Cannot get candidate viewpoints, skipping this round");
      return;
    }

    UpdateKeyposeGraph();

    // After UpdateViewPoints, which rebuilds the candidate set this snaps onto, and before
    // GlobalPlanning, whose UpdateCellStatus honours the priority cells it marks.
    UpdateTargetViewPoints();

    int uncovered_point_num = 0;
    int uncovered_frontier_point_num = 0;
    if (!exploration_finished_ || !kNoExplorationReturnHome) {
      UpdateViewPointCoverage();
      UpdateCoveredAreas(uncovered_point_num, uncovered_frontier_point_num);
    } else {
      viewpoint_manager_->ResetViewPointCoverage();
    }

    update_representation_timer.Stop(false);
    update_representation_runtime_ +=
        update_representation_timer.GetDuration("ms");

    // Global TSP
    std::vector<int> global_cell_tsp_order;
    exploration_path_ns::ExplorationPath global_path;
    GlobalPlanning(global_cell_tsp_order, global_path);

    // Local TSP
    exploration_path_ns::ExplorationPath local_path;
    LocalPlanning(uncovered_point_num, uncovered_frontier_point_num,
                  global_path, local_path);

    near_home_ = GetRobotToHomeDistance() < kRushHomeDist;
    at_home_ = GetRobotToHomeDistance() < kAtHomeDistThreshold;

    double current_time = this->now().seconds();
    double delta_time = current_time - start_time_;

    // Never declare exploration finished while targets are outstanding. The
    // IsLocalCoverageComplete() guard alone is not enough: it is held false by
    // HasTargetViewPoints(), which counts only *accepted* targets, so a robot whose targets
    // are all still `far` -- i.e. has not reached any of them yet -- sails straight through
    // it, parks at home, and burns the rest of the window. That is exactly what happened at
    // t+56.5s with nine goals pending.
    // Un-latch. `exploration_finished_` gates UpdateViewPointCoverage (the else branch above
    // resets coverage every tick) and, with kNoExplorationReturnHome, parks the robot at home
    // -- so once it latches, accepted target viewpoints are handed to a local planner that has
    // nothing left to cover and the robot never moves again. Measured: chinese_room latched at
    // t+16.5s with eight outstanding requests and sat still for the remaining 134s of its
    // window, and the stall watchdog could not report it because `stopped_` had switched the
    // watchdog off. Outstanding target work re-opens exploration.
    if ((exploration_finished_ || stopped_) && TargetWorkOutstanding()) {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 10000,
                           "%d target viewpoint(s) still outstanding -- resuming exploration",
                           static_cast<int>(target_viewpoint_positions_.size()));
      exploration_finished_ = false;
      stopped_ = false;
    }

    // `!TargetWorkOutstanding()` rather than `!TargetPreemptActive()`: preempt is a policy
    // flag (may targets restrict the global tour), and target_explorer holds it false while
    // any label is still undiscovered so the frontier tour is not starved. That left the
    // latch open in the one case it was written to close -- every request still `far` -- so
    // gate on the fact instead.
    if (has_registered_scan_ &&
        grid_world_->IsReturningHome() &&
        local_coverage_planner_->IsLocalCoverageComplete() &&
        !TargetWorkOutstanding() && (current_time - start_time_) > 5) {
      if (!exploration_finished_) {
        // PrintExplorationStatus("Exploration completed, returning home", false);
        // Distinct from smart_vlm's EXPLORATION END, which is the mission clock closing the
        // window. This is TARE's own verdict. Three things now keep it from landing while
        // targets are outstanding, and it took all three: HasTargetViewPoints() for accepted
        // targets, !TargetPreemptActive() above for ones still `far`, and the priority-cell
        // seeding in SolveGlobalTSP so return_home_ is not latched underneath us.
        RCLCPP_INFO(this->get_logger(),
                    "TARE EXPLORATION FINISHED at t+%.1fs (returning home)",
                    delta_time);
      }
      exploration_finished_ = true;
    }

    if (exploration_finished_ && at_home_ && !stopped_) {
      PrintExplorationStatus("Return home completed", false);
      stopped_ = true;
    }

    exploration_path_ = ConcatenateGlobalLocalPath(global_path, local_path);

    PublishExplorationState();

    lookahead_point_update_ =
        GetLookAheadPoint(exploration_path_, global_path, lookahead_point_);
    lookahead_point_valid_ = lookahead_point_valid_ || lookahead_point_update_;
    PublishWaypoint();

    overall_processing_timer.Stop(false);
    overall_runtime_ = overall_processing_timer.GetDuration("ms");

    visualizer_->GetGlobalSubspaceMarker(grid_world_, global_cell_tsp_order);
    Eigen::Vector3d viewpoint_origin = viewpoint_manager_->GetOrigin();
    visualizer_->GetLocalPlanningHorizonMarker(
        viewpoint_origin.x(), viewpoint_origin.y(), robot_position_.z);
    visualizer_->PublishMarkers();

    PublishLocalPlanningVisualization(local_path);
    PublishGlobalPlanningVisualization(global_path, local_path);
    PublishRuntime();
  }
}
} // namespace sensor_coverage_planner_3d_ns

/**
 * @file sensor_coverage_planner_ground.h
 * @author Chao Cao (ccao1@andrew.cmu.edu)
 * @brief Class that does the job of exploration
 * @version 0.1
 * @date 2020-06-03
 *
 * @copyright Copyright (c) 2021
 *
 */
#pragma once

#include <cmath>
#include <vector>

#include <Eigen/Core>
// ROS
#include <geometry_msgs/msg/point_stamped.hpp>
#include <geometry_msgs/msg/polygon_stamped.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joy.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/int32_multi_array.hpp>
#include <tf2/transform_datatypes.h>
// PCL
#include <pcl/PointIndices.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl_conversions/pcl_conversions.h>
// Third parties
#include <utils/misc_utils.h>
#include <utils/pointcloud_utils.h>
// Components
#include "exploration_path/exploration_path.h"
#include "grid_world/grid_world.h"
#include "keypose_graph/keypose_graph.h"
#include "local_coverage_planner/local_coverage_planner.h"
#include "planning_env/planning_env.h"
#include "rolling_occupancy_grid/rolling_occupancy_grid.h"
#include "tare_visualizer/tare_visualizer.h"
#include "viewpoint_manager/viewpoint_manager.h"

#define cursup "\033[A"
#define cursclean "\033[2K"
#define curshome "\033[0;0H"

namespace sensor_coverage_planner_3d_ns {
const std::string kWorldFrameID = "map";
typedef pcl::PointXYZRGBNormal PlannerCloudPointType;
typedef pcl::PointCloud<PlannerCloudPointType> PlannerCloudType;
typedef misc_utils_ns::Timer Timer;

class SensorCoveragePlanner3D : public rclcpp::Node {
public:
  explicit SensorCoveragePlanner3D();
  bool initialize();
  void execute();
  ~SensorCoveragePlanner3D() = default;

private:
  // Parameters
  // String
  std::string sub_start_exploration_topic_;
  std::string sub_keypose_topic_;
  std::string sub_state_estimation_topic_;
  std::string sub_registered_scan_topic_;
  std::string sub_terrain_map_topic_;
  std::string sub_terrain_map_ext_topic_;
  std::string sub_coverage_boundary_topic_;
  std::string sub_viewpoint_boundary_topic_;
  std::string sub_nogo_boundary_topic_;
  std::string sub_joystick_topic_;
  std::string sub_reset_waypoint_topic_;
  // Semantic steering input: places the perception side wants the robot to stand, so an
  // under-observed target object gets seen from a direction it has not been seen from.
  std::string sub_target_viewpoints_topic_;
  // Which requested standing positions were placed and which the local horizon refused, so
  // the semantic side can tell "not there yet" from "nothing can stand there".
  std::string pub_target_feedback_topic_;
  // Bool: may targets preempt frontier exploration right now. Owned by target_explorer,
  // which is the only thing that knows whether every target label has been found yet.
  std::string sub_target_preempt_topic_;

  std::string pub_exploration_finish_topic_;
  std::string pub_runtime_breakdown_topic_;
  std::string pub_runtime_topic_;
  std::string pub_waypoint_topic_;
  std::string pub_momentum_activation_count_topic_;

  // Bool
  bool kAutoStart;
  bool kRushHome;
  bool kUseTerrainHeight;
  bool kCheckTerrainCollision;
  bool kExtendWayPoint;
  bool kUseLineOfSightLookAheadPoint;
  bool kNoExplorationReturnHome;
  bool kUseMomentum;
  bool kUseTargetViewPoints;
  bool kUseWaypointStallWatchdog;

  // Double
  double kKeyposeCloudDwzFilterLeafSize;
  double kRushHomeDist;
  double kAtHomeDistThreshold;
  double kTerrainCollisionThreshold;
  double kLookAheadDistance;
  double kExtendWayPointDistanceBig;
  double kExtendWayPointDistanceSmall;
  // Poses older than this are ignored, so a dead target_explorer reverts TARE to stock
  // coverage instead of pinning it to the last thing it happened to ask for.
  double kTargetViewPointTimeout;
  // A request is honoured only if a real candidate viewpoint sits within this of it. This
  // is what makes an unreachable request (a bin pointing into the wall behind a window)
  // cost nothing: no candidate is near it, so it is simply dropped.
  double kTargetViewPointSnapMaxDist;
  // Stall watchdog. PublishWaypoint() only runs inside `if (keypose_cloud_update_)`, which
  // is set on every 5th registered scan and which two paths return out of before reaching
  // it -- and waypoint_converter ships yawConfig -1 ("reach waypoint and stop"), so a
  // skipped round leaves the robot coasting to a halt at a waypoint nobody will replace.
  double kWaypointStallTimeout;
  double kWaypointStallEscalateTimeout;
  double kStallProgressDist;
  double kStallBlacklistRadius;

  // Int
  int kDirectionChangeCounterThr;
  int kDirectionNoChangeCounterThr;
  int kResetWaypointJoystickAxesID;
  // Kept small on purpose: must-visit viewpoints have their covered points marked covered
  // before EnqueueViewpointCandidates runs, so each one slightly depresses the marginal
  // gain of genuine coverage viewpoints. A few tens would suppress frontier selection.
  int kMaxTargetViewPointNum;
  int kMaxStallBlacklistNum;

  std::shared_ptr<pointcloud_utils_ns::PCLCloud<PlannerCloudPointType>>
      keypose_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZ>>
      registered_scan_stack_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      registered_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      large_terrain_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      terrain_collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      terrain_ext_collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      viewpoint_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      grid_world_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      selected_viewpoint_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      exploring_cell_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      exploration_path_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      lookahead_point_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      keypose_graph_vis_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      viewpoint_in_collision_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      point_cloud_manager_neighbor_cloud_;
  std::shared_ptr<pointcloud_utils_ns::PCLCloud<pcl::PointXYZI>>
      reordered_global_subspace_cloud_;

  nav_msgs::msg::Odometry keypose_;
  geometry_msgs::msg::Point robot_position_;
  geometry_msgs::msg::Point last_robot_position_;
  lidar_model_ns::LiDARModel robot_viewpoint_;
  exploration_path_ns::ExplorationPath exploration_path_;
  Eigen::Vector3d lookahead_point_;
  Eigen::Vector3d lookahead_point_direction_;
  Eigen::Vector3d moving_direction_;
  double robot_yaw_;
  bool moving_forward_;
  std::vector<Eigen::Vector3d> visited_positions_;
  int cur_keypose_node_ind_;
  Eigen::Vector3d initial_position_;

  std::shared_ptr<keypose_graph_ns::KeyposeGraph> keypose_graph_;
  std::shared_ptr<planning_env_ns::PlanningEnv> planning_env_;
  std::shared_ptr<viewpoint_manager_ns::ViewPointManager> viewpoint_manager_;
  std::shared_ptr<local_coverage_planner_ns::LocalCoveragePlanner>
      local_coverage_planner_;
  std::shared_ptr<grid_world_ns::GridWorld> grid_world_;
  std::shared_ptr<tare_visualizer_ns::TAREVisualizer> visualizer_;

  std::shared_ptr<misc_utils_ns::Marker> keypose_graph_node_marker_;
  std::shared_ptr<misc_utils_ns::Marker> keypose_graph_edge_marker_;
  std::shared_ptr<misc_utils_ns::Marker> nogo_boundary_marker_;
  std::shared_ptr<misc_utils_ns::Marker> grid_world_marker_;

  bool keypose_cloud_update_;
  bool initialized_;
  bool has_registered_scan_;
  bool lookahead_point_update_;
  bool relocation_;
  bool start_exploration_;
  bool exploration_finished_;
  bool near_home_;
  bool at_home_;
  bool stopped_;
  bool test_point_update_;
  bool viewpoint_ind_update_;
  bool step_;
  bool use_momentum_;
  bool lookahead_point_in_line_of_sight_;
  bool reset_waypoint_;
  // Requested standing positions and when they arrived, in map frame.
  std::vector<Eigen::Vector3d> target_viewpoint_positions_;
  double target_viewpoints_receive_time_;
  // Where the robot was when it last made real progress, and how long ago that was.
  Eigen::Vector3d stall_reference_position_;
  double stall_reference_time_;
  // When a waypoint last actually went out on the wire. Stamped in SendWaypoint(), which
  // every publisher funnels through, so this cannot disagree with what the topic saw.
  // Distinct from stall_reference_time_ on purpose: that one tracks the ROBOT, this one
  // tracks the TOPIC, and a robot still coasting toward a stale goal keeps the first fresh
  // while the second ages without limit.
  double last_waypoint_publish_time_;
  // How many times the freshness half has had to re-send. Reported, not acted on: a run with
  // a high count had a planning round that kept failing to publish, which is a different bug
  // from the robot being physically stuck, and the two were previously indistinguishable in
  // the logs because this half was silent.
  int waypoint_refresh_count_;
  // Latches once any planning round produces a lookahead. Distinct from
  // lookahead_point_update_, which is re-derived every round and goes false again whenever
  // a round finds none -- which is exactly when the watchdog most wants to re-send the last
  // good one. Before this latches, lookahead_point_ is uninitialised Eigen memory.
  bool lookahead_point_valid_;
  // Latest value of /exploration/target_preempt, and when it arrived. Stale means stock.
  bool target_preempt_;
  double target_preempt_receive_time_;
  // The target the lookahead is currently pinned to. Held so the stall watchdog can retire
  // the TARGET when the robot cannot reach it -- blacklisting the waypoint alone leaves the
  // pin in place and the robot re-adopts the same unreachable target every cycle.
  Eigen::Vector3d preempt_target_position_;
  bool preempt_target_valid_;
  // Places the robot demonstrably could not reach. Kept as world positions, not viewpoint
  // indices, because the viewpoint lattice rolls with the robot and indices do not survive.
  std::vector<Eigen::Vector3d> stall_blacklist_;
  // Target viewpoints TARE actually snapped this cycle, so the lookahead can be pointed at
  // the nearest one. Rebuilt every cycle alongside the local planner's copy.
  std::vector<int> accepted_target_viewpoint_indices_;
  pointcloud_utils_ns::PointCloudDownsizer<pcl::PointXYZ> pointcloud_downsizer_;

  int update_representation_runtime_;
  int local_viewpoint_sampling_runtime_;
  int local_path_finding_runtime_;
  int global_planning_runtime_;
  int trajectory_optimization_runtime_;
  int overall_runtime_;
  int registered_cloud_count_;
  int keypose_count_;
  int direction_change_count_;
  int direction_no_change_count_;
  int momentum_activation_count_;

  double start_time_;
  double global_direction_switch_time_;
  double reset_waypoint_joystick_axis_value_;

  rclcpp::TimerBase::SharedPtr execution_timer_;

  // ROS subscribers
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr exploration_start_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      registered_scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      terrain_map_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
      terrain_map_ext_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr
      state_estimation_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr
      coverage_boundary_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr
      viewpoint_boundary_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PolygonStamped>::SharedPtr
      nogo_boundary_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr joystick_sub_;
  rclcpp::Subscription<std_msgs::msg::Empty>::SharedPtr reset_waypoint_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr
      target_viewpoints_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr target_feedback_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr target_preempt_sub_;

  // ROS publishers
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_full_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr global_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr old_global_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr
      to_nearest_global_subspace_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_tsp_path_publisher_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr exploration_path_publisher_;
  // Pose2D on /way_point_with_heading: the only drive output the challenge
  // firewall relays out of the AI module's domain (challenge_topics_bridge.yaml).
  // Upstream published PointStamped on /way_point, which is the *system's* topic,
  // written by waypoint_converter downstream of us.
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr waypoint_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr exploration_finish_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32MultiArray>::SharedPtr
      runtime_breakdown_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr runtime_pub_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr
      momentum_activation_count_pub_;
  // Debug
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr
      pointcloud_manager_neighbor_cells_origin_pub_;

  void ReadParameters();
  void InitializeData();

  // Callback functions
  void
  ExplorationStartCallback(const std_msgs::msg::Bool::ConstSharedPtr start_msg);
  void StateEstimationCallback(
      const nav_msgs::msg::Odometry::ConstSharedPtr state_estimation_msg);
  void RegisteredScanCallback(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr registered_cloud_msg);
  void TerrainMapCallback(
      const sensor_msgs::msg::PointCloud2::ConstSharedPtr terrain_map_msg);
  void TerrainMapExtCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr
                                 terrain_cloud_large_msg);
  void CoverageBoundaryCallback(
      const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg);
  void ViewPointBoundaryCallback(
      const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg);
  void NogoBoundaryCallback(
      const geometry_msgs::msg::PolygonStamped::ConstSharedPtr polygon_msg);
  void JoystickCallback(const sensor_msgs::msg::Joy::ConstSharedPtr joy_msg);
  void
  ResetWaypointCallback(const std_msgs::msg::Empty::ConstSharedPtr empty_msg);
  void TargetViewPointsCallback(
      const geometry_msgs::msg::PoseArray::ConstSharedPtr pose_array_msg);
  // Snap the requested positions onto real candidate viewpoints and hand them to the local
  // planner as must-visit. Returns the cells of the requests that fell outside the local
  // horizon, so the global layer can keep those subspaces EXPLORING.
  void UpdateTargetViewPoints();
  void TargetPreemptCallback(const std_msgs::msg::Bool::ConstSharedPtr preempt_msg);
  /** Is target preemption active AND fresh? Stale or false leaves TARE bit-for-bit stock. */
  bool TargetPreemptActive() const;
  /** Does the semantic layer still have places it wants visited?
   *
   * A FACT about outstanding work, unlike TargetPreemptActive() which is a policy flag about
   * whether targets may restrict the global tour. The two diverge exactly where it hurt: a
   * run whose requests are all still `far` has preempt off yet eight outstanding requests, so
   * the finish latch sailed through and the robot parked at home for the rest of the window.
   */
  bool TargetWorkOutstanding() const;
  /** Which accepted target to drive at, or -1. Order, not distance -- see the definition. */
  int PreferredTargetViewPointInd() const;
  // Runs on every tick, including the ones that never reach PublishWaypoint.
  void CheckWaypointStall();
  bool IsStallBlacklisted(const Eigen::Vector3d &position) const;

  void SendInitialWaypoint();
  void UpdateKeyposeGraph();
  int UpdateViewPoints();
  void UpdateViewPointCoverage();
  void UpdateRobotViewPointCoverage();
  void UpdateCoveredAreas(int &uncovered_point_num,
                          int &uncovered_frontier_point_num);
  void UpdateVisitedPositions();
  void UpdateGlobalRepresentation();
  void GlobalPlanning(std::vector<int> &global_cell_tsp_order,
                      exploration_path_ns::ExplorationPath &global_path);
  void PublishGlobalPlanningVisualization(
      const exploration_path_ns::ExplorationPath &global_path,
      const exploration_path_ns::ExplorationPath &local_path);
  void LocalPlanning(int uncovered_point_num, int uncovered_frontier_point_num,
                     const exploration_path_ns::ExplorationPath &global_path,
                     exploration_path_ns::ExplorationPath &local_path);
  void PublishLocalPlanningVisualization(
      const exploration_path_ns::ExplorationPath &local_path);
  exploration_path_ns::ExplorationPath ConcatenateGlobalLocalPath(
      const exploration_path_ns::ExplorationPath &global_path,
      const exploration_path_ns::ExplorationPath &local_path);

  void PublishRuntime();
  double GetRobotToHomeDistance();
  void PublishExplorationState();
  void PublishWaypoint();
  // Single funnel for every waypoint this node emits, so the message type and the
  // heading convention live in exactly one place.
  void SendWaypoint(double x, double y);
  bool
  GetLookAheadPoint(const exploration_path_ns::ExplorationPath &local_path,
                    const exploration_path_ns::ExplorationPath &global_path,
                    Eigen::Vector3d &lookahead_point);

  void PrintExplorationStatus(std::string status, bool clear_last_line = true);
  void CountDirectionChange();
};

} // namespace sensor_coverage_planner_3d_ns

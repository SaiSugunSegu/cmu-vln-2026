# Common ops — run from the repo's docker/ folder:  make -f ../scripts/Makefile <target>

# X display the simulator renders into. :1 is the Xvnc server on the dev box;
# override with `make sim SIM_DISPLAY=:0` if you have a real X session. (Named
# SIM_DISPLAY rather than DISPLAY so an inherited DISPLAY can't shadow it.)
SIM_DISPLAY ?= :1

# Challenge sim runs the simulator in domain 42; plain system_simulation.sh
# stays in domain 0. Override with `make foxglove DOMAIN=0`.
DOMAIN ?= 42

SYS_RUN = docker exec -it -e DISPLAY=$(SIM_DISPLAY) iros2026_system bash -c
# vglrun -d egl puts Unity on the GPU; without it Unity falls back to llvmpipe.
VGL = cd /home/docker/autonomy_stack_mecanum_wheel_platform && vglrun -d egl

up:            ## build + start both containers (GPU) using pure Docker image
	docker compose -f compose_gpu.yml up --build -d

down:
	docker compose -f compose_gpu.yml down

sim:           ## launch simulator + base autonomy + rviz2 (blocks; terminal A)
	$(SYS_RUN) "$(VGL) ./system_simulation.sh"

sim-noviz:     ## same as sim but without rviz2 (use with Foxglove)
	$(SYS_RUN) "$(VGL) ./system_simulation_noviz.sh"

challenge:     ## launch sim behind the 6-topic domain firewall, no rviz2
	$(SYS_RUN) "$(VGL) ./challenge_simulation.sh --noviz"

ai:            ## launch smart_vlm (blocks; terminal B)
	docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm smart_vlm.launch"

# make ask q="How many books are on the sofa"
ask:
	docker exec -it iros2026_ai_module bash -c "ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: '$(q)'}\""

foxglove:      ## start Foxglove bridge on host port 8765 (blocks; tunnel from laptop)
	docker exec -it -e ROS_DOMAIN_ID=$(DOMAIN) iros2026_ai_module bash -c "source /opt/ros/jazzy/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765 address:=0.0.0.0"

topics:        ## sanity-check input topic rates
	docker exec -it iros2026_ai_module bash -c "ros2 topic hz /camera/image & ros2 topic hz /registered_scan & ros2 topic hz /state_estimation & sleep 12; kill %1 %2 %3"

# make bag name=scene01_run1
bag:
	docker exec -it iros2026_ai_module bash -c "ros2 bag record /camera/image /registered_scan /sensor_scan /terrain_map /terrain_map_ext /state_estimation /challenge_question -o /data/bags/$(name)"

shell-sys:
	docker exec -it iros2026_system bash

shell-ai:
	docker exec -it iros2026_ai_module bash

.PHONY: up down sim sim-noviz challenge ai ask foxglove topics bag shell-sys shell-ai

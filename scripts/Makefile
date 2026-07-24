# Common ops — run from the repo's docker/ folder:  make -f ../scripts/Makefile <target>

up:            ## build + start both containers (GPU) using pure Docker image
	docker compose -f compose_gpu.yml up --build -d

down:
	docker compose -f compose_gpu.yml down

sim:           ## launch simulator + base autonomy (blocks; terminal A)
	docker exec -it iros2026_system bash -c "/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh"

ai:            ## launch smart_vlm (blocks; terminal B)
	docker exec -it iros2026_ai_module bash -c "source /home/docker/ai_module/install/setup.bash && ros2 launch smart_vlm smart_vlm.launch"

# make ask q="How many books are on the sofa"
ask:
	docker exec -it iros2026_ai_module bash -c "ros2 topic pub --once /challenge_question std_msgs/msg/String \"{data: '$(q)'}\""

topics:        ## sanity-check input topic rates
	docker exec -it iros2026_ai_module bash -c "ros2 topic hz /camera/image & ros2 topic hz /registered_scan & ros2 topic hz /state_estimation & sleep 12; kill %1 %2 %3"

# make bag name=scene01_run1
bag:
	docker exec -it iros2026_ai_module bash -c "ros2 bag record /camera/image /registered_scan /sensor_scan /terrain_map /terrain_map_ext /state_estimation /challenge_question -o /data/bags/$(name)"

shell-sys:
	docker exec -it iros2026_system bash

shell-ai:
	docker exec -it iros2026_ai_module bash

.PHONY: up down sim ai ask topics bag shell-sys shell-ai
